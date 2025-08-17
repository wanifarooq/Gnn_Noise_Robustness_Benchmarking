import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv, GINConv, GATConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, negative_sampling
import time

from model.gnns import GCN, GIN, GAT, GAT2

class EstimateAdj(nn.Module):
    def __init__(self, nfea, args, idx_train, device='cuda'):
        super().__init__()
        self.estimator = GCN(nfea, args['edge_hidden'], args['edge_hidden'], dropout=0.0)
        self.device = device
        self.args = args

    def forward(self, features, edge_index):
        data = Data(x=features, edge_index=edge_index)
        return self.estimator(data)

    def get_estimated_weights(self, edge_index, representations):
        if edge_index.numel() == 0:
            return torch.empty(0, device=self.device)
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        estimated_weights = torch.sigmoid((x0 * x1).sum(dim=1))
        estimated_weights = torch.where(
            estimated_weights < self.args['t_small'],
            torch.tensor(0.0, device=self.device),
            estimated_weights
        )
        return estimated_weights
    
    def reconstruct_loss(self, edge_index, representations):
        num_nodes = representations.size(0)
        if edge_index.numel() == 0:
            return torch.tensor(0.0, device=self.device)
    
        neg_samples = negative_sampling(
            edge_index,
            num_nodes=num_nodes,
            num_neg_samples=min(self.args['n_n'] * num_nodes, edge_index.size(1))
        )
    
        mask_pos = edge_index[0] < edge_index[1]
        mask_neg = neg_samples[0] < neg_samples[1]
    
        pos_edge_index = edge_index[:, mask_pos]
        neg_edge_index = neg_samples[:, mask_neg]
    
        if pos_edge_index.size(1) == 0 or neg_edge_index.size(1) == 0:
            return torch.tensor(0.0, device=self.device)
    
        pos = torch.sigmoid((representations[pos_edge_index[0]] * representations[pos_edge_index[1]]).sum(dim=1))
        neg = torch.sigmoid((representations[neg_edge_index[0]] * representations[neg_edge_index[1]]).sum(dim=1))
    
        loss_pos = F.mse_loss(pos, torch.ones_like(pos), reduction='mean')
        loss_neg = F.mse_loss(neg, torch.zeros_like(neg), reduction='mean')
    
        rec_loss = 0.5 * (loss_pos + loss_neg)
        return rec_loss


class NRGNN:
    def __init__(self, args, device, gnn_type='GCN'):
        self.device = device
        self.args = args
        self.gnn_type = gnn_type
        self.best_val_acc = 0.0
        self.best_acc_pred_val = 0.0
        self.best_pred = None
        self.weights = None
        self.predictor_weights = None

    def fit(self, features, adj, labels, idx_train, idx_val=None):
        edge_index, _ = from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        if sp.issparse(features):
            features = torch.tensor(features.todense(), dtype=torch.float32, device=self.device)
        else:
            features = features.float().to(self.device)

        if isinstance(labels, torch.Tensor):
            labels = labels.to(torch.long).to(self.device)
        else:
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        self.features = features
        self.labels = labels
        self.edge_index = edge_index

        if idx_val is None:
            if isinstance(idx_train, torch.Tensor):
                idx_train_list = idx_train.detach().cpu().tolist()
            else:
                idx_train_list = list(idx_train)
            val_size = max(1, len(idx_train_list) // 5)
            rng = np.random.default_rng(42)
            val_indices = rng.choice(idx_train_list, val_size, replace=False)
            idx_val = torch.tensor(val_indices, dtype=torch.long, device=self.device)
            idx_train = torch.tensor([i for i in idx_train_list if i not in set(val_indices.tolist())],
                                     dtype=torch.long, device=self.device)
        else:
            if not isinstance(idx_val, torch.Tensor):
                idx_val = torch.tensor(idx_val, dtype=torch.long, device=self.device)
            else:
                idx_val = idx_val.to(self.device)
            if not isinstance(idx_train, torch.Tensor):
                idx_train = torch.tensor(idx_train, dtype=torch.long, device=self.device)
            else:
                idx_train = idx_train.to(self.device)

        self.idx_unlabel = torch.tensor(
            list(set(range(features.size(0))) - set(idx_train.detach().cpu().tolist())),
            dtype=torch.long, device=self.device
        )

        hidden_dim = self.args.get('hidden_channels', 64)
        dropout = self.args.get('dropout', 0.5)
        out_dim = int(labels.max().item()) + 1

        if self.gnn_type == 'GCN':
            self.predictor = GCN(features.size(1), hidden_dim, out_dim, dropout=dropout, self_loop=True).to(self.device)
            self.model = GCN(features.size(1), hidden_dim, out_dim, dropout=dropout, self_loop=True).to(self.device)
        elif self.gnn_type == 'GIN':
            self.predictor = GIN(features.size(1), hidden_dim, out_dim, dropout=dropout).to(self.device)
            self.model = GIN(features.size(1), hidden_dim, out_dim, dropout=dropout).to(self.device)
        elif self.gnn_type == 'GAT':
            self.predictor = GAT(features.size(1), hidden_dim, out_dim, dropout=dropout).to(self.device)
            self.model = GAT(features.size(1), hidden_dim, out_dim, dropout=dropout).to(self.device)
        elif self.gnn_type == 'GAT2':
            self.predictor = GAT2(features.size(1), hidden_dim, out_dim, dropout=dropout).to(self.device)
            self.model = GAT2(features.size(1), hidden_dim, out_dim, dropout=dropout).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.gnn_type}")

        self.estimator = EstimateAdj(features.size(1), self.args, idx_train, device=self.device).to(self.device)
        self.pred_edge_index = self.get_train_edge(edge_index, features, self.args.get('n_p', 5), idx_train)

        params = list(self.model.parameters()) + list(self.estimator.parameters()) + list(self.predictor.parameters())
        self.optimizer = optim.Adam(params, lr=self.args.get('lr', 0.01), weight_decay=self.args.get('weight_decay', 5e-4))

        t_total = time.time()
        for epoch in range(self.args.get('epochs', 200)):
            self.train_epoch(epoch, features, edge_index, idx_train, idx_val)

            if (epoch + 1) % 10 == 0:
                acc_val = self.evaluate(idx_val)
                print(f"Epoch {epoch+1:03d} | Val Acc: {acc_val:.4f} | Best: {self.best_val_acc:.4f}")

        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        if self.predictor_weights is not None:
            self.predictor.load_state_dict(self.predictor_weights)

        return self.test(idx_val)

    def train_epoch(self, epoch, features, edge_index, idx_train, idx_val):
        self.model.train()
        self.predictor.train()
        self.optimizer.zero_grad()

        representations = self.estimator(features, edge_index)
        rec_loss = self.estimator.reconstruct_loss(edge_index, representations)

        predictor_weights = self.estimator.get_estimated_weights(self.pred_edge_index, representations)
        pred_edge_index = torch.cat([edge_index, self.pred_edge_index], dim=1)
        predictor_edge_weights = torch.cat([
            torch.ones(edge_index.size(1), device=self.device),
            predictor_weights
        ], dim=0)

        pred_data = Data(x=features, edge_index=pred_edge_index, edge_weight=predictor_edge_weights)
        log_pred = self.predictor(pred_data)

        if self.best_pred is None:
            pred = F.softmax(log_pred, dim=1).detach()
            self.best_pred = pred
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(self.best_pred)
        else:
            pred = self.best_pred

        if self.unlabel_edge_index.size(1) > 0:
            extra_w = self.estimator.get_estimated_weights(self.unlabel_edge_index, representations)
            estimated_weights = torch.cat([predictor_edge_weights, extra_w], dim=0)
            model_edge_index = torch.cat([pred_edge_index, self.unlabel_edge_index], dim=1)
        else:
            estimated_weights = predictor_edge_weights
            model_edge_index = pred_edge_index

        model_data = Data(x=features, edge_index=model_edge_index, edge_weight=estimated_weights)
        log_out = self.model(model_data)
        prob_out = F.softmax(log_out, dim=1).clamp(1e-8, 1 - 1e-8)

        if len(self.idx_add) > 0:
            loss_add = (-torch.sum(pred[self.idx_add] * torch.log(prob_out[self.idx_add]), dim=1)).mean()
        else:
            loss_add = torch.tensor(0.0, device=self.device)

        loss_pred = F.cross_entropy(log_pred[idx_train], self.labels[idx_train])
        loss_gcn = F.cross_entropy(log_out[idx_train], self.labels[idx_train])
        loss = loss_pred + self.args.get('alpha', 1.0) * loss_add + self.args.get('beta', 1.0) * rec_loss + loss_gcn

        loss.backward()
        self.optimizer.step()
        self.evaluate_and_update(idx_val)

    def evaluate_and_update(self, idx_val):
        self.model.eval()
        self.predictor.eval()
        with torch.no_grad():
            pred_val = self.predictor(Data(x=self.features, edge_index=self.edge_index))
            pred_val = F.softmax(pred_val, dim=1)
            acc_val = accuracy_score(self.labels[idx_val].cpu(), pred_val[idx_val].argmax(dim=1).cpu())
            if acc_val > self.best_val_acc:
                self.best_val_acc = acc_val
                self.weights = deepcopy(self.model.state_dict())
                self.predictor_weights = deepcopy(self.predictor.state_dict())

    def evaluate(self, idx_val):
        self.model.eval()
        self.predictor.eval()
        with torch.no_grad():
            pred_val = self.predictor(Data(x=self.features, edge_index=self.edge_index))
            pred_val = F.softmax(pred_val, dim=1)
            acc_val = accuracy_score(
                self.labels[idx_val].cpu(),
                pred_val[idx_val].argmax(dim=1).cpu()
            )
        return acc_val


    def get_train_edge(self, edge_index, features, n_p, idx_train):
        return edge_index

    def get_model_edge(self, pred):
        return self.edge_index, []

    def test(self, idx_test):
        self.model.eval()
        with torch.no_grad():
            out = self.model(Data(x=self.features, edge_index=self.edge_index))
            pred = out.argmax(dim=1)
            acc = accuracy_score(self.labels[idx_test].cpu(), pred[idx_test].cpu())
        return acc
