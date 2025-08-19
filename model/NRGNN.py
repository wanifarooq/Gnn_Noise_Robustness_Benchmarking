import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, negative_sampling
import time

from model.GNNs import GCN, GIN, GAT, GAT2

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class EstimateAdj(nn.Module):
    
    def __init__(self, nfea, args, idx_train, device='cuda'):
        super(EstimateAdj, self).__init__()
        
        self.estimator = GCN(nfea, args.get('edge_hidden', 64), args.get('edge_hidden', 64), 
                           dropout=0.0, self_loop=True)
        self.device = device
        self.args = args

    def forward(self, edge_index, features):
        edge_weights = torch.ones(edge_index.shape[1], device=self.device).float()

        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weights)
        representations = self.estimator(data)
        
        rec_loss = self.reconstruct_loss(edge_index, representations)
        
        return representations, rec_loss

    def get_estimated_weigths(self, edge_index, representations):
        if edge_index.numel() == 0:
            return torch.empty(0, device=self.device)
            
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0, x1), dim=1)
        
        estimated_weights = F.relu(output)
        estimated_weights[estimated_weights < self.args.get('t_small', 0.01)] = 0.0
        
        return estimated_weights
    
    def reconstruct_loss(self, edge_index, representations):
        num_nodes = representations.shape[0]
        
        randn = negative_sampling(edge_index, num_nodes=num_nodes, 
                                num_neg_samples=self.args.get('n_n', 1) * num_nodes)
        randn = randn[:, randn[0] < randn[1]]
        
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0, neg1), dim=1)
        
        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0, pos1), dim=1)
        
        rec_loss = (F.mse_loss(neg, torch.zeros_like(neg), reduction='sum') + 
                   F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) * \
                   num_nodes / (randn.shape[1] + edge_index.shape[1])
        
        return rec_loss


class NRGNN:
    def __init__(self, args, device, gnn_type='GCN'):
        self.device = device
        self.args = args
        self.gnn_type = gnn_type
        
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0
        self.best_pred = None
        self.best_graph = None
        self.best_model_index = None
        self.best_pred_graph = None
        self.weights = None
        self.predictor_model_weigths = None
        self.estimator = None
        self.model = None
        self.predictor = None
        self.pred_edge_index = None
        self.unlabel_edge_index = None
        self.idx_add = []

    def fit(self, features, adj, labels, idx_train, idx_val=None):
        args = self.args
        
        edge_index, _ = from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)
        
        if sp.issparse(features):
            features = torch.tensor(features.todense(), dtype=torch.float32)
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)

        labels = torch.LongTensor(np.array(labels)).to(self.device)

        self.edge_index = edge_index
        self.features = features
        self.labels = labels

        self.idx_unlabel = torch.LongTensor(
            list(set(range(features.shape[0])) - set(idx_train))
        ).to(self.device)

        if idx_val is None:
            val_size = max(1, len(idx_train) // 5)
            rng = np.random.default_rng(42)
            val_indices = rng.choice(idx_train, val_size, replace=False)
            idx_val = torch.tensor(val_indices, dtype=torch.long, device=self.device)
            idx_train = torch.tensor(
                [i for i in idx_train if i not in set(val_indices.tolist())],
                dtype=torch.long, device=self.device
            )
        else:
            idx_train = torch.tensor(idx_train, dtype=torch.long, device=self.device)
            idx_val = torch.tensor(idx_val, dtype=torch.long, device=self.device)
        
        hidden_dim = args.get('hidden', 64)
        out_dim = int(labels.max().item()) + 1
        dropout = args.get('dropout', 0.5)
        
        if self.gnn_type == 'GCN':
            self.predictor = GCN(features.shape[1], hidden_dim, out_dim, 
                               dropout=dropout, self_loop=True).to(self.device)
            self.model = GCN(features.shape[1], hidden_dim, out_dim, 
                           dropout=dropout, self_loop=True).to(self.device)
        elif self.gnn_type == 'GIN':
            self.predictor = GIN(features.shape[1], hidden_dim, out_dim, 
                               dropout=dropout).to(self.device)
            self.model = GIN(features.shape[1], hidden_dim, out_dim, 
                           dropout=dropout).to(self.device)
        elif self.gnn_type == 'GAT':
            self.predictor = GAT(features.shape[1], hidden_dim, out_dim, 
                               dropout=dropout).to(self.device)
            self.model = GAT(features.shape[1], hidden_dim, out_dim, 
                           dropout=dropout).to(self.device)
        elif self.gnn_type == 'GAT2':
            self.predictor = GAT2(features.shape[1], hidden_dim, out_dim, 
                                dropout=dropout).to(self.device)
            self.model = GAT2(features.shape[1], hidden_dim, out_dim, 
                            dropout=dropout).to(self.device)
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")
        
        self.estimator = EstimateAdj(features.shape[1], args, idx_train, 
                                   device=self.device).to(self.device)
        
        self.pred_edge_index = self.get_train_edge(edge_index, features, 
                                                 args.get('n_p', 5), idx_train)
        
        self.optimizer = optim.Adam(
            list(self.model.parameters()) +
            list(self.estimator.parameters()) +
            list(self.predictor.parameters()),
            lr=float(args.get('lr', 0.01)), 
            weight_decay=float(args.get('weight_decay', 5e-4))
        )

        t_total = time.time()
        for epoch in range(args.get('epochs', 200)):
            self.train(epoch, features, edge_index, idx_train, idx_val)
        
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        if self.predictor_model_weigths is not None:
            self.predictor.load_state_dict(self.predictor_model_weigths)
        
        val_acc = self.test(idx_val)
        
        return val_acc

    def train(self, epoch, features, edge_index, idx_train, idx_val):
        args = self.args
        
        t = time.time()
        self.model.train()
        self.predictor.train()
        self.optimizer.zero_grad()
        
        representations, rec_loss = self.estimator(edge_index, features)
        
        predictor_weights = self.estimator.get_estimated_weigths(self.pred_edge_index, representations)
        pred_edge_index = torch.cat([edge_index, self.pred_edge_index], dim=1)
        predictor_weights = torch.cat([
            torch.ones([edge_index.shape[1]], device=self.device),
            predictor_weights
        ], dim=0)

        pred_data = Data(x=features, edge_index=pred_edge_index, edge_weight=predictor_weights)
        log_pred = self.predictor(pred_data)

        if self.best_pred is None:
            pred = F.softmax(log_pred, dim=1).detach()
            self.best_pred = pred
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(self.best_pred)
        else:
            pred = self.best_pred
        
        estimated_weights = self.estimator.get_estimated_weigths(self.unlabel_edge_index, representations)
        estimated_weights = torch.cat([predictor_weights, estimated_weights], dim=0)
        model_edge_index = torch.cat([pred_edge_index, self.unlabel_edge_index], dim=1)

        model_data = Data(x=features, edge_index=model_edge_index, edge_weight=estimated_weights)
        output = self.model(model_data)
        pred_model = F.softmax(output, dim=1)
        
        eps = 1e-8
        pred_model = pred_model.clamp(eps, 1-eps)
        
        if len(self.idx_add) > 0:
            loss_add = (-torch.sum(pred[self.idx_add] * torch.log(pred_model[self.idx_add]), dim=1)).mean()
        else:
            loss_add = torch.tensor(0.0, device=self.device)
        
        loss_pred = F.cross_entropy(log_pred[idx_train], self.labels[idx_train])
        loss_gcn = F.cross_entropy(output[idx_train], self.labels[idx_train])
        
        total_loss = (loss_gcn + loss_pred + 
                     args.get('alpha', 1.0) * rec_loss + 
                     args.get('beta', 1.0) * loss_add)
        
        total_loss.backward()
        self.optimizer.step()
        
        acc_train = accuracy(output[idx_train].detach(), self.labels[idx_train])
        
        self.model.eval()
        self.predictor.eval()
        
        with torch.no_grad():
            pred = F.softmax(self.predictor(pred_data), dim=1)
            output = self.model(Data(x=features, edge_index=model_edge_index, 
                                   edge_weight=estimated_weights.detach()))
            
            acc_pred_val = accuracy(pred[idx_val], self.labels[idx_val])
            acc_val = accuracy(output[idx_val], self.labels[idx_val])
        
        if acc_pred_val > self.best_acc_pred_val:
            self.best_acc_pred_val = acc_pred_val
            self.best_pred_graph = predictor_weights.detach()
            self.best_pred = pred.detach()
            self.predictor_model_weigths = deepcopy(self.predictor.state_dict())
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(pred)
        
        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = estimated_weights.detach()
            self.best_model_index = model_edge_index
            self.weights = deepcopy(self.model.state_dict())
            if args.get('debug', False):
                print('\t=== saving current graph/gcn, best_val_acc: {:.4f}'.format(self.best_val_acc.item()))
        
        if args.get('debug', True):
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                  'loss_pred: {:.4f}'.format(loss_pred.item()),
                  'loss_add: {:.4f}'.format(loss_add.item()),
                  'rec_loss: {:.4f}'.format(rec_loss.item()),
                  'loss_total: {:.4f}'.format(total_loss.item()))
            print('Epoch: {:04d}'.format(epoch+1),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'acc_pred_val: {:.4f}'.format(acc_pred_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))
            print('Size of add idx is {}'.format(len(self.idx_add)))

    def test(self, idx_test):
        features = self.features
        labels = self.labels
        
        self.predictor.eval()
        estimated_weights = self.best_pred_graph
        pred_edge_index = torch.cat([self.edge_index, self.pred_edge_index], dim=1)
        
        with torch.no_grad():
            pred_data = Data(x=features, edge_index=pred_edge_index, edge_weight=estimated_weights)
            output = self.predictor(pred_data)
            loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
        
        print("\tPredictor results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        
        self.model.eval()
        estimated_weights = self.best_graph
        model_edge_index = self.best_model_index
        
        with torch.no_grad():
            model_data = Data(x=features, edge_index=model_edge_index, edge_weight=estimated_weights)
            output = self.model(model_data)
            loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
        
        print("\tGCN classifier results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        
        return float(acc_test)

    def get_train_edge(self, edge_index, features, n_p, idx_train):
        if n_p == 0:
            return torch.empty((2, 0), device=self.device, dtype=torch.long)
        
        poten_edges = []
        
        if n_p > len(idx_train) or n_p < 0:
            for i in range(len(features)):
                indices = set(idx_train.cpu().numpy())
                existing_neighbors = set(edge_index[1, edge_index[0] == i].cpu().numpy())
                indices = indices - existing_neighbors
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:

            for i in range(len(features)):
                sim = torch.div(
                    torch.matmul(features[i], features[idx_train].T),
                    features[i].norm() * features[idx_train].norm(dim=1)
                )
                _, rank = sim.topk(n_p)
                
                if rank.max() < len(features) and rank.min() >= 0:
                    indices = idx_train[rank.cpu().numpy()]
                    indices = set(indices.cpu().numpy() if hasattr(indices, 'cpu') else indices)
                else:
                    indices = set()

                existing_neighbors = set(edge_index[1, edge_index[0] == i].cpu().numpy())
                indices = indices - existing_neighbors
                
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        
        if len(poten_edges) == 0:
            return torch.empty((2, 0), device=self.device, dtype=torch.long)
        
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = to_undirected(poten_edges, len(features)).to(self.device)
        
        return poten_edges

    def get_model_edge(self, pred):
        idx_add = self.idx_unlabel[
            (pred.max(dim=1)[0][self.idx_unlabel] > self.args.get('p_u', 0.8))
        ]
        
        if len(idx_add) == 0 or len(self.idx_unlabel) == 0:
            return torch.empty((2, 0), device=self.device, dtype=torch.long), idx_add
        
        row = self.idx_unlabel.repeat(len(idx_add))
        col = idx_add.repeat(len(self.idx_unlabel), 1).T.flatten()
        mask = (row != col)
        unlabel_edge_index = torch.stack([row[mask], col[mask]], dim=0)
        
        return unlabel_edge_index, idx_add