import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, negative_sampling
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score

def accuracy(output, labels):

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

class GCNEstimator(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, device=None):
        super(GCNEstimator, self).__init__()
        self.device = device
        self.gc1 = GCNConv(nfeat, nhid, bias=True, add_self_loops=True)
        self.gc2 = GCNConv(nhid, nclass, bias=True, add_self_loops=True)
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.gc1(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight)
        return x

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

class GNNs(nn.Module):

    def __init__(self, base_model, device):
        super().__init__()
        self.base_model = base_model
        self.device = device
        
    def forward(self, x, edge_index, edge_weight=None):

        class DataLike:
            def __init__(self, x, edge_index, edge_weight=None):
                self.x = x
                self.edge_index = edge_index
                self.edge_weight = edge_weight
        
        data = DataLike(x, edge_index, edge_weight)
        return self.base_model(data)
    
    def reset_parameters(self):
        if hasattr(self.base_model, 'initialize'):
            self.base_model.initialize()
        else:
            for module in self.base_model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

class EstimateAdj(nn.Module):

    def __init__(self, nfeat, args, device='cuda'):
        super(EstimateAdj, self).__init__()
        self.estimator = GCNEstimator(nfeat, args.edge_hidden, args.edge_hidden, 
                                     dropout=0.0, device=device)
        self.device = device
        self.args = args

    def forward(self, edge_index, features):

        edge_weight = torch.ones(edge_index.shape[1], device=self.device, dtype=torch.float32)
        representations = self.estimator(features, edge_index, edge_weight)
        rec_loss = self.reconstruct_loss(edge_index, representations)
        return representations, rec_loss
    
    def get_estimated_weights(self, edge_index, representations):
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(x0 * x1, dim=1)
        estimated_weights = F.relu(output)
        
        mask = estimated_weights >= self.args.t_small
        estimated_weights = estimated_weights * mask.float()
        
        return estimated_weights
    
    def reconstruct_loss(self, edge_index, representations):
        num_nodes = representations.shape[0]
        
        neg_edge_index = negative_sampling(
            edge_index, num_nodes=num_nodes, 
            num_neg_samples=self.args.n_n * num_nodes
        )
        
        neg_edge_index = neg_edge_index[:, neg_edge_index[0] < neg_edge_index[1]]
        pos_edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        
        if neg_edge_index.shape[1] == 0 or pos_edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        neg_sim = torch.sum(
            representations[neg_edge_index[0]] * representations[neg_edge_index[1]], 
            dim=1
        )
        pos_sim = torch.sum(
            representations[pos_edge_index[0]] * representations[pos_edge_index[1]], 
            dim=1
        )
        
        neg_loss = F.mse_loss(neg_sim, torch.zeros_like(neg_sim), reduction='sum')
        pos_loss = F.mse_loss(pos_sim, torch.ones_like(pos_sim), reduction='sum')
        
        total_edges = neg_edge_index.shape[1] + pos_edge_index.shape[1]
        rec_loss = (neg_loss + pos_loss) * num_nodes / total_edges
        
        return rec_loss

class NRGNN:
    def __init__(self, args, device, base_model=None):
        self.device = device
        self.args = args
        self.base_model_class = base_model
        
        self.best_val_loss = float('inf')
        self.best_acc_pred_val = 0.0
        self.early_stop_counter = 0
        self.patience = getattr(args, "patience", 10)
        
        self.predictor = None
        self.model = None
        self.estimator = None
        self.optimizer = None
        
        self.weights = None
        self.predictor_model_weights = None
        self.best_pred = None
        self.best_graph = None
        self.best_pred_graph = None
        self.best_model_index = None
        
        self.edge_index = None
        self.pred_edge_index = None
        self.unlabel_edge_index = None
        self.idx_add = []
        self.idx_unlabel = None

    def fit(self, features, adj, labels, idx_train, idx_val):

        args = self.args
        
        self._prepare_data(features, adj, labels, idx_train)
        
        self._initialize_models()
        
        t_total = time.time()
        try:
            for epoch in range(args.epochs):
                self._train_epoch(epoch, idx_train, idx_val)
        except StopIteration:
            if args.debug:
                print(f"Early stopping at epoch {epoch+1}")

        print(f"Total time elapsed: {time.time() - t_total:.4f}s")
        
        self._load_best_models()

    def _prepare_data(self, features, adj, labels, idx_train):

        self.edge_index, _ = from_scipy_sparse_matrix(adj)
        self.edge_index = self.edge_index.to(self.device)
        
        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense()
        else:
            features = torch.FloatTensor(np.array(features))
        self.features = features.to(self.device)
        
        self.labels = torch.LongTensor(np.array(labels)).to(self.device)
        
        all_indices = set(range(self.features.shape[0]))
        train_set = set(idx_train.tolist() if isinstance(idx_train, np.ndarray) else idx_train)
        unlabel_list = list(all_indices - train_set)
        self.idx_unlabel = torch.LongTensor(unlabel_list).to(self.device)

    def _initialize_models(self):

        predictor_base = deepcopy(self.base_model_class)
        model_base = deepcopy(self.base_model_class)
        
        self.predictor = GNNs(predictor_base, self.device).to(self.device)
        self.model = GNNs(model_base, self.device).to(self.device)
        self.estimator = EstimateAdj(self.features.shape[1], self.args, device=self.device).to(self.device)
        
        self.pred_edge_index = self._get_train_edges()
        
        all_params = (list(self.model.parameters()) + 
                     list(self.estimator.parameters()) + 
                     list(self.predictor.parameters()))
        
        self.optimizer = optim.Adam(
            all_params, 
            lr=float(self.args.lr), 
            weight_decay=float(self.args.weight_decay)
        )

    def _get_train_edges(self):
        if self.args.n_p <= 0:
            return None
            
        idx_train_tensor = torch.arange(self.features.shape[0], device=self.device)
        
        poten_edges = []
        n_p = min(self.args.n_p, self.features.shape[0] - 1)
        
        features_norm = F.normalize(self.features, p=2, dim=1)
        
        for i in range(self.features.shape[0]):
            similarities = torch.mm(features_norm[i:i+1], features_norm.t()).squeeze()
            
            similarities[i] = -1
            _, top_indices = similarities.topk(n_p)
            
            existing_neighbors = set(self.edge_index[1][self.edge_index[0] == i].cpu().numpy())
            
            for j in top_indices:
                j_item = j.item()
                if j_item not in existing_neighbors:
                    poten_edges.append([i, j_item])
        
        if not poten_edges:
            return None
            
        poten_edges = torch.tensor(poten_edges, dtype=torch.long, device=self.device).t()
        poten_edges = to_undirected(poten_edges, num_nodes=self.features.shape[0])
        
        return poten_edges

    def _get_model_edges(self, pred):
        if len(self.idx_unlabel) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), []

        max_probs = pred.max(dim=1)[0]
        confident_mask = max_probs[self.idx_unlabel] > self.args.p_u
        idx_add = self.idx_unlabel[confident_mask]
        
        if len(idx_add) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), idx_add.cpu().numpy()

        num_unlabel = len(self.idx_unlabel)
        num_add = len(idx_add)
        
        if num_unlabel == 0 or num_add == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), idx_add.cpu().numpy()
        
        row = self.idx_unlabel.repeat(num_add)
        col = idx_add.repeat(num_unlabel)
        
        mask = row != col
        
        if mask.sum() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), idx_add.cpu().numpy()
        
        unlabel_edge_index = torch.stack([row[mask], col[mask]], dim=0)
        
        return unlabel_edge_index, idx_add.cpu().numpy()

    def _train_epoch(self, epoch, idx_train, idx_val):
        t = time.time()
        self.model.train()
        self.predictor.train()
        self.estimator.train()
        
        self.optimizer.zero_grad()
        
        representations, rec_loss = self.estimator(self.edge_index, self.features)
        
        if self.pred_edge_index is not None and self.pred_edge_index.shape[1] > 0:
            predictor_weights = self.estimator.get_estimated_weights(self.pred_edge_index, representations)
            pred_edge_index = torch.cat([self.edge_index, self.pred_edge_index], dim=1)
            all_pred_weights = torch.cat([
                torch.ones(self.edge_index.shape[1], device=self.device), 
                predictor_weights
            ], dim=0)
        else:
            pred_edge_index = self.edge_index
            all_pred_weights = torch.ones(self.edge_index.shape[1], device=self.device)

        log_pred = self.predictor(self.features, pred_edge_index, all_pred_weights)
        
        if self.best_pred is None:
            with torch.no_grad():
                pred = F.softmax(log_pred, dim=1)
                self.best_pred = pred.detach()
                self.unlabel_edge_index, self.idx_add = self._get_model_edges(self.best_pred)
        
        if self.unlabel_edge_index is not None and self.unlabel_edge_index.shape[1] > 0:
            model_edge_weights = self.estimator.get_estimated_weights(self.unlabel_edge_index, representations)
            estimated_weights = torch.cat([all_pred_weights, model_edge_weights], dim=0)
            model_edge_index = torch.cat([pred_edge_index, self.unlabel_edge_index], dim=1)
        else:
            estimated_weights = all_pred_weights
            model_edge_index = pred_edge_index

        output = self.model(self.features, model_edge_index, estimated_weights)
        
        loss_pred = F.cross_entropy(log_pred[idx_train], self.labels[idx_train])
        loss_gcn = F.cross_entropy(output[idx_train], self.labels[idx_train])
        
        if len(self.idx_add) > 0:
            pred_model = F.softmax(output, dim=1)
            pred_model = torch.clamp(pred_model, 1e-8, 1 - 1e-8)
            
            kl_loss = F.kl_div(
                torch.log(pred_model[self.idx_add]), 
                self.best_pred[self.idx_add], 
                reduction='batchmean'
            )
            loss_add = kl_loss
        else:
            loss_add = torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = (loss_gcn + loss_pred + 
                     self.args.alpha * rec_loss + 
                     self.args.beta * loss_add)

        if torch.isnan(total_loss):
            print(f"NaN detected! Skipping epoch {epoch}")
            return

        total_loss.backward()
        self.optimizer.step()

        self._evaluate_epoch(epoch, idx_train, idx_val, pred_edge_index, all_pred_weights, 
                            model_edge_index, estimated_weights, total_loss, t)

    def _evaluate_epoch(self, epoch, idx_train, idx_val, pred_edge_index, all_pred_weights,
                        model_edge_index, estimated_weights, total_loss, start_time):
        self.model.eval()
        self.predictor.eval()
        self.estimator.eval()
        
        with torch.no_grad():

            pred_output = self.predictor(self.features, pred_edge_index, all_pred_weights)
            pred_probs = F.softmax(pred_output, dim=1)

            model_output = self.model(self.features, model_edge_index, estimated_weights.detach())
            
            train_loss = F.cross_entropy(model_output[idx_train], self.labels[idx_train]).item()
            val_loss = F.cross_entropy(model_output[idx_val], self.labels[idx_val]).item()
            
            train_acc = accuracy(model_output[idx_train], self.labels[idx_train])
            val_acc = accuracy(model_output[idx_val], self.labels[idx_val])
            acc_pred_val = accuracy(pred_probs[idx_val], self.labels[idx_val])
            
            train_f1 = f1_score(self.labels[idx_train].cpu(), model_output[idx_train].argmax(dim=1).cpu(), average='macro')
            val_f1 = f1_score(self.labels[idx_val].cpu(), model_output[idx_val].argmax(dim=1).cpu(), average='macro')

            if self.args.debug:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                    f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f} | Time: {time.time() - start_time:.2f}s")
                
                print(f"  Pred Val Acc: {acc_pred_val:.4f} | Add Nodes: {len(self.idx_add)}")

            if acc_pred_val > self.best_acc_pred_val:
                self.best_acc_pred_val = acc_pred_val
                self.best_pred_graph = all_pred_weights.detach()
                self.best_pred = pred_probs.detach()
                self.predictor_model_weights = deepcopy(self.predictor.state_dict())
                
                self.unlabel_edge_index, self.idx_add = self._get_model_edges(self.best_pred)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_graph = estimated_weights.detach()
                self.best_model_index = model_edge_index
                self.weights = deepcopy(self.model.state_dict())
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                if self.args.debug:
                    print(f"Early stopping at epoch {epoch+1}, best val loss: {self.best_val_loss:.4f}")
                raise StopIteration

    def _load_best_models(self):
        print("Loading best model according to validation performance")
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        if self.predictor_model_weights is not None:
            self.predictor.load_state_dict(self.predictor_model_weights)

    def test(self, idx_test):
        self.model.eval()
        self.predictor.eval()
        
        with torch.no_grad():
            if self.best_pred_graph is not None and self.pred_edge_index is not None:
                pred_edge_index = torch.cat([self.edge_index, self.pred_edge_index], dim=1)
                pred_output = self.predictor(self.features, pred_edge_index, self.best_pred_graph)
                pred_probs = F.softmax(pred_output, dim=1)
                
                pred_acc = accuracy(pred_probs[idx_test], self.labels[idx_test])
                y_true = self.labels[idx_test].cpu().numpy()
                y_pred = pred_probs[idx_test].argmax(dim=1).cpu().numpy()
                f1_test = f1_score(y_true, y_pred, average='macro')
                
                print(f"Predictor Test Acc: {pred_acc:.4f} | Test F1: {f1_test:.4f}")

            if self.best_graph is not None and self.best_model_index is not None:
                model_output = self.model(self.features, self.best_model_index, self.best_graph)
                
                model_loss = F.cross_entropy(model_output[idx_test], self.labels[idx_test]).item()
                model_acc = accuracy(model_output[idx_test], self.labels[idx_test])
                
                y_true = self.labels[idx_test].cpu().numpy()
                y_pred = model_output[idx_test].argmax(dim=1).cpu().numpy()
                f1_test = f1_score(y_true, y_pred, average='macro')
                
                print(f"Test Loss: {model_loss:.4f} | Test Acc: {model_acc:.4f} | Test F1: {f1_test:.4f}")
                return float(model_acc)
        
        return 0.0
