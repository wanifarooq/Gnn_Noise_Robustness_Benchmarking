import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix, negative_sampling
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data
from sklearn.metrics import f1_score

from model.GNNs import GCN, GIN, GAT, GAT2

class DualGNN(nn.Module):
    def __init__(self, gnn_type: str, nfeat: int, nhid: int, nclass: int,
                 dropout: float = 0.5, use_edge_weight: bool = False, heads: int = 4,
                 n_layers: int = None, device=None, self_loop: bool = False):
        super().__init__()
        self.device = device
        self.gnn_type = gnn_type.lower()
        self.use_edge_weight = use_edge_weight
        self.dropout = dropout
        self.self_loop = self_loop

        def create_gnn():
            if self.gnn_type == 'gcn':
                return GCN(nfeat, nhid, nclass, n_layers=n_layers or 2,
                           dropout=dropout, self_loop=self.self_loop)
            elif self.gnn_type == 'gin':
                return GIN(nfeat, nhid, nclass, n_layers=n_layers or 3,
                           mlp_layers=2, dropout=dropout)
            elif self.gnn_type == 'gat' and not use_edge_weight:
                return GAT(nfeat, nhid, nclass, n_layers=n_layers or 3,
                           heads=heads, dropout=dropout, self_loop=self.self_loop)
            elif self.gnn_type == 'gat2' and not use_edge_weight:
                return GAT2(nfeat, nhid, nclass, n_layers=n_layers or 3,
                            heads=heads, dropout=dropout, self_loop=self.self_loop)
            else:
                raise ValueError(f"GNN type {gnn_type} not supported")

        self.branch1 = create_gnn()
        self.branch2 = create_gnn()

    def forward(self, x, edge_index, edge_weight=None):
        edge_attr = edge_weight.unsqueeze(-1) if self.use_edge_weight and edge_weight is not None else None

        data = Data(x=x, edge_index=edge_index)
        if edge_attr is not None:
            data.edge_attr = edge_attr
        if self.device is not None:
            data = data.to(self.device)

        out1 = self.branch1(data)
        out2 = self.branch2(data)
        return out1, out2

    def reset_parameters(self):
        if hasattr(self.branch1, "reset_parameters"):
            self.branch1.reset_parameters()
        if hasattr(self.branch2, "reset_parameters"):
            self.branch2.reset_parameters()

class AdaptiveLoss(nn.Module):
    def __init__(self, args):
        super(AdaptiveLoss, self).__init__()
        self.args = args
        self.epochs = args.epochs
        self.increment = 0.5 / self.epochs

    def forward(self, y1, y2, targets, epoch=0):
        loss1 = F.cross_entropy(y1, targets, reduction='none')
        loss2 = F.cross_entropy(y2, targets, reduction='none')
        total_loss = loss1 + loss2
        
        if epoch == 0:
            return total_loss.mean()
        
        sorted_indices = torch.argsort(total_loss)
        forget_rate = min(0.5, self.increment * epoch)
        remember_rate = max(0.5, 1 - forget_rate)
        
        num_remember = int(remember_rate * len(total_loss))
        clean_indices = sorted_indices[:num_remember]

        clean_loss = total_loss[clean_indices].mean()
        kl_loss = self._kl_divergence(y1, y2) + self._kl_divergence(y2, y1)

        noisy_indices = sorted_indices[num_remember:]
        if len(noisy_indices) > 0:
            p1, p2 = F.softmax(y1, dim=1), F.softmax(y2, dim=1)
            pred1, pred2 = y1.max(1)[1], y2.max(1)[1]
            conf1, conf2 = p1.max(1)[0], p2.max(1)[0]
            
            agree_mask = (pred1[noisy_indices] == pred2[noisy_indices])
            high_conf_mask = (conf1[noisy_indices] * conf2[noisy_indices] > 0.5)
            correction_mask = agree_mask & high_conf_mask
            
            if correction_mask.sum() > 0:
                correct_indices = noisy_indices[correction_mask]
                weights = (conf1[correct_indices] * conf2[correct_indices]).sqrt()
                correction_loss = weights * (
                    F.cross_entropy(y1[correct_indices], pred1[correct_indices], reduction='none') +
                    F.cross_entropy(y2[correct_indices], pred1[correct_indices], reduction='none')
                )
                clean_loss = clean_loss + correction_loss.mean()
        
        return clean_loss + self.args.co_lambda * kl_loss
    
    def _kl_divergence(self, pred, target):
        return F.kl_div(
            F.log_softmax(pred, dim=1),
            F.softmax(target.detach(), dim=1),
            reduction='batchmean'
        )


class GraphAdjEstimator(nn.Module):
    def __init__(self, nfeat, nhid, args, device):
        super(GraphAdjEstimator, self).__init__()
        self.device = device
        self.args = args
        
        self.encoder = GCNConv(nfeat, nhid, add_self_loops=True)
        self.tau = args.tau
        
    def forward(self, x, edge_index):
        h = F.normalize(F.relu(self.encoder(x, edge_index)), dim=1)
        pos_loss, neg_loss = self._reconstruction_loss(h, edge_index)
        rec_loss = pos_loss + neg_loss
        return h, rec_loss
    
    def get_edge_weights(self, edge_index, representations, base_weights=None):
        src, dst = edge_index[0], edge_index[1]
        edge_weights = (representations[src] * representations[dst]).sum(dim=1)
        edge_weights = F.relu(edge_weights)
        edge_weights = edge_weights * (edge_weights >= self.tau).float()
        
        if base_weights is not None:
            edge_weights = base_weights + edge_weights * (1 - base_weights)
            
        return edge_weights
    
    def _reconstruction_loss(self, h, edge_index):
        num_nodes = h.size(0)
        pos_edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        pos_scores = (h[pos_edge_index[0]] * h[pos_edge_index[1]]).sum(dim=1)
        pos_loss = F.mse_loss(pos_scores, torch.ones_like(pos_scores))

        neg_edge_index = negative_sampling(
            edge_index, num_nodes=num_nodes, 
            num_neg_samples=min(pos_edge_index.size(1), self.args.n_neg * num_nodes)
        )
        neg_edge_index = neg_edge_index[:, neg_edge_index[0] < neg_edge_index[1]]
        neg_scores = (h[neg_edge_index[0]] * h[neg_edge_index[1]]).sum(dim=1)
        neg_loss = F.mse_loss(neg_scores, torch.zeros_like(neg_scores))
        
        return pos_loss, neg_loss

class RTGNN(nn.Module):
    def __init__(self, nfeat, nclass, args, device, gnn_type='gcn'):
        super().__init__()
        self.device = device
        self.args = args
        self.gnn_type = gnn_type.lower()

        gnn_kwargs = self._get_gnn_kwargs()
        self.predictor = DualGNN(
            gnn_type=self.gnn_type,
            nfeat=nfeat,
            nhid=args.hidden,
            nclass=nclass,
            dropout=args.dropout,
            device=device,
            **gnn_kwargs
        )
        
        self.adj_estimator = GraphAdjEstimator(nfeat, args.edge_hidden, args, device)
        self.adaptive_loss = AdaptiveLoss(args)
        
        self.best_val_acc = 0
        self.best_state = None
        
        print(f"Initialized RTGNN with {self.gnn_type.upper()} backbone")
        
    def _get_gnn_kwargs(self):
        base = {'n_layers': getattr(self.args, 'n_layers', 2)}
        if self.gnn_type == 'gcn':
            base.update({'self_loop': getattr(self.args, 'self_loop', True)})
        elif self.gnn_type == 'gin':
            base.update({'mlp_layers': getattr(self.args, 'mlp_layers', 2),
                         'train_eps': getattr(self.args, 'train_eps', True)})
        elif self.gnn_type in ['gat', 'gat2']:
            base.update({'heads': getattr(self.args, 'heads', 8),
                         'self_loop': getattr(self.args, 'self_loop', True)})
        return base

    def forward(self, x, edge_index, edge_weight=None):
        return self.predictor(x, edge_index, edge_weight)
    
    def fit(self, features, adj, labels, idx_train, idx_val, idx_test=None, 
            noise_idx=None, clean_idx=None):

        edge_index, _ = from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)
        features = features.to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        
        knn_edges = self._generate_knn_edges(features, edge_index, idx_train)
        
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, 
                             weight_decay=self.args.weight_decay)

        patience = getattr(self.args, 'patience', 20)
        counter = 0
        best_val_acc = 0
        
        print(f"Starting RTGNN training with {self.gnn_type.upper()}...")
        
        for epoch in range(self.args.epochs):
            self.train()
            optimizer.zero_grad()
            
            representations, rec_loss = self.adj_estimator(features, edge_index)
            
            if knn_edges.size(1) > 0:
                combined_edges = torch.cat([edge_index, knn_edges], dim=1)
                base_weights = torch.cat([
                    torch.ones(edge_index.size(1)),
                    torch.zeros(knn_edges.size(1))
                ]).to(self.device)
            else:
                combined_edges = edge_index
                base_weights = torch.ones(edge_index.size(1)).to(self.device)
            
            edge_weights = self.adj_estimator.get_edge_weights(
                combined_edges, representations, base_weights
            )
            
            valid_mask = edge_weights > 0
            final_edges = combined_edges[:, valid_mask]
            final_weights = edge_weights[valid_mask]

            output1, output2 = self.predictor(features, final_edges, final_weights)

            pred_loss = self.adaptive_loss(
                output1[idx_train], output2[idx_train], 
                labels[idx_train], epoch
            )

            pseudo_loss = self._compute_pseudo_loss(output1, output2, idx_train)

            total_loss = (pred_loss + self.args.alpha * rec_loss + 
                          self.args.co_lambda * pseudo_loss)
            
            total_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():

                output1_train, output2_train = self.predictor(features, final_edges, final_weights)
                loss_train1 = F.cross_entropy(output1_train[idx_train], labels[idx_train])
                loss_train2 = F.cross_entropy(output2_train[idx_train], labels[idx_train])
                loss_train = (loss_train1 + loss_train2) / 2

                train_pred = (output1_train[idx_train] + output2_train[idx_train]) / 2
                train_acc = (train_pred.argmax(dim=1) == labels[idx_train]).float().mean().item()
                train_f1 = f1_score(labels[idx_train].cpu(), train_pred.argmax(dim=1).cpu(), average='macro')

                output1_val, output2_val = self.predictor(features, final_edges, final_weights)
                val_loss1 = F.cross_entropy(output1_val[idx_val], labels[idx_val])
                val_loss2 = F.cross_entropy(output2_val[idx_val], labels[idx_val])
                val_loss = (val_loss1 + val_loss2) / 2

                val_pred = (output1_val[idx_val] + output2_val[idx_val]) / 2
                val_acc = (val_pred.argmax(dim=1) == labels[idx_val]).float().mean().item()
                val_f1 = f1_score(labels[idx_val].cpu(), val_pred.argmax(dim=1).cpu(), average='macro')

            print(f"Epoch {epoch:03d} | Train Loss: {loss_train:.4f}, Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | "
                  f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            if epoch == 0:
                best_val_loss = val_loss
                counter = 0
                self.best_state = {
                    'model': deepcopy(self.state_dict()),
                    'edges': final_edges.clone(),
                    'weights': final_weights.clone()
                }
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                self.best_state = {
                    'model': deepcopy(self.state_dict()),
                    'edges': final_edges.clone(),
                    'weights': final_weights.clone()
                }
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement per {patience} epoche)")
                    break

        if self.best_state is not None:
            self.load_state_dict(self.best_state['model'])
            print(f"Training completed! Best validation loss: {best_val_loss:.4f}")

    def test(self, features, labels, idx_test):
        if self.best_state is None:
            print("Model not trained yet!")
            return 0.0

        self.eval()
        with torch.no_grad():
            features = features.to(self.device)
            labels = torch.LongTensor(labels).to(self.device)
            
            output1, output2 = self.predictor(
                features, 
                self.best_state['edges'], 
                self.best_state['weights']
            )

            test_loss1 = F.cross_entropy(output1[idx_test], labels[idx_test])
            test_loss2 = F.cross_entropy(output2[idx_test], labels[idx_test])
            test_loss = (test_loss1 + test_loss2) / 2

            test_pred = (output1[idx_test] + output2[idx_test]) / 2
            test_acc = (test_pred.argmax(dim=1) == labels[idx_test]).float().mean().item()
            test_f1 = f1_score(labels[idx_test].cpu(), test_pred.argmax(dim=1).cpu(), average='macro')

            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
            
            return test_acc

        
    def _generate_knn_edges(self, features, edge_index, idx_train, k=None):
        if k is None:
            k = self.args.K
            
        if k == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)
        
        existing = set(map(tuple, edge_index.t().cpu().numpy()))
        features_norm = F.normalize(features, dim=1)
        similarity = torch.mm(features_norm, features_norm.t())
        
        new_edges = []
        idx_train = torch.tensor(idx_train, device=self.device)
        all_nodes = torch.arange(features.size(0), device=self.device)
        unlabeled = all_nodes[~torch.isin(all_nodes, idx_train)]
        
        for train_node in idx_train:
            if len(unlabeled) == 0:
                continue
                
            sim_scores = similarity[train_node, unlabeled]
            _, top_k = sim_scores.topk(min(k, len(unlabeled)))
            
            for idx in top_k:
                neighbor = unlabeled[idx].item()
                edge = (train_node.item(), neighbor)
                if edge not in existing and edge[::-1] not in existing:
                    new_edges.append([train_node.item(), neighbor])
                    new_edges.append([neighbor, train_node.item()])
        
        if new_edges:
            return torch.tensor(new_edges, device=self.device).t()
        else:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)
    
    def _compute_pseudo_loss(self, output1, output2, idx_train):
        all_nodes = torch.arange(output1.size(0), device=self.device)
        unlabeled = all_nodes[~torch.isin(all_nodes, torch.tensor(idx_train, device=self.device))]
        
        if len(unlabeled) == 0:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            pred1 = F.softmax(output1[unlabeled], dim=1)
            pred2 = F.softmax(output2[unlabeled], dim=1)
            
            conf1, class1 = pred1.max(dim=1)
            conf2, class2 = pred2.max(dim=1)
            
            consistent = (class1 == class2)
            confident = (conf1 * conf2) > (self.args.th ** 2)
            reliable = consistent & confident
            
            if reliable.sum() == 0:
                return torch.tensor(0.0, device=self.device)
            
            pseudo_nodes = unlabeled[reliable]
            pseudo_labels = class1[reliable]
        
        loss1 = F.cross_entropy(output1[pseudo_nodes], pseudo_labels)
        loss2 = F.cross_entropy(output2[pseudo_nodes], pseudo_labels)
        
        return (loss1 + loss2) / 2
    
    def _evaluate(self, features, edge_index, edge_weight, labels, idx_val):
        self.eval()
        with torch.no_grad():
            output1, output2 = self.predictor(features, edge_index, edge_weight)
            pred1 = output1[idx_val].max(1)[1]
            pred2 = output2[idx_val].max(1)[1]
            
            acc1 = accuracy_score(labels[idx_val].cpu(), pred1.cpu())
            acc2 = accuracy_score(labels[idx_val].cpu(), pred2.cpu())
            
            return (acc1 + acc2) / 2
