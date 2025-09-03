import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from copy import deepcopy
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix, negative_sampling
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score

from model.GNNs import GCN, GIN, GAT, GATv2
from model.evaluation import OversmoothingMetrics

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
            elif self.gnn_type == 'gatv2' and not use_edge_weight:
                return GATv2(nfeat, nhid, nclass, n_layers=n_layers or 3,
                            heads=heads, dropout=dropout, self_loop=self.self_loop)
            else:
                raise ValueError(f"GNN type {self.gnn_type} not supported")

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
        super().__init__()
        self.args = args
        self.epochs = args.epochs
        self.increment = 0.5 / self.epochs
        self.decay_w = getattr(args, "decay_w", 1.0)

    def forward(self, y1, y2, targets, epoch=0):
        loss1 = F.cross_entropy(y1, targets, reduction='none')
        loss2 = F.cross_entropy(y2, targets, reduction='none')
        total_loss = loss1 + loss2
        
        if epoch == 0:
            return total_loss.mean()

        sorted_idx = torch.argsort(total_loss)
        forget_rate = self.increment * epoch
        remember_rate = max(0.5, 1 - forget_rate)
        num_remember = int(remember_rate * len(total_loss))
        clean_idx = sorted_idx[:num_remember]
        noisy_idx = sorted_idx[num_remember:]

        clean_loss = total_loss[clean_idx].mean()

        correction_loss = torch.tensor(0.0, device=y1.device)
        if len(noisy_idx) > 0:
            p1, p2 = F.softmax(y1, dim=1), F.softmax(y2, dim=1)
            pred1, pred2 = y1.max(1)[1], y2.max(1)[1]
            conf1, conf2 = p1.max(1)[0], p2.max(1)[0]

            agree = pred1[noisy_idx] == pred2[noisy_idx]
            high_conf = (conf1[noisy_idx] * conf2[noisy_idx] >
                         (1 - (1 - min(0.5, 1/y1.size(0))) * epoch/self.epochs))
            mask = agree & high_conf
            if mask.sum() > 0:
                idx = noisy_idx[mask]
                weights = (conf1[idx] * conf2[idx])**(0.5 - 0.5*epoch/self.epochs)
                correction_loss = (weights * (
                    F.cross_entropy(y1[idx], pred1[idx], reduction='none') +
                    F.cross_entropy(y2[idx], pred1[idx], reduction='none')
                )).mean()

        residual_loss = self.decay_w * total_loss[noisy_idx].mean() if len(noisy_idx) > 0 else 0.0

        kl_loss = self._kl(y1, y2) + self._kl(y2, y1)

        return clean_loss + correction_loss + residual_loss + self.args.co_lambda * kl_loss

    def _kl(self, p, q):
        return F.kl_div(F.log_softmax(p, dim=1),
                        F.softmax(q.detach(), dim=1),
                        reduction='batchmean')
class IntraviewReg(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def forward(self, y1, y2, edge_index, edge_weight, idx_label):
        if isinstance(idx_label, list):
            idx_label = torch.tensor(idx_label, device=self.device)
        
        if idx_label.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        weighted_adj = to_scipy_sparse_matrix(edge_index, edge_weight.detach().cpu())
        colsum = np.array(weighted_adj.sum(0))
        r_inv = np.power(colsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        norm_adj = weighted_adj.dot(sp.diags(r_inv))

        norm_idx, norm_w = from_scipy_sparse_matrix(norm_adj)
        norm_idx, norm_w = norm_idx.to(self.device), norm_w.to(self.device)

        mask_edges = (torch.isin(norm_idx[1], idx_label))
        edge_index_f = norm_idx[:, mask_edges]
        edge_weight_f = norm_w[mask_edges]

        if edge_index_f.size(1) == 0:
            return torch.tensor(0.0, device=self.device)

        loss = (edge_weight_f * self._kl(y1[edge_index_f[1]], y1[edge_index_f[0]].detach())).sum()
        loss += (edge_weight_f * self._kl(y2[edge_index_f[1]], y2[edge_index_f[0]].detach())).sum()
        loss = loss / idx_label.size(0)
        return loss

    def _kl(self, p, q):
        return F.kl_div(F.log_softmax(p, dim=1),
                        F.softmax(q, dim=1),
                        reduction='batchmean')


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
        self.intra_reg = IntraviewReg(device=device)

        
        self.best_val_acc = 0
        self.best_state = None

        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        
        print(f"Initialized RTGNN with {self.gnn_type.upper()} backbone")
    
    def _compute_oversmoothing_for_mask(self, embeddings, edge_index, mask, labels=None):
        try:
            mask_indices = torch.where(mask)[0]
            mask_embeddings = embeddings[mask]
            
            mask_set = set(mask_indices.cpu().numpy())
            edge_mask = torch.tensor([
                src.item() in mask_set and tgt.item() in mask_set
                for src, tgt in edge_index.t()
            ], device=edge_index.device)
            
            if not edge_mask.any():
                return {
                    'NumRank': float(min(mask_embeddings.shape)),
                    'Erank': float(min(mask_embeddings.shape)),
                    'EDir': 0.0,
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0
                }
            
            masked_edges = edge_index[:, edge_mask]
            node_mapping = {orig_idx.item(): local_idx for local_idx, orig_idx in enumerate(mask_indices)}
            
            remapped_edges = torch.stack([
                torch.tensor([node_mapping[src.item()] for src in masked_edges[0]], device=edge_index.device),
                torch.tensor([node_mapping[tgt.item()] for tgt in masked_edges[1]], device=edge_index.device)
            ])
            
            graphs_in_class = [{
                'X': mask_embeddings,
                'edge_index': remapped_edges,
                'edge_weight': None
            }]
            
            return self.oversmoothing_evaluator.compute_all_metrics(
                X=mask_embeddings,
                edge_index=remapped_edges,
                graphs_in_class=graphs_in_class
            )
            
        except Exception as e:
            print(f"Warning: Could not compute oversmoothing metrics for mask: {e}")
            return None
        
    def _get_gnn_kwargs(self):
        base = {'n_layers': getattr(self.args, 'n_layers', 2)}
        if self.gnn_type == 'gcn':
            base.update({'self_loop': getattr(self.args, 'self_loop', True)})
        elif self.gnn_type == 'gin':
            base.update({'mlp_layers': getattr(self.args, 'mlp_layers', 2),
                         'train_eps': getattr(self.args, 'train_eps', True)})
        elif self.gnn_type in ['gat', 'gatv2']:
            base.update({'heads': getattr(self.args, 'heads', 8),
                         'self_loop': getattr(self.args, 'self_loop', True)})
        return base

    def forward(self, x, edge_index, edge_weight=None):
        return self.predictor(x, edge_index, edge_weight)
        
    def compute_metrics_and_de(self, features, final_edges, final_weights, labels, 
                              idx_train, idx_val, idx_test=None):
        self.eval()
        with torch.no_grad():
            output1, output2 = self.predictor(features, final_edges, final_weights)
            output_avg = (output1 + output2) / 2
            
            data = Data(x=features, edge_index=final_edges)
            if final_weights is not None:
                data.edge_attr = final_weights.unsqueeze(-1) if final_weights.dim() == 1 else final_weights
            data = data.to(self.device)

            try:
                h1 = self.predictor.branch1.get_embeddings(data) if hasattr(self.predictor.branch1, 'get_embeddings') else self.predictor.branch1(data)
                h2 = self.predictor.branch2.get_embeddings(data) if hasattr(self.predictor.branch2, 'get_embeddings') else self.predictor.branch2(data)
                h_avg = (h1 + h2) / 2
            except:
                h_avg = output_avg
            
            metrics = {}
            
            loss_train1 = F.cross_entropy(output1[idx_train], labels[idx_train])
            loss_train2 = F.cross_entropy(output2[idx_train], labels[idx_train])
            metrics['train_loss'] = (loss_train1 + loss_train2) / 2
            
            train_pred = output_avg[idx_train].argmax(dim=1)
            metrics['train_acc'] = (train_pred == labels[idx_train]).float().mean().item()
            metrics['train_f1'] = f1_score(labels[idx_train].cpu(), train_pred.cpu(), average='macro')
            
            val_ce_loss1 = F.cross_entropy(output1[idx_val], labels[idx_val])
            val_ce_loss2 = F.cross_entropy(output2[idx_val], labels[idx_val])
            metrics['val_loss'] = (val_ce_loss1 + val_ce_loss2) / 2
            
            val_pred = output_avg[idx_val].argmax(dim=1)
            metrics['val_acc'] = (val_pred == labels[idx_val]).float().mean().item()
            metrics['val_f1'] = f1_score(labels[idx_val].cpu(), val_pred.cpu(), average='macro')
            
            train_mask = torch.zeros(features.size(0), dtype=torch.bool, device=self.device)
            train_mask[idx_train] = True
            val_mask = torch.zeros(features.size(0), dtype=torch.bool, device=self.device)
            val_mask[idx_val] = True
            
            train_oversmoothing = self._compute_oversmoothing_for_mask(h_avg, final_edges, train_mask, labels)
            val_oversmoothing = self._compute_oversmoothing_for_mask(h_avg, final_edges, val_mask, labels)

            if idx_test is not None:
                test_loss1 = F.cross_entropy(output1[idx_test], labels[idx_test])
                test_loss2 = F.cross_entropy(output2[idx_test], labels[idx_test])
                metrics['test_loss'] = (test_loss1 + test_loss2) / 2
                
                test_pred = output_avg[idx_test].argmax(dim=1)
                metrics['test_acc'] = (test_pred == labels[idx_test]).float().mean().item()
                metrics['test_f1'] = f1_score(labels[idx_test].cpu(), test_pred.cpu(), average='macro')
                
                test_mask = torch.zeros(features.size(0), dtype=torch.bool, device=self.device)
                test_mask[idx_test] = True
                test_oversmoothing = self._compute_oversmoothing_for_mask(h_avg, final_edges, test_mask, labels)
                
                return metrics, train_oversmoothing, val_oversmoothing, test_oversmoothing
            
            return metrics, train_oversmoothing, val_oversmoothing

    def fit(self, features, adj, labels, idx_train, idx_val, idx_test=None, 
            noise_idx=None, clean_idx=None):

        start_time = time.time()
        
        edge_index, _ = from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)
        features = features.to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        
        knn_edges = self._generate_knn_edges(features, edge_index, idx_train)
        
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, 
                             weight_decay=self.args.weight_decay)

        patience = getattr(self.args, 'patience', 8)
        counter = 0
        best_val_f1 = 0
        
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

            neighbor_kl_loss = self.intra_reg(output1, output2, final_edges, final_weights, idx_train)

            total_loss = (pred_loss +
                          self.args.alpha * rec_loss +
                          pseudo_loss +
                          self.args.co_lambda * neighbor_kl_loss)

            total_loss.backward()
            optimizer.step()

            metrics, train_oversmoothing, val_oversmoothing = self.compute_metrics_and_de(
                features, final_edges, final_weights, labels, idx_train, idx_val
            )

            train_de_edir = train_oversmoothing['EDir'] if train_oversmoothing else 0.0
            train_de_traditional = train_oversmoothing['EDir_traditional'] if train_oversmoothing else 0.0
            train_eproj = train_oversmoothing['EProj'] if train_oversmoothing else 0.0
            train_mad = train_oversmoothing['MAD'] if train_oversmoothing else 0.0
            train_num_rank = train_oversmoothing['NumRank'] if train_oversmoothing else 0.0
            train_eff_rank = train_oversmoothing['Erank'] if train_oversmoothing else 0.0

            val_de_edir = val_oversmoothing['EDir'] if val_oversmoothing else 0.0
            val_de_traditional = val_oversmoothing['EDir_traditional'] if val_oversmoothing else 0.0
            val_eproj = val_oversmoothing['EProj'] if val_oversmoothing else 0.0
            val_mad = val_oversmoothing['MAD'] if val_oversmoothing else 0.0
            val_num_rank = val_oversmoothing['NumRank'] if val_oversmoothing else 0.0
            val_eff_rank = val_oversmoothing['Erank'] if val_oversmoothing else 0.0

            print(f"Epoch {epoch:03d} | Train Acc: {metrics['train_acc']:.4f}, Val Acc: {metrics['val_acc']:.4f} | "
                  f"Train F1: {metrics['train_f1']:.4f}, Val F1: {metrics['val_f1']:.4f}")
            print(f"Train DE: {train_de_edir:.4f}, Val DE: {val_de_edir:.4f} | "
                  f"Train DE_trad: {train_de_traditional:.4f}, Val DE_trad: {val_de_traditional:.4f} | "
                  f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                  f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                  f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                  f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")

            if epoch == 0:
                best_val_f1 = metrics['val_f1']
                counter = 0
                self.best_state = {
                    'model': deepcopy(self.state_dict()),
                    'edges': final_edges.clone(),
                    'weights': final_weights.clone()
                }
            elif metrics['val_f1'] > best_val_f1:
                best_val_f1 = metrics['val_f1']
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
            
        total_time = time.time() - start_time
        
        if idx_test is not None:
            final_metrics, final_train_oversmoothing, final_val_oversmoothing, final_test_oversmoothing = self.compute_metrics_and_de(
                features, self.best_state['edges'], self.best_state['weights'], 
                labels, idx_train, idx_val, idx_test
            )
            
            print(f"\nTraining completed in {total_time:.2f}s")
            print(f"Test Loss: {final_metrics['test_loss']:.4f} | Test Acc: {final_metrics['test_acc']:.4f} | Test F1: {final_metrics['test_f1']:.4f}")
            print("Final Oversmoothing Metrics:")
            
            if final_train_oversmoothing is not None:
                print(f"Train: EDir: {final_train_oversmoothing['EDir']:.4f}, EDir_traditional: {final_train_oversmoothing['EDir_traditional']:.4f}, "
                      f"EProj: {final_train_oversmoothing['EProj']:.4f}, MAD: {final_train_oversmoothing['MAD']:.4f}, "
                      f"NumRank: {final_train_oversmoothing['NumRank']:.4f}, Erank: {final_train_oversmoothing['Erank']:.4f}")
            
            if final_val_oversmoothing is not None:
                print(f"Val: EDir: {final_val_oversmoothing['EDir']:.4f}, EDir_traditional: {final_val_oversmoothing['EDir_traditional']:.4f}, "
                      f"EProj: {final_val_oversmoothing['EProj']:.4f}, MAD: {final_val_oversmoothing['MAD']:.4f}, "
                      f"NumRank: {final_val_oversmoothing['NumRank']:.4f}, Erank: {final_val_oversmoothing['Erank']:.4f}")
            
            if final_test_oversmoothing is not None:
                print(f"Test: EDir: {final_test_oversmoothing['EDir']:.4f}, EDir_traditional: {final_test_oversmoothing['EDir_traditional']:.4f}, "
                      f"EProj: {final_test_oversmoothing['EProj']:.4f}, MAD: {final_test_oversmoothing['MAD']:.4f}, "
                      f"NumRank: {final_test_oversmoothing['NumRank']:.4f}, Erank: {final_test_oversmoothing['Erank']:.4f}")
            
        print(f"Training completed! Best validation F1: {best_val_f1:.4f}")

    def test(self, features, labels, idx_test):
        if self.best_state is None:
            print("Model not trained yet.")
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

            test_mask = torch.zeros(features.size(0), dtype=torch.bool, device=self.device)
            test_mask[idx_test] = True
            
            data = Data(x=features, edge_index=self.best_state['edges'])
            if self.best_state['weights'] is not None:
                data.edge_attr = self.best_state['weights'].unsqueeze(-1) if self.best_state['weights'].dim() == 1 else self.best_state['weights']
            data = data.to(self.device)
            
            h1 = self.predictor.branch1.get_embeddings(data) if hasattr(self.predictor.branch1, 'get_embeddings') else self.predictor.branch1(data)
            h2 = self.predictor.branch2.get_embeddings(data) if hasattr(self.predictor.branch2, 'get_embeddings') else self.predictor.branch2(data)
            h_avg = (h1 + h2) / 2

            test_oversmoothing = self._compute_oversmoothing_for_mask(h_avg, self.best_state['edges'], test_mask, labels)

            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
            
            if test_oversmoothing is not None:
                print(f"Test: EDir: {test_oversmoothing['EDir']:.4f}, EDir_traditional: {test_oversmoothing['EDir_traditional']:.4f}, "
                      f"EProj: {test_oversmoothing['EProj']:.4f}, MAD: {test_oversmoothing['MAD']:.4f}, "
                      f"NumRank: {test_oversmoothing['NumRank']:.4f}, Erank: {test_oversmoothing['Erank']:.4f}")
            
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
            mask = consistent & confident
            if mask.sum() == 0:
                return torch.tensor(0.0, device=self.device)
            pseudo_nodes = unlabeled[mask]
            pseudo_labels = class1[mask]

        loss_ce = (
            F.cross_entropy(output1[pseudo_nodes], pseudo_labels) +
            F.cross_entropy(output2[pseudo_nodes], pseudo_labels)
        ) / 2

        loss_kl = (
            F.kl_div(F.log_softmax(output1[pseudo_nodes], dim=1),
                    F.softmax(output2[pseudo_nodes], dim=1), reduction='batchmean') +
            F.kl_div(F.log_softmax(output2[pseudo_nodes], dim=1),
                    F.softmax(output1[pseudo_nodes], dim=1), reduction='batchmean')
        )

        return loss_ce + self.args.co_lambda * loss_kl

    
    def _evaluate(self, features, edge_index, edge_weight, labels, idx_val):
        self.eval()
        with torch.no_grad():
            output1, output2 = self.predictor(features, edge_index, edge_weight)
            pred1 = output1[idx_val].max(1)[1]
            pred2 = output2[idx_val].max(1)[1]
            
            acc1 = accuracy_score(labels[idx_val].cpu(), pred1.cpu())
            acc2 = accuracy_score(labels[idx_val].cpu(), pred2.cpu())
            
            return (acc1 + acc2) / 2