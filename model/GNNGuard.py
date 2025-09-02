import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix, diags
import scipy.sparse as sp
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score

from model.evaluation import OversmoothingMetrics

class GNNGuard(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 attention=True, device=None, P0=0.5, K=2, D2=16):
        super(GNNGuard, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay if attention else 0
        self.attention = attention

        self.P0 = P0
        self.K = K
        self.D2 = D2

        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.labels = None

        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        self.oversmoothing_history = {
            'train': [],
            'val': [],
            'test': []
        }

        self.gate = Parameter(torch.rand(1))
        self.test_value = Parameter(torch.rand(1))

        self.gc_layers = nn.ModuleList()
        self.gc_layers.append(GCNConv(nfeat, nhid, bias=True))
        if self.K > 1:
            self.gc_layers.append(GCNConv(nhid, self.D2, bias=True))
        self.gc_layers.append(GCNConv(D2 if self.K > 1 else nhid, nclass, bias=True))

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

    def forward(self, x, adj):
        x = x.to_dense() if x.is_sparse else x
        x = x.to(self.device)

        for i, gc in enumerate(self.gc_layers[:-1]):
            if self.attention:
                adj_mod = self.att_coef(x, adj, i=i)
                edge_index = adj_mod._indices()
                adj_values = adj_mod._values()
            else:
                edge_index = adj._indices()
                adj_values = adj._values()

            x = gc(x, edge_index, edge_weight=adj_values)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        gc_last = self.gc_layers[-1]
        if self.attention:
            adj_mod = self.att_coef(x, adj, i=self.K)
            edge_index = adj_mod._indices()
            adj_values = adj_mod._values()
        else:
            edge_index = adj._indices()
            adj_values = adj._values()

        x = gc_last(x, edge_index, edge_weight=adj_values)
        return F.log_softmax(x, dim=1)

    def att_coef(self, fea, edge_index, is_lil=False, i=0, debug=False):
        if not is_lil:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()

        n_node = fea.shape[0]

        row, col = edge_index[0].cpu().data.numpy(), edge_index[1].cpu().data.numpy()
        row = np.array(row, dtype=np.int64, copy=True)
        col = np.array(col, dtype=np.int64, copy=True)

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)
        sim_matrix = np.array(sim_matrix, dtype=np.float32, copy=True)
        sim = sim_matrix[row, col].copy()
        sim[sim < self.P0] = 0


        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim

        if att_dense[0, 0] == 1:
            att_dense -= sp.diags(att_dense.diagonal(), offsets=0, format="lil")

        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        if att_dense_norm[0, 0] == 0:
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_edge_weight = att[row, col]
        att_edge_weight = np.array(att_edge_weight, dtype=np.float32, copy=True).flatten()
        att_edge_weight = np.exp(att_edge_weight)
        att_edge_weight = torch.tensor(att_edge_weight, dtype=torch.float32).to(self.device)

        att_adj = torch.tensor(np.vstack((row, col)), dtype=torch.int64).to(self.device)
        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape).to(self.device)

        return new_adj


    def add_loop_sparse(self, adj, fill_value=1):
        row = torch.arange(adj.shape[0], dtype=torch.int64)
        i = torch.stack((row, row), dim=0)
        v = torch.ones(adj.shape[0], dtype=torch.float32)
        I_n = torch.sparse.FloatTensor(i, v, adj.shape).to(self.device)
        return adj + I_n

    def initialize(self):
        for gc in self.gc_layers:
            gc.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val, idx_test=None,
            epochs=200, verbose=True, patience=5):
        self.initialize()

        if not isinstance(adj, torch.Tensor):
            adj = self.sparse_mx_to_torch_sparse_tensor(adj)

        self.features = features.to(self.device)
        self.adj_norm = self.add_loop_sparse(adj.to(self.device))
        self.labels = labels.to(self.device)

        self._train_with_early_stopping(labels, idx_train, idx_val, idx_test, epochs, patience, verbose)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)).to(self.device)
        values = torch.from_numpy(sparse_mx.data).to(self.device)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32, device=self.device)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, idx_test, epochs, patience, verbose):
        if verbose:
            print('Training GNNGuard model')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = float('inf')
        patience_counter = early_stopping
        weights = deepcopy(self.state_dict())

        for i in range(epochs):
            self.train()
            optimizer.zero_grad()
            output_train = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output_train[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                output_val = self.forward(self.features, self.adj_norm)
                loss_val = F.nll_loss(output_val[idx_val], labels[idx_val])

                pred_train_labels = output_train[idx_train].max(1)[1].cpu().numpy()
                pred_val_labels = output_val[idx_val].max(1)[1].cpu().numpy()
                true_train_labels = labels[idx_train].cpu().numpy()
                true_val_labels = labels[idx_val].cpu().numpy()

                train_acc = accuracy_score(true_train_labels, pred_train_labels)
                val_acc = accuracy_score(true_val_labels, pred_val_labels)
                train_f1 = f1_score(true_train_labels, pred_train_labels, average='macro')
                val_f1 = f1_score(true_val_labels, pred_val_labels, average='macro')

                train_mask = torch.zeros(self.features.size(0), dtype=torch.bool, device=self.device)
                val_mask = torch.zeros(self.features.size(0), dtype=torch.bool, device=self.device)
                train_mask[idx_train] = True
                val_mask[idx_val] = True

                edge_index = self.adj_norm._indices()
                train_oversmoothing = self._compute_oversmoothing_for_mask(output_train, edge_index, train_mask, labels)
                val_oversmoothing = self._compute_oversmoothing_for_mask(output_val, edge_index, val_mask, labels)

                if train_oversmoothing is not None:
                    self.oversmoothing_history['train'].append(train_oversmoothing)
                if val_oversmoothing is not None:
                    self.oversmoothing_history['val'].append(val_oversmoothing)

            if verbose:
                train_de = train_oversmoothing['EDir'] if train_oversmoothing else 0.0
                train_de_traditional = train_oversmoothing['EDir_traditional'] if train_oversmoothing else 0.0
                train_eproj = train_oversmoothing['EProj'] if train_oversmoothing else 0.0
                train_mad = train_oversmoothing['MAD'] if train_oversmoothing else 0.0
                train_num_rank = train_oversmoothing['NumRank'] if train_oversmoothing else 0.0
                train_eff_rank = train_oversmoothing['Erank'] if train_oversmoothing else 0.0
                
                val_de = val_oversmoothing['EDir'] if val_oversmoothing else 0.0
                val_de_traditional = val_oversmoothing['EDir_traditional'] if val_oversmoothing else 0.0
                val_eproj = val_oversmoothing['EProj'] if val_oversmoothing else 0.0
                val_mad = val_oversmoothing['MAD'] if val_oversmoothing else 0.0
                val_num_rank = val_oversmoothing['NumRank'] if val_oversmoothing else 0.0
                val_eff_rank = val_oversmoothing['Erank'] if val_oversmoothing else 0.0
                
                print(f"Epoch {i+1:03d} | Train Loss: {loss_train:.4f}, Val Loss: {loss_val:.4f} | "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                    f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
                print(f"Train DE: {train_de:.4f}, Val DE: {val_de:.4f} | "
                    f"Train DE_trad: {train_de_traditional:.4f}, Val DE_trad: {val_de_traditional:.4f} | "
                    f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                    f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                    f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                    f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")

            if loss_val < best_loss_val:
                best_loss_val = loss_val
                self.output = output_val
                weights = deepcopy(self.state_dict())
                patience_counter = early_stopping
            else:
                patience_counter -= 1

            if patience_counter <= 0:
                if verbose:
                    print(f'Early stopping at epoch {i}, best_loss_val={best_loss_val:.4f}')
                break

        self.load_state_dict(weights)
        
        if idx_test is not None:
            test_mask = torch.zeros(self.features.size(0), dtype=torch.bool, device=self.device)
            train_mask_final = torch.zeros(self.features.size(0), dtype=torch.bool, device=self.device)
            val_mask_final = torch.zeros(self.features.size(0), dtype=torch.bool, device=self.device)
            
            train_mask_final[idx_train] = True
            val_mask_final[idx_val] = True
            test_mask[idx_test] = True
            
            self.eval()
            with torch.no_grad():
                final_output = self.forward(self.features, self.adj_norm)
                
                pred_test_labels = final_output[idx_test].max(1)[1].cpu().numpy()
                true_test_labels = self.labels[idx_test].cpu().numpy()
                test_acc = accuracy_score(true_test_labels, pred_test_labels)
                test_f1 = f1_score(true_test_labels, pred_test_labels, average='macro')
                test_loss = F.nll_loss(final_output[idx_test], self.labels[idx_test])
                
                edge_index = self.adj_norm._indices()
                
                final_train_oversmoothing = self._compute_oversmoothing_for_mask(final_output, edge_index, train_mask_final, self.labels)
                final_val_oversmoothing = self._compute_oversmoothing_for_mask(final_output, edge_index, val_mask_final, self.labels)
                final_test_oversmoothing = self._compute_oversmoothing_for_mask(final_output, edge_index, test_mask, self.labels)
                
                if final_test_oversmoothing is not None:
                    self.oversmoothing_history['test'].append(final_test_oversmoothing)
            
            final_metrics = {
                'test_loss': test_loss.item(),
                'test_acc': test_acc,
                'test_f1': test_f1
            }
            
            if verbose:
                print(f"\nTraining completed")
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

    def test(self, idx_test):
        self.eval()
        with torch.no_grad():
            output = self.forward(self.features, self.adj_norm)
            loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
            pred_test_labels = output[idx_test].max(1)[1].cpu().numpy()
            true_test_labels = self.labels[idx_test].cpu().numpy()
            acc_test = accuracy_score(true_test_labels, pred_test_labels)
            f1_test = f1_score(true_test_labels, pred_test_labels, average='macro')
            
            test_mask = torch.zeros(self.features.size(0), dtype=torch.bool, device=self.device)
            test_mask[idx_test] = True
            edge_index = self.adj_norm._indices()
            test_oversmoothing = self._compute_oversmoothing_for_mask(output, edge_index, test_mask, self.labels)

        print(f"Test Loss: {loss_test:.4f} | Test Acc: {acc_test:.4f} | Test F1: {f1_test:.4f}")
        
        if test_oversmoothing is not None:
            print(f"Test Oversmoothing: EDir: {test_oversmoothing['EDir']:.4f}, EDir_traditional: {test_oversmoothing['EDir_traditional']:.4f}, "
                  f"EProj: {test_oversmoothing['EProj']:.4f}, MAD: {test_oversmoothing['MAD']:.4f}, "
                  f"NumRank: {test_oversmoothing['NumRank']:.4f}, Erank: {test_oversmoothing['Erank']:.4f}")
        
        return acc_test, f1_test
    
    def get_oversmoothing_history(self):
        return self.oversmoothing_history