
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

class GNNGuard(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, 
                 attention=True, device=None):
        super(GNNGuard, self).__init__()
        
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay if attention else 0
        self.attention = attention
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        
        self.gate = Parameter(torch.rand(1))
        self.test_value = Parameter(torch.rand(1))
        self.drop_learn_1 = nn.Linear(2, 1)
        self.drop_learn_2 = nn.Linear(2, 1)
        
        self.gc1 = GCNConv(nfeat, nhid, bias=True)
        self.gc2 = GCNConv(nhid, nclass, bias=True)

    def forward(self, x, adj):
        x = x.to_dense() if x.is_sparse else x
        x = x.to(self.device)
        
        if self.attention:
            adj = self.att_coef(x, adj, i=0)
        edge_index = adj._indices()
        x = self.gc1(x, edge_index, edge_weight=adj._values())
        x = F.relu(x)
        
        if self.attention:
            adj_2 = self.att_coef(x, adj, i=1)
            adj_memory = adj_2.to_dense()
            row, col = adj_memory.nonzero()[:,0], adj_memory.nonzero()[:,1]
            edge_index = torch.stack((row, col), dim=0)
            adj_values = adj_memory[row, col]
        else:
            edge_index = adj._indices()
            adj_values = adj._values()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight=adj_values)
        
        return F.log_softmax(x, dim=1)

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()

        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0

        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        if att_dense_norm[0, 0] == 0:
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32).to(self.device)
        att_adj = torch.tensor(att_adj, dtype=torch.int64).to(self.device)

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape).to(self.device)
        return new_adj

    def add_loop_sparse(self, adj, fill_value=1):
        row = torch.arange(0, int(adj.shape[0]), dtype=torch.int64)
        i = torch.stack((row, row), dim=0)
        v = torch.ones(adj.shape[0], dtype=torch.float32)
        shape = adj.shape
        I_n = torch.sparse.FloatTensor(i, v, shape)
        return adj + I_n.to(self.device)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.drop_learn_1.reset_parameters()
        self.drop_learn_2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val, idx_test=None, 
        train_iters=200, verbose=False, patience=5):
        self.initialize()
        
        if type(adj) is not torch.Tensor:
            adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        
        features = features.to(self.device)
        adj = adj.to(self.device)
        labels = labels.to(self.device)
        
        adj = self.add_loop_sparse(adj)
        self.adj_norm = adj
        self.features = features
        self.labels = labels

        self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        ).to(self.device)
        values = torch.from_numpy(sparse_mx.data).to(self.device)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32, device=self.device)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('Training GNNGuard model')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = float('inf')
        patience_counter = early_stopping
        weights = deepcopy(self.state_dict())

        for i in range(train_iters):
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

            if verbose:
                print(f"Epoch {i:03d} | Train Loss: {loss_train:.4f}, Val Loss: {loss_val:.4f} | "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                    f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

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

    def test(self, idx_test):
        self.eval()
        with torch.no_grad():
            output = self.forward(self.features, self.adj_norm)
            loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
            pred_test_labels = output[idx_test].max(1)[1].cpu().numpy()
            true_test_labels = self.labels[idx_test].cpu().numpy()
            acc_test = accuracy_score(true_test_labels, pred_test_labels)
            f1_test = f1_score(true_test_labels, pred_test_labels, average='macro')

        print(f"Test Loss: {loss_test:.4f} | Test Acc: {acc_test:.4f} | Test F1: {f1_test:.4f}")
        return acc_test, f1_test
