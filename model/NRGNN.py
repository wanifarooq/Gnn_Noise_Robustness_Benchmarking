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
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, self_loop=True ,device=None):
        super(GCN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GCNConv(nfeat, nhid, bias=with_bias,add_self_loops=self_loop)
        self.gc2 = GCNConv(nhid, nclass, bias=with_bias,add_self_loops=self_loop)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        
    def forward(self, x, edge_index, edge_weight):
        if self.with_relu:
            x = F.relu(self.gc1(x, edge_index,edge_weight))
        else:
            x = self.gc1(x, edge_index,edge_weight)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)
        return x

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, **kwargs):
        if initialize:
            self.initialize()
        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(adj)
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.float().to(self.device)
        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        self.features = features.to(self.device)
        self.labels = torch.LongTensor(np.array(labels)).to(self.device)
        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose, patience=10):
        if verbose:
            print('Training gcn model')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = float('inf')
        best_acc_val = 0
        best_weights = None
        patience_counter = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            if loss_val.item() < best_loss_val:
                best_loss_val = loss_val.item()
                best_acc_val = acc_val
                best_weights = deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if verbose and i % 10 == 0:
                print(f"Epoch {i}, train loss: {loss_train.item():.4f}, "
                      f"val loss: {loss_val.item():.4f}, val acc: {acc_val.item():.4f}")
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {i}, best val loss: {best_loss_val:.4f}")
                break
        self.load_state_dict(best_weights)
        self.eval()
        self.output = self.forward(self.features, self.edge_index, self.edge_weight)

    def test(self, idx_test):
        self.eval()
        output = self.forward(self.features, self.edge_index,self.edge_weight)
        loss_test = F.cross_entropy(output[idx_test], self.labels[idx_test])
        acc_test = accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return output

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
    
    def initialize(self):
        if hasattr(self.base_model, 'initialize'):
            self.base_model.initialize()
        else:
            for module in self.base_model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

class NRGNN:
    def __init__(self, args, device, base_model=None):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0
        self.best_pred = None
        self.best_graph = None
        self.best_model_index = None
        self.weights = None
        self.estimator = None
        self.model = None
        self.pred_edge_index = None
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.patience = getattr(args, "patience", 10)
        
        self.base_model_class = base_model

    def fit(self, features, adj, labels, idx_train, idx_val):
        args = self.args

        edge_index, _ = from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
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

        predictor_base = deepcopy(self.base_model_class)
        model_base = deepcopy(self.base_model_class)
            
        self.predictor = GNNs(predictor_base, self.device).to(self.device)
        self.model = GNNs(model_base, self.device).to(self.device)

        self.estimator = EstimateAdj(features.shape[1], args, idx_train, device=self.device).to(self.device)
        self.pred_edge_index = self.get_train_edge(edge_index, features, self.args.n_p, idx_train)

        self.optimizer = optim.Adam(
            list(self.model.parameters())
            + list(self.estimator.parameters())
            + list(self.predictor.parameters()),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.patience = getattr(args, "patience", 10)

        t_total = time.time()
        try:
            for epoch in range(args.epochs):
                self.train(epoch, features, edge_index, idx_train, idx_val)
        except StopIteration:
            if args.debug:
                print(f"Training stopped early at epoch {epoch+1}")

        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print("Picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)
        self.predictor.load_state_dict(self.predictor_model_weights)
        self.test(idx_val)

    def train(self, epoch, features, edge_index, idx_train, idx_val):
        args = self.args

        t = time.time()
        self.model.train()
        self.predictor.train()
        self.optimizer.zero_grad()

        representations, rec_loss = self.estimator(edge_index, features)

        if self.pred_edge_index is not None and self.pred_edge_index.shape[1] > 0:
            predictor_weights = self.estimator.get_estimated_weigths(self.pred_edge_index, representations)
            pred_edge_index = torch.cat([edge_index, self.pred_edge_index], dim=1)
            predictor_weights = torch.cat(
                [torch.ones([edge_index.shape[1]], device=self.device), predictor_weights], dim=0
            )
        else:
            pred_edge_index = edge_index
            predictor_weights = torch.ones([edge_index.shape[1]], device=self.device)

        log_pred = self.predictor(features, pred_edge_index, predictor_weights)

        if self.best_pred is None:
            pred = F.softmax(log_pred, dim=1).detach()
            self.best_pred = pred
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(self.best_pred)
        else:
            pred = self.best_pred

        if self.unlabel_edge_index is not None and self.unlabel_edge_index.shape[1] > 0:
            estimated_weights = self.estimator.get_estimated_weigths(self.unlabel_edge_index, representations)
            estimated_weights = torch.cat([predictor_weights, estimated_weights], dim=0)
            model_edge_index = torch.cat([pred_edge_index, self.unlabel_edge_index], dim=1)
        else:
            estimated_weights = predictor_weights
            model_edge_index = pred_edge_index

        output = self.model(features, model_edge_index, estimated_weights)
        pred_model = F.softmax(output, dim=1)

        eps = 1e-8
        pred_model = pred_model.clamp(eps, 1 - eps)

        if len(self.idx_add) > 0:
            loss_add = (-torch.sum(pred[self.idx_add] * torch.log(pred_model[self.idx_add]), dim=1)).mean()
        else:
            loss_add = torch.tensor(0.0, device=self.device, requires_grad=True)

        loss_pred = F.cross_entropy(log_pred[idx_train], self.labels[idx_train])
        loss_gcn = F.cross_entropy(output[idx_train], self.labels[idx_train])

        total_loss = loss_gcn + loss_pred + self.args.alpha * rec_loss + self.args.beta * loss_add

        if torch.isnan(total_loss):
            print(f"NaN detected! loss_gcn: {loss_gcn.item()}, loss_pred: {loss_pred.item()}, "
                  f"loss_add: {loss_add.item()}, rec_loss: {rec_loss.item()}")
            return

        total_loss.backward()
        self.optimizer.step()

        train_loss = total_loss.item()
        train_acc = accuracy(output[idx_train].detach(), self.labels[idx_train])
        train_f1 = f1_score(self.labels[idx_train].cpu(),
                            output[idx_train].argmax(dim=1).cpu(),
                            average="macro")

        self.model.eval()
        self.predictor.eval()
        with torch.no_grad():
            pred = F.softmax(self.predictor(features, pred_edge_index, predictor_weights), dim=1)
            output = self.model(features, model_edge_index, estimated_weights.detach())

            acc_pred_val = accuracy(pred[idx_val], self.labels[idx_val])
            val_loss = F.cross_entropy(output[idx_val], self.labels[idx_val]).item()
            val_acc = accuracy(output[idx_val], self.labels[idx_val])
            val_f1 = f1_score(self.labels[idx_val].cpu(),
                            output[idx_val].argmax(dim=1).cpu(),
                            average="macro")

            if acc_pred_val > getattr(self, "best_acc_pred_val", 0):
                self.best_acc_pred_val = acc_pred_val
                self.best_pred_graph = predictor_weights.detach()
                self.best_pred = pred.detach()
                self.predictor_model_weights = deepcopy(self.predictor.state_dict())

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_graph = estimated_weights.detach()
            self.best_model_index = model_edge_index
            self.weights = deepcopy(self.model.state_dict())
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        if self.early_stop_counter >= self.patience:
            if args.debug:
                print(f"Early stopping at epoch {epoch+1}, best val loss: {self.best_val_loss:.4f}")
            raise StopIteration

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
              f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

        if args.debug:
            print(f"Epoch: {epoch+1:04d}",
                  f"loss_gcn: {loss_gcn.item():.4f}",
                  f"loss_pred: {loss_pred.item():.4f}",
                  f"loss_add: {loss_add.item() if not torch.isnan(loss_add) else 0.0:.4f}",
                  f"rec_loss: {rec_loss.item():.4f}",
                  f"loss_total: {total_loss.item():.4f}")
            print(f"Epoch: {epoch+1:04d}",
                  f"acc_train: {train_acc:.4f}",
                  f"acc_val: {val_acc:.4f}",
                  f"val_loss: {val_loss:.4f}",
                  f"acc_pred_val: {acc_pred_val:.4f}",
                  f"time: {time.time() - t:.4f}s")
            print(f"Size of add idx is {len(self.idx_add)}")

    def test(self, idx_test):
        features = self.features
        labels = self.labels

        self.predictor.eval()
        estimated_weights = self.best_pred_graph
        pred_edge_index = torch.cat([self.edge_index,self.pred_edge_index],dim=1)
        output = self.predictor(features, pred_edge_index,estimated_weights)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tPredictor results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))

        self.model.eval()
        estimated_weights = self.best_graph
        model_edge_index = self.best_model_index
        output = self.model(features, model_edge_index, estimated_weights)

        loss_test = F.cross_entropy(output[idx_test], labels[idx_test]).item()
        acc_test = accuracy(output[idx_test], labels[idx_test])
        f1_test = f1_score(labels[idx_test].cpu(),
                          output[idx_test].argmax(dim=1).cpu(),
                          average="macro")

        print(f"Test Loss: {loss_test:.4f} | Test Acc: {acc_test:.4f} | Test F1: {f1_test:.4f}")
        return float(acc_test)

    def get_train_edge(self, edge_index, features, n_p, idx_train):
        if n_p == 0:
            return None

        poten_edges = []
        if n_p > len(idx_train) or n_p < 0:
            for i in range(len(features)):
                indices = set(idx_train)
                indices = indices - set(edge_index[1,edge_index[0]==i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            for i in range(len(features)):
                sim = torch.div(torch.matmul(features[i],features[idx_train].T), features[i].norm()*features[idx_train].norm(dim=1))
                _,rank = sim.topk(n_p)
                if rank.max() < len(features) and rank.min() >= 0:
                    indices = idx_train[rank.cpu().numpy()]
                    indices = set(indices)
                else:
                    indices = set()
                indices = indices - set(edge_index[1,edge_index[0]==i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = to_undirected(poten_edges,len(features)).to(self.device)
        return poten_edges

    def get_model_edge(self, pred):
        if len(self.idx_unlabel) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), []

        confidence_mask = pred.max(dim=1)[0][self.idx_unlabel] > self.args.p_u
        idx_add = self.idx_unlabel[confidence_mask]
        
        if len(idx_add) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), idx_add

        row = self.idx_unlabel.repeat(len(idx_add))
        col = idx_add.repeat(len(self.idx_unlabel),1).T.flatten()
        mask = (row != col)
        
        if mask.sum() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), idx_add
        
        unlabel_edge_index = torch.stack([row[mask], col[mask]], dim=0)
        return unlabel_edge_index, idx_add
                    
class EstimateAdj(nn.Module):
    def __init__(self, nfea, args, idx_train ,device='cuda'):
        super(EstimateAdj, self).__init__()
        
        self.estimator = GCN(nfea, args.edge_hidden, args.edge_hidden,dropout=0.0,device=device)
        self.device = device
        self.args = args
        self.representations = 0

    def forward(self, edge_index, features):
        representations = self.estimator(features,edge_index,\
                                        torch.ones([edge_index.shape[1]]).to(self.device).float())
        rec_loss = self.reconstruct_loss(edge_index, representations)
        return representations,rec_loss
    
    def get_estimated_weigths(self, edge_index, representations):
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0,x1),dim=1)
        estimated_weights = F.relu(output)
        estimated_weights = torch.where(estimated_weights < self.args.t_small, 
                                       torch.zeros_like(estimated_weights), 
                                       estimated_weights)
        return estimated_weights
    
    def reconstruct_loss(self, edge_index, representations):
        num_nodes = representations.shape[0]
        randn = negative_sampling(edge_index,num_nodes=num_nodes, num_neg_samples=self.args.n_n*num_nodes)
        randn = randn[:,randn[0]<randn[1]]
        edge_index = edge_index[:, edge_index[0]<edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0,neg1),dim=1)
        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0,pos1),dim=1)
        rec_loss = (F.mse_loss(neg,torch.zeros_like(neg), reduction='sum') \
                    + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                    * num_nodes/(randn.shape[1] + edge_index.shape[1]) 
        return rec_loss