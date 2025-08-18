import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import copy
import time
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph
from torch_geometric.data import Data as PyGData

def BinaryLabelToPosNeg(labels):
    if isinstance(labels, torch.Tensor):
        return 2 * labels - 1
    else:
        return 2 * labels - 1

def accuracy(preds, labels):
    preds = preds.long() if isinstance(preds, torch.Tensor) else torch.tensor(preds).long()
    labels = labels.long() if isinstance(labels, torch.Tensor) else torch.tensor(labels).long()
    return (preds == labels).float().mean().item()

def preprocess_graph_torch(adj):
    device = adj.device
    adj = adj + torch.eye(adj.size(0), device=device)
    rowsum = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

def row_normalize_torch(mx):
    rowsum = mx.sum(dim=1)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0
    r_mat_inv = torch.diag(r_inv)
    return r_mat_inv @ mx

def debug_data_dict(data_dict):
    print("Data dict debug")
    print(f"N (total nodes): {data_dict['N']}")
    print(f"K (num classes): {data_dict['K']}")
    print(f"_z_obs shape: {data_dict['_z_obs'].shape}")
    print(f"_Z_obs shape: {data_dict['_Z_obs'].shape}")
    print(f"Classes in _z_obs: {np.unique(data_dict['_z_obs'])}")
    print(f"Max class: {data_dict['_z_obs'].max()}")
    print(f"Min class: {data_dict['_z_obs'].min()}")
    
    if 'nodes_AB_all' in data_dict:
        print(f"nodes_AB_all length: {len(data_dict['nodes_AB_all'])}")
        print(f"Classes in AB subset: {np.unique(data_dict['_z_obs'][data_dict['nodes_AB_all']])}")
    
    print("-"*50)

def prepare_simpledata_attrs(pyg_data):
    row, col = pyg_data.edge_index.cpu().numpy()
    N = pyg_data.num_nodes
    A_obs = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
    A_obs = A_obs + A_obs.T
    A_obs.data = np.ones_like(A_obs.data)
    A_obs.eliminate_zeros()

    X_obs = pyg_data.x.cpu().numpy()
    z_obs = pyg_data.y.cpu().numpy()

    K = int(z_obs.max() + 1)
    print(f"Detected {K} classes in data: {np.unique(z_obs)}")

    split_train = np.where(pyg_data.train_mask.cpu().numpy())[0]
    split_val = np.where(pyg_data.val_mask.cpu().numpy())[0]
    split_test = np.where(pyg_data.test_mask.cpu().numpy())[0]

    train_mask = pyg_data.train_mask.cpu().numpy()
    val_mask = pyg_data.val_mask.cpu().numpy()
    test_mask = pyg_data.test_mask.cpu().numpy()

    Z_obs = np.eye(K)[z_obs].copy()
    
    assert Z_obs.shape == (N, K), f"Z_obs shape {Z_obs.shape} should be ({N}, {K})"
    assert z_obs.max() < K, f"Max label {z_obs.max()} should be < K={K}"

    data_dict = {
        '_A_obs': A_obs,
        '_X_obs': X_obs,
        '_z_obs': z_obs,
        'N': N,
        'K': K,
        'split_train': split_train,
        'split_val': split_val,
        'split_test': split_test,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        '_Z_obs': Z_obs,
    }
    
    debug_data_dict(data_dict)
    
    return data_dict

def resetBinaryClass_init(data_dict, a, b):

    nodes_A = np.where(data_dict['_z_obs'] == a)[0]
    nodes_B = np.where(data_dict['_z_obs'] == b)[0]
    nodes_AB = np.concatenate([nodes_A, nodes_B])

    nodes_AB_train = np.intersect1d(nodes_AB, data_dict['split_train'])
    nodes_AB_val = np.intersect1d(nodes_AB, data_dict['split_val'])
    nodes_AB_test = np.intersect1d(nodes_AB, data_dict['split_test'])

    nodes_AB_all = np.concatenate([
        nodes_AB_train,
        np.setdiff1d(nodes_AB, nodes_AB_train)
    ])

    mask_train_rel = np.isin(nodes_AB_all, nodes_AB_train)
    split_unlabeled = np.where(~mask_train_rel)[0]

    labels_AB = data_dict['_z_obs'][nodes_AB_all].copy()
    labels_AB[labels_AB == a] = 0
    labels_AB[labels_AB == b] = 1

    data_dict['nodes_AB_train'] = nodes_AB_train
    data_dict['nodes_AB_val'] = nodes_AB_val
    data_dict['nodes_AB_test'] = nodes_AB_test
    data_dict['nodes_AB_all'] = nodes_AB_all
    data_dict['split_unlabeled'] = split_unlabeled
    data_dict['_z_obs_bin'] = labels_AB

def recover_data(data_dict):
    data_dict['_z_obs'] = np.argmax(data_dict['_Z_obs'], axis=1)

def resetz_by_Z(data_dict):
    data_dict['_z_obs'] = np.argmax(data_dict['_Z_obs'], axis=1)

def preprocess_graph_numpy(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

def row_normalize_numpy(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class Attack:
    def __init__(self, data_dict, gpu_id, atkEpoch, gcnL2):
        self.data = data_dict
        self.gpu_id = gpu_id
        self.atkEpoch = atkEpoch
        self.gcnL2 = gcnL2
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    def GNN_test(self, gnn_model, epochs=200, early_stopping=10):
        model = copy.deepcopy(gnn_model).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
    
        X = torch.tensor(self.data['_X_obs'][self.data['nodes_AB_all']], dtype=torch.float32, device=self.device)
        y = torch.tensor(self.data['_z_obs_bin'], dtype=torch.long, device=self.device)
    
        train_mask = torch.tensor(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_train']), device=self.device)
        val_mask = torch.tensor(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_val']), device=self.device)
        test_mask = torch.tensor(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_test']), device=self.device)
    
        A = sp.coo_matrix(self.data['_A_obs'])
        edge_index, edge_weight = from_scipy_sparse_matrix(A)
        edge_index = edge_index.to(self.device)
    
        nodes = torch.tensor(self.data['nodes_AB_all'], dtype=torch.long, device=self.device)
        edge_index_sub, _ = subgraph(nodes, edge_index, relabel_nodes=True, num_nodes=A.shape[0])
    
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            data_batch = PyGData(x=X, edge_index=edge_index_sub, y=y)
            
            out = model(data_batch)
            
            loss = criterion(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
    
            model.eval()
            with torch.no_grad():
                data_batch = PyGData(x=X, edge_index=edge_index_sub, y=y)
                out_val = model(data_batch)
                val_loss = criterion(out_val[val_mask], y[val_mask]).item()
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping:
                    break
    
        model.load_state_dict(best_model_wts)
        model.eval()
        with torch.no_grad():
            data_batch = PyGData(x=X, edge_index=edge_index_sub, y=y)
            out = model(data_batch)
            
            probs = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)
    
            correct = pred[test_mask] == y[test_mask]
            acc = int(correct.sum()) / int(test_mask.sum())
    
            eachAcc = []
            for c in range(probs.shape[1]):
                mask_c = (y[test_mask] == c)
                if mask_c.sum() == 0:
                    eachAcc.append(None)
                else:
                    eachAcc.append(int((pred[test_mask][mask_c] == c).sum()) / int(mask_c.sum()))
    
            loss = criterion(out[test_mask], y[test_mask]).item()
    
        return acc, loss, eachAcc
    
    def getK_GCN(self):
        A = self.data['_A_obs'][np.ix_(self.data['nodes_AB_all'], self.data['nodes_AB_all'])]
        if hasattr(A, "toarray"):
            A = A.toarray()
        A = torch.tensor(A, dtype=torch.float32, device=self.device)
        A = preprocess_graph_torch(A)
        
        X = self.data['_X_obs'][self.data['nodes_AB_all']]
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = row_normalize_torch(X)
        
        X_bar = A @ A @ X
        
        train_indices_in_AB = np.where(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_train']))[0]
        X_bar_l = X_bar[train_indices_in_AB, :]
        
        reg_identity = self.gcnL2 * torch.eye(X_bar_l.size(1), device=self.device)
        tmp = X_bar_l.T @ X_bar_l + reg_identity
        tmp = torch.linalg.pinv(tmp)
        tmp = tmp @ X_bar_l.T
        
        K = X_bar @ tmp
        K = K[self.data['split_unlabeled'], :]
        
        return K.cpu().numpy()
    
    def gradient_attack_sgd_pytorch(self, c_max):
        train_indices_in_AB = np.where(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_train']))[0]
        unlabeled_indices_in_AB = self.data['split_unlabeled']

        self.data['train_indices_in_AB'] = train_indices_in_AB
        self.data['unlabeled_indices_in_AB'] = unlabeled_indices_in_AB

        train_labels_binary = self.data['_z_obs_bin'][train_indices_in_AB]
        unlabeled_labels_binary = self.data['_z_obs_bin'][unlabeled_indices_in_AB]
        
        y_l = BinaryLabelToPosNeg(train_labels_binary).reshape(-1, 1)
        y_u = BinaryLabelToPosNeg(unlabeled_labels_binary).reshape(-1, 1)
    
        device = self.device
        y_l_tensor = torch.tensor(y_l, dtype=torch.float32, device=device)
        y_u_tensor = torch.tensor(y_u, dtype=torch.float32, device=device)
        K_tensor = torch.tensor(self.K, dtype=torch.float32, device=device)

        alpha = torch.nn.Parameter(0.5 * torch.ones_like(y_l_tensor, device=device))
        optimizer = optim.SGD([alpha], lr=1e-4)
        tau = self.tau
    
        for step in range(self.atkEpoch):
            optimizer.zero_grad()
            u1 = torch.rand_like(alpha)
            u2 = torch.rand_like(alpha)
            gumbel1 = -torch.log(-torch.log(u1 + 1e-20) + 1e-20)
            gumbel2 = -torch.log(-torch.log(u2 + 1e-20) + 1e-20)
            epsilon = gumbel1 - gumbel2
    
            logit = torch.log(alpha / (1 - alpha) + 1e-20)
            tmp = torch.exp((logit + epsilon) / 0.5)
            z = 2 / (1 + tmp) - 1
    
            y_l_tmp = y_l_tensor * z

            y_u_preds = torch.tanh(tau * torch.matmul(K_tensor, y_l_tmp))
            loss = torch.mean(y_u_preds * y_u_tensor)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                alpha.clamp_(0, 1)
    
        alpha_np = alpha.detach().cpu().numpy().reshape(-1)
        idx = np.argsort(alpha_np)[::-1]
        d_y = np.ones(len(train_indices_in_AB))
        count = 0
        flip = {}
        
        for i in idx:
            if alpha_np[i] > 0.5 and count < c_max:
                d_y[i] = -1
                count += 1

                original_node_idx = self.data['nodes_AB_all'][train_indices_in_AB[i]]
                flip[original_node_idx] = alpha_np[i]
                if count == c_max:
                    break
    
        trainLabels = y_l.reshape(-1)
        acc_cln, pred_closed_cln = self.closedForm_bin(trainLabels)
        
        trainLabels_atk = trainLabels * d_y
        acc_atk, pred_closed_atk = self.closedForm_bin(trainLabels_atk)
    
        _z_obs_bin_atk = self.data['_z_obs_bin'].copy()
        for i in range(len(train_indices_in_AB)):
            if d_y[i] == -1:
                ab_idx = train_indices_in_AB[i]
                _z_obs_bin_atk[ab_idx] = 1 - _z_obs_bin_atk[ab_idx]
    
        return d_y, _z_obs_bin_atk, acc_cln, acc_atk, pred_closed_cln, pred_closed_atk

    def closedForm_bin(self, trainLabels):
        trainLabels_tensor = torch.tensor(trainLabels, dtype=torch.float32, device=self.device)
        K_tensor = torch.tensor(self.K, dtype=torch.float32, device=self.device)
        y_pred = K_tensor @ trainLabels_tensor
        y_pred_sign = torch.sign(y_pred)
        
        unlabeled_labels_binary = self.data['_z_obs_bin'][self.data['unlabeled_indices_in_AB']]
        true_labels_binary = BinaryLabelToPosNeg(unlabeled_labels_binary)
        true_labels_tensor = torch.tensor(true_labels_binary, device=self.device, dtype=torch.float32)

        acc = accuracy(y_pred_sign.cpu(), true_labels_tensor.cpu())
        return acc, y_pred.cpu().numpy()

    def optimize_tau(self, c_max, tau_candidates=[0.1, 0.5, 1.0, 2.0, 5.0]):
        best_tau = None
        best_obj = -float('inf')
        best_results = None
        for tau in tau_candidates:
            self.tau = tau
            d_y, _, acc_bin_cln, acc_bin_atk, preds_closed_cln, preds_closed_atk = self.gradient_attack_sgd_pytorch(c_max)
            if acc_bin_cln - acc_bin_atk > best_obj:
                best_obj = acc_bin_cln - acc_bin_atk
                best_tau = tau
                best_results = (d_y, _, acc_bin_cln, acc_bin_atk, preds_closed_cln, preds_closed_atk)
        self.tau = best_tau
        print(f"Best tau found: {best_tau}")
        return best_results

    def binaryAttack_multiclass_with_clean(self, c_max, a=2, b=3, gnn_model=None):
        time1 = time.time()
        resetBinaryClass_init(self.data, a, b)
        
        self.K = self.getK_GCN()
        time2 = time.time()
        
        best_results = self.optimize_tau(c_max)
        
        if best_results:
            d_y, _, acc_bin_cln, acc_bin_atk, preds_closed_cln, preds_closed_atk = best_results
            print(f"Binary Clean accuracy: {acc_bin_cln:.4f}")
            print(f"Binary Attacked accuracy: {acc_bin_atk:.4f}")
            print(f"Binary Attack success: {acc_bin_cln - acc_bin_atk:.4f}")
        else:
            acc_bin_cln = acc_bin_atk = 0
            d_y = np.ones(len(np.where(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_train']))[0]))
    
        train_indices_in_AB = np.where(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_train']))[0]
        train_indices_in_AB = np.where(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_train']))[0]
        flip_mask_in_train = (d_y == -1)
        flipNodes = self.data['nodes_AB_all'][train_indices_in_AB[flip_mask_in_train]]

    
        time3 = time.time()
        print(f"overall: {time3 - time1}, pre-process: {time2 - time1}, optimize: {time3 - time2}")
    
        recover_data(self.data)
    
        print(f"Debug info:")
        print(f"Classes a={a}, b={b}")
        print(f"_Z_obs shape: {self.data['_Z_obs'].shape}")
        print(f"Number of classes K: {self.data['K']}")
        print(f"Flip nodes: {flipNodes}")
        
        if a >= self.data['K'] or b >= self.data['K']:
            raise ValueError(f"Class indices a={a} or b={b} are out of bounds. Dataset has {self.data['K']} classes (0 to {self.data['K']-1})")
    
        acc_clean_gnn = None
        if gnn_model is not None:
            print("Testing GNN with clean data...")
            acc_clean_gnn, _, _ = self.GNN_test(gnn_model)
            print(f"GNN Clean accuracy: {acc_clean_gnn:.4f}")
    
        _Z_obs_atk = np.array(self.data['_Z_obs'])
        
        if _Z_obs_atk.shape[1] != self.data['K']:
            print(f"Warning: _Z_obs shape {_Z_obs_atk.shape} doesn't match K={self.data['K']}")
            _Z_obs_atk = np.eye(self.data['K'])[self.data['_z_obs']]
        
        for i in flipNodes:
            if self.data['_z_obs'][i] == a:
                _Z_obs_atk[i] = np.zeros(self.data['K'])
                _Z_obs_atk[i][b] = 1
            elif self.data['_z_obs'][i] == b:
                _Z_obs_atk[i] = np.zeros(self.data['K'])
                _Z_obs_atk[i][a] = 1
            else:
                print(f"no change (not in target classes)")
        
        self.data['_Z_obs'] = _Z_obs_atk
        resetz_by_Z(self.data)
        resetBinaryClass_init(self.data, a=a, b=b)

        acc_attacked_gnn = None
        if gnn_model is not None:
            acc_test_mul_atk = []
            for run in range(5):
                print(f"Running attacked GNN test {run+1}/5...")
                acc, _, _ = self.GNN_test(gnn_model)
                acc_test_mul_atk.append(acc)
            
            acc_attacked_gnn = sum(acc_test_mul_atk) / len(acc_test_mul_atk)
            print(f"Test accuracies after attack: {acc_test_mul_atk}")
            print(f"Average test accuracy after attack: {acc_attacked_gnn:.4f}")
    
        recover_data(self.data)
        
        return {
            'binary_clean_acc': acc_bin_cln,
            'binary_attacked_acc': acc_bin_atk,
            'gnn_clean_acc': acc_clean_gnn,
            'gnn_attacked_acc': acc_attacked_gnn,
            'attack_success': acc_bin_cln - acc_bin_atk if best_results else 0,
            'gnn_attack_success': acc_clean_gnn - acc_attacked_gnn if (acc_clean_gnn and acc_attacked_gnn) else 0,
            'flipped_nodes': flipNodes
        }
    
    def binaryAttack_multiclass(self, c_max, a=2, b=3, gnn_model=None):
        time1 = time.time()
        resetBinaryClass_init(self.data, a, b)
        
        self.K = self.getK_GCN()
        time2 = time.time()
        d_y, _, acc_bin_cln, acc_bin_atk, preds_closed_cln, preds_closed_atk = self.optimize_tau(c_max)

        train_indices_in_AB = np.where(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_train']))[0]
        train_indices_in_AB = np.where(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_train']))[0]
        flip_mask_in_train = (d_y == -1)
        flipNodes = self.data['nodes_AB_all'][train_indices_in_AB[flip_mask_in_train]]

    
        time3 = time.time()
        print(f"overall: {time3 - time1}, pre-process: {time2 - time1}, optimize: {time3 - time2}")
    
        recover_data(self.data)
    
        print(f"Debug info:")
        print(f"Classes a={a}, b={b}")
        print(f"_Z_obs shape: {self.data['_Z_obs'].shape}")
        print(f"Number of classes K: {self.data['K']}")
        print(f"Flip nodes: {flipNodes}")
        
        if a >= self.data['K'] or b >= self.data['K']:
            raise ValueError(f"Class indices a={a} or b={b} are out of bounds. Dataset has {self.data['K']} classes (0 to {self.data['K']-1})")
    
        _Z_obs_atk = np.array(self.data['_Z_obs'])
        
        if _Z_obs_atk.shape[1] != self.data['K']:
            print(f"Warning: _Z_obs shape {_Z_obs_atk.shape} doesn't match K={self.data['K']}")
            _Z_obs_atk = np.eye(self.data['K'])[self.data['_z_obs']]
        
        for i in flipNodes:

            if self.data['_z_obs'][i] == a:
                _Z_obs_atk[i] = np.zeros(self.data['K'])
                _Z_obs_atk[i][b] = 1
                print(f"class {b}")
            elif self.data['_z_obs'][i] == b:
                _Z_obs_atk[i] = np.zeros(self.data['K'])
                _Z_obs_atk[i][a] = 1
                print(f"class {a}")
            else:
                print(f"no change (not in target classes)")
        
        self.data['_Z_obs'] = _Z_obs_atk
        resetz_by_Z(self.data)
        resetBinaryClass_init(self.data, a=a, b=b)

        if gnn_model is not None:
            acc_test_mul_atk = []
            for run in range(5):
                print(f"Running GNN test {run+1}/5...")
                acc, _, _ = self.GNN_test(gnn_model)
                acc_test_mul_atk.append(acc)
            
            recover_data(self.data)
            print(f"Test accuracies: {acc_test_mul_atk}")
            acc_mul = sum(acc_test_mul_atk) / len(acc_test_mul_atk)
            print(f"Average test accuracy after attack: {acc_mul:.4f}")
            return acc_mul
        else:
            recover_data(self.data)
            return acc_bin_atk
