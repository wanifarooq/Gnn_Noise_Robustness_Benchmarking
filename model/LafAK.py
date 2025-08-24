import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import copy
import time
import scipy
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph
from torch_geometric.data import Data as PyGData
from sklearn.metrics import f1_score

def BinaryLabelToPosNeg(labels):
    if isinstance(labels, torch.Tensor):
        return 2 * labels - 1
    else:
        return 2 * labels - 1

def BinaryLabelTo01(labels):
    if isinstance(labels, torch.Tensor):
        return (labels + 1) / 2
    else:
        return (labels + 1) / 2
    

def accuracy_torch(preds, labels):
    pred_labels = preds.argmax(dim=1)
    correct = (pred_labels == labels).sum().item()
    return correct / len(labels)

def f1_torch(preds, labels):
    pred_labels = preds.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return f1_score(labels, pred_labels, average="macro")

def accuracy(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    if preds.dtype == np.float32 or preds.dtype == np.float64:
        preds = np.sign(preds)
    
    return (preds == labels).mean()

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
        '_z_obs': z_obs.copy(),
        '_z_obs_original': z_obs.copy(),
        'N': N,
        'K': K,
        'split_train': split_train,
        'split_val': split_val,
        'split_test': split_test,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        '_Z_obs': Z_obs.copy(),
        '_Z_obs_original': Z_obs.copy(),
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
        nodes_AB_val,
        nodes_AB_test
    ])

    mask_train_rel = np.isin(nodes_AB_all, nodes_AB_train)
    split_train_AB = np.where(mask_train_rel)[0]
    split_unlabeled = np.where(~mask_train_rel)[0]

    labels_AB = data_dict['_z_obs'][nodes_AB_all].copy()
    labels_AB[labels_AB == a] = 0
    labels_AB[labels_AB == b] = 1

    data_dict.update({
        'nodes_AB_train': nodes_AB_train,
        'nodes_AB_val': nodes_AB_val,
        'nodes_AB_test': nodes_AB_test,
        'nodes_AB_all': nodes_AB_all,
        'split_train_AB': split_train_AB,
        'split_unlabeled': split_unlabeled,
        '_z_obs_bin': labels_AB,
        'binary_classes': (a, b)
    })
    
    print(f"Binary setup: {len(nodes_AB_train)} train, {len(nodes_AB_val)} val, {len(nodes_AB_test)} test")
    print(f"Total AB nodes: {len(nodes_AB_all)}, unlabeled: {len(split_unlabeled)}")

def recover_data(data_dict):

    data_dict['_z_obs'] = data_dict['_z_obs_original'].copy()
    data_dict['_Z_obs'] = data_dict['_Z_obs_original'].copy()

def resetz_by_Z(data_dict):

    data_dict['_z_obs'] = np.argmax(data_dict['_Z_obs'], axis=1)

class Attack:
    def __init__(self, data_dict, gpu_id, atkEpoch, gcnL2, smooth_coefficient, c_max):
        self.data = data_dict
        self.gpu_id = gpu_id
        self.atkEpoch = atkEpoch
        self.gcnL2 = gcnL2
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.tau = smooth_coefficient
        self.c_max = c_max

    def GNN_test(self, gnn_model, epochs=200, early_stopping=10):
        model = copy.deepcopy(gnn_model).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
    
        X = torch.tensor(self.data['_X_obs'][self.data['nodes_AB_all']], dtype=torch.float32, device=self.device)
        y = torch.tensor(self.data['_z_obs_bin'], dtype=torch.long, device=self.device)
    
        train_mask = torch.tensor(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_train']), device=self.device)
        val_mask = torch.tensor(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_val']), device=self.device)
        test_mask = torch.tensor(np.isin(self.data['nodes_AB_all'], self.data['nodes_AB_test']), device=self.device)
    
        A = self.data['_A_obs']
        if hasattr(A, 'tocoo'):
            A = A.tocoo()
        edge_index, edge_weight = from_scipy_sparse_matrix(A)
        edge_index = edge_index.to(self.device)
    
        nodes = torch.tensor(self.data['nodes_AB_all'], dtype=torch.long, device=self.device)
        edge_index_sub, _ = subgraph(nodes, edge_index, relabel_nodes=True, num_nodes=A.shape[0])
    
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    
        for epoch in range(1, epochs+1):
            model.train()
            optimizer.zero_grad()
            
            data_batch = PyGData(x=X, edge_index=edge_index_sub, y=y)
            out = model(data_batch)
            
            loss = criterion(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(data_batch)

                train_loss = loss.item()
                val_loss = criterion(out_val[val_mask], y[val_mask]).item()

                train_acc = accuracy_torch(out_val[train_mask], y[train_mask])
                val_acc = accuracy_torch(out_val[val_mask], y[val_mask])
                train_f1 = f1_torch(out_val[train_mask], y[train_mask])
                val_f1 = f1_torch(out_val[val_mask], y[val_mask])

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping:
                    print(f"Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_model_wts)
        model.eval()
        with torch.no_grad():
            data_batch = PyGData(x=X, edge_index=edge_index_sub, y=y)
            out = model(data_batch)

            test_loss = criterion(out[test_mask], y[test_mask]).item()
            acc = accuracy_torch(out[test_mask], y[test_mask])
            f1 = f1_torch(out[test_mask], y[test_mask])

            print(f"Test Loss: {test_loss:.4f} | Test Acc: {acc:.4f} | Test F1: {f1:.4f}")

        return acc, f1, out[test_mask].cpu().numpy()

    def getK_GCN(self):

        A_sub = self.data['_A_obs'][np.ix_(self.data['nodes_AB_all'], self.data['nodes_AB_all'])]
        A_processed = preprocess_graph_numpy(A_sub).tocsr()
        
        X_sub = self.data['_X_obs'][self.data['nodes_AB_all']]
        if hasattr(X_sub, 'toarray'):
            X_sub = X_sub.toarray()
        X_normalized = row_normalize_numpy(sp.csr_matrix(X_sub)).tocsr()
        
        X_bar = A_processed.dot(A_processed).dot(X_normalized).tocsr()
        
        X_bar_l = X_bar[self.data['split_train_AB'], :]
        
        tmp = X_bar_l.T.dot(X_bar_l)
        reg_matrix = self.gcnL2 * np.identity(tmp.shape[0])
        tmp_reg = tmp.toarray() + reg_matrix
        tmp_inv = sp.csr_matrix(scipy.linalg.pinv(tmp_reg))
        tmp_final = tmp_inv.dot(X_bar_l.T)
        
        K = X_bar.dot(tmp_final)[self.data['split_unlabeled'], :].toarray()
        
        return K

    def closedForm_bin(self, trainLabels):

        if isinstance(trainLabels, torch.Tensor):
            trainLabels = trainLabels.cpu().numpy()
        
        trainLabels = trainLabels.reshape(-1, 1)
        y_pred = np.dot(self.K, trainLabels).flatten()
    
        unlabeled_labels = self.data['_z_obs_bin'][self.data['split_unlabeled']]
        true_labels = BinaryLabelToPosNeg(unlabeled_labels)
        
        mse_org = accuracy(np.sign(y_pred), true_labels)
        
        return mse_org, y_pred

    def gradient_attack_sgd(self):

        train_labels_bin = self.data['_z_obs_bin'][self.data['split_train_AB']]
        unlabeled_labels_bin = self.data['_z_obs_bin'][self.data['split_unlabeled']]
        
        y_l = BinaryLabelToPosNeg(train_labels_bin).reshape(-1, 1)
        y_u = BinaryLabelToPosNeg(unlabeled_labels_bin).reshape(-1, 1)
        
        y_l_tensor = torch.tensor(y_l, dtype=torch.float32, device=self.device)
        y_u_tensor = torch.tensor(y_u, dtype=torch.float32, device=self.device)
        K_tensor = torch.tensor(self.K, dtype=torch.float32, device=self.device)

        alpha = torch.nn.Parameter(0.5 * torch.ones_like(y_l_tensor, device=self.device))
        optimizer = optim.SGD([alpha], lr=1e-4)
        
        for step in range(self.atkEpoch):
            optimizer.zero_grad()
            
            u1 = torch.rand_like(alpha)
            u2 = torch.rand_like(alpha)
            gumbel1 = -torch.log(-torch.log(u1 + 1e-20) + 1e-20)
            gumbel2 = -torch.log(-torch.log(u2 + 1e-20) + 1e-20)
            epsilon = gumbel1 - gumbel2
            
            logit = torch.log(alpha / (1 - alpha + 1e-20) + 1e-20)
            tmp = torch.exp((logit + epsilon) / 0.5)
            z = 2.0 / (1.0 + tmp) - 1.0
            
            y_l_tmp = y_l_tensor * z
            
            y_u_preds = torch.tanh(self.tau * torch.matmul(K_tensor, y_l_tmp))
            
            loss = torch.mean(y_u_preds * y_u_tensor)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                alpha.clamp_(0, 1)
        
        alpha_final = alpha.detach().cpu().numpy().flatten()
        
        idx = np.argsort(alpha_final)[::-1]
        d_y = np.ones(len(self.data['split_train_AB']))
        count = 0
        flip_info = {}
        
        max_flips = min(self.c_max, len(self.data['split_train_AB']))
        
        for i in idx:
            if alpha_final[i] > 0.5 and count < max_flips:
                d_y[i] = -1
                count += 1
                global_node_idx = self.data['nodes_AB_all'][self.data['split_train_AB'][i]]
                flip_info[global_node_idx] = alpha_final[i]
                if count == max_flips:
                    break
        
        print(f"Flipped {count}/{max_flips} nodes (alpha > 0.5 constraint)")
        
        trainLabels_clean = y_l.flatten()
        acc_cln, pred_closed_cln = self.closedForm_bin(trainLabels_clean)
        
        trainLabels_atk = trainLabels_clean * d_y
        acc_atk, pred_closed_atk = self.closedForm_bin(trainLabels_atk)
        
        _z_obs_bin_atk = self.data['_z_obs_bin'].copy()
        for i, flip in enumerate(d_y):
            if flip == -1:
                ab_idx = self.data['split_train_AB'][i]
                _z_obs_bin_atk[ab_idx] = 1 - _z_obs_bin_atk[ab_idx]
        
        return d_y, BinaryLabelTo01(BinaryLabelToPosNeg(_z_obs_bin_atk)), acc_cln, acc_atk, pred_closed_cln, pred_closed_atk

    def get_tau(self):
        tau_list = [1, 2, 4, 8, 16, 32, 64, 128]
        maxTau = 0
        maxAcc = 100000
        maxOuts = None
        
        print(f"Optimizing tau with c_max = {self.c_max}")
        
        for tau in tau_list:
            self.tau = tau
            print(f"Testing tau = {tau}")
            outs = self.gradient_attack_sgd()
            acc_atk = outs[3]
            
            print(f"  Tau {tau}: attacked accuracy = {acc_atk:.4f}")
            
            if acc_atk <= maxAcc:
                maxAcc = acc_atk
                maxTau = tau
                maxOuts = outs
                
        self.tau = maxTau
        print(f"Best tau found: {self.tau}, with attacked accuracy: {maxAcc:.4f}")
        return maxOuts

    def binaryAttack_multiclass(self, a=2, b=3, gnn_model=None):
        time1 = time.time()
        
        print(f"Starting binary attack with c_max = {self.c_max}, classes {a} vs {b}")
        
        resetBinaryClass_init(self.data, a, b)
        
        self.K = self.getK_GCN()
        time2 = time.time()

        d_y, _, acc_bin_cln, acc_bin_atk, preds_closed_cln, preds_closed_atk = self.get_tau()
        
        flip_mask = (d_y != 1)
        flipNodes = self.data['nodes_AB_all'][self.data['split_train_AB'][flip_mask]]
        
        time3 = time.time()
        print(f"Timing - Overall: {time3-time1:.2f}s, Preprocess: {time2-time1:.2f}s, Optimize: {time3-time2:.2f}s")
        print(f"Binary attack - Clean: {acc_bin_cln:.4f}, Attacked: {acc_bin_atk:.4f}")
        print(f"Flipped {len(flipNodes)} nodes: {flipNodes}")
        
        recover_data(self.data)
        
        _Z_obs_atk = self.data['_Z_obs'].copy()
        
        for node_idx in flipNodes:
            current_class = self.data['_z_obs'][node_idx]
            if current_class == a:
                _Z_obs_atk[node_idx] = np.zeros(self.data['K'])
                _Z_obs_atk[node_idx][b] = 1
            elif current_class == b:
                _Z_obs_atk[node_idx] = np.zeros(self.data['K'])
                _Z_obs_atk[node_idx][a] = 1
            else:
                print(f"Warning: Node {node_idx} has class {current_class}, not in target classes {a},{b}")
        
        self.data['_Z_obs'] = _Z_obs_atk
        resetz_by_Z(self.data)
        
        if gnn_model is not None:
            resetBinaryClass_init(self.data, a=a, b=b)
            
            acc_test_runs = []
            for run in range(5):
                print(f"Test run {run+1}/5...")
                acc, _, _ = self.GNN_test(gnn_model)
                acc_test_runs.append(acc)
            
            avg_acc = sum(acc_test_runs) / len(acc_test_runs)
            print(f"Test accuracies: {acc_test_runs}")
            print(f"Average attacked accuracy: {avg_acc:.4f}")
            
            recover_data(self.data)
            
            return avg_acc
        else:
            recover_data(self.data)
            return acc_bin_atk

    def binaryAttack_multiclass_with_clean(self, a=2, b=3, gnn_model=None):
        time1 = time.time()
        
        print(f"Starting comprehensive binary attack with c_max = {self.c_max}, classes {a} vs {b}")
        
        resetBinaryClass_init(self.data, a, b)
        
        self.K = self.getK_GCN()
        time2 = time.time()
        
        d_y, _, acc_bin_cln, acc_bin_atk, preds_closed_cln, preds_closed_atk = self.get_tau()
        
        flip_mask = (d_y != 1)
        flipNodes = self.data['nodes_AB_all'][self.data['split_train_AB'][flip_mask]]
        
        time3 = time.time()
        print(f"Timing - Overall: {time3-time1:.2f}s, Preprocess: {time2-time1:.2f}s, Optimize: {time3-time2:.2f}s")
        print(f"Binary attack - Clean: {acc_bin_cln:.4f}, Attacked: {acc_bin_atk:.4f}")
        print(f"Flipped {len(flipNodes)} nodes: {flipNodes}")
        
        acc_clean_gnn = None
        if gnn_model is not None:
            print("Testing clean GNN accuracy...")
            acc_clean_gnn, _, _ = self.GNN_test(gnn_model)
            print(f"Clean GNN accuracy: {acc_clean_gnn:.4f}")
        
        recover_data(self.data)

        _Z_obs_atk = self.data['_Z_obs'].copy()
        
        for node_idx in flipNodes:
            current_class = self.data['_z_obs'][node_idx]
            if current_class == a:
                _Z_obs_atk[node_idx] = np.zeros(self.data['K'])
                _Z_obs_atk[node_idx][b] = 1
            elif current_class == b:
                _Z_obs_atk[node_idx] = np.zeros(self.data['K'])
                _Z_obs_atk[node_idx][a] = 1
            else:
                print(f"Warning: Node {node_idx} has class {current_class}, not in target classes {a},{b}")
        
        self.data['_Z_obs'] = _Z_obs_atk
        resetz_by_Z(self.data)

        acc_attacked_gnn = None
        if gnn_model is not None:
            resetBinaryClass_init(self.data, a=a, b=b)
            
            acc_test_runs = []
            for run in range(5):
                print(f"Test run {run+1}/5...")
                acc, _, _ = self.GNN_test(gnn_model)
                acc_test_runs.append(acc)
            
            acc_attacked_gnn = sum(acc_test_runs) / len(acc_test_runs)
            print(f"Test accuracies: {acc_test_runs}")
            print(f"Average attacked GNN accuracy: {acc_attacked_gnn:.4f}")
        
        recover_data(self.data)
        
        return {
            'binary_clean_acc': acc_bin_cln,
            'binary_attacked_acc': acc_bin_atk,
            'gnn_clean_acc': acc_clean_gnn,
            'gnn_attacked_acc': acc_attacked_gnn,
            'attack_success': acc_bin_cln - acc_bin_atk,
            'gnn_attack_success': acc_clean_gnn - acc_attacked_gnn if (acc_clean_gnn and acc_attacked_gnn) else 0,
            'flipped_nodes': flipNodes
        }