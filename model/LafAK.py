import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import copy
import time
import scipy
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph
from torch_geometric.data import Data as PyGData
try:
    import networkx as nx
except ImportError:
    nx = None
from sklearn.metrics import f1_score

from model.evaluation import OversmoothingMetrics

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
        self.oversmoothing_evaluator = OversmoothingMetrics(device=self.device)

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

    def GNN_test(self, gnn_model, epochs=200, early_stopping=10):
        import time
        start_time = time.time()
        
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
                metrics = {}
                metrics['val_loss'] = criterion(out_val[val_mask], y[val_mask]).item()
                metrics['train_acc'] = accuracy_torch(out_val[train_mask], y[train_mask])
                metrics['val_acc'] = accuracy_torch(out_val[val_mask], y[val_mask])
                metrics['train_f1'] = f1_torch(out_val[train_mask], y[train_mask])
                metrics['val_f1'] = f1_torch(out_val[val_mask], y[val_mask])
                
                train_oversmoothing = self._compute_oversmoothing_for_mask(out_val, edge_index_sub, train_mask, y)
                val_oversmoothing = self._compute_oversmoothing_for_mask(out_val, edge_index_sub, val_mask, y)
                
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

                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Val Loss: {metrics['val_loss']:.4f} | "
                    f"Train Acc: {metrics['train_acc']:.4f}, Val Acc: {metrics['val_acc']:.4f} | "
                    f"Train F1: {metrics['train_f1']:.4f}, Val F1: {metrics['val_f1']:.4f}")
                print(f"Train DE: {train_de:.4f}, Val DE: {val_de:.4f} | "
                    f"Train DE_trad: {train_de_traditional:.4f}, Val DE_trad: {val_de_traditional:.4f} | "
                    f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                    f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                    f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                    f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")
        

            if metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
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

            final_metrics = {}
            final_metrics['test_loss'] = criterion(out[test_mask], y[test_mask]).item()
            final_metrics['test_acc'] = accuracy_torch(out[test_mask], y[test_mask])
            final_metrics['test_f1'] = f1_torch(out[test_mask], y[test_mask])
            
            final_train_oversmoothing = self._compute_oversmoothing_for_mask(out, edge_index_sub, train_mask, y)
            final_val_oversmoothing = self._compute_oversmoothing_for_mask(out, edge_index_sub, val_mask, y)  
            final_test_oversmoothing = self._compute_oversmoothing_for_mask(out, edge_index_sub, test_mask, y)

        total_time = time.time() - start_time
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


        return final_metrics['test_acc'], final_metrics['test_f1'], out[test_mask].cpu().numpy()

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
    
class CommunityDefense:
    def __init__(self, 
                 pyg_data: PyGData,
                 community_method: str = "louvain",
                 num_communities: int = None,
                 lambda_comm: float = 2.0,
                 pos_weight: float = 1.0,
                 neg_weight: float = 2.0,
                 margin: float = 1.5,
                 num_neg_samples: int = 3,
                 device: torch.device = None,
                 verbose: bool = True):
        
        self.data = pyg_data
        self.community_method = community_method
        self.num_communities = num_communities
        self.lambda_comm = float(lambda_comm)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.margin = float(margin)
        self.num_neg_samples = int(num_neg_samples)
        self.verbose = verbose
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        
        self.oversmoothing_evaluator = OversmoothingMetrics(device=self.device)
        
        self._prepare_graph()
        self.comm_labels = self._compute_communities()
        
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

    def _prepare_graph(self):
        row, col = self.data.edge_index.cpu().numpy()
        N = self.data.num_nodes
        A = sp.coo_matrix((np.ones_like(row, dtype=np.float32), (row, col)), shape=(N, N))
        A = A + A.T
        A.data = np.ones_like(A.data, dtype=np.float32)
        A.eliminate_zeros()
        self.A = A.tocsr()
        self.N = N

        assert len(self.data.train_mask) == N, f"Train mask dim {len(self.data.train_mask)} != N {N}"
        assert len(self.data.val_mask) == N, f"Val mask dim {len(self.data.val_mask)} != N {N}"
        assert len(self.data.test_mask) == N, f"Test mask dim {len(self.data.test_mask)} != N {N}"
        
        self.train_mask = torch.as_tensor(self.data.train_mask, dtype=torch.bool, device=self.device)
        self.val_mask = torch.as_tensor(self.data.val_mask, dtype=torch.bool, device=self.device)
        self.test_mask = torch.as_tensor(self.data.test_mask, dtype=torch.bool, device=self.device)

    def _compute_communities(self):
        if self.community_method.lower() == "louvain":
            if nx is None:
                raise ImportError("For 'louvain' you need networkx: pip install networkx python-louvain")
            
            community_louvain = None

            try:
                import community.community_louvain as community_louvain
                if self.verbose:
                    print("Using community.community_louvain")
            except ImportError:
                pass
            
            if community_louvain is None:
                try:
                    import community as community_module
                    if hasattr(community_module, 'best_partition'):
                        community_louvain = community_module
                        if self.verbose:
                            print("Using community module with best_partition")
                    else:
                        if self.verbose:
                            print("community module found but no best_partition attribute")
                except ImportError:
                    pass
            
            if community_louvain is None:
                try:
                    import community_louvain
                    if self.verbose:
                        print("Using direct community_louvain import")
                except ImportError:
                    pass

            if community_louvain is None:
                try:
                    import networkx.algorithms.community as nx_community
                    if self.verbose:
                        print("Using NetworkX community detection as fallback")
                    
                    G = nx.Graph()
                    G.add_nodes_from(range(self.N))
                    r, c = self.A.nonzero()
                    edges = list(zip(r.tolist(), c.tolist()))
                    G.add_edges_from(edges)
                    
                    if hasattr(nx_community, 'louvain_communities'):
                        communities = nx_community.louvain_communities(G, seed=0)
                        partition = {}
                        for comm_id, nodes in enumerate(communities):
                            for node in nodes:
                                partition[node] = comm_id
                    else:
                        communities = nx_community.greedy_modularity_communities(G)
                        partition = {}
                        for comm_id, nodes in enumerate(communities):
                            for node in nodes:
                                partition[node] = comm_id
                    
                    labels = np.array([partition.get(i, 0) for i in range(self.N)], dtype=np.int64)
                    if self.verbose:
                        k = len(set(labels.tolist()))
                        print(f"Defense: NetworkX community detection: found {k} community")
                    return labels
                    
                except (ImportError, AttributeError) as e:
                    if self.verbose:
                        print(f"NetworkX community detection also failed: {e}")
            
            if community_louvain is None:
                raise ImportError(
                    "Could not import Louvain community detection."
                )

            G = nx.Graph()
            G.add_nodes_from(range(self.N))
            r, c = self.A.nonzero()
            edges = list(zip(r.tolist(), c.tolist()))
            G.add_edges_from(edges)

            partition = community_louvain.best_partition(G, random_state=0)
            labels = np.array([partition[i] for i in range(self.N)], dtype=np.int64)
            if self.verbose:
                k = len(set(labels.tolist()))
                print(f"Defense: Louvain: trovate {k} comunità")
            return labels

        elif self.community_method.lower() == "spectral":
            from sklearn.cluster import KMeans
            from scipy.sparse import csgraph

            deg = np.array(self.A.sum(1)).flatten()
            d_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
            D_inv_sqrt = sp.diags(d_inv_sqrt)
            L = sp.eye(self.N) - D_inv_sqrt @ self.A @ D_inv_sqrt

            if self.num_communities is None:
                if hasattr(self.data, "y") and self.data.y.dim() == 1:
                    self.num_communities = int(self.data.y.max().item()) + 1
                else:
                    self.num_communities = 8

            from scipy.sparse.linalg import eigsh
            vals, vecs = eigsh(L.asfptype(), k=self.num_communities, which="SM")
            X = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
            km = KMeans(n_clusters=self.num_communities, n_init=10, random_state=0)
            labels = km.fit_predict(X).astype(np.int64)
            if self.verbose:
                print(f"Defense: Spectral: k={self.num_communities} community")
            return labels
        else:
            raise ValueError("community_method: 'louvain' o 'spectral'")

    def _build_pos_neg_pairs(self, labels_np):

        r, c = self.A.nonzero()

        mask_upper = r < c
        r = r[mask_upper]; c = c[mask_upper]

        same = labels_np[r] == labels_np[c]
        pos_pairs = np.stack([r[same], c[same]], axis=1) if np.any(same) else np.zeros((0,2), dtype=np.int64)

        neg_pairs = []
        rng = np.random.default_rng(0)
        for i in range(self.N):

            diff_idx = np.where(labels_np != labels_np[i])[0]
            if diff_idx.size == 0:
                continue
            if diff_idx.size <= self.num_neg_samples:
                chosen = diff_idx
            else:
                chosen = rng.choice(diff_idx, size=self.num_neg_samples, replace=False)
            for j in chosen:
                if i < j:
                    neg_pairs.append((i, j))
                elif j < i:
                    neg_pairs.append((j, i))
        if neg_pairs:
            neg_pairs = np.unique(np.array(neg_pairs, dtype=np.int64), axis=0)
        else:
            neg_pairs = np.zeros((0,2), dtype=np.int64)

        return pos_pairs, neg_pairs

    def community_loss(self, z: torch.Tensor, labels_np: np.ndarray) -> torch.Tensor:

        pos_pairs, neg_pairs = self._build_pos_neg_pairs(labels_np)
        
        if pos_pairs.shape[0] == 0 and neg_pairs.shape[0] == 0:
            return torch.zeros([], device=z.device)

        z_norm = F.normalize(z, p=2, dim=1, eps=1e-8)
        loss = 0.0
        cnt = 0

        if pos_pairs.shape[0] > 0:
            i = torch.as_tensor(pos_pairs[:,0], dtype=torch.long, device=z.device)
            j = torch.as_tensor(pos_pairs[:,1], dtype=torch.long, device=z.device)
            
            cos_sim = torch.clamp((z_norm[i] * z_norm[j]).sum(dim=1), -1.0 + 1e-7, 1.0 - 1e-7)
            pos_term = (1.0 - cos_sim).mean()
            loss = loss + self.pos_weight * pos_term
            cnt += 1

        if neg_pairs.shape[0] > 0:
            i = torch.as_tensor(neg_pairs[:,0], dtype=torch.long, device=z.device)
            j = torch.as_tensor(neg_pairs[:,1], dtype=torch.long, device=z.device)
            
            cos_sim = torch.clamp((z_norm[i] * z_norm[j]).sum(dim=1), -1.0 + 1e-7, 1.0 - 1e-7)
            adaptive_margin = max(0.1, 1.0 - self.margin/len(np.unique(labels_np)))
            neg_term = F.relu(cos_sim - adaptive_margin).mean()
            loss = loss + self.neg_weight * neg_term
            cnt += 1

        return loss / max(cnt, 1)

    def train_with_defense(self,
                        gnn_model,
                        epochs: int = 200,
                        early_stopping: int = 20,
                        lr: float = 0.005,
                        weight_decay: float = 1e-3,
                        debug: bool = True):
        import time
        start_time = time.time()
        
        model = gnn_model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        ce = torch.nn.CrossEntropyLoss()

        A = self.A
        if hasattr(A, "tocoo"):
            A = A.tocoo()
        edge_index, _ = from_scipy_sparse_matrix(A)
        edge_index = edge_index.to(self.device)

        X = self.data.x.to(self.device, dtype=torch.float32)
        y = self.data.y.to(self.device, dtype=torch.long)

        best_val = float("inf")
        best_state = None
        no_improve = 0
        comm_np = self.comm_labels.copy()

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            batch = PyGData(x=X, edge_index=edge_index, y=y)
            out = model(batch)

            if isinstance(out, tuple) and len(out) == 2:
                logits, z = out
            else:
                logits = out
                if hasattr(model, 'last_hidden'):
                    z = model.last_hidden
                else:
                    z = F.dropout(logits, p=0.3, training=True)

            loss_sup = ce(logits[self.train_mask], y[self.train_mask])
            
            adaptive_lambda = self.lambda_comm * min(1.0, epoch / 50.0)
            loss_comm = self.community_loss(z, comm_np) if adaptive_lambda > 0 else torch.zeros([], device=self.device)
            
            loss = loss_sup + adaptive_lambda * loss_comm
            train_loss = loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_eval = model(batch)
                logits_eval = out_eval[0] if (isinstance(out_eval, tuple) and len(out_eval) == 2) else out_eval
                
                metrics = {}
                metrics['val_loss'] = ce(logits_eval[self.val_mask], y[self.val_mask]).item()
                pred = logits_eval.argmax(dim=1).cpu().numpy()
                y_true = y.cpu().numpy()

                metrics['train_acc'] = (pred[self.train_mask.cpu().numpy()] == y_true[self.train_mask.cpu().numpy()]).mean()
                metrics['val_acc'] = (pred[self.val_mask.cpu().numpy()] == y_true[self.val_mask.cpu().numpy()]).mean()

                metrics['train_f1'] = f1_score(y_true[self.train_mask.cpu().numpy()],
                                    pred[self.train_mask.cpu().numpy()],
                                    average="macro")
                metrics['val_f1'] = f1_score(y_true[self.val_mask.cpu().numpy()],
                                pred[self.val_mask.cpu().numpy()],
                                average="macro")

                train_oversmoothing = self._compute_oversmoothing_for_mask(logits_eval, edge_index, self.train_mask, y)
                val_oversmoothing = self._compute_oversmoothing_for_mask(logits_eval, edge_index, self.val_mask, y)
                
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

            scheduler.step(metrics['val_loss'])

            if debug:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Val Loss: {metrics['val_loss']:.4f} | "
                      f"Train Acc: {metrics['train_acc']:.4f}, Val Acc: {metrics['val_acc']:.4f} | "
                      f"Train F1: {metrics['train_f1']:.4f}, Val F1: {metrics['val_f1']:.4f}")
                print(f"Train DE: {train_de:.4f}, Val DE: {val_de:.4f} | "
                      f"Train DE_trad: {train_de_traditional:.4f}, Val DE_trad: {val_de_traditional:.4f} | "
                      f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                      f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                      f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                      f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")

            if metrics['val_loss'] < best_val - 1e-4:
                best_val = metrics['val_loss']
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= early_stopping:
                if self.verbose:
                    print(f"Defense: Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            batch = PyGData(x=X, edge_index=edge_index, y=y)
            out = model(batch)
            logits = out[0] if (isinstance(out, tuple) and len(out) == 2) else out
            pred = logits.argmax(dim=1).cpu().numpy()
            y_true = y.cpu().numpy()

            final_metrics = {}
            final_metrics['test_loss'] = ce(logits[self.test_mask], y[self.test_mask]).item()
            final_metrics['test_acc'] = (pred[self.test_mask.cpu().numpy()] == y_true[self.test_mask.cpu().numpy()]).mean()
            final_metrics['test_f1'] = f1_score(y_true[self.test_mask.cpu().numpy()],
                            pred[self.test_mask.cpu().numpy()],
                            average="macro")
            
            final_train_oversmoothing = self._compute_oversmoothing_for_mask(logits, edge_index, self.train_mask, y)
            final_val_oversmoothing = self._compute_oversmoothing_for_mask(logits, edge_index, self.val_mask, y)  
            final_test_oversmoothing = self._compute_oversmoothing_for_mask(logits, edge_index, self.test_mask, y)

        total_time = time.time() - start_time
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

        return final_metrics['test_acc']
