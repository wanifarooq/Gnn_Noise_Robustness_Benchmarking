import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_dense_adj
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple
import copy

def dirichlet_energy(x, edge_index):
    row, col = edge_index
    diff = x[row] - x[col]
    return (diff ** 2).sum(dim=1).mean()

def compute_dirichlet_energy_splits(features, data):
    train_features = features[data.train_mask]
    val_features = features[data.val_mask] 
    test_features = features[data.test_mask]
    
    train_nodes = torch.where(data.train_mask)[0]
    val_nodes = torch.where(data.val_mask)[0] 
    test_nodes = torch.where(data.test_mask)[0]
    
    def filter_edges_for_nodes(edge_index, node_set):
        node_mapping = {node.item(): i for i, node in enumerate(node_set)}
        
        mask = torch.isin(edge_index[0], node_set) & torch.isin(edge_index[1], node_set)
        filtered_edges = edge_index[:, mask]

        if filtered_edges.size(1) > 0:
            remapped_edges = torch.zeros_like(filtered_edges)
            for i in range(filtered_edges.size(1)):
                remapped_edges[0, i] = node_mapping[filtered_edges[0, i].item()]
                remapped_edges[1, i] = node_mapping[filtered_edges[1, i].item()]
            return remapped_edges
        else:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    
    train_edges = filter_edges_for_nodes(data.edge_index, train_nodes)
    val_edges = filter_edges_for_nodes(data.edge_index, val_nodes)
    test_edges = filter_edges_for_nodes(data.edge_index, test_nodes)
    
    train_de = dirichlet_energy(train_features, train_edges) if train_edges.size(1) > 0 else torch.tensor(0.0)
    val_de = dirichlet_energy(val_features, val_edges) if val_edges.size(1) > 0 else torch.tensor(0.0)
    test_de = dirichlet_energy(test_features, test_edges) if test_edges.size(1) > 0 else torch.tensor(0.0)
    
    return train_de.item(), val_de.item(), test_de.item()

class MaximalCodingRateReduction(nn.Module):
    def __init__(self, gam1: float = 1.0, gam2: float = 1.0, eps: float = 0.01, corafull: bool = False):
        super().__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.corafull = corafull

    def compute_discrimn_loss_empirical(self, W: torch.Tensor) -> torch.Tensor:
        p, m = W.shape
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W @ W.T)
        return logdet / 2.

    def compute_discrimn_loss_theoretical(self, W: torch.Tensor) -> torch.Tensor:
        p, m = W.shape
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W @ W.T)
        return logdet / 2.

    def compute_compress_loss_empirical_manyclass(self, W: torch.Tensor, L_all: torch.Tensor) -> torch.Tensor:
        p, m = W.shape
        k = torch.max(L_all) + 1
        I = torch.eye(p, device=W.device)
        compress_loss = 0.
        
        for j in range(k):
            mask = torch.where(L_all == j)[0]
            if len(mask) == 0:
                continue
            W_masked = W[:, mask]
            trPi = W_masked.shape[1] + 1e-8
            scalar = p / (trPi * self.eps)
            WWT = W_masked @ W_masked.T
            if WWT.shape[0] < p:
                padding = (0, p - WWT.shape[0], 0, p - WWT.shape[0])
                WWT = F.pad(WWT, padding)
            log_det = torch.logdet(I + scalar * WWT)
            compress_loss += log_det * trPi / m
            
        return compress_loss / 2.

    def compute_compress_loss_empirical(self, W: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device)
        compress_loss = 0.
        
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W @ Pi[j] @ W.T)
            compress_loss += log_det * trPi / m
            
        return compress_loss / 2.

    def compute_compress_loss_theoretical(self, W: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W @ Pi[j] @ W.T)
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, X: torch.Tensor, data, A: torch.Tensor, 
                Y_all: torch.Tensor, alpha: float, beta: float, T: int, 
                train_mask: torch.Tensor):
        
        num_classes = data.y.max().item() + 1
        X_train = X[train_mask]
        W = X.T

        Y_all = self._compute_semantic_labels(X_train, Y_all, X, train_mask, beta, num_classes)

        for _ in range(T):
            Y_all = (1 - alpha) * (A @ Y_all) + alpha * Y_all

        L_all = torch.argmax(Y_all, dim=1)

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)

        if self.corafull:
            compress_loss_empi = self.compute_compress_loss_empirical_manyclass(W, L_all)
            total_loss_empi = -self.gam2 * discrimn_loss_empi + self.gam1 * compress_loss_empi
            return total_loss_empi, [discrimn_loss_empi.item(), compress_loss_empi.item()], L_all

        else:

            Pi = self._label_to_membership(L_all.cpu(), num_classes)
            Pi = torch.tensor(Pi, dtype=torch.float32, device=X.device)

            compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
            compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)
            discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)

            total_loss_empi = -self.gam2 * discrimn_loss_empi + self.gam1 * compress_loss_empi

            return (total_loss_empi,
                    [discrimn_loss_empi.item(), compress_loss_empi.item()],
                    [discrimn_loss_theo.item(), compress_loss_theo.item()],
                    L_all)

    def _label_to_membership(self, targets: torch.Tensor, num_classes: int) -> np.ndarray:
        targets_onehot = F.one_hot(targets, num_classes).numpy()
        num_samples, num_classes = targets_onehot.shape
        Pi = np.zeros((num_classes, num_samples, num_samples))
        max_indices = np.argmax(targets_onehot, axis=1)
        Pi[max_indices, np.arange(num_samples), np.arange(num_samples)] = 1
        return Pi
    
    def _compute_semantic_labels(self, X_train: torch.Tensor, Y_all: torch.Tensor,
                                X: torch.Tensor, train_mask: torch.Tensor, 
                                beta: float, num_classes: int) -> torch.Tensor:
        Y_train = Y_all[train_mask].argmax(dim=1)
        centroids = []
        for i in range(num_classes):
            class_mask = (Y_train == i)
            if class_mask.sum() > 0:
                centroids.append(X_train[class_mask].mean(dim=0))
            else:
                centroids.append(torch.zeros(X_train.shape[1], device=X_train.device))
        centroids = torch.stack(centroids)
        X_normalized = F.normalize(X, p=2, dim=1)
        centroids_normalized = F.normalize(centroids, p=2, dim=1)
        cos_sim = torch.abs(X_normalized @ centroids_normalized.T)
        Y_all = beta * Y_all + (1 - beta) * cos_sim
        return F.softmax(Y_all, dim=1)

def normalize_adj_row(adj: torch.Tensor) -> torch.Tensor:
    rowsum = adj.sum(1, keepdim=True)
    r_inv = torch.where(rowsum > 0, 1.0 / rowsum, torch.zeros_like(rowsum))
    return adj * r_inv

def preprocess_labels(data, corrupted_labels: torch.Tensor, 
                     alpha: float, T: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_classes = data.y.max().item() + 1
    num_nodes = data.x.shape[0]
    
    Y_all = torch.zeros(num_nodes, num_classes, device=device)
    Y_onehot = F.one_hot(corrupted_labels, num_classes).float().to(device)
    Y_all[data.train_mask] = Y_onehot[data.train_mask]
    
    A = to_dense_adj(data.edge_index)[0].to(device)
    A_train = A.clone()
    
    train_mask_matrix = data.train_mask.float().unsqueeze(1) @ data.train_mask.float().unsqueeze(0)
    A_train = A_train * train_mask_matrix
    A_train = normalize_adj_row(A_train)
    A = normalize_adj_row(A)
    
    for _ in range(T):
        Y_all = (1 - alpha) * (A_train @ Y_all) + alpha * Y_all
    
    predicted_labels = Y_all.argmax(dim=1)
    return predicted_labels, Y_all, A


def evaluate_linear_probe(features: torch.Tensor, data, 
                         noisy_labels: torch.Tensor, clean_labels: torch.Tensor) -> Tuple[float, float, float]:
    features_norm = normalize(features.detach().cpu().numpy(), norm='l2')
    
    train_features = features_norm[data.train_mask.cpu()]
    val_features = features_norm[data.val_mask.cpu()] 
    test_features = features_norm[data.test_mask.cpu()]
    
    train_labels = noisy_labels[data.train_mask].cpu().numpy()
    val_clean = clean_labels[data.val_mask].cpu().numpy()
    test_clean = clean_labels[data.test_mask].cpu().numpy()
    train_clean = clean_labels[data.train_mask].cpu().numpy()
    
    clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, random_state=42)
    clf.fit(train_features, train_labels)
    
    train_acc = accuracy_score(train_clean, clf.predict(train_features))
    val_acc = accuracy_score(val_clean, clf.predict(val_features))  
    test_acc = accuracy_score(test_clean, clf.predict(test_features))
    
    return train_acc, val_acc, test_acc

class EnhancedGNN(nn.Module):

    def __init__(self, base_model, use_layer_norm=False, use_residual=False, 
                 use_residual_linear=False, final_activation='relu', 
                 final_normalization=None):
        super().__init__()
        
        self.base_model = base_model
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_residual_linear = use_residual_linear
        self.final_activation = final_activation
        self.final_normalization = final_normalization
        
        self._setup_enhancement_layers()
        
    def _setup_enhancement_layers(self):

        if self.final_normalization == 'layer_norm':
 
            self.final_layer_norm = None
            
        if self.use_residual_linear and self.use_residual:
 
            self.residual_projections = nn.ModuleDict()
    
    def _create_dynamic_layers(self, input_features, output_features):

        if self.final_normalization == 'layer_norm' and self.final_layer_norm is None:
            self.final_layer_norm = nn.LayerNorm(output_features.size(-1))
            self.final_layer_norm = self.final_layer_norm.to(output_features.device)
            
        if (self.use_residual_linear and self.use_residual and 
            input_features.size(-1) != output_features.size(-1)):
            key = f"{input_features.size(-1)}_{output_features.size(-1)}"
            if key not in self.residual_projections:
                proj = nn.Linear(input_features.size(-1), output_features.size(-1))
                self.residual_projections[key] = proj.to(output_features.device)
    
    def forward(self, data):

        if hasattr(data, 'x'):
            original_x = data.x.clone()
        else:
            original_x = data.clone()
        
        x = self.base_model(data)
        
        if hasattr(data, 'x'):
            self._create_dynamic_layers(original_x, x)
        else:
            self._create_dynamic_layers(original_x, x)
        
        if self.use_residual:
            if self.use_residual_linear:
                key = f"{original_x.size(-1)}_{x.size(-1)}"
                if key in self.residual_projections:
                    x = x + self.residual_projections[key](original_x)
            elif original_x.size(-1) == x.size(-1):
                x = x + original_x
        
        if self.final_normalization == 'layer_norm' and self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        elif self.final_normalization == 'l1':
            x = F.normalize(x, p=1, dim=1)
        elif self.final_normalization == 'l2':
            x = F.normalize(x, p=2, dim=1)
        
        if self.final_activation == 'relu':
            x = F.relu(x)
        elif self.final_activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.final_activation == 'elu':
            x = F.elu(x)
        elif self.final_activation == 'tanh':
            x = torch.tanh(x)
        elif self.final_activation is None:
            pass
            
        return x
    
    def reset_parameters(self):
        if hasattr(self.base_model, 'reset_parameters'):
            self.base_model.reset_parameters()
        elif hasattr(self.base_model, 'initialize'):
            self.base_model.initialize()
            
        if hasattr(self, 'final_layer_norm') and self.final_layer_norm is not None:
            self.final_layer_norm.reset_parameters()
            
        for proj in self.residual_projections.values():
            proj.reset_parameters()

def create_enhanced_model(get_model_fn, model_name, enhancement_config=None, **model_kwargs):

    default_enhancement = {
        'use_layer_norm': False,
        'use_residual': False,
        'use_residual_linear': False,
        'final_activation': 'relu',
        'final_normalization': None  # None, 'layer_norm', 'l1', 'l2'
    }
    
    if enhancement_config:
        default_enhancement.update(enhancement_config)
    
    base_model = get_model_fn(model_name=model_name, **model_kwargs)
    
    if not any(default_enhancement.values()):
        return base_model
    
    return EnhancedGNN(
        base_model=base_model,
        **default_enhancement
    )

class ERASETrainer:
    def __init__(self, config, device, num_classes, get_model_fn):
        self.config = config
        self.device = device
        self.num_classes = num_classes
        self.get_model_fn = get_model_fn
        
    def train(self, data, debug=False):
        enhancement_config = {
            'use_layer_norm': self.config.get('use_layer_norm', False),
            'use_residual': self.config.get('use_residual', False),
            'use_residual_linear': self.config.get('use_residual_linear', False),
            'final_activation': 'relu',
            'final_normalization': 'l1'
        }
        
        model = create_enhanced_model(
            get_model_fn=self.get_model_fn,
            model_name=self.config.get('erase_gnn_type', 'gcn').lower(),
            enhancement_config=enhancement_config,
            in_channels=data.num_features,
            hidden_channels=self.config.get('hidden_channels', 128),
            out_channels=self.config.get('n_embedding', 512),
            n_layers=self.config.get('n_layers', 2),
            dropout=self.config.get('dropout', 0.5),
            self_loop=self.config.get('self_loop', True),
            mlp_layers=self.config.get('mlp_layers', 2),
            train_eps=self.config.get('train_eps', True),
            heads=self.config.get('n_heads', 8)
        ).to(self.device)
        
        noisy_labels = getattr(data, 'y_noisy', data.y_original)

        L_all, Y_all, A = preprocess_labels(
            data, noisy_labels, 
            self.config.get('alpha', 0.6), 
            self.config.get('T', 5), 
            str(self.device)
        )

        loss_fn = MaximalCodingRateReduction(
            gam1=self.config.get('gam1', 1.0),
            gam2=self.config.get('gam2', 2.0), 
            eps=self.config.get('eps', 0.05),
            corafull=self.config.get('corafull', False)
        )
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.get('lr', 0.001),
            weight_decay=self.config.get('weight_decay', 0.0005),
            amsgrad=True
        )
        
        best_val_acc = 0
        best_val_loss = float('inf')
        best_train_acc = 0
        patience_counter = 0
        num_epochs = self.config.get('total_epochs', 200)
        patience = self.config.get('patience', 50)
        
        for epoch in range(1, num_epochs + 1):
            train_result = self._train_epoch(model, data, optimizer, loss_fn, A, Y_all, L_all)
            train_loss, loss_components, L_all = train_result
            
            train_acc, val_acc, val_loss = self._evaluate_train_val(model, data, L_all)

            train_f1, val_f1 = self._compute_f1_scores(model, data, L_all)
            
            train_de, val_de = self._compute_dirichlet_energy_train_val(model, data)
                    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_acc = train_acc
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if debug:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                      f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f} | "
                      f"Train DE: {train_de:.4f}, Val DE: {val_de:.4f}")

            if patience_counter >= patience:
                if debug:
                    print(f'Early stopping at epoch {epoch}')
                break

        model.load_state_dict(best_model_state)
        
        test_acc, test_f1, test_loss = self._evaluate_test_final(model, data, L_all)
        
        final_train_de, final_val_de, final_test_de = self._compute_dirichlet_energy(model, data)

        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
        print(f"Final Dirichlet Energy - Train: {final_train_de:.4f}, Val: {final_val_de:.4f}, Test: {final_test_de:.4f}")

        return {
            'train': best_train_acc,
            'val': best_val_acc,
            'test': test_acc,
            'dirichlet_energy': {
                'train': final_train_de,
                'val': final_val_de,
                'test': final_test_de
            }
        }

    @torch.no_grad()
    def _compute_dirichlet_energy_train_val(self, model, data):
        model.eval()
        features = model(data)
        train_de, val_de, _ = compute_dirichlet_energy_splits(features, data)
        return train_de, val_de

    @torch.no_grad()
    def _compute_dirichlet_energy(self, model, data):
        model.eval()
        features = model(data)
        return compute_dirichlet_energy_splits(features, data)

    @torch.no_grad()
    def _compute_f1_scores(self, model, data, L_all):
        model.eval()
        features = model(data)

        features_norm = normalize(features.detach().cpu().numpy(), norm='l2')
        
        train_features = features_norm[data.train_mask.cpu()]
        val_features = features_norm[data.val_mask.cpu()]
        
        train_labels = L_all[data.train_mask].cpu().numpy()
        y_true = data.y.cpu().numpy()
        
        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, random_state=42)
        clf.fit(train_features, train_labels)
        
        train_pred = clf.predict(train_features)
        val_pred = clf.predict(val_features)
        
        train_f1 = f1_score(
            y_true[data.train_mask.cpu()], 
            train_pred, 
            average='weighted', 
            zero_division=0
        )
        val_f1 = f1_score(
            y_true[data.val_mask.cpu()], 
            val_pred, 
            average='weighted', 
            zero_division=0
        )
        
        return train_f1, val_f1

    @torch.no_grad()
    def _evaluate_test_final(self, model, data, L_all):
        model.eval()
        features = model(data)

        _, _, test_acc = evaluate_linear_probe(features, data, L_all, data.y)

        features_norm = normalize(features.detach().cpu().numpy(), norm='l2')
        
        train_features = features_norm[data.train_mask.cpu()]
        test_features = features_norm[data.test_mask.cpu()]
        
        train_labels = L_all[data.train_mask].cpu().numpy()
        y_true = data.y.cpu().numpy()
        
        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, random_state=42)
        clf.fit(train_features, train_labels)
        
        test_pred = clf.predict(test_features)
        test_f1 = f1_score(
            y_true[data.test_mask.cpu()], 
            test_pred, 
            average='macro',
            zero_division=0
        )
        
        test_loss = self._compute_val_loss_ce(model, data, data.test_mask)
        
        return test_acc, test_f1, test_loss.item()

    @torch.no_grad()
    def _compute_val_loss_ce(self, model, data, val_mask):
        features = model(data)
        y_val = data.y[val_mask]
        ce_loss = torch.nn.CrossEntropyLoss()
        val_loss = ce_loss(features[val_mask], y_val)
        return val_loss

    @torch.no_grad()
    def _evaluate_train_val(self, model, data, L_all):
        model.eval()
        features = model(data)

        train_acc, val_acc, _ = evaluate_linear_probe(features, data, L_all, data.y)

        val_mask = data.val_mask
        val_loss = self._compute_val_loss_ce(model, data, val_mask)

        return train_acc, val_acc, val_loss

    @torch.no_grad()
    def _evaluate_test(self, model, data, L_all):
        model.eval()
        features = model(data)

        _, _, test_acc = evaluate_linear_probe(features, data, L_all, data.y)
        return test_acc

    def _train_epoch(self, model, data, optimizer, loss_fn, A, Y_all, L_all):
        model.train()
        optimizer.zero_grad()
        
        features = model(data)
        result = loss_fn(features, data, A, Y_all, 
                        self.config.get('alpha', 0.6), 
                        self.config.get('beta', 0.6), 
                        self.config.get('T', 5), 
                        data.train_mask)
        
        if len(result) == 3:
            loss, loss_components, L_all = result
            loss.backward()
            optimizer.step()
            return loss.item(), loss_components, L_all
        else:
            loss, loss_components, L_all = result[0], result[1], result[3]
            loss.backward()
            optimizer.step()
            return loss.item(), loss_components, L_all

    @torch.no_grad()
    def _test_epoch(self, model, data, noisy_labels, clean_labels):
        model.eval()
        features = model(data)
        return evaluate_linear_probe(features, data, noisy_labels, clean_labels)