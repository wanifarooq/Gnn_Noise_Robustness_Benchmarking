import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix, negative_sampling
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp

from model.gnns import GCN, GIN, GAT, GATv2, GPS
from model.evaluation import (OversmoothingMetrics, ClassificationMetrics,
                              compute_oversmoothing_for_mask, evaluate_model,
                              DEFAULT_OVERSMOOTHING, ZERO_CLS)
from model.base import BaseTrainer
from model.registry import register


class DualBranchGNNModel(nn.Module):
    
    def __init__(self, gnn_type: str, input_features: int, hidden_dim: int, num_classes: int,
                 dropout_rate: float = 0.5, attention_heads: int = 4,
                 num_layers: int | None = None, device=None, add_self_loops: bool = False,
                 attn_type: str = 'multihead', use_pe: bool = False, pe_dim: int = 8):
        super().__init__()
        self.device = device
        self.gnn_type = gnn_type.lower()
        self.dropout_rate = dropout_rate
        self.add_self_loops = add_self_loops
        self.attn_type = attn_type
        self.use_pe = use_pe
        self.pe_dim = pe_dim

        def create_gnn_branch():
            if self.gnn_type == 'gcn':
                return GCN(input_features, hidden_dim, num_classes, n_layers=num_layers or 2,
                          dropout=dropout_rate, self_loop=self.add_self_loops)
            elif self.gnn_type == 'gin':
                return GIN(input_features, hidden_dim, num_classes, n_layers=num_layers or 3,
                          mlp_layers=2, dropout=dropout_rate)
            elif self.gnn_type == 'gat':
                return GAT(input_features, hidden_dim, num_classes, n_layers=num_layers or 3,
                          heads=attention_heads, dropout=dropout_rate, self_loop=self.add_self_loops)
            elif self.gnn_type == 'gatv2':
                return GATv2(input_features, hidden_dim, num_classes, n_layers=num_layers or 3,
                            heads=attention_heads, dropout=dropout_rate, self_loop=self.add_self_loops)
            elif self.gnn_type == 'gps':
                return GPS(input_features, hidden_dim, num_classes, n_layers=num_layers or 3,
                          heads=attention_heads, dropout=dropout_rate, 
                          attn_type=self.attn_type, use_pe=self.use_pe, pe_dim=self.pe_dim)
            else:
                raise ValueError(f"GNN type {self.gnn_type} not supported")

        self.first_branch = create_gnn_branch()
        self.second_branch = create_gnn_branch()

    def _to_graph_data(self, node_features, edge_indices, edge_weights=None):
        graph_data = Data(x=node_features, edge_index=edge_indices)
        if edge_weights is not None:
            graph_data.edge_weight = edge_weights
        if self.device is not None:
            graph_data = graph_data.to(self.device)
        return graph_data

    def forward(self, node_features, edge_indices, edge_weights=None):
        """Return (out_channels-dim logits, out_channels-dim logits) from both branches."""
        graph_data = self._to_graph_data(node_features, edge_indices, edge_weights)
        first_branch_output = self.first_branch(graph_data)
        second_branch_output = self.second_branch(graph_data)
        return first_branch_output, second_branch_output

    def get_embeddings(self, node_features, edge_indices, edge_weights=None):
        """Return averaged hidden_channels-dim representations from both branches."""
        graph_data = self._to_graph_data(node_features, edge_indices, edge_weights)
        first_emb = self.first_branch.get_embeddings(graph_data)
        second_emb = self.second_branch.get_embeddings(graph_data)
        return (first_emb + second_emb) / 2

    def reset_parameters(self):
        if hasattr(self.first_branch, "reset_parameters"):
            self.first_branch.reset_parameters()
        if hasattr(self.second_branch, "reset_parameters"):
            self.second_branch.reset_parameters()


class AdaptiveCoTeachingLoss(nn.Module):
    #Adaptive co-teaching loss
    
    def __init__(self, training_config):
        super().__init__()
        self.training_config = training_config
        self.total_epochs = training_config.epochs
        # Forget rate starting small and increasing to a max (e.g. noise rate)
        self.forget_rate_max = 0.5 # default max
        self.forget_rate_increment = self.forget_rate_max / self.total_epochs
        self.residual_weight_decay = 1.0

    def forward(self, first_branch_logits, second_branch_logits, target_labels, current_epoch=0):
        # R-2 Fix: Proper Co-teaching logic. 
        # Each branch calculates its own loss, but only trains on the samples 
        # that the OTHER branch thinks are clean.
        
        loss1 = F.cross_entropy(first_branch_logits, target_labels, reduction='none')
        loss2 = F.cross_entropy(second_branch_logits, target_labels, reduction='none')
        
        if current_epoch == 0:
            return loss1.mean() + loss2.mean()

        # Calculate how many samples to "remember"
        current_forget_rate = self.forget_rate_increment * current_epoch
        num_to_remember = int((1 - current_forget_rate) * len(target_labels))
        num_to_remember = max(int(0.5 * len(target_labels)), num_to_remember)

        # Branch 1 picks clean samples for Branch 2
        idx1 = torch.argsort(loss1)[:num_to_remember]
        # Branch 2 picks clean samples for Branch 1
        idx2 = torch.argsort(loss2)[:num_to_remember]

        # Cross-update: Branch 1 trains on idx2, Branch 2 trains on idx1
        clean_loss = loss1[idx2].mean() + loss2[idx1].mean()

        # Correction loss for noisy samples (using pseudo-labels)
        correction_loss = torch.tensor(0.0, device=first_branch_logits.device)
        
        # Identified noisy samples (those not picked by either branch)
        noisy_mask = torch.ones(len(target_labels), dtype=torch.bool, device=first_branch_logits.device)
        noisy_mask[idx1] = False
        noisy_mask[idx2] = False
        noisy_indices = torch.where(noisy_mask)[0]

        if len(noisy_indices) > 0:
            p1 = F.softmax(first_branch_logits, dim=1)
            p2 = F.softmax(second_branch_logits, dim=1)
            
            c1, pred1 = p1.max(1)
            c2, pred2 = p2.max(1)

            # R-3 Fix: Confidence threshold should start high and decrease
            # (Paper: starts strict at 1.0, drops toward a lower bound)
            conf_threshold = 1.0 - (0.5 * current_epoch / self.total_epochs)
            
            agree = (pred1[noisy_indices] == pred2[noisy_indices])
            high_conf = (c1[noisy_indices] * c2[noisy_indices] > (conf_threshold**2))
            
            corr_mask = agree & high_conf
            if corr_mask.sum() > 0:
                corr_idx = noisy_indices[corr_mask]
                # R-5 Fix: Detach confidence weights to prevent secondary gradient paths
                weights = (c1[corr_idx] * c2[corr_idx]).detach()
                
                # Align both branches with the agreed pseudo-label
                correction_loss = (weights * (
                    F.cross_entropy(first_branch_logits[corr_idx], pred1[corr_idx], reduction='none') +
                    F.cross_entropy(second_branch_logits[corr_idx], pred1[corr_idx], reduction='none')
                )).mean()

        # R-4 Fix: Total loss should only include the core co-teaching/correction terms.
        # co_lambda regularization (KL) is handled separately in RTGNN training step.
        return clean_loss + correction_loss

    def _compute_kl_divergence(self, logits_p, logits_q):
        return F.kl_div(F.log_softmax(logits_p, dim=1),
                        F.softmax(logits_q.detach(), dim=1),
                        reduction='batchmean')


class IntraViewRegularization(nn.Module):
    #Intra-view regularization
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def forward(self, first_branch_output, second_branch_output, edge_indices, edge_weights, labeled_node_indices):
        if isinstance(labeled_node_indices, list):
            labeled_node_indices = torch.tensor(labeled_node_indices, device=self.device)
        
        if labeled_node_indices.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        weighted_adjacency = to_scipy_sparse_matrix(edge_indices, edge_weights.detach().cpu())
        column_sums = np.array(weighted_adjacency.sum(0))
        reciprocal_col_sums = np.power(column_sums, -1).flatten()
        reciprocal_col_sums[np.isinf(reciprocal_col_sums)] = 0
        normalized_adjacency = weighted_adjacency.dot(sp.diags(reciprocal_col_sums))

        normalized_edge_indices, normalized_edge_weights = from_scipy_sparse_matrix(normalized_adjacency)
        normalized_edge_indices = normalized_edge_indices.to(self.device)
        normalized_edge_weights = normalized_edge_weights.to(self.device)

        labeled_edges_mask = torch.isin(normalized_edge_indices[1], labeled_node_indices)
        filtered_edge_indices = normalized_edge_indices[:, labeled_edges_mask]
        filtered_edge_weights = normalized_edge_weights[labeled_edges_mask]

        if filtered_edge_indices.size(1) == 0:
            return torch.tensor(0.0, device=self.device)

        first_branch_loss = (filtered_edge_weights * self._compute_kl_divergence(
            first_branch_output[filtered_edge_indices[1]], 
            first_branch_output[filtered_edge_indices[0]].detach()
        )).sum()
        
        second_branch_loss = (filtered_edge_weights * self._compute_kl_divergence(
            second_branch_output[filtered_edge_indices[1]], 
            second_branch_output[filtered_edge_indices[0]].detach()
        )).sum()
        
        total_loss = (first_branch_loss + second_branch_loss) / labeled_node_indices.size(0)
        return total_loss

    def _compute_kl_divergence(self, logits_p, logits_q):
        return F.kl_div(F.log_softmax(logits_p, dim=1),
                        F.softmax(logits_q, dim=1),
                        reduction='batchmean')


class GraphStructureEstimator(nn.Module):
    #Graph structure estimator
    
    def __init__(self, input_features, hidden_dim, training_config, device):
        super(GraphStructureEstimator, self).__init__()
        self.device = device
        self.training_config = training_config
        
        self.feature_encoder = GCNConv(input_features, hidden_dim, add_self_loops=True)
        self.similarity_threshold = training_config.tau
        
    def forward(self, node_features, edge_indices):
        encoded_features = F.normalize(F.relu(self.feature_encoder(node_features, edge_indices)), dim=1)
        positive_loss, negative_loss = self._compute_reconstruction_loss(encoded_features, edge_indices)
        reconstruction_loss = positive_loss + negative_loss
        return encoded_features, reconstruction_loss
    
    def compute_adaptive_edge_weights(self, edge_indices, node_representations, base_edge_weights=None):
        source_nodes, target_nodes = edge_indices[0], edge_indices[1]
        similarity_scores = (node_representations[source_nodes] * node_representations[target_nodes]).sum(dim=1)
        adaptive_weights = F.relu(similarity_scores)
        adaptive_weights = adaptive_weights * (adaptive_weights >= self.similarity_threshold).float()
        
        if base_edge_weights is not None:
            adaptive_weights = base_edge_weights + adaptive_weights * (1 - base_edge_weights)
            
        return adaptive_weights
    
    def _compute_reconstruction_loss(self, node_representations, edge_indices):
        num_nodes = node_representations.size(0)
        positive_edges = edge_indices[:, edge_indices[0] < edge_indices[1]]
        positive_similarities = (node_representations[positive_edges[0]] * node_representations[positive_edges[1]]).sum(dim=1)
        positive_loss = F.mse_loss(positive_similarities, torch.ones_like(positive_similarities))

        negative_edges = negative_sampling(
            edge_indices, num_nodes=num_nodes, 
            num_neg_samples=min(positive_edges.size(1), self.training_config.n_neg * num_nodes)
        )
        negative_edges = negative_edges[:, negative_edges[0] < negative_edges[1]]
        negative_similarities = (node_representations[negative_edges[0]] * node_representations[negative_edges[1]]).sum(dim=1)
        negative_loss = F.mse_loss(negative_similarities, torch.zeros_like(negative_similarities))
        
        return positive_loss, negative_loss
    
class RTGNNTrainingConfig:
    def __init__(self, config_dict):
        rtgnn_params = config_dict.get('rtgnn_params', {})
        training_params = config_dict.get('training', {})
        model_params = config_dict.get('model', {})
                
        self.epochs = training_params.get('epochs', 200)
        self.lr = training_params.get('lr', 0.001)  
        self.weight_decay = float(training_params.get('weight_decay', 5e-4))
        self.patience = training_params.get('patience', 8)
        self.dropout = model_params.get('dropout', 0.5)

        self.hidden = model_params.get('hidden_channels', 128)
        self.edge_hidden = rtgnn_params.get('edge_hidden', 64)
        self.n_layers = model_params.get('n_layers', 2)
        self.self_loop = model_params.get('self_loop', True)
        self.mlp_layers = model_params.get('mlp_layers', 2)
        self.train_eps = model_params.get('train_eps', True)
        self.heads = model_params.get('heads', 8)
                
        self.co_lambda = rtgnn_params.get('co_lambda', 0.1)
        self.alpha = rtgnn_params.get('alpha', 0.3)
        self.th = rtgnn_params.get('th', 0.8)
        self.K = rtgnn_params.get('K', 50)
        self.tau = rtgnn_params.get('tau', 0.05)
        self.n_neg = rtgnn_params.get('n_neg', 100)
        self.oversmoothing_every = training_params.get('oversmoothing_every', 20)

class RTGNN(nn.Module):

    def __init__(self, training_config, device, gnn_backbone='gcn', data_for_training=None):
        super().__init__()
        self.device = device
        self.training_config = training_config
        self.gnn_backbone = gnn_backbone.lower()

        if data_for_training is not None:
            (self.node_features, self.node_labels, self.train_node_indices, self.val_node_indices, self.test_node_indices, self.adjacency_matrix) = self._prepare_data(data_for_training)
            input_features = self.node_features.shape[1]
            num_classes = len(np.unique(self.node_labels))
            self.num_classes = num_classes

        gnn_specific_config = self._get_gnn_specific_configuration()
        
        self.dual_branch_predictor = DualBranchGNNModel(
            gnn_type=self.gnn_backbone,
            input_features=input_features,
            hidden_dim=training_config.hidden,
            num_classes=num_classes,
            dropout_rate=training_config.dropout,
            device=device,
            **gnn_specific_config
        )
        
        # Initialize components
        self.structure_estimator = GraphStructureEstimator(input_features, training_config.edge_hidden, training_config, device)
        self.adaptive_loss_function = AdaptiveCoTeachingLoss(training_config)
        self.intraview_regularizer = IntraViewRegularization(device=device)

        self.best_validation_performance = 0
        self.best_model_state = None

        # Oversmoothing evaluation
        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        self.cls_evaluator = ClassificationMetrics(average='macro')
        
        print(f"Initialized RTGNN with {self.gnn_backbone.upper()} backbone")

    def _prepare_data(self, data_for_training):

        node_features = data_for_training.x.cpu().numpy()
        node_labels = data_for_training.y.cpu().numpy()
        
        print("[DEBUG] Checking label corruption in RTGNN training:")
        if hasattr(data_for_training, 'y_original'):
            corrupted_count = (data_for_training.y[data_for_training.train_mask] != data_for_training.y_original[data_for_training.train_mask]).sum().item()
            print(f"[DEBUG] Training labels corrupted: {corrupted_count}/{data_for_training.train_mask.sum().item()} nodes")
            val_clean = (data_for_training.y[data_for_training.val_mask] == data_for_training.y_original[data_for_training.val_mask]).all().item()
            test_clean = (data_for_training.y[data_for_training.test_mask] == data_for_training.y_original[data_for_training.test_mask]).all().item()
            print(f"[DEBUG] Validation labels clean: {val_clean}, Test labels clean: {test_clean}")
        
        adjacency_matrix = to_scipy_sparse_matrix(data_for_training.edge_index, num_nodes=data_for_training.num_nodes)
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        adjacency_matrix = adjacency_matrix.tolil()
        adjacency_matrix[adjacency_matrix > 1] = 1
        adjacency_matrix.setdiag(0)
        
        sparse_features = sp.csr_matrix(node_features)
        row_sums = np.array(sparse_features.sum(1))
        reciprocal_row_sums = np.power(row_sums, -1).flatten()
        reciprocal_row_sums[np.isinf(reciprocal_row_sums)] = 0.
        row_normalization_matrix = sp.diags(reciprocal_row_sums)
        normalized_features = row_normalization_matrix.dot(sparse_features)
        normalized_features = torch.FloatTensor(np.array(normalized_features.todense()))

        if hasattr(data_for_training, 'train_mask'):
            train_node_indices = data_for_training.train_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
            val_node_indices = data_for_training.val_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
            test_node_indices = data_for_training.test_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
        else:
            num_nodes = data_for_training.num_nodes
            train_node_indices = list(range(min(140, num_nodes // 5)))
            val_node_indices = list(range(len(train_node_indices), min(len(train_node_indices) + 500, num_nodes // 2)))
            test_node_indices = list(range(max(len(train_node_indices) + len(val_node_indices), num_nodes // 2), num_nodes))
        
        return normalized_features, node_labels, train_node_indices, val_node_indices, test_node_indices, adjacency_matrix
    
    def _get_gnn_specific_configuration(self):
        base_config = {'num_layers': getattr(self.training_config, 'n_layers', 2)}
            
        if self.gnn_backbone == 'gcn':
            base_config.update({'add_self_loops': getattr(self.training_config, 'self_loop', True)})
            
        elif self.gnn_backbone == 'gin':
            base_config.update({
                'mlp_layers': getattr(self.training_config, 'mlp_layers', 2),
                'train_eps': getattr(self.training_config, 'train_eps', True)
            })
            
        elif self.gnn_backbone in ['gat', 'gatv2']:
            base_config.update({
                'attention_heads': getattr(self.training_config, 'heads', 8),
                'add_self_loops': getattr(self.training_config, 'self_loop', True)
            })
            
        elif self.gnn_backbone == 'gps':
            base_config.update({
                'attention_heads': getattr(self.training_config, 'heads', 4),
                'attn_type': getattr(self.training_config, 'attn_type', 'multihead'),
                'use_pe': getattr(self.training_config, 'use_pe', False),
                'pe_dim': getattr(self.training_config, 'pe_dim', 8)
            })
        
        return base_config

    def estimate_noise_transition_matrix(self, val_predictions, val_labels):
        """R-1 Implementation: Estimate noise transition matrix from validation data."""
        num_classes = self.num_classes
        transition_matrix = torch.zeros((num_classes, num_classes), device=self.device)
        
        # Count transitions from true class (approximated by prediction) to noisy label
        for i in range(num_classes):
            idx = (val_predictions == i)
            if idx.sum() > 0:
                for j in range(num_classes):
                    transition_matrix[i, j] = (val_labels[idx] == j).sum().float() / idx.sum().float()
            else:
                transition_matrix[i, i] = 1.0
                
        return transition_matrix

    def forward(self, node_features, edge_indices, edge_weights=None):
        return self.dual_branch_predictor(node_features, edge_indices, edge_weights)
        
    def evaluate_model_performance(self, node_features, final_edge_indices, final_edge_weights, node_labels, 
                              train_node_indices, val_node_indices, test_node_indices=None):

        self.eval()
        with torch.no_grad():
            first_branch_output, second_branch_output = self.dual_branch_predictor(
                node_features, final_edge_indices, final_edge_weights
            )
            averaged_output = (first_branch_output + second_branch_output) / 2
            
            averaged_embeddings = self.dual_branch_predictor.get_embeddings(
                node_features, final_edge_indices, final_edge_weights
            )
            
            performance_metrics = {}
            performance_metrics['_predictions'] = averaged_output.argmax(dim=1)

            train_loss_branch1 = F.cross_entropy(first_branch_output[train_node_indices], node_labels[train_node_indices])
            train_loss_branch2 = F.cross_entropy(second_branch_output[train_node_indices], node_labels[train_node_indices])
            performance_metrics['train_loss'] = (train_loss_branch1 + train_loss_branch2) / 2
            
            train_predictions = averaged_output[train_node_indices].argmax(dim=1)
            performance_metrics['train_acc'] = (train_predictions == node_labels[train_node_indices]).float().mean().item()
            performance_metrics['train_f1'] = self.cls_evaluator.compute_f1(train_predictions, node_labels[train_node_indices])
            
            val_loss_branch1 = F.cross_entropy(first_branch_output[val_node_indices], node_labels[val_node_indices])
            val_loss_branch2 = F.cross_entropy(second_branch_output[val_node_indices], node_labels[val_node_indices])
            performance_metrics['val_loss'] = (val_loss_branch1 + val_loss_branch2) / 2
            
            val_predictions = averaged_output[val_node_indices].argmax(dim=1)
            performance_metrics['val_acc'] = (val_predictions == node_labels[val_node_indices]).float().mean().item()
            performance_metrics['val_f1'] = self.cls_evaluator.compute_f1(val_predictions, node_labels[val_node_indices])
            
            train_mask = torch.zeros(node_features.size(0), dtype=torch.bool, device=self.device)
            train_mask[train_node_indices] = True
            val_mask = torch.zeros(node_features.size(0), dtype=torch.bool, device=self.device)
            val_mask[val_node_indices] = True
            
            train_oversmoothing = compute_oversmoothing_for_mask(self.oversmoothing_evaluator, averaged_embeddings, final_edge_indices, train_mask)
            val_oversmoothing = compute_oversmoothing_for_mask(self.oversmoothing_evaluator, averaged_embeddings, final_edge_indices, val_mask)

            if test_node_indices is not None:
                test_loss_branch1 = F.cross_entropy(first_branch_output[test_node_indices], node_labels[test_node_indices])
                test_loss_branch2 = F.cross_entropy(second_branch_output[test_node_indices], node_labels[test_node_indices])
                performance_metrics['test_loss'] = (test_loss_branch1 + test_loss_branch2) / 2
                
                test_predictions = averaged_output[test_node_indices].argmax(dim=1)
                performance_metrics['test_acc'] = (test_predictions == node_labels[test_node_indices]).float().mean().item()
                performance_metrics['test_f1'] = self.cls_evaluator.compute_f1(test_predictions, node_labels[test_node_indices])
                
                test_mask = torch.zeros(node_features.size(0), dtype=torch.bool, device=self.device)
                test_mask[test_node_indices] = True
                test_oversmoothing = compute_oversmoothing_for_mask(self.oversmoothing_evaluator, averaged_embeddings, final_edge_indices, test_mask)
                
                return performance_metrics, train_oversmoothing, val_oversmoothing, test_oversmoothing
            
            return performance_metrics, train_oversmoothing, val_oversmoothing

    def evaluate_final_performance(self, clean_labels=None):

        node_features = self.node_features.to(self.device)
        node_labels = torch.as_tensor(clean_labels if clean_labels is not None else self.node_labels, dtype=torch.long, device=self.device)

        if self.best_model_state is None:
            print("Model not trained yet.")
            return {
                'test_cls': dict(ZERO_CLS),
                'train_cls': dict(ZERO_CLS),
                'val_cls': dict(ZERO_CLS),
                'test_oversmoothing': dict(DEFAULT_OVERSMOOTHING),
                'train_oversmoothing_final': dict(DEFAULT_OVERSMOOTHING),
                'val_oversmoothing_final': dict(DEFAULT_OVERSMOOTHING),
            }

        self.eval()
        with torch.no_grad():
            num_nodes = node_features.size(0)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            train_mask[self.train_node_indices] = True
            val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            val_mask[self.val_node_indices] = True
            test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            test_mask[self.test_node_indices] = True

            best_edges = self.best_model_state['edges']
            best_weights = self.best_model_state['weights']

            def _get_predictions():
                out1, out2 = self.dual_branch_predictor(node_features, best_edges, best_weights)
                return ((out1 + out2) / 2).argmax(dim=1)

            def _get_embeddings():
                return self.dual_branch_predictor.get_embeddings(node_features, best_edges, best_weights)

            results = evaluate_model(
                _get_predictions, _get_embeddings, node_labels,
                train_mask, val_mask, test_mask,
                best_edges, self.device
            )

            print(f"Test Acc: {results['test_cls']['accuracy']:.4f} | Test F1: {results['test_cls']['f1']:.4f} | "
                  f"Precision: {results['test_cls']['precision']:.4f}, Recall: {results['test_cls']['recall']:.4f}")
            print(f"Test Oversmoothing: {results['test_oversmoothing']}")

            return results
        
    def _generate_knn_edge_connections(self, node_features, original_edge_indices, train_node_indices, k=None):
        #Generate KNN edges
        if k is None:
            k = self.training_config.K
            
        if k == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)
        
        existing_edges = set(map(tuple, original_edge_indices.t().cpu().numpy()))
        
        normalized_features = F.normalize(node_features, dim=1)
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        
        new_edge_list = []
        train_node_tensor = torch.tensor(train_node_indices, device=self.device)
        all_nodes = torch.arange(node_features.size(0), device=self.device)
        unlabeled_nodes = all_nodes[~torch.isin(all_nodes, train_node_tensor)]
        
        # Find k most similar unlabeled nodes
        for train_node in train_node_tensor:
            if len(unlabeled_nodes) == 0:
                continue
                
            # Compute similarities
            similarity_scores = similarity_matrix[train_node, unlabeled_nodes]
            _, top_k_indices = similarity_scores.topk(min(k, len(unlabeled_nodes)))
            
            for idx in top_k_indices:
                neighbor_node = unlabeled_nodes[idx].item()
                edge = (train_node.item(), neighbor_node)
                reverse_edge = (neighbor_node, train_node.item())
                
                if edge not in existing_edges and reverse_edge not in existing_edges:
                    new_edge_list.append([train_node.item(), neighbor_node])
                    new_edge_list.append([neighbor_node, train_node.item()])
        
        if new_edge_list:
            return torch.tensor(new_edge_list, device=self.device).t()
        else:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)
    
    def _compute_pseudo_labeling_loss(self, first_branch_output, second_branch_output, train_node_indices):
        #Compute pseudo-labeling loss
        all_nodes = torch.arange(first_branch_output.size(0), device=self.device)
        unlabeled_nodes = all_nodes[~torch.isin(all_nodes, torch.tensor(train_node_indices, device=self.device))]
        
        if len(unlabeled_nodes) == 0:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            first_branch_probs = F.softmax(first_branch_output[unlabeled_nodes], dim=1)
            second_branch_probs = F.softmax(second_branch_output[unlabeled_nodes], dim=1)
            first_confidences, first_predictions = first_branch_probs.max(dim=1)
            second_confidences, second_predictions = second_branch_probs.max(dim=1)
            
            predictions_consistent = (first_predictions == second_predictions)
            confidence_threshold = self.training_config.th ** 2
            high_confidence = (first_confidences * second_confidences) > confidence_threshold
            
            reliable_pseudo_mask = predictions_consistent & high_confidence
            if reliable_pseudo_mask.sum() == 0:
                return torch.tensor(0.0, device=self.device)
                
            reliable_pseudo_nodes = unlabeled_nodes[reliable_pseudo_mask]
            pseudo_labels = first_predictions[reliable_pseudo_mask]

        cross_entropy_loss = (
            F.cross_entropy(first_branch_output[reliable_pseudo_nodes], pseudo_labels) +
            F.cross_entropy(second_branch_output[reliable_pseudo_nodes], pseudo_labels)
        ) / 2

        # Compute KL divergence loss
        kl_divergence_loss = (
            F.kl_div(F.log_softmax(first_branch_output[reliable_pseudo_nodes], dim=1),
                    F.softmax(second_branch_output[reliable_pseudo_nodes], dim=1), reduction='batchmean') +
            F.kl_div(F.log_softmax(second_branch_output[reliable_pseudo_nodes], dim=1),
                    F.softmax(first_branch_output[reliable_pseudo_nodes], dim=1), reduction='batchmean')
        )

        return cross_entropy_loss + self.training_config.co_lambda * kl_divergence_loss


@register('rtgnn')
class RTGNNMethodTrainer(BaseTrainer):

    def train(self):
        from methods.registry import get_helper
        from training.training_loop import TrainingLoop
        d = self.init_data
        self._helper = get_helper('rtgnn')
        self._loop = TrainingLoop(self._helper, log_epoch_fn=self.log_epoch)
        result = self._loop.run(
            d['backbone_model'], d['data_for_training'],
            self.config, d['device'], d,
        )
        self._rtgnn = self._loop.state['rtgnn']
        return result

    def _get_state(self):
        if hasattr(self, '_loop') and hasattr(self._loop, '_state'):
            return self._loop.state
        return None

    def get_checkpoint_state(self) -> dict:
        return self._helper.get_checkpoint_state(self._get_state())

    def _setup_for_eval(self, checkpoint_state):
        """Create state via helper so load_checkpoint_state can populate it."""
        from methods.registry import get_helper
        from training.training_loop import TrainingLoop
        d = self.init_data
        if not hasattr(self, '_helper'):
            self._helper = get_helper('rtgnn')
        self._loop = TrainingLoop(self._helper)
        state = self._helper.setup(
            d['backbone_model'], d['data_for_training'],
            self.config, d['device'], d,
        )
        self._loop._state = state
        self._rtgnn = state['rtgnn']

    def load_checkpoint_state(self, state):
        if not hasattr(self, '_rtgnn'):
            self._setup_for_eval(state)
        self._helper.load_checkpoint_state(self._get_state(), state)

    def profile_flops(self):
        from util.profiling import profile_model_flops
        d = self.init_data
        rtgnn = self._rtgnn
        dual = rtgnn.dual_branch_predictor
        data = d['data_for_training']

        def fwd():
            return dual(data.x, data.edge_index)

        return profile_model_flops(dual, data, d['device'], forward_fn=fwd)

    def profile_training_step(self):
        """Profile one training step for RTGNN (structure_estimator + dual_branch).

        Simplified vs the real step — omits KNN edges, pseudo-labeling,
        intra-view regularization, and co-teaching loss (which involve scipy
        ops the profiler cannot track).  Captures the dominant neural-network
        forward+backward cost.
        """
        from util.profiling import profile_training_step_flops
        rtgnn = self._rtgnn
        d = self.init_data

        node_features = rtgnn.node_features.to(rtgnn.device)
        node_labels = torch.as_tensor(rtgnn.node_labels, dtype=torch.long,
                                      device=rtgnn.device)
        edge_indices, _ = from_scipy_sparse_matrix(rtgnn.adjacency_matrix)
        edge_indices = edge_indices.to(rtgnn.device)
        train_indices = rtgnn.train_node_indices

        def step_fn():
            node_reps, recon_loss = rtgnn.structure_estimator(
                node_features, edge_indices
            )
            base_weights = torch.ones(edge_indices.size(1), device=rtgnn.device)
            adaptive_weights = rtgnn.structure_estimator.compute_adaptive_edge_weights(
                edge_indices, node_reps, base_weights
            )
            out1, out2 = rtgnn.dual_branch_predictor(
                node_features, edge_indices, adaptive_weights
            )
            loss = (F.cross_entropy(out1[train_indices], node_labels[train_indices])
                    + F.cross_entropy(out2[train_indices], node_labels[train_indices]))
            return loss

        models = [rtgnn.dual_branch_predictor, rtgnn.structure_estimator]
        return profile_training_step_flops(models, d['device'], step_fn)

    def evaluate(self):
        d = self.init_data
        clean_labels = d['data'].y_original.cpu().numpy()
        return self._rtgnn.evaluate_final_performance(clean_labels=clean_labels)
