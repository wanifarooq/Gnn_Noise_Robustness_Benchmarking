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
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

from model.gnns import GCN, GIN, GAT, GATv2, GPS
from model.evaluation import OversmoothingMetrics


class DualBranchGNNModel(nn.Module):
    
    def __init__(self, gnn_type: str, input_features: int, hidden_dim: int, num_classes: int,
                 dropout_rate: float = 0.5, use_edge_weights: bool = False, attention_heads: int = 4,
                 num_layers: int = None, device=None, add_self_loops: bool = False,
                 attn_type: str = 'multihead', use_pe: bool = False, pe_dim: int = 8):
        super().__init__()
        self.device = device
        self.gnn_type = gnn_type.lower()
        self.use_edge_weights = use_edge_weights
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
            elif self.gnn_type == 'gat' and not use_edge_weights:
                return GAT(input_features, hidden_dim, num_classes, n_layers=num_layers or 3,
                          heads=attention_heads, dropout=dropout_rate, self_loop=self.add_self_loops)
            elif self.gnn_type == 'gatv2' and not use_edge_weights:
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

    def forward(self, node_features, edge_indices, edge_weights=None):
        edge_attributes = edge_weights.unsqueeze(-1) if self.use_edge_weights and edge_weights is not None else None

        graph_data = Data(x=node_features, edge_index=edge_indices)
        if edge_attributes is not None:
            graph_data.edge_attr = edge_attributes
        if self.device is not None:
            graph_data = graph_data.to(self.device)

        first_branch_output = self.first_branch(graph_data)
        second_branch_output = self.second_branch(graph_data)
        return first_branch_output, second_branch_output

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
        self.forget_rate_increment = 0.5 / self.total_epochs
        self.residual_weight_decay = 1.0

    def forward(self, first_branch_logits, second_branch_logits, target_labels, current_epoch=0):
        first_branch_loss = F.cross_entropy(first_branch_logits, target_labels, reduction='none')
        second_branch_loss = F.cross_entropy(second_branch_logits, target_labels, reduction='none')
        combined_loss = first_branch_loss + second_branch_loss
        
        if current_epoch == 0:
            return combined_loss.mean()

        # Sort samples by loss
        sorted_loss_indices = torch.argsort(combined_loss)
        current_forget_rate = self.forget_rate_increment * current_epoch
        remember_rate = max(0.5, 1 - current_forget_rate)
        num_samples_to_remember = int(remember_rate * len(combined_loss))
        
        clean_sample_indices = sorted_loss_indices[:num_samples_to_remember]
        noisy_sample_indices = sorted_loss_indices[num_samples_to_remember:]

        # Loss for clean samples
        clean_samples_loss = combined_loss[clean_sample_indices].mean()

        # Correction loss for noisy samples
        correction_loss = torch.tensor(0.0, device=first_branch_logits.device)
        if len(noisy_sample_indices) > 0:
            first_probabilities = F.softmax(first_branch_logits, dim=1)
            second_probabilities = F.softmax(second_branch_logits, dim=1)
            first_predictions = first_branch_logits.max(1)[1]
            second_predictions = second_branch_logits.max(1)[1]
            first_confidences = first_probabilities.max(1)[0]
            second_confidences = second_probabilities.max(1)[0]

            predictions_agree = first_predictions[noisy_sample_indices] == second_predictions[noisy_sample_indices]
            confidence_threshold = 1 - (1 - min(0.5, 1/first_branch_logits.size(0))) * current_epoch/self.total_epochs
            high_confidence = (first_confidences[noisy_sample_indices] * second_confidences[noisy_sample_indices] > confidence_threshold)
            
            correction_mask = predictions_agree & high_confidence
            if correction_mask.sum() > 0:
                correction_indices = noisy_sample_indices[correction_mask]
                confidence_weights = (first_confidences[correction_indices] * second_confidences[correction_indices])**(0.5 - 0.5*current_epoch/self.total_epochs)
                correction_loss = (confidence_weights * (
                    F.cross_entropy(first_branch_logits[correction_indices], first_predictions[correction_indices], reduction='none') +
                    F.cross_entropy(second_branch_logits[correction_indices], first_predictions[correction_indices], reduction='none')
                )).mean()

        # Residual loss
        residual_loss = self.residual_weight_decay * combined_loss[noisy_sample_indices].mean() if len(noisy_sample_indices) > 0 else 0.0

        # KL divergence regularization
        kl_regularization = self._compute_kl_divergence(first_branch_logits, second_branch_logits) + \
                           self._compute_kl_divergence(second_branch_logits, first_branch_logits)

        return clean_samples_loss + correction_loss + residual_loss + self.training_config.co_lambda * kl_regularization

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
        
        print(f"Initialized RTGNN with {self.gnn_backbone.upper()} backbone")

    def _prepare_data(self, data_for_training):

        node_features = data_for_training.x.cpu().numpy()
        node_labels = data_for_training.y.cpu().numpy()
        
        print(f"[DEBUG] Checking label corruption in RTGNN training:")
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
    
    def _compute_oversmoothing_metrics_for_subset(self, node_embeddings, edge_indices, subset_mask, node_labels=None):
        try:
            subset_node_indices = torch.where(subset_mask)[0]
            subset_embeddings = node_embeddings[subset_mask]
            
            subset_nodes_set = set(subset_node_indices.cpu().numpy())
            edge_in_subset_mask = torch.tensor([
                source.item() in subset_nodes_set and target.item() in subset_nodes_set
                for source, target in edge_indices.t()
            ], device=edge_indices.device)
            
            if not edge_in_subset_mask.any():
                return {
                    'NumRank': float(min(subset_embeddings.shape)),
                    'Erank': float(min(subset_embeddings.shape)),
                    'EDir': 0.0,
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0
                }
            
            subset_edges = edge_indices[:, edge_in_subset_mask]
            node_index_mapping = {original_idx.item(): local_idx for local_idx, original_idx in enumerate(subset_node_indices)}
            
            remapped_edges = torch.stack([
                torch.tensor([node_index_mapping[source.item()] for source in subset_edges[0]], device=edge_indices.device),
                torch.tensor([node_index_mapping[target.item()] for target in subset_edges[1]], device=edge_indices.device)
            ])

            subset_graph_data = [{
                'X': subset_embeddings,
                'edge_index': remapped_edges,
                'edge_weight': None
            }]
            
            return self.oversmoothing_evaluator.compute_all_metrics(
                X=subset_embeddings,
                edge_index=remapped_edges,
                graphs_in_class=subset_graph_data
            )
            
        except Exception as e:
            print(f"Warning: Could not compute oversmoothing metrics for subset: {e}")
            return None
            
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
            
            graph_data = Data(x=node_features, edge_index=final_edge_indices)
            if final_edge_weights is not None:
                graph_data.edge_attr = final_edge_weights.unsqueeze(-1) if final_edge_weights.dim() == 1 else final_edge_weights
            graph_data = graph_data.to(self.device)

            try:
                first_embeddings = self.dual_branch_predictor.first_branch.get_embeddings(graph_data) if hasattr(self.dual_branch_predictor.first_branch, 'get_embeddings') else self.dual_branch_predictor.first_branch(graph_data)
                second_embeddings = self.dual_branch_predictor.second_branch.get_embeddings(graph_data) if hasattr(self.dual_branch_predictor.second_branch, 'get_embeddings') else self.dual_branch_predictor.second_branch(graph_data)
                averaged_embeddings = (first_embeddings + second_embeddings) / 2
            except:
                averaged_embeddings = averaged_output
            
            performance_metrics = {}
            
            train_loss_branch1 = F.cross_entropy(first_branch_output[train_node_indices], node_labels[train_node_indices])
            train_loss_branch2 = F.cross_entropy(second_branch_output[train_node_indices], node_labels[train_node_indices])
            performance_metrics['train_loss'] = (train_loss_branch1 + train_loss_branch2) / 2
            
            train_predictions = averaged_output[train_node_indices].argmax(dim=1)
            performance_metrics['train_acc'] = (train_predictions == node_labels[train_node_indices]).float().mean().item()
            performance_metrics['train_f1'] = f1_score(node_labels[train_node_indices].cpu(), train_predictions.cpu(), average='macro')
            
            val_loss_branch1 = F.cross_entropy(first_branch_output[val_node_indices], node_labels[val_node_indices])
            val_loss_branch2 = F.cross_entropy(second_branch_output[val_node_indices], node_labels[val_node_indices])
            performance_metrics['val_loss'] = (val_loss_branch1 + val_loss_branch2) / 2
            
            val_predictions = averaged_output[val_node_indices].argmax(dim=1)
            performance_metrics['val_acc'] = (val_predictions == node_labels[val_node_indices]).float().mean().item()
            performance_metrics['val_f1'] = f1_score(node_labels[val_node_indices].cpu(), val_predictions.cpu(), average='macro')
            
            train_mask = torch.zeros(node_features.size(0), dtype=torch.bool, device=self.device)
            train_mask[train_node_indices] = True
            val_mask = torch.zeros(node_features.size(0), dtype=torch.bool, device=self.device)
            val_mask[val_node_indices] = True
            
            train_oversmoothing = self._compute_oversmoothing_metrics_for_subset(averaged_embeddings, final_edge_indices, train_mask, node_labels)
            val_oversmoothing = self._compute_oversmoothing_metrics_for_subset(averaged_embeddings, final_edge_indices, val_mask, node_labels)

            if test_node_indices is not None:
                test_loss_branch1 = F.cross_entropy(first_branch_output[test_node_indices], node_labels[test_node_indices])
                test_loss_branch2 = F.cross_entropy(second_branch_output[test_node_indices], node_labels[test_node_indices])
                performance_metrics['test_loss'] = (test_loss_branch1 + test_loss_branch2) / 2
                
                test_predictions = averaged_output[test_node_indices].argmax(dim=1)
                performance_metrics['test_acc'] = (test_predictions == node_labels[test_node_indices]).float().mean().item()
                performance_metrics['test_f1'] = f1_score(node_labels[test_node_indices].cpu(), test_predictions.cpu(), average='macro')
                
                test_mask = torch.zeros(node_features.size(0), dtype=torch.bool, device=self.device)
                test_mask[test_node_indices] = True
                test_oversmoothing = self._compute_oversmoothing_metrics_for_subset(averaged_embeddings, final_edge_indices, test_mask, node_labels)
                
                return performance_metrics, train_oversmoothing, val_oversmoothing, test_oversmoothing
            
            return performance_metrics, train_oversmoothing, val_oversmoothing

    def train_model(self):


        per_epochs_oversmoothing = defaultdict(list)

        training_start_time = time.time()

        node_features = self.node_features.to(self.device)
        node_labels = torch.as_tensor(self.node_labels, dtype=torch.long, device=self.device)
        train_indices = self.train_node_indices
        val_indices = self.val_node_indices
        test_indices = self.test_node_indices
        adjacency_matrix = self.adjacency_matrix
        
        edge_indices, _ = from_scipy_sparse_matrix(adjacency_matrix)
        edge_indices = edge_indices.to(self.device)
        node_features = node_features.to(self.device)
        
        # Generate KNN edges
        knn_edge_indices = self._generate_knn_edge_connections(node_features, edge_indices, train_indices)
        
        optimizer = optim.Adam(self.parameters(), lr=self.training_config.lr, 
                             weight_decay=self.training_config.weight_decay)

        # Early stopping
        early_stop_patience = getattr(self.training_config, 'patience', 8)
        patience_counter = 0
        best_validation_loss = float('inf') #best_validation_accuracy = 0.0
        
        print(f"Starting RTGNN training with {self.gnn_backbone.upper()}")
        
        for epoch in range(self.training_config.epochs):
            self.train()
            optimizer.zero_grad()
            
            node_representations, reconstruction_loss = self.structure_estimator(node_features, edge_indices)
            
            # Combine original and KNN edges
            if knn_edge_indices.size(1) > 0:
                combined_edge_indices = torch.cat([edge_indices, knn_edge_indices], dim=1)
                base_edge_weights = torch.cat([
                    torch.ones(edge_indices.size(1)),
                    torch.zeros(knn_edge_indices.size(1))
                ]).to(self.device)
            else:
                combined_edge_indices = edge_indices
                base_edge_weights = torch.ones(edge_indices.size(1)).to(self.device)
            
            # Compute adaptive edge weights
            adaptive_edge_weights = self.structure_estimator.compute_adaptive_edge_weights(
                combined_edge_indices, node_representations, base_edge_weights
            )
            
            # Filter edges with positive weights
            valid_edge_mask = adaptive_edge_weights > 0
            final_edge_indices = combined_edge_indices[:, valid_edge_mask]
            final_edge_weights = adaptive_edge_weights[valid_edge_mask]

            first_branch_output, second_branch_output = self.dual_branch_predictor(
                node_features, final_edge_indices, final_edge_weights
            )

            main_prediction_loss = self.adaptive_loss_function(
                first_branch_output[train_indices], second_branch_output[train_indices], 
                node_labels[train_indices], epoch
            )

            pseudo_labeling_loss = self._compute_pseudo_labeling_loss(first_branch_output, second_branch_output, train_indices)

            # Compute intra-view regularization loss
            intraview_consistency_loss = self.intraview_regularizer(
                first_branch_output, second_branch_output, final_edge_indices, final_edge_weights, train_indices
            )

            # Combine all loss components
            total_training_loss = (main_prediction_loss +
                          self.training_config.alpha * reconstruction_loss +
                          pseudo_labeling_loss +
                          self.training_config.co_lambda * intraview_consistency_loss)

            total_training_loss.backward()
            optimizer.step()

            compute_oversmoothing = (epoch + 1) % 20 == 0 or epoch == 0
            if compute_oversmoothing:
                performance_metrics, train_oversmoothing, val_oversmoothing = self.evaluate_model_performance(
                    node_features, final_edge_indices, final_edge_weights, node_labels, train_indices, val_indices
                )

                for key, value in train_oversmoothing.items():
                    per_epochs_oversmoothing[key].append(value)

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

                print(f"Epoch {epoch:03d} | Train Acc: {performance_metrics['train_acc']:.4f}, Val Acc: {performance_metrics['val_acc']:.4f} | "
                      f"Train F1: {performance_metrics['train_f1']:.4f}, Val F1: {performance_metrics['val_f1']:.4f}")
                print(f"Train DE: {train_de_edir:.4f}, Val DE: {val_de_edir:.4f} | "
                      f"Train DE_trad: {train_de_traditional:.4f}, Val DE_trad: {val_de_traditional:.4f} | "
                      f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                      f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                      f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                      f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")
            else:
                performance_metrics, _, _ = self.evaluate_model_performance(
                    node_features, final_edge_indices, final_edge_weights, node_labels, train_indices, val_indices
                )
                print(f"Epoch {epoch:03d} | Train Acc: {performance_metrics['train_acc']:.4f}, Val Acc: {performance_metrics['val_acc']:.4f} | "
                      f"Train F1: {performance_metrics['train_f1']:.4f}, Val F1: {performance_metrics['val_f1']:.4f}")

            # Early stopping
            current_val_loss = performance_metrics['val_loss'].item() #current_val_accuracy = performance_metrics['val_acc']
            if current_val_loss < best_validation_loss: #if current_val_accuracy > best_validation_accuracy:
                best_validation_loss = current_val_loss # best_validation_accuracy = current_val_accuracy
                patience_counter = 0
                self.best_model_state = {
                    'model': deepcopy(self.state_dict()),
                    'edges': final_edge_indices.clone(),
                    'weights': final_edge_weights.clone()
                }
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                    break

        if self.best_model_state is not None:
            self.load_state_dict(self.best_model_state['model'])
            
        total_training_time = time.time() - training_start_time
        
        if test_indices is not None:
            final_performance_metrics, final_train_oversmoothing, final_val_oversmoothing, final_test_oversmoothing = self.evaluate_model_performance(
                node_features, self.best_model_state['edges'], self.best_model_state['weights'], 
                node_labels, train_indices, val_indices, test_indices
            )
            
            print(f"\nTraining completed in {total_training_time:.2f}s")
            print(f"Test Loss: {final_performance_metrics['test_loss']:.4f} | Test Acc: {final_performance_metrics['test_acc']:.4f} | Test F1: {final_performance_metrics['test_f1']:.4f}")
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
            
        print(f"Training completed! Best validation loss: {best_validation_loss:.4f}") #print(f"Training completed! Best validation accuracy: {best_validation_accuracy:.4f}")
        return per_epochs_oversmoothing


    def evaluate_final_performance(self, clean_labels=None):

        node_features = self.node_features.to(self.device)
        node_labels = torch.as_tensor(clean_labels if clean_labels is not None else self.node_labels, dtype=torch.long, device=self.device)
        test_indices = self.test_node_indices

        if self.best_model_state is None:
            print("Model not trained yet.")
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'oversmoothing': {
                    'EDir': 0.0,
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0,
                    'NumRank': 0.0,
                    'Erank': 0.0
                }
            }

        self.eval()
        with torch.no_grad():
            node_features = node_features.to(self.device)
            node_labels = torch.as_tensor(node_labels, dtype=torch.long, device=self.device)
            
            first_branch_output, second_branch_output = self.dual_branch_predictor(
                node_features, 
                self.best_model_state['edges'], 
                self.best_model_state['weights']
            )

            test_loss_branch1 = F.cross_entropy(first_branch_output[test_indices], node_labels[test_indices])
            test_loss_branch2 = F.cross_entropy(second_branch_output[test_indices], node_labels[test_indices])
            test_loss = (test_loss_branch1 + test_loss_branch2) / 2

            averaged_test_predictions = (first_branch_output[test_indices] + second_branch_output[test_indices]) / 2
            predicted_test_labels = averaged_test_predictions.argmax(dim=1)
            
            true_labels = node_labels[test_indices].cpu().numpy()
            predicted_labels = predicted_test_labels.cpu().numpy()
            
            test_accuracy = (predicted_test_labels == node_labels[test_indices]).float().mean().item()
            test_f1 = f1_score(true_labels, predicted_labels, average='macro')
            test_precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
            test_recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)

            test_mask = torch.zeros(node_features.size(0), dtype=torch.bool, device=self.device)
            test_mask[test_indices] = True
            
            graph_data = Data(x=node_features, edge_index=self.best_model_state['edges'])
            if self.best_model_state['weights'] is not None:
                graph_data.edge_attr = self.best_model_state['weights'].unsqueeze(-1) if self.best_model_state['weights'].dim() == 1 else self.best_model_state['weights']
            graph_data = graph_data.to(self.device)
            
            first_embeddings = self.dual_branch_predictor.first_branch.get_embeddings(graph_data) if hasattr(self.dual_branch_predictor.first_branch, 'get_embeddings') else self.dual_branch_predictor.first_branch(graph_data)
            second_embeddings = self.dual_branch_predictor.second_branch.get_embeddings(graph_data) if hasattr(self.dual_branch_predictor.second_branch, 'get_embeddings') else self.dual_branch_predictor.second_branch(graph_data)
            averaged_embeddings = (first_embeddings + second_embeddings) / 2

            test_oversmoothing = self._compute_oversmoothing_metrics_for_subset(
                averaged_embeddings, self.best_model_state['edges'], test_mask, node_labels
            )

            if test_oversmoothing is not None:
                oversmoothing_dict = {
                    'EDir': test_oversmoothing['EDir'],
                    'EDir_traditional': test_oversmoothing['EDir_traditional'],
                    'EProj': test_oversmoothing['EProj'],
                    'MAD': test_oversmoothing['MAD'],
                    'NumRank': test_oversmoothing['NumRank'],
                    'Erank': test_oversmoothing['Erank']
                }
            else:
                oversmoothing_dict = {
                    'EDir': 0.0,
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0,
                    'NumRank': 0.0,
                    'Erank': 0.0
                }

            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | Test F1: {test_f1:.4f}")
            print(f"Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f}")
            
            if test_oversmoothing is not None:
                print(f"Test Oversmoothing - EDir: {oversmoothing_dict['EDir']:.4f}, "
                      f"EDir_traditional: {oversmoothing_dict['EDir_traditional']:.4f}, "
                      f"EProj: {oversmoothing_dict['EProj']:.4f}, MAD: {oversmoothing_dict['MAD']:.4f}, "
                      f"NumRank: {oversmoothing_dict['NumRank']:.4f}, Erank: {oversmoothing_dict['Erank']:.4f}")
            
            return {
                'accuracy': test_accuracy,
                'f1': test_f1,
                'precision': test_precision,
                'recall': test_recall,
                'oversmoothing': oversmoothing_dict,
            }
        
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
