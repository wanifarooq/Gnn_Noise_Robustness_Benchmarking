import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_dense_adj
from typing import Tuple
from scipy.sparse import csr_matrix

from model.evaluation import evaluate_model
from model.base import BaseTrainer
from model.registry import register


class MaximalCodingRateReductionLoss(nn.Module):
    #Maximal Coding Rate Reduction loss

    def __init__(self, compression_weight: float = 1.0, discrimination_weight: float = 1.0,
                 eps: float = 0.01):
        super().__init__()
        self.compression_weight = compression_weight
        self.discrimination_weight = discrimination_weight
        self.eps = eps

    def compute_discrimination_loss_empirical(self, feature_matrix: torch.Tensor) -> torch.Tensor:
        #Compute empirical discrimination loss term
        num_features, num_samples = feature_matrix.shape
        identity_matrix = torch.eye(num_features, device=feature_matrix.device)
        scalar = num_features / (num_samples * self.eps)
        _, log_determinant = torch.linalg.slogdet(
            identity_matrix +
            self.discrimination_weight * scalar * feature_matrix @ feature_matrix.T
        )

        return log_determinant / 2.

    def compute_discrimination_loss_theoretical(self, feature_matrix: torch.Tensor) -> torch.Tensor:
        #Compute theoretical discrimination loss term
        num_features, num_samples = feature_matrix.shape
        identity_matrix = torch.eye(num_features, device=feature_matrix.device)
        scalar = num_features / (num_samples * self.eps)
        log_determinant = torch.logdet(identity_matrix + scalar * feature_matrix @ feature_matrix.T)
        return log_determinant / 2.

    def compute_compression_loss_empirical_multiclass(self, feature_matrix: torch.Tensor,
                                                     predicted_labels: torch.Tensor) -> torch.Tensor:
        #Compute compression loss for multiple classes
        num_features, num_samples = feature_matrix.shape
        num_classes = torch.max(predicted_labels) + 1
        identity_matrix = torch.eye(num_features, device=feature_matrix.device)
        compression_loss: torch.Tensor = torch.tensor(0.0, device=feature_matrix.device)

        for class_idx in range(num_classes):
            class_mask = torch.where(predicted_labels == class_idx)[0]
            if len(class_mask) == 0:
                continue
            class_features = feature_matrix[:, class_mask]
            class_trace = class_features.shape[1] + 1e-8
            scalar = num_features / (class_trace * self.eps)
            feature_covariance = class_features @ class_features.T
            if feature_covariance.shape[0] < num_features:
                padding = (0, num_features - feature_covariance.shape[0], 0, num_features - feature_covariance.shape[0])
                feature_covariance = F.pad(feature_covariance, padding)
            log_det = torch.logdet(identity_matrix + scalar * feature_covariance)
            compression_loss += log_det * class_trace / num_samples

        return compression_loss / 2.

    def compute_compression_loss_empirical(self, feature_matrix: torch.Tensor,
                                         membership_matrices: torch.Tensor) -> torch.Tensor:
        #Compute empirical compression loss with membership matrices
        num_features, num_samples = feature_matrix.shape
        num_classes, _, _ = membership_matrices.shape
        identity_matrix = torch.eye(num_features, device=feature_matrix.device)
        compression_loss: torch.Tensor = torch.tensor(0.0, device=feature_matrix.device)

        for class_idx in range(num_classes):
            membership_trace = torch.trace(membership_matrices[class_idx]) + 1e-8
            scalar = num_features / (membership_trace * self.eps)
            log_det = torch.logdet(identity_matrix + scalar * feature_matrix @ membership_matrices[class_idx] @ feature_matrix.T)
            compression_loss += log_det * membership_trace / num_samples

        return compression_loss / 2.

    def compute_compression_loss_theoretical(self, feature_matrix: torch.Tensor,
                                           membership_matrices: torch.Tensor) -> torch.Tensor:
        # Compute theoretical compression loss with membership matrices
        num_features, num_samples = feature_matrix.shape
        num_classes, _, _ = membership_matrices.shape
        identity_matrix = torch.eye(num_features, device=feature_matrix.device)
        compression_loss: torch.Tensor = torch.tensor(0.0, device=feature_matrix.device)
        for class_idx in range(num_classes):
            membership_trace = torch.trace(membership_matrices[class_idx]) + 1e-8
            scalar = num_features / (membership_trace * self.eps)
            log_det = torch.logdet(identity_matrix + scalar * feature_matrix @ membership_matrices[class_idx] @ feature_matrix.T)
            compression_loss += membership_trace / (2 * num_samples) * log_det
        return compression_loss

    def forward(self, node_features: torch.Tensor, graph_data, adjacency_matrix: torch.Tensor,
                semantic_labels: torch.Tensor, label_prop_alpha: float, cosine_weight_beta: float,
                propagation_steps: int, train_node_mask: torch.Tensor):

        num_classes = int(graph_data.y.max().item() + 1)
        train_features = node_features[train_node_mask]
        feature_matrix = node_features.T

        # Update semantic labels with cosine similarity
        semantic_labels = self._update_semantic_labels_with_cosine_similarity(
            train_features, semantic_labels, node_features, train_node_mask, cosine_weight_beta, num_classes
        )

        # Apply label propagation (maintain gradients)
        for _ in range(propagation_steps):
            semantic_labels = (1 - label_prop_alpha) * (adjacency_matrix @ semantic_labels) + label_prop_alpha * semantic_labels

        # E-1 Fix: Use soft probabilities instead of hard argmax to keep graph differentiable
        # This allows the Coding Rate loss to provide gradients back to the backbone.
        soft_labels = F.softmax(semantic_labels, dim=1)
        predicted_labels = torch.argmax(soft_labels, dim=1)

        # Compute discrimination loss
        discrimination_loss_empirical = self.compute_discrimination_loss_empirical(feature_matrix)

        # E-1 Fix: Compute membership matrices differentiably
        membership_matrices = self._convert_labels_to_membership_matrices_differentiable(soft_labels, num_classes)

        compression_loss_empirical = self.compute_compression_loss_empirical(feature_matrix, membership_matrices)
        compression_loss_theoretical = self.compute_compression_loss_theoretical(feature_matrix, membership_matrices)
        discrimination_loss_theoretical = self.compute_discrimination_loss_theoretical(feature_matrix)

        total_loss_empirical = -self.discrimination_weight * discrimination_loss_empirical + self.compression_weight * compression_loss_empirical

        return (total_loss_empirical,
                [discrimination_loss_empirical.item(), compression_loss_empirical.item()],
                [discrimination_loss_theoretical.item(), compression_loss_theoretical.item()],
                predicted_labels)

    def _convert_labels_to_membership_matrices_differentiable(self, soft_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """E-1 Implementation: Differentiable conversion to membership matrices."""
        num_samples = soft_labels.shape[0]
        # Membership matrix Pi_j is diag(soft_labels[:, j])
        # We return a (C, N, N) tensor where each slice is a diagonal matrix.
        membership_matrices = torch.zeros((num_classes, num_samples, num_samples), device=soft_labels.device)
        for j in range(num_classes):
            membership_matrices[j] = torch.diag(soft_labels[:, j])
        return membership_matrices

    def _convert_labels_to_membership_matrices(self, target_labels: torch.Tensor, num_classes: int) -> np.ndarray:
        # Legacy non-differentiable version
        targets_onehot = F.one_hot(target_labels, num_classes).cpu().numpy()
        num_samples, num_classes = targets_onehot.shape
        membership_matrices = np.zeros((num_classes, num_samples, num_samples))
        max_indices = np.argmax(targets_onehot, axis=1)
        membership_matrices[max_indices, np.arange(num_samples), np.arange(num_samples)] = 1
        return membership_matrices

    def _update_semantic_labels_with_cosine_similarity(self, train_features: torch.Tensor, semantic_labels: torch.Tensor,
                                                      all_features: torch.Tensor, train_node_mask: torch.Tensor,
                                                      cosine_weight: float, num_classes: int) -> torch.Tensor:

        train_labels = semantic_labels[train_node_mask].argmax(dim=1)

        centroids_list = []
        for class_idx in range(num_classes):
            class_mask = (train_labels == class_idx)
            if class_mask.sum() > 0:
                centroids_list.append(train_features[class_mask].mean(dim=0))
            else:
                centroids_list.append(torch.zeros(train_features.shape[1], device=train_features.device))

        class_centroids = torch.stack(centroids_list)

        normalized_features = F.normalize(all_features, p=2, dim=1)
        normalized_centroids = F.normalize(class_centroids, p=2, dim=1)
        cosine_similarities = torch.abs(normalized_features @ normalized_centroids.T)

        updated_semantic_labels = cosine_weight * semantic_labels + (1 - cosine_weight) * cosine_similarities

        return F.softmax(updated_semantic_labels, dim=1)

class AdjacencyMatrixProcessor:

    @staticmethod
    def _normalize_adj_row(adj_matrix):
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()

        if not isinstance(adj_matrix, csr_matrix):
            adj_matrix = csr_matrix(adj_matrix)

        row_sums = np.array(adj_matrix.sum(1)).flatten()
        row_sums[row_sums == 0] = 1
        row_inv = 1.0 / row_sums
        row_inv_diag = csr_matrix((row_inv, (np.arange(len(row_inv)), np.arange(len(row_inv)))),
                                  shape=(len(row_inv), len(row_inv)))
        return row_inv_diag @ adj_matrix

    @staticmethod
    def normalize_adjacency_by_rows(adjacency_matrix: torch.Tensor) -> torch.Tensor:

        adj_numpy = adjacency_matrix.cpu().numpy()
        normalized_adj = AdjacencyMatrixProcessor._normalize_adj_row(adj_numpy)
        return torch.from_numpy(normalized_adj.todense()).to(adjacency_matrix.device).float()

    @staticmethod
    def preprocess_semantic_labels_with_propagation(graph_data, corrupted_node_labels: torch.Tensor,
                                                   label_prop_alpha: float, propagation_steps: int,
                                                   device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        num_classes = graph_data.y.max().item() + 1
        num_nodes = graph_data.x.shape[0]

        semantic_labels_matrix = torch.zeros(num_nodes, num_classes, device=device)
        corrupted_onehot = F.one_hot(corrupted_node_labels, num_classes).float().to(device)
        semantic_labels_matrix[graph_data.train_mask] = corrupted_onehot[graph_data.train_mask]

        full_adjacency = to_dense_adj(graph_data.edge_index)[0].cpu()
        full_adjacency = full_adjacency.to(device)

        train_only_adjacency = full_adjacency.detach().clone()

        int_train_mask = graph_data.train_mask.cpu().numpy().astype(int).reshape(-1, 1)
        M = np.matmul(int_train_mask, int_train_mask.T).astype(bool)
        M_tensor = torch.from_numpy(M).to(device)

        train_only_adjacency = train_only_adjacency * M_tensor.float()

        train_only_adjacency_normalized = AdjacencyMatrixProcessor.normalize_adjacency_by_rows(train_only_adjacency)
        full_adjacency_normalized = AdjacencyMatrixProcessor.normalize_adjacency_by_rows(full_adjacency)

        for _ in range(propagation_steps):
            semantic_labels_matrix = ((1 - label_prop_alpha) *
                                    torch.matmul(train_only_adjacency_normalized, semantic_labels_matrix) +
                                    label_prop_alpha * semantic_labels_matrix)

        initial_predicted_labels = semantic_labels_matrix.argmax(dim=1)

        return initial_predicted_labels, semantic_labels_matrix, full_adjacency_normalized


class EnhancedGNNWrapper(nn.Module):

    def __init__(self, base_gnn_model, use_layer_normalization=False, use_residual_connections=False,
                 use_learnable_residual_projection=False, final_activation_function='relu',
                 final_feature_normalization=None):
        super().__init__()

        self.base_gnn_model = base_gnn_model
        self.use_layer_normalization = use_layer_normalization
        self.use_residual_connections = use_residual_connections
        self.use_learnable_residual_projection = use_learnable_residual_projection
        self.final_activation_function = final_activation_function
        self.final_feature_normalization = final_feature_normalization

        self._initialize_enhancement_layers()

    def _initialize_enhancement_layers(self):

        if self.final_feature_normalization == 'layer_norm':
            self.final_layer_norm = None

        if self.use_learnable_residual_projection and self.use_residual_connections:
            self.residual_projection_layers = nn.ModuleDict()

    def _create_dynamic_layers_if_needed(self, input_node_features, output_node_features):

        if self.final_feature_normalization == 'layer_norm' and self.final_layer_norm is None:
            self.final_layer_norm = nn.LayerNorm(output_node_features.size(-1))
            self.final_layer_norm = self.final_layer_norm.to(output_node_features.device)

        if (self.use_learnable_residual_projection and self.use_residual_connections and
            input_node_features.size(-1) != output_node_features.size(-1)):
            dimension_key = f"{input_node_features.size(-1)}_{output_node_features.size(-1)}"
            if dimension_key not in self.residual_projection_layers:
                projection_layer = nn.Linear(input_node_features.size(-1), output_node_features.size(-1))
                self.residual_projection_layers[dimension_key] = projection_layer.to(output_node_features.device)

    def forward(self, graph_data):

        if hasattr(graph_data, 'x'):
            original_node_features = graph_data.x.clone()
        else:
            original_node_features = graph_data.clone()

        enhanced_features = self.base_gnn_model(graph_data)

        if hasattr(graph_data, 'x'):
            self._create_dynamic_layers_if_needed(original_node_features, enhanced_features)
        else:
            self._create_dynamic_layers_if_needed(original_node_features, enhanced_features)

        # Apply residual connections
        if self.use_residual_connections:
            if self.use_learnable_residual_projection:
                dimension_key = f"{original_node_features.size(-1)}_{enhanced_features.size(-1)}"
                if dimension_key in self.residual_projection_layers:
                    enhanced_features = enhanced_features + self.residual_projection_layers[dimension_key](original_node_features)
            elif original_node_features.size(-1) == enhanced_features.size(-1):
                enhanced_features = enhanced_features + original_node_features

        # Apply normalization
        if self.final_feature_normalization == 'layer_norm' and self.final_layer_norm is not None:
            enhanced_features = self.final_layer_norm(enhanced_features)
        elif self.final_feature_normalization == 'l1':
            enhanced_features = F.normalize(enhanced_features, p=1, dim=1)
        elif self.final_feature_normalization == 'l2':
            enhanced_features = F.normalize(enhanced_features, p=2, dim=1)

        # Apply final activation
        if self.final_activation_function == 'relu':
            enhanced_features = F.relu(enhanced_features)
        elif self.final_activation_function == 'leaky_relu':
            enhanced_features = F.leaky_relu(enhanced_features)
        elif self.final_activation_function == 'elu':
            enhanced_features = F.elu(enhanced_features)
        elif self.final_activation_function == 'tanh':
            enhanced_features = torch.tanh(enhanced_features)
        elif self.final_activation_function is None:
            pass

        return enhanced_features

    def get_embeddings(self, graph_data):
        """Return hidden_channels-dim representations from the base GNN."""
        return self.base_gnn_model.get_embeddings(graph_data)

    def reset_parameters(self):

        if hasattr(self.base_gnn_model, 'reset_parameters'):
            self.base_gnn_model.reset_parameters()
        elif hasattr(self.base_gnn_model, 'initialize'):
            self.base_gnn_model.initialize()

        if hasattr(self, 'final_layer_norm') and self.final_layer_norm is not None:
            self.final_layer_norm.reset_parameters()

        for projection_layer in self.residual_projection_layers.values():
            projection_layer.reset_parameters()


@register('erase')
class ERASEMethodTrainer(BaseTrainer):
    def train(self):
        from methods.registry import get_helper
        from training.training_loop import TrainingLoop

        d = self.init_data
        self._helper = get_helper('erase')
        self._loop = TrainingLoop(self._helper, log_epoch_fn=self.log_epoch)
        return self._loop.run(
            d['backbone_model'], d['data_for_training'],
            self.config, d['device'], d,
        )

    def _get_state(self):
        """Get helper state — works both during and after training."""
        if hasattr(self, '_loop') and hasattr(self._loop, '_state'):
            return self._loop.state
        return self._loop_state

    def get_checkpoint_state(self) -> dict:
        return self._helper.get_checkpoint_state(self._get_state())

    def load_checkpoint_state(self, state):
        if not hasattr(self, '_helper'):
            from methods.registry import get_helper
            self._helper = get_helper('erase')
            d = self.init_data
            self._loop_state = self._helper.setup(
                d['backbone_model'], d['data_for_training'],
                self.config, d['device'], d,
            )
        else:
            self._loop_state = self._get_state()
        self._helper.load_checkpoint_state(self._loop_state, state)

    def profile_flops(self):
        from util.profiling import profile_model_flops
        d = self.init_data
        state = self._get_state()
        fwd = self._helper.get_inference_forward_fn(state, d['data_for_training'])
        return profile_model_flops(state['models'][0], d['data_for_training'],
                                   d['device'], forward_fn=fwd)

    def profile_training_step(self):
        from util.profiling import profile_training_step_flops
        d = self.init_data
        state = self._get_state()
        step_fn = self._helper.get_training_step_fn(state, d['data_for_training'])
        return profile_training_step_flops(state['models'][0], d['device'], step_fn)

    def evaluate(self):
        d = self.init_data
        state = self._get_state()
        data = d['data_for_training']
        clean_labels = getattr(data, 'y_original', data.y)

        def get_predictions():
            return self._helper.get_predictions(state, data)

        def get_embeddings():
            return self._helper.get_embeddings(state, data)

        return evaluate_model(
            get_predictions, get_embeddings, clean_labels,
            data.train_mask, data.val_mask, data.test_mask,
            data.edge_index, d['device'],
        )
