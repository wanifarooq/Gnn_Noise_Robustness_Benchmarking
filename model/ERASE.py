import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_dense_adj
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple, Dict, Any
import copy
from scipy.sparse import csr_matrix

from model.evaluation import OversmoothingMetrics


def normalize_adj_row(adj_matrix):

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
        log_determinant = torch.logdet(identity_matrix + self.discrimination_weight * scalar * feature_matrix @ feature_matrix.T)
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
        compression_loss = 0.
        
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
        compression_loss = 0.
        
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
        compression_loss = 0.
        for class_idx in range(num_classes):
            membership_trace = torch.trace(membership_matrices[class_idx]) + 1e-8
            scalar = num_features / (membership_trace * self.eps)
            log_det = torch.logdet(identity_matrix + scalar * feature_matrix @ membership_matrices[class_idx] @ feature_matrix.T)
            compression_loss += membership_trace / (2 * num_samples) * log_det
        return compression_loss

    def forward(self, node_features: torch.Tensor, graph_data, adjacency_matrix: torch.Tensor, 
                semantic_labels: torch.Tensor, label_prop_alpha: float, cosine_weight_beta: float, 
                propagation_steps: int, train_node_mask: torch.Tensor):
        
        num_classes = graph_data.y.max().item() + 1
        train_features = node_features[train_node_mask]
        feature_matrix = node_features.T

        # Update semantic labels with cosine similarity
        semantic_labels = self._update_semantic_labels_with_cosine_similarity(
            train_features, semantic_labels, node_features, train_node_mask, cosine_weight_beta, num_classes
        )

        # Apply label propagation
        for _ in range(propagation_steps):
            semantic_labels = (1 - label_prop_alpha) * (adjacency_matrix @ semantic_labels) + label_prop_alpha * semantic_labels

        predicted_labels = torch.argmax(semantic_labels, dim=1)

        # Compute discrimination loss
        discrimination_loss_empirical = self.compute_discrimination_loss_empirical(feature_matrix)

        # Standard variant with membership matrices
        membership_matrices = self._convert_labels_to_membership_matrices(predicted_labels.cpu(), num_classes)
        membership_matrices = torch.tensor(membership_matrices, dtype=torch.float32, device=node_features.device)

        compression_loss_empirical = self.compute_compression_loss_empirical(feature_matrix, membership_matrices)
        compression_loss_theoretical = self.compute_compression_loss_theoretical(feature_matrix, membership_matrices)
        discrimination_loss_theoretical = self.compute_discrimination_loss_theoretical(feature_matrix)

        total_loss_empirical = -self.discrimination_weight * discrimination_loss_empirical + self.compression_weight * compression_loss_empirical

        return (total_loss_empirical,
                [discrimination_loss_empirical.item(), compression_loss_empirical.item()],
                [discrimination_loss_theoretical.item(), compression_loss_theoretical.item()],
                predicted_labels)

    def _convert_labels_to_membership_matrices(self, target_labels: torch.Tensor, num_classes: int) -> np.ndarray:
        targets_onehot = F.one_hot(target_labels, num_classes).numpy()
        num_samples, num_classes = targets_onehot.shape
        membership_matrices = np.zeros((num_classes, num_samples, num_samples))
        max_indices = np.argmax(targets_onehot, axis=1)
        membership_matrices[max_indices, np.arange(num_samples), np.arange(num_samples)] = 1
        return membership_matrices
    
    def _update_semantic_labels_with_cosine_similarity(self, train_features: torch.Tensor, semantic_labels: torch.Tensor,
                                                      all_features: torch.Tensor, train_node_mask: torch.Tensor, 
                                                      cosine_weight: float, num_classes: int) -> torch.Tensor:

        train_labels = semantic_labels[train_node_mask].argmax(dim=1)
        
        class_centroids = []
        for class_idx in range(num_classes):
            class_mask = (train_labels == class_idx)
            if class_mask.sum() > 0:
                class_centroids.append(train_features[class_mask].mean(dim=0))
            else:
                class_centroids.append(torch.zeros(train_features.shape[1], device=train_features.device))
        
        class_centroids = torch.stack(class_centroids)
        
        normalized_features = F.normalize(all_features, p=2, dim=1)
        normalized_centroids = F.normalize(class_centroids, p=2, dim=1)
        cosine_similarities = torch.abs(normalized_features @ normalized_centroids.T)
        
        updated_semantic_labels = cosine_weight * semantic_labels + (1 - cosine_weight) * cosine_similarities
        
        return F.softmax(updated_semantic_labels, dim=1)

class AdjacencyMatrixProcessor:

    @staticmethod
    def normalize_adjacency_by_rows(adjacency_matrix: torch.Tensor) -> torch.Tensor:

        adj_numpy = adjacency_matrix.cpu().numpy()
        normalized_adj = normalize_adj_row(adj_numpy)
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


class LinearProbeEvaluator:
    @staticmethod
    def evaluate_with_linear_probe(learned_features: torch.Tensor, graph_data, 
                                predicted_labels_all_nodes: torch.Tensor, 
                                clean_node_labels: torch.Tensor,
                                random_state: int = 42) -> Tuple[float, float, float]:

        features_normalized = normalize(learned_features.detach().cpu().numpy(), norm='l2')
        clean_labels_np = clean_node_labels.cpu().numpy()
        
        train_features = features_normalized[graph_data.train_mask.cpu().numpy()]
        val_features = features_normalized[graph_data.val_mask.cpu().numpy()]
        test_features = features_normalized[graph_data.test_mask.cpu().numpy()]
        
        train_noisy_labels_np = predicted_labels_all_nodes[graph_data.train_mask].cpu().numpy()
        
        clf = LogisticRegression(
            solver='lbfgs', 
            multi_class='auto', 
            max_iter=1000, 
            random_state=random_state
        ).fit(train_features, train_noisy_labels_np.ravel())
        
        def accuracy_helper(classifier, features, labels):
            y_pred = classifier.predict(features)
            return accuracy_score(labels, y_pred)
        
        train_acc = accuracy_helper(clf, train_features, clean_labels_np[graph_data.train_mask.cpu().numpy()])
        val_acc = accuracy_helper(clf, val_features, clean_labels_np[graph_data.val_mask.cpu().numpy()])
        test_acc = accuracy_helper(clf, test_features, clean_labels_np[graph_data.test_mask.cpu().numpy()])
        
        return train_acc, val_acc, test_acc

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
    
    def reset_parameters(self):

        if hasattr(self.base_gnn_model, 'reset_parameters'):
            self.base_gnn_model.reset_parameters()
        elif hasattr(self.base_gnn_model, 'initialize'):
            self.base_gnn_model.initialize()
            
        if hasattr(self, 'final_layer_norm') and self.final_layer_norm is not None:
            self.final_layer_norm.reset_parameters()
            
        for projection_layer in self.residual_projection_layers.values():
            projection_layer.reset_parameters()


class ERASETrainer:

    def __init__(self, training_config: Dict[str, Any], computation_device: torch.device, 
                 num_node_classes: int, model_creation_function):
        self.training_config = training_config
        self.computation_device = computation_device
        self.num_node_classes = num_node_classes
        self.model_creation_function = model_creation_function
        self.oversmoothing_metrics_calculator = OversmoothingMetrics(device=computation_device)
        
    def train_erase_model(self, graph_data, enable_debug_output=False):
        # Create enhanced model
        enhanced_model = self._create_enhanced_gnn_model()
        
        noisy_node_labels = graph_data.y

        initial_predicted_labels, semantic_labels_matrix, adjacency_matrix = (
            AdjacencyMatrixProcessor.preprocess_semantic_labels_with_propagation(
                graph_data, noisy_node_labels, 
                self.training_config.get('alpha', 0.6), 
                self.training_config.get('T', 5), 
                str(self.computation_device)
            )
        )

        # Create loss function
        mcr2_loss_function = MaximalCodingRateReductionLoss(
            compression_weight=self.training_config.get('gam1', 1.0),
            discrimination_weight=self.training_config.get('gam2', 2.0), 
            eps=self.training_config.get('eps', 0.05)
        )
        
        model_optimizer = torch.optim.Adam(
            enhanced_model.parameters(),
            lr=self.training_config.get('lr', 0.001),
            weight_decay=self.training_config.get('weight_decay', 0.0005),
            amsgrad=True
        )
        
        training_results = self._execute_training_loop(
            enhanced_model, graph_data, model_optimizer, mcr2_loss_function, 
            adjacency_matrix, semantic_labels_matrix, initial_predicted_labels, enable_debug_output
        )
        
        return training_results

    def _create_enhanced_gnn_model(self):
        #Create enhanced GNN model
        enhancement_configuration = {
            'use_layer_normalization': self.training_config.get('use_layer_norm', False),
            'use_residual_connections': self.training_config.get('use_residual', False),
            'use_learnable_residual_projection': self.training_config.get('use_residual_linear', False),
            'final_activation_function': 'relu',
            'final_feature_normalization': 'l1'
        }
        
        base_gnn_model = self.model_creation_function(
            model_name=self.training_config.get('erase_gnn_type', 'gcn').lower(),
            in_channels=self.training_config.get('in_channels'),
            hidden_channels=self.training_config.get('hidden_channels', 128),
            out_channels=self.training_config.get('n_embedding', 512),
            n_layers=self.training_config.get('n_layers', 2),
            dropout=self.training_config.get('dropout', 0.5),
            self_loop=self.training_config.get('self_loop', True),
            mlp_layers=self.training_config.get('mlp_layers', 2),
            train_eps=self.training_config.get('train_eps', True),
            heads=self.training_config.get('n_heads', 8)
        )
        
        if not any(enhancement_configuration.values()):
            return base_gnn_model.to(self.computation_device)
        
        enhanced_model = EnhancedGNNWrapper(
            base_gnn_model=base_gnn_model,
            **enhancement_configuration
        )
        return enhanced_model.to(self.computation_device)

    def _execute_training_loop(self, model, graph_data, optimizer, loss_function, 
                                  adjacency_matrix, semantic_labels_matrix, predicted_labels, debug_mode):

            best_validation_accuracy = 0
            best_validation_loss = float('inf')
            best_training_accuracy = 0
            patience_counter = 0
            max_epochs = self.training_config.get('total_epochs', 200)
            patience_limit = self.training_config.get('patience', 50)

            best_oversmoothing_metrics = None
            
            for current_epoch in range(max_epochs):
 
                epoch_training_results = self._execute_single_training_epoch(
                    model, graph_data, optimizer, loss_function, adjacency_matrix, 
                    semantic_labels_matrix, predicted_labels
                )
                epoch_loss, loss_components, predicted_labels = epoch_training_results
                
                train_accuracy, validation_accuracy, validation_loss = self._evaluate_training_and_validation_performance(
                    model, graph_data, predicted_labels
                )
                train_f1, validation_f1 = self._compute_f1_scores_for_splits(model, graph_data, predicted_labels)

                should_compute_oversmoothing = (current_epoch + 1) % 20 == 0
                if should_compute_oversmoothing:
                    oversmoothing_metrics_by_split = self._compute_oversmoothing_metrics_for_all_splits(model, graph_data)
                    train_oversmoothing = oversmoothing_metrics_by_split.get('train', {})
                    validation_oversmoothing = oversmoothing_metrics_by_split.get('val', {})
                else:
                    train_oversmoothing = {}
                    validation_oversmoothing = {}
                
                if debug_mode:
                    self._print_epoch_debug_information(current_epoch, train_accuracy, validation_accuracy, 
                                                      train_f1, validation_f1, train_oversmoothing, validation_oversmoothing,
                                                      should_print_oversmoothing=should_compute_oversmoothing)

                # Early stopping
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_training_accuracy = train_accuracy
                    best_validation_accuracy = validation_accuracy
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                    if should_compute_oversmoothing:
                        best_oversmoothing_metrics = oversmoothing_metrics_by_split
                else:
                    patience_counter += 1

                if patience_counter >= patience_limit:
                    if debug_mode:
                        print(f'Early stopping triggered at epoch {current_epoch}')
                    break

            model.load_state_dict(best_model_state)
            final_test_results = self._evaluate_final_test_performance(model, graph_data, predicted_labels)
            final_oversmoothing_metrics = self._compute_oversmoothing_metrics_for_all_splits(model, graph_data)

            if debug_mode:
                self._print_final_results(final_test_results, final_oversmoothing_metrics)

            return {
                'accuracy': final_test_results[0],
                'f1': final_test_results[1],
                'precision': final_test_results[2],
                'recall': final_test_results[3],
                'oversmoothing': final_oversmoothing_metrics.get('test', {})
            }

    def _execute_single_training_epoch(self, model, graph_data, optimizer, loss_function, 
                                     adjacency_matrix, semantic_labels_matrix, predicted_labels):

        model.train()
        optimizer.zero_grad()
        
        node_features = model(graph_data)
        
        loss_computation_result = loss_function(
            node_features, graph_data, adjacency_matrix, semantic_labels_matrix, 
            self.training_config.get('alpha', 0.6), 
            self.training_config.get('beta', 0.6), 
            self.training_config.get('T', 5), 
            graph_data.train_mask
        )
        
        if len(loss_computation_result) == 3:
            total_loss, loss_components, updated_predicted_labels = loss_computation_result
        else:
            total_loss, loss_components, updated_predicted_labels = (
                loss_computation_result[0], loss_computation_result[1], loss_computation_result[3]
            )
        
        total_loss.backward()
        optimizer.step()
        return total_loss.item(), loss_components, updated_predicted_labels

    @torch.no_grad()
    def _evaluate_training_and_validation_performance(self, model, graph_data, predicted_labels):

        model.eval()
        learned_node_features = model(graph_data)

        train_accuracy, validation_accuracy, _ = LinearProbeEvaluator.evaluate_with_linear_probe(
            learned_node_features, 
            graph_data, 
            predicted_labels, 
            graph_data.y,
            random_state=self.training_config.get('seed', 42)
        )

        validation_loss = self._compute_cross_entropy_loss_for_split(model, graph_data, graph_data.val_mask)
        return train_accuracy, validation_accuracy, validation_loss

    @torch.no_grad()
    def _compute_f1_scores_for_splits(self, model, graph_data, predicted_labels):

        model.eval()
        learned_node_features = model(graph_data)

        normalized_features = normalize(learned_node_features.detach().cpu().numpy(), norm='l2')
        
        train_features = normalized_features[graph_data.train_mask.cpu()]
        validation_features = normalized_features[graph_data.val_mask.cpu()]
        
        train_noisy_labels = predicted_labels[graph_data.train_mask].cpu().numpy()
        true_labels = graph_data.y.cpu().numpy()
        
        linear_classifier = LogisticRegression(
            solver='lbfgs', 
            multi_class='auto', 
            max_iter=1000, 
            random_state=self.training_config.get('seed', 42)
        )
        linear_classifier.fit(train_features, train_noisy_labels)
        
        train_predictions = linear_classifier.predict(train_features)
        validation_predictions = linear_classifier.predict(validation_features)
        
        train_f1 = f1_score(
            true_labels[graph_data.train_mask.cpu()], 
            train_predictions, 
            average='weighted', 
            zero_division=0
        )
        validation_f1 = f1_score(
            true_labels[graph_data.val_mask.cpu()], 
            validation_predictions, 
            average='weighted', 
            zero_division=0
        )
        
        return train_f1, validation_f1

    @torch.no_grad()
    def _evaluate_final_test_performance(self, model, graph_data, predicted_labels):

        model.eval()
        learned_node_features = model(graph_data)

        _, _, test_accuracy = LinearProbeEvaluator.evaluate_with_linear_probe(
            learned_node_features, 
            graph_data, 
            predicted_labels, 
            graph_data.y,
            random_state=self.training_config.get('seed', 42)
        )

        normalized_features = normalize(learned_node_features.detach().cpu().numpy(), norm='l2')
        
        train_features = normalized_features[graph_data.train_mask.cpu()]
        test_features = normalized_features[graph_data.test_mask.cpu()]
        
        train_noisy_labels = predicted_labels[graph_data.train_mask].cpu().numpy()
        true_test_labels = graph_data.y.cpu().numpy()
        
        linear_classifier = LogisticRegression(
            solver='lbfgs', 
            multi_class='auto', 
            max_iter=1000, 
            random_state=self.training_config.get('seed', 42)
        )
        linear_classifier.fit(train_features, train_noisy_labels)
        
        test_predictions = linear_classifier.predict(test_features)
        
        test_f1 = f1_score(true_test_labels[graph_data.test_mask.cpu()], test_predictions, 
                        average='macro', zero_division=0)
        test_precision = precision_score(true_test_labels[graph_data.test_mask.cpu()], test_predictions, 
                                    average='macro', zero_division=0)
        test_recall = recall_score(true_test_labels[graph_data.test_mask.cpu()], test_predictions, 
                                average='macro', zero_division=0)
        
        test_loss = self._compute_cross_entropy_loss_for_split(model, graph_data, graph_data.test_mask)
        
        return test_accuracy, test_f1, test_precision, test_recall, test_loss.item()

    @torch.no_grad()
    def _compute_cross_entropy_loss_for_split(self, model, graph_data, split_mask):

        learned_features = model(graph_data)
        split_true_labels = graph_data.y[split_mask]
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        split_loss = cross_entropy_loss(learned_features[split_mask], split_true_labels)
        return split_loss

    @torch.no_grad()
    def _compute_oversmoothing_metrics_for_all_splits(self, model, graph_data):

        model.eval()
        learned_node_features = model(graph_data)
        
        def extract_subgraph_for_split(node_mask):

            split_nodes = torch.where(node_mask)[0]
            node_index_mapping = {node.item(): idx for idx, node in enumerate(split_nodes)}
            
            edge_mask = torch.isin(graph_data.edge_index[0], split_nodes) & torch.isin(graph_data.edge_index[1], split_nodes)
            filtered_edge_index = graph_data.edge_index[:, edge_mask]
            
            if filtered_edge_index.size(1) > 0:
                remapped_edge_index = torch.zeros_like(filtered_edge_index)
                for edge_idx in range(filtered_edge_index.size(1)):
                    remapped_edge_index[0, edge_idx] = node_index_mapping[filtered_edge_index[0, edge_idx].item()]
                    remapped_edge_index[1, edge_idx] = node_index_mapping[filtered_edge_index[1, edge_idx].item()]
                return learned_node_features[node_mask], remapped_edge_index
            else:
                return learned_node_features[node_mask], torch.empty((2, 0), dtype=torch.long, device=graph_data.edge_index.device)

        oversmoothing_metrics_by_split = {}
        for split_name, split_mask in [('train', graph_data.train_mask), ('val', graph_data.val_mask), ('test', graph_data.test_mask)]:
            split_features, split_edge_index = extract_subgraph_for_split(split_mask)
            
            if split_edge_index.size(1) > 0:
                subgraphs_for_evaluation = [{
                    'X': split_features,
                    'edge_index': split_edge_index,
                    'edge_weight': None
                }]
                
                split_oversmoothing_metrics = self.oversmoothing_metrics_calculator.compute_all_metrics(
                    X=split_features,
                    edge_index=split_edge_index,
                    edge_weight=None,
                    graphs_in_class=subgraphs_for_evaluation
                )
                oversmoothing_metrics_by_split[split_name] = split_oversmoothing_metrics
            else:

                oversmoothing_metrics_by_split[split_name] = {
                    'EDir': 0.0,
                    'NumRank': float(min(split_features.shape)),
                    'Erank': float(min(split_features.shape)),
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0
                }
        
        return oversmoothing_metrics_by_split

    def _print_epoch_debug_information(self, epoch_number, train_acc, val_acc, train_f1, val_f1, 
                                     train_oversmoothing, val_oversmoothing, should_print_oversmoothing=True):

        print(f"Epoch {epoch_number+1:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
              f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        
        if should_print_oversmoothing and train_oversmoothing and val_oversmoothing:
            train_edir = train_oversmoothing.get('EDir', 0.0)
            train_edir_trad = train_oversmoothing.get('EDir_traditional', 0.0)
            train_eproj = train_oversmoothing.get('EProj', 0.0)
            train_mad = train_oversmoothing.get('MAD', 0.0)
            train_num_rank = train_oversmoothing.get('NumRank', 0.0)
            train_eff_rank = train_oversmoothing.get('Erank', 0.0)

            val_edir = val_oversmoothing.get('EDir', 0.0)
            val_edir_trad = val_oversmoothing.get('EDir_traditional', 0.0)
            val_eproj = val_oversmoothing.get('EProj', 0.0)
            val_mad = val_oversmoothing.get('MAD', 0.0)
            val_num_rank = val_oversmoothing.get('NumRank', 0.0)
            val_eff_rank = val_oversmoothing.get('Erank', 0.0)

            print(f"[Oversmoothing Metrics] Train EDir: {train_edir:.4f}, Val EDir: {val_edir:.4f} | "
                  f"Train EDir_trad: {train_edir_trad:.4f}, Val EDir_trad: {val_edir_trad:.4f}")
            print(f"[Oversmoothing Metrics] Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                  f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f}")
            print(f"[Oversmoothing Metrics] Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                  f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")
        elif should_print_oversmoothing:
            print("[Oversmoothing Metrics] Computed every 20 epochs - skipping this epoch")

    def _print_final_results(self, test_results, final_oversmoothing_metrics):

        test_acc, test_f1, test_precision, test_recall, test_loss = test_results
        
        final_train_oversmoothing = final_oversmoothing_metrics.get('train', {})
        final_val_oversmoothing = final_oversmoothing_metrics.get('val', {})
        final_test_oversmoothing = final_oversmoothing_metrics.get('test', {})

        print(f"\nERASE Training completed!")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print("Final Oversmoothing Metrics:")

        if final_train_oversmoothing:
            print(f"Train: EDir: {final_train_oversmoothing['EDir']:.4f}, EDir_traditional: {final_train_oversmoothing['EDir_traditional']:.4f}, "
                f"EProj: {final_train_oversmoothing['EProj']:.4f}, MAD: {final_train_oversmoothing['MAD']:.4f}, "
                f"NumRank: {final_train_oversmoothing['NumRank']:.4f}, Erank: {final_train_oversmoothing['Erank']:.4f}")

        if final_val_oversmoothing:
            print(f"Val: EDir: {final_val_oversmoothing['EDir']:.4f}, EDir_traditional: {final_val_oversmoothing['EDir_traditional']:.4f}, "
                f"EProj: {final_val_oversmoothing['EProj']:.4f}, MAD: {final_val_oversmoothing['MAD']:.4f}, "
                f"NumRank: {final_val_oversmoothing['NumRank']:.4f}, Erank: {final_val_oversmoothing['Erank']:.4f}")

        if final_test_oversmoothing:
            print(f"Test: EDir: {final_test_oversmoothing['EDir']:.4f}, EDir_traditional: {final_test_oversmoothing['EDir_traditional']:.4f}, "
                f"EProj: {final_test_oversmoothing['EProj']:.4f}, MAD: {final_test_oversmoothing['MAD']:.4f}, "
                f"NumRank: {final_test_oversmoothing['NumRank']:.4f}, Erank: {final_test_oversmoothing['Erank']:.4f}")


def create_enhanced_gnn_model(model_creation_function, gnn_model_name, enhancement_configuration=None, **model_creation_kwargs):

    default_enhancement_config = {
        'use_layer_normalization': False,
        'use_residual_connections': False,
        'use_learnable_residual_projection': False,
        'final_activation_function': 'relu',
        'final_feature_normalization': None  # None, 'layer_norm', 'l1', 'l2'
    }
    
    if enhancement_configuration:
        default_enhancement_config.update(enhancement_configuration)
    
    base_gnn_model = model_creation_function(model_name=gnn_model_name, **model_creation_kwargs)
    
    if not any(default_enhancement_config.values()):
        return base_gnn_model
    
    return EnhancedGNNWrapper(
        base_gnn_model=base_gnn_model,
        **default_enhancement_config
    )


def analyze_oversmoothing_evolution_during_training(erase_trainer, trained_model, graph_data, 
                                                   epochs_to_analyze=None):

    if epochs_to_analyze is None:
        epochs_to_analyze = list(range(10, 201, 10))
    
    oversmoothing_evolution = {
        epoch: erase_trainer._compute_oversmoothing_metrics_for_all_splits(trained_model, graph_data) 
        for epoch in epochs_to_analyze
    }
    
    return oversmoothing_evolution


