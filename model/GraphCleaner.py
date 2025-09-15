import os
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, 
    matthews_corrcoef, roc_auc_score
)
from torch_geometric.utils import to_scipy_sparse_matrix
from cleanlab.count import estimate_latent, compute_confident_joint

from model.evaluation import OversmoothingMetrics


class GraphCleanerNoiseDetector:

    def __init__(self, configuration_params, computation_device):

        self.config = configuration_params
        self.device = computation_device
        
        self.neighborhood_depth = configuration_params.get('graphcleaner_params', {}).get('k', 3)
        self.negative_sample_ratio = configuration_params.get('graphcleaner_params', {}).get('sample_rate', 0.5)
        self.classifier_max_iterations = configuration_params.get('graphcleaner_params', {}).get('max_iter_classifier', 3000)
        self.validation_split_type = configuration_params.get('graphcleaner_params', {}).get('held_split', 'valid')
        
        self.learning_rate = configuration_params.get('training', {}).get('lr', 0.001)
        self.weight_decay_factor = float(configuration_params.get('training', {}).get('weight_decay', 5e-4))
        self.training_epochs = configuration_params.get('training', {}).get('epochs', 200)
        self.early_stopping_patience = configuration_params.get('training', {}).get('patience', 10)

        self.oversmoothing_calculator = OversmoothingMetrics(device=computation_device)
        

    def _calculate_oversmoothing_metrics_for_subset(self, node_embeddings, graph_edges, node_subset_mask, node_labels=None):
        
        try:
            subset_node_indices = torch.where(node_subset_mask)[0]
            subset_embeddings = node_embeddings[subset_node_indices]
            
            subset_indices_set = set(subset_node_indices.cpu().numpy())
            
            # Filter edges
            edge_within_subset_mask = torch.tensor([
                source_node.item() in subset_indices_set and target_node.item() in subset_indices_set
                for source_node, target_node in graph_edges.t()
            ], device=graph_edges.device)
            
            if not edge_within_subset_mask.any():
                return {
                    'NumRank': float(min(subset_embeddings.shape)),
                    'Erank': float(min(subset_embeddings.shape)),
                    'EDir': 0.0,
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0
                }
            
            subset_edges = graph_edges[:, edge_within_subset_mask]
            index_mapping = {original_idx.item(): new_idx for new_idx, original_idx in enumerate(subset_node_indices)}
            
            remapped_edge_indices = torch.stack([
                torch.tensor([index_mapping[src.item()] for src in subset_edges[0]], device=graph_edges.device),
                torch.tensor([index_mapping[tgt.item()] for tgt in subset_edges[1]], device=graph_edges.device)
            ])
            
            subset_graph_data = [{
                'X': subset_embeddings,
                'edge_index': remapped_edge_indices,
                'edge_weight': None
            }]
            
            return self.oversmoothing_calculator.compute_all_metrics(
                X=subset_embeddings,
                edge_index=remapped_edge_indices,
                graphs_in_class=subset_graph_data
            )
            
        except Exception as error:
            print(f"Warning: Could not compute oversmoothing metrics for subset: {error}")
            return None
        
    def _convert_logits_to_probabilities(self, prediction_logits):

        if isinstance(prediction_logits, np.ndarray):

            exp_values = np.exp(prediction_logits - np.max(prediction_logits, axis=1, keepdims=True))
            return exp_values / exp_values.sum(axis=1, keepdims=True)
        elif isinstance(prediction_logits, torch.Tensor):
            return F.softmax(prediction_logits, dim=1).cpu().numpy()
        else:
            return F.softmax(torch.tensor(prediction_logits), dim=1).cpu().numpy()

    def _calculate_dirichlet_energy(self, graph_data, model_predictions, node_subset_mask):
 
        if model_predictions.dim() > 1 and model_predictions.size(1) > 1:
            prediction_probabilities = F.softmax(model_predictions, dim=1)
        else:
            prediction_probabilities = model_predictions
            
        graph_edge_index = graph_data.edge_index
        total_nodes = graph_data.num_nodes

        accumulated_energy = 0.0
        edge_count = 0

        for edge_idx in range(graph_edge_index.size(1)):
            source_node, target_node = graph_edge_index[0, edge_idx].item(), graph_edge_index[1, edge_idx].item()
            
            if node_subset_mask[source_node] or node_subset_mask[target_node]:
                probability_difference = torch.norm(prediction_probabilities[source_node] - prediction_probabilities[target_node], p=2) ** 2
                accumulated_energy += probability_difference.item()
                edge_count += 1
        
        return accumulated_energy / edge_count if edge_count > 0 else 0.0

    def _calculate_training_metrics(self, graph_data, neural_network_model, model_predictions):

        metrics_dict = {}

        training_predictions = model_predictions[graph_data.train_mask].argmax(dim=-1).cpu()
        training_ground_truth = graph_data.y[graph_data.train_mask].cpu()
        metrics_dict['train_loss'] = F.cross_entropy(model_predictions[graph_data.train_mask], graph_data.y_noisy[graph_data.train_mask]).item()
        metrics_dict['train_acc'] = accuracy_score(training_ground_truth, training_predictions)
        metrics_dict['train_f1'] = f1_score(training_ground_truth, training_predictions, average='micro')

        validation_predictions = model_predictions[graph_data.val_mask].argmax(dim=-1).cpu()
        validation_ground_truth = graph_data.y[graph_data.val_mask].cpu()
        metrics_dict['val_loss'] = F.cross_entropy(model_predictions[graph_data.val_mask], graph_data.y_noisy[graph_data.val_mask]).item()
        metrics_dict['val_acc'] = accuracy_score(validation_ground_truth, validation_predictions)
        metrics_dict['val_f1'] = f1_score(validation_ground_truth, validation_predictions, average='micro')
        
        return metrics_dict

    def _train_base_neural_network(self, graph_data, neural_network_model, total_training_epochs=200):

        print("Training base GNN for GraphCleaner")
        training_start_time = time.time()
        
        model_optimizer = torch.optim.Adam(
            neural_network_model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay_factor
        )
        
        prediction_history_list = []
        optimal_predictions = []
        optimal_model_state = None
        
        graph_data = graph_data.to(self.device)
        neural_network_model = neural_network_model.to(self.device)
        
        best_validation_loss = float('inf')
        optimal_model_state = None
        best_f1_score = 0
        patience_counter = 0

        for current_epoch in range(total_training_epochs):

            neural_network_model.train()
            model_optimizer.zero_grad()

            model_output = neural_network_model(graph_data)
            training_loss = F.cross_entropy(model_output[graph_data.train_mask], graph_data.y_noisy[graph_data.train_mask])

            training_loss.backward()
            model_optimizer.step()

            neural_network_model.eval()
            with torch.no_grad():
                model_output = neural_network_model(graph_data)
                current_metrics = self._calculate_training_metrics(graph_data, neural_network_model, model_output)
                
                if (current_epoch + 1) % 20 == 0:
                    train_oversmoothing_metrics = self._calculate_oversmoothing_metrics_for_subset(
                        model_output, graph_data.edge_index, graph_data.train_mask, graph_data.y)
                    val_oversmoothing_metrics = self._calculate_oversmoothing_metrics_for_subset(
                        model_output, graph_data.edge_index, graph_data.val_mask, graph_data.y)
                    
                    train_dirichlet_energy = train_oversmoothing_metrics.get('EDir', 0.0) if train_oversmoothing_metrics else 0.0
                    train_dirichlet_traditional = train_oversmoothing_metrics.get('EDir_traditional', 0.0) if train_oversmoothing_metrics else 0.0
                    train_effective_projection = train_oversmoothing_metrics.get('EProj', 0.0) if train_oversmoothing_metrics else 0.0
                    train_mean_absolute_deviation = train_oversmoothing_metrics.get('MAD', 0.0) if train_oversmoothing_metrics else 0.0
                    train_numerical_rank = train_oversmoothing_metrics.get('NumRank', 0.0) if train_oversmoothing_metrics else 0.0
                    train_effective_rank = train_oversmoothing_metrics.get('Erank', 0.0) if train_oversmoothing_metrics else 0.0
                    
                    val_dirichlet_energy = val_oversmoothing_metrics.get('EDir', 0.0) if val_oversmoothing_metrics else 0.0
                    val_dirichlet_traditional = val_oversmoothing_metrics.get('EDir_traditional', 0.0) if val_oversmoothing_metrics else 0.0
                    val_effective_projection = val_oversmoothing_metrics.get('EProj', 0.0) if val_oversmoothing_metrics else 0.0
                    val_mean_absolute_deviation = val_oversmoothing_metrics.get('MAD', 0.0) if val_oversmoothing_metrics else 0.0
                    val_numerical_rank = val_oversmoothing_metrics.get('NumRank', 0.0) if val_oversmoothing_metrics else 0.0
                    val_effective_rank = val_oversmoothing_metrics.get('Erank', 0.0) if val_oversmoothing_metrics else 0.0
                    
                    print(f"Epoch {current_epoch:03d} | Train Loss: {current_metrics['train_loss']:.4f}, Val Loss: {current_metrics['val_loss']:.4f} | "
                        f"Train Acc: {current_metrics['train_acc']:.4f}, Val Acc: {current_metrics['val_acc']:.4f} | "
                        f"Train F1: {current_metrics['train_f1']:.4f}, Val F1: {current_metrics['val_f1']:.4f}")
                    print(f"Train DE: {train_dirichlet_energy:.4f}, Val DE: {val_dirichlet_energy:.4f} | "
                        f"Train DE_trad: {train_dirichlet_traditional:.4f}, Val DE_trad: {val_dirichlet_traditional:.4f} | "
                        f"Train EProj: {train_effective_projection:.4f}, Val EProj: {val_effective_projection:.4f} | "
                        f"Train MAD: {train_mean_absolute_deviation:.4f}, Val MAD: {val_mean_absolute_deviation:.4f} | "
                        f"Train NumRank: {train_numerical_rank:.4f}, Val NumRank: {val_numerical_rank:.4f} | "
                        f"Train Erank: {train_effective_rank:.4f}, Val Erank: {val_effective_rank:.4f}")
                else:

                    print(f"Epoch {current_epoch:03d} | Train Loss: {current_metrics['train_loss']:.4f}, Val Loss: {current_metrics['val_loss']:.4f} | "
                        f"Train Acc: {current_metrics['train_acc']:.4f}, Val Acc: {current_metrics['val_acc']:.4f} | "
                        f"Train F1: {current_metrics['train_f1']:.4f}, Val F1: {current_metrics['val_f1']:.4f}")

                if current_metrics['val_loss'] < best_validation_loss:
                    best_validation_loss = current_metrics['val_loss']
                    optimal_model_state = copy.deepcopy(neural_network_model.state_dict())
                    optimal_predictions = model_output.cpu().detach().numpy()
                    best_f1_score = current_metrics['val_f1']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {current_epoch+1}")
                        break

            if (current_epoch + 1) % 20 == 0 or current_epoch == current_epoch:
                prediction_history_list.append(model_output.cpu().detach().numpy())

        if len(prediction_history_list) == 0:
            neural_network_model.eval()
            with torch.no_grad():
                final_output = neural_network_model(graph_data)
                prediction_history_list.append(final_output.cpu().detach().numpy())

        if optimal_model_state is not None:
            neural_network_model.load_state_dict(optimal_model_state)

        neural_network_model.eval()
        with torch.no_grad():
            final_model_output = neural_network_model(graph_data)
            final_metrics_dict = {}
            
            test_predictions = final_model_output[graph_data.test_mask].argmax(dim=-1).cpu()
            test_ground_truth = graph_data.y[graph_data.test_mask].cpu()
            final_metrics_dict['test_loss'] = F.cross_entropy(final_model_output[graph_data.test_mask], graph_data.y_noisy[graph_data.test_mask]).item()
            final_metrics_dict['test_acc'] = accuracy_score(test_ground_truth, test_predictions)
            final_metrics_dict['test_f1'] = f1_score(test_ground_truth, test_predictions, average='micro')
            
            final_train_oversmoothing = self._calculate_oversmoothing_metrics_for_subset(
                final_model_output, graph_data.edge_index, graph_data.train_mask, graph_data.y)
            final_val_oversmoothing = self._calculate_oversmoothing_metrics_for_subset(
                final_model_output, graph_data.edge_index, graph_data.val_mask, graph_data.y)
            final_test_oversmoothing = self._calculate_oversmoothing_metrics_for_subset(
                final_model_output, graph_data.edge_index, graph_data.test_mask, graph_data.y)

        total_training_time = time.time() - training_start_time

        print(f"\nTraining completed in {total_training_time:.2f}s")
        print(f"Test Loss: {final_metrics_dict['test_loss']:.4f} | Test Acc: {final_metrics_dict['test_acc']:.4f} | Test F1: {final_metrics_dict['test_f1']:.4f}")
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

        return np.array(prediction_history_list), optimal_predictions, neural_network_model
    
    def _estimate_noise_transition_matrix(self, corrupted_labels, prediction_probabilities, total_classes):

        print(f"Using {len(corrupted_labels)} samples for confident joint estimation.")
        
        # Compute confident joint matrix
        confident_joint_matrix = compute_confident_joint(corrupted_labels, prediction_probabilities)
        
        # Estimate latent class distribution and noise matrices
        estimated_class_priors, estimated_noise_matrix, inverse_noise_matrix = estimate_latent(
            confident_joint_matrix, corrupted_labels)
        
        print(f"Estimated prior probabilities: {estimated_class_priors}")
        print("Estimated noise transition matrix:")
        print(estimated_noise_matrix)
        
        return estimated_class_priors, estimated_noise_matrix, inverse_noise_matrix
    
    def _generate_negative_samples(self, original_graph_data, noise_transition_matrix, num_classes, sampling_rate=None):
        
        if sampling_rate is None:
            sampling_rate = self.negative_sample_ratio
        import random

        corrupted_graph_data = copy.deepcopy(original_graph_data)
    
        training_node_indices = np.argwhere(corrupted_graph_data.held_mask.cpu().numpy() == True).flatten()
        training_node_labels = corrupted_graph_data.y[corrupted_graph_data.held_mask].cpu().numpy()

        # Filter out invalid classes
        valid_sample_indices = set(range(len(training_node_labels)))
        for class_idx in range(num_classes):
            if (class_idx >= noise_transition_matrix.shape[1] or 
                np.isnan(noise_transition_matrix[0, class_idx]) or 
                np.max(noise_transition_matrix[:, class_idx]) == 1):
                print(f"Class {class_idx} is invalid for negative sampling.")

                valid_indices = set(np.where(training_node_labels != class_idx)[0])
                valid_sample_indices = valid_sample_indices & valid_indices
                
        training_node_indices = training_node_indices[list(valid_sample_indices)]

        num_negative_samples = int(np.round(sampling_rate * len(training_node_indices)))
        negative_sample_indices = random.sample(list(training_node_indices), num_negative_samples)
        
        for node_index in negative_sample_indices:
            current_label = int(corrupted_graph_data.y[node_index])

            new_corrupted_label = current_label
            while new_corrupted_label == current_label:
                new_corrupted_label = np.random.choice(
                    range(num_classes), 
                    p=noise_transition_matrix[:, current_label]
                )
            corrupted_graph_data.y[node_index] = new_corrupted_label

        return corrupted_graph_data, negative_sample_indices

    def _construct_detection_features(self, neighborhood_depth, original_graph_data, corrupted_graph_data, 
                                    negative_sample_indices, prediction_history, total_classes):

        print("Generating detection features")

        # Create binary indicator for negative samples
        negative_sample_indicator = np.zeros(original_graph_data.num_nodes)
        negative_sample_indicator[negative_sample_indices] = 1

        adjacency_matrix = to_scipy_sparse_matrix(original_graph_data.edge_index, num_nodes=original_graph_data.num_nodes)
        num_nodes = adjacency_matrix.shape[0]

        # Create normalized graph Laplacian
        normalized_laplacian = (
            spdiags(np.squeeze((1e-10 + np.array(adjacency_matrix.sum(1)))**-0.5), 0, num_nodes, num_nodes) @ 
            adjacency_matrix @ 
            spdiags(np.squeeze((1e-10 + np.array(adjacency_matrix.sum(0)))**-0.5), 0, num_nodes, num_nodes)
        )
        
        laplacian_squared = normalized_laplacian @ normalized_laplacian
        laplacian_cubed = normalized_laplacian @ laplacian_squared
        
        laplacian_squared.setdiag(np.zeros(num_nodes))
        laplacian_cubed.setdiag(np.zeros(num_nodes))
        
        print("Normalized Laplacian matrices calculated.")

        # Prepare label matrices
        negative_indicator_matrix = negative_sample_indicator[:, np.newaxis]
        original_label_matrix = np.eye(total_classes)[original_graph_data.y.cpu().numpy()]
        
        # Create corrected label matrix
        corrected_label_matrix = (negative_indicator_matrix * np.eye(total_classes)[corrupted_graph_data.y.cpu().numpy()] + 
                                (1 - negative_indicator_matrix) * original_label_matrix)
        
        label_propagation_1hop = normalized_laplacian @ original_label_matrix
        label_propagation_2hop = laplacian_squared @ original_label_matrix  
        label_propagation_3hop = laplacian_cubed @ original_label_matrix

        # Convert latest predictions to probabilities
        latest_prediction_probs = self._convert_logits_to_probabilities(torch.tensor(prediction_history[-1]))
        
        prediction_propagation_1hop = normalized_laplacian @ latest_prediction_probs
        prediction_propagation_2hop = laplacian_squared @ latest_prediction_probs
        prediction_propagation_3hop = laplacian_cubed @ latest_prediction_probs

        # Compute feature interactions
        feature_list = [
            np.sum(corrected_label_matrix * latest_prediction_probs, axis=1, keepdims=True),
            np.sum(corrected_label_matrix * prediction_propagation_1hop, axis=1, keepdims=True),
            np.sum(corrected_label_matrix * prediction_propagation_2hop, axis=1, keepdims=True),
            np.sum(corrected_label_matrix * label_propagation_1hop, axis=1, keepdims=True),
            np.sum(corrected_label_matrix * label_propagation_2hop, axis=1, keepdims=True),
            np.sum(corrected_label_matrix * label_propagation_3hop, axis=1, keepdims=True)
        ]

        if neighborhood_depth >= 4:
            laplacian_fourth = normalized_laplacian @ laplacian_cubed
            laplacian_fourth.setdiag(np.zeros(num_nodes))
            label_propagation_4hop = laplacian_fourth @ original_label_matrix
            prediction_propagation_4hop = laplacian_fourth @ latest_prediction_probs
            feature_list += [
                np.sum(corrected_label_matrix * prediction_propagation_4hop, axis=1, keepdims=True),
                np.sum(corrected_label_matrix * label_propagation_4hop, axis=1, keepdims=True)
            ]

        if neighborhood_depth >= 5:
            laplacian_fifth = normalized_laplacian @ laplacian_fourth
            laplacian_fifth.setdiag(np.zeros(num_nodes))
            label_propagation_5hop = laplacian_fifth @ original_label_matrix
            prediction_propagation_5hop = laplacian_fifth @ latest_prediction_probs
            feature_list += [
                np.sum(corrected_label_matrix * prediction_propagation_5hop, axis=1, keepdims=True),
                np.sum(corrected_label_matrix * label_propagation_5hop, axis=1, keepdims=True)
            ]

        feature_matrix = np.hstack(feature_list)
        return feature_matrix, negative_sample_indicator

    def execute_noise_detection_pipeline(self, graph_data, neural_network_model, num_classes):

        print("Starting GraphCleaner Detection Pipeline")

        # Determine which split to use for noise matrix estimation
        if self.validation_split_type == 'train':
            graph_data.held_mask = graph_data.train_mask
        elif self.validation_split_type == 'valid':
            graph_data.held_mask = graph_data.val_mask
        else:
            graph_data.held_mask = graph_data.test_mask

        # Train base GNN model
        prediction_history, optimal_predictions, trained_neural_network = self._train_base_neural_network(
            graph_data, neural_network_model, self.training_epochs
        )
        
        # Extract validation data
        validation_corrupted_labels = graph_data.y_noisy[graph_data.val_mask].cpu().numpy()
        validation_prediction_probs = self._convert_logits_to_probabilities(
            torch.tensor(optimal_predictions))[graph_data.val_mask.cpu().numpy()]
        
        # Estimate noise transition matrix
        class_priors, noise_transition_matrix, inverse_noise_matrix = self._estimate_noise_transition_matrix(
            validation_corrupted_labels, validation_prediction_probs, num_classes
        )
        
        # Generate negative samples
        artificially_corrupted_data, artificial_corruption_indices = self._generate_negative_samples(
            graph_data, noise_transition_matrix, num_classes)
        print(f"{len(artificial_corruption_indices)} negative samples generated")

        # Generate features for binary classification
        detection_features, binary_noise_labels = self._construct_detection_features(
            self.neighborhood_depth, graph_data, artificially_corrupted_data, 
            artificial_corruption_indices, prediction_history, num_classes
        )

        print("Training binary noise detector")

        # Prepare training and test data for binary classifier
        validation_mask_cpu = graph_data.val_mask.cpu().numpy()
        test_mask_cpu = graph_data.test_mask.cpu().numpy()
        
        classifier_training_features = detection_features[validation_mask_cpu].reshape(
            detection_features[validation_mask_cpu].shape[0], -1)
        classifier_training_labels = binary_noise_labels[validation_mask_cpu]

        classifier_test_features = detection_features[test_mask_cpu].reshape(
            detection_features[test_mask_cpu].shape[0], -1)
        
        # Train binary logistic regression classifier
        binary_noise_classifier = LogisticRegression(
            max_iter=self.classifier_max_iterations, 
            random_state=42
        )
        binary_noise_classifier.fit(classifier_training_features, classifier_training_labels)

        # Generate final predictions
        test_confidence_scores = binary_noise_classifier.predict_proba(classifier_test_features)[:, 1]
        test_binary_predictions = test_confidence_scores > 0.5
        
        return test_binary_predictions, test_confidence_scores, binary_noise_classifier, trained_neural_network
    
    def clean_training_data(self, graph_data, neural_network_model, num_classes):
 
        print("Starting GraphCleaner Training Data Cleaning")

        # Set validation split for noise matrix estimation
        if self.validation_split_type == 'train':
            graph_data.held_mask = graph_data.train_mask
        elif self.validation_split_type == 'valid':
            graph_data.held_mask = graph_data.val_mask
        else:
            graph_data.held_mask = graph_data.test_mask

        # Train base GNN model on noisy training data
        prediction_history, optimal_predictions, trained_neural_network = self._train_base_neural_network(
            graph_data, neural_network_model, self.training_epochs
        )
        
        # Extract validation data
        validation_corrupted_labels = graph_data.y[graph_data.val_mask].cpu().numpy()
        validation_prediction_probs = self._convert_logits_to_probabilities(
            torch.tensor(optimal_predictions))[graph_data.val_mask.cpu().numpy()]
        
        # Estimate noise transition matrix
        class_priors, noise_transition_matrix, inverse_noise_matrix = self._estimate_noise_transition_matrix(
            validation_corrupted_labels, validation_prediction_probs, num_classes
        )
        
        # Generate negative samples
        artificially_corrupted_data, artificial_corruption_indices = self._generate_negative_samples(
            graph_data, noise_transition_matrix, num_classes)
        print(f"{len(artificial_corruption_indices)} negative samples generated for detector training")

        # Generate features for binary classification
        detection_features, binary_noise_labels = self._construct_detection_features(
            self.neighborhood_depth, graph_data, artificially_corrupted_data, 
            artificial_corruption_indices, prediction_history, num_classes
        )

        # Train binary noise detector
        validation_mask_cpu = graph_data.val_mask.cpu().numpy()
        classifier_training_features = detection_features[validation_mask_cpu].reshape(
            detection_features[validation_mask_cpu].shape[0], -1)
        classifier_training_labels = binary_noise_labels[validation_mask_cpu]

        binary_noise_classifier = LogisticRegression(
            max_iter=self.classifier_max_iterations, 
            random_state=42
        )
        binary_noise_classifier.fit(classifier_training_features, classifier_training_labels)

        # Apply detector
        train_mask_cpu = graph_data.train_mask.cpu().numpy()
        train_detection_features = detection_features[train_mask_cpu].reshape(
            detection_features[train_mask_cpu].shape[0], -1)
        
        train_noise_predictions = binary_noise_classifier.predict(train_detection_features)
        train_confidence_scores = binary_noise_classifier.predict_proba(train_detection_features)[:, 1]
        
        train_indices = torch.where(graph_data.train_mask)[0]
        detected_noisy_indices = train_indices[torch.tensor(train_noise_predictions.astype(bool))]
        
        clean_train_mask = graph_data.train_mask.clone()
        clean_train_mask[detected_noisy_indices] = False
        
        print(f"GraphCleaner detected {len(detected_noisy_indices)} noisy nodes out of {graph_data.train_mask.sum()} training nodes")
        print(f"Clean training set size: {clean_train_mask.sum()} nodes")
        
        return clean_train_mask, graph_data

    def evaluate_detection_performance(self, noise_predictions, ground_truth_labels, confidence_scores, trained_model=None, graph_data=None):

        detection_accuracy = accuracy_score(ground_truth_labels, noise_predictions)
        detection_f1 = f1_score(ground_truth_labels, noise_predictions)
        detection_precision = precision_score(ground_truth_labels, noise_predictions)
        detection_recall = recall_score(ground_truth_labels, noise_predictions)
        detection_mcc = matthews_corrcoef(ground_truth_labels, noise_predictions)
        detection_auc = roc_auc_score(ground_truth_labels, confidence_scores)
        
        oversmoothing_results = None
        if trained_model is not None and graph_data is not None:
            trained_model.eval()
            with torch.no_grad():
                model_predictions = trained_model(graph_data)
                oversmoothing_results = self._calculate_oversmoothing_metrics_for_subset(
                    model_predictions, graph_data.edge_index, graph_data.test_mask, graph_data.y
                )
        
        print(f"GraphCleaner Training completed!")
        print(f"Test Accuracy: {detection_accuracy:.4f}")
        print(f"Test F1: {detection_f1:.4f}")
        print(f"Test Precision: {detection_precision:.4f}")
        print(f"Test Recall: {detection_recall:.4f}")
        print(f"Test MCC: {detection_mcc:.4f}")
        print(f"Test AUC: {detection_auc:.4f}")
        print(f"Samples detected as noisy: {np.sum(noise_predictions)}")
        print(f"Actual noisy samples: {np.sum(ground_truth_labels)}")
        
        return {
            'accuracy': detection_accuracy,
            'f1': detection_f1,
            'precision': detection_precision,
            'recall': detection_recall,
            'mcc': detection_mcc,
            'auc': detection_auc,
            'oversmoothing': oversmoothing_results
        }
