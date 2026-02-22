import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict

from model.evaluation import OversmoothingMetrics

class LabelWeightingNetwork(nn.Module):

    def __init__(self, input_dimensions=2, hidden_dimensions=32):
        super().__init__()
        self.weighting_network = nn.Sequential(
            nn.Linear(input_dimensions, hidden_dimensions),
            nn.ReLU(),
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.ReLU(),
            nn.Linear(hidden_dimensions, 1),
            nn.Sigmoid()
        )
    
    def forward(self, loss_noisy_labels, loss_pseudo_labels):
        combined_losses = torch.stack([loss_noisy_labels, loss_pseudo_labels], dim=1)
        return self.weighting_network(combined_losses).squeeze(-1)

class GNNCleanerTrainer:
    def __init__(self, configuration, graph_data, computation_device, number_of_classes, base_gnn_model=None):
        self.config = configuration
        self.data = graph_data
        self.device = computation_device
        self.num_classes = number_of_classes
        
        # Training hyperparameters
        self.max_training_epochs = configuration.get('max_epochs', 200)
        self.model_learning_rate = configuration.get('model_learning_rate', 0.01)
        self.network_learning_rate = configuration.get('net_learning_rate', 0.001)
        self.weight_decay_coefficient = float(configuration.get('weight_decay', 5e-4))
        
        # Label propagation parameters
        self.label_propagation_iterations = configuration.get('label_propagation_iterations', 50)
        self.similarity_epsilon = float(configuration.get('similarity_epsilon', 1e-8))
        
        self.clean_set_ratio = configuration.get('clean_set_ratio', 0.1)
        
        # Early stopping
        self.early_stopping_patience = int(configuration.get('early_stopping_patience', 10))

        if base_gnn_model is not None:
            self.gnn_model = base_gnn_model.to(computation_device)
        else:
            raise ValueError("GNN Cleaner requires a base GNN model")
        
        self.label_weighting_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(), 
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(computation_device)
        
        self.gnn_optimizer = torch.optim.Adam(
            self.gnn_model.parameters(), 
            lr=self.model_learning_rate, 
            weight_decay=self.weight_decay_coefficient
        )
        
        self.network_optimizer = torch.optim.Adam(
            self.label_weighting_net.parameters(),
            lr=self.network_learning_rate
        )
        
        self._initialize_clean_set()
        
        self.training_results = {'train': -1, 'val': -1, 'test': -1}
        self._display_dataset_statistics()

        # Initialize oversmoothing evaluation
        self.oversmoothing_metric_evaluator = OversmoothingMetrics(device=computation_device)
        self.oversmoothing_training_history = {
            'train': [],
            'val': [],
            'test': []
        }

    def _initialize_clean_set(self):
        if not hasattr(self.data, 'y_original'):
            raise ValueError("Need y_original to identify clean samples")
        
        train_indices = self.data.train_mask.nonzero(as_tuple=True)[0]
        clean_train_indices = []
        
        for idx in train_indices:
            if self.data.y[idx] == self.data.y_original[idx]:
                clean_train_indices.append(idx)
        
        clean_train_indices = torch.tensor(clean_train_indices, device=self.device)
        
        if len(clean_train_indices) < int(train_indices.size(0) * self.clean_set_ratio):
            print(f"Warning: Only {len(clean_train_indices)} truly clean nodes found in training set")
            print(f"Needed: {int(train_indices.size(0) * self.clean_set_ratio)} clean nodes")
            print("Selecting additional nodes randomly from training set as approximation")
            
            remaining_train = train_indices[~torch.isin(train_indices, clean_train_indices)]
            needed = int(train_indices.size(0) * self.clean_set_ratio) - len(clean_train_indices)
            
            if len(remaining_train) > 0:
                additional_indices = remaining_train[torch.randperm(len(remaining_train))[:needed]]
                clean_train_indices = torch.cat([clean_train_indices, additional_indices])
        
        max_clean = max(1, int(train_indices.size(0) * self.clean_set_ratio))
        clean_train_indices = clean_train_indices[:max_clean]
        
        self.initial_clean_mask = torch.zeros(self.data.y.size(0), dtype=torch.bool, device=self.device)
        self.initial_clean_mask[clean_train_indices] = True
        
        self.expanding_clean_sample_mask = self.initial_clean_mask.clone()
        
        truly_clean = (self.data.y[clean_train_indices] == self.data.y_original[clean_train_indices]).sum()
        print(f"Initialized clean set with {self.initial_clean_mask.sum()} nodes from training set")
        print(f"Of these, {truly_clean} are truly clean, {len(clean_train_indices) - truly_clean} are approximations")

    def _execute_label_propagation(self, similarity_matrix, initial_node_labels, clean_nodes_mask, propagation_iterations=50):
        total_nodes = initial_node_labels.size(0)
        
        label_probabilities = torch.zeros(total_nodes, self.num_classes, device=self.device)
        
        # Set clean node labels
        clean_node_indices = clean_nodes_mask.nonzero(as_tuple=True)[0]
        for node_idx in clean_node_indices:
            if hasattr(self.data, 'y_original'):
                node_label = self.data.y_original[node_idx].item()  # Use ground truth
            else:
                node_label = initial_node_labels[node_idx].item()
            label_probabilities[node_idx, node_label] = 1.0
        
        similarity_tensor = torch.from_numpy(similarity_matrix.toarray()).float().to(self.device)
        
        # Iterative label propagation
        for iteration in range(propagation_iterations):
            label_probabilities = torch.matmul(similarity_tensor, label_probabilities)
            
            # Reset clean node labels
            for node_idx in clean_node_indices:
                if hasattr(self.data, 'y_original'):
                    node_label = self.data.y_original[node_idx].item()
                else:
                    node_label = initial_node_labels[node_idx].item()
                label_probabilities[node_idx] = 0.0
                label_probabilities[node_idx, node_label] = 1.0
        
        return label_probabilities

    def _perform_training_step(self, current_epoch):

        self.gnn_model.train()
        self.label_weighting_net.train()
        
        node_features = self.data.x.to(self.device)
        edge_connectivity = self.data.edge_index.to(self.device)
        noisy_node_labels = self.data.y.to(self.device)
        ground_truth_labels = self.data.y_original.to(self.device) if hasattr(self.data, 'y_original') else self.data.y.to(self.device)

        try:
            node_embeddings = self.gnn_model(self.data)
        except:
            node_embeddings = self.gnn_model(node_features, edge_connectivity)

        # Similarity matrix
        similarity_matrix = self._build_node_similarity_matrix(edge_connectivity, node_embeddings.detach())
        
        # Label propagation from clean nodes
        propagated_labels = self._execute_label_propagation(
            similarity_matrix, ground_truth_labels, self.expanding_clean_sample_mask, self.label_propagation_iterations
        )

        # Sample Selection
        D_select = torch.zeros_like(self.data.train_mask, dtype=torch.bool)
        D_left = torch.zeros_like(self.data.train_mask, dtype=torch.bool)
        
        train_node_indices = self.data.train_mask.nonzero(as_tuple=True)[0]
        for node_idx in train_node_indices:
            pseudo_label = propagated_labels[node_idx].argmax().item()
            given_label = noisy_node_labels[node_idx].item()
            
            if pseudo_label == given_label:
                D_select[node_idx] = True
            else:
                D_left[node_idx] = True
        
        # Update expanding clean set
        self.expanding_clean_sample_mask = (self.expanding_clean_sample_mask | D_select).detach()
        
        total_loss = 0.0
        
        # Training on selected node
        if D_select.sum() > 0:
            selected_embeddings = node_embeddings[D_select]
            selected_labels = ground_truth_labels[D_select]
            
            select_loss = F.cross_entropy(selected_embeddings, selected_labels)
            
            # One-step optimization
            self.gnn_optimizer.zero_grad()
            select_loss.backward(retain_graph=True)
            self.gnn_optimizer.step()
            
            total_loss += select_loss.item()

        # Label Correction
        if D_left.sum() > 0:
            try:
                updated_embeddings = self.gnn_model(self.data)
            except:
                updated_embeddings = self.gnn_model(node_features, edge_connectivity)
            
            left_indices = D_left.nonzero(as_tuple=True)[0]

            corrected_loss = 0.0
            for node_idx in left_indices:

                y_hat = updated_embeddings[node_idx].unsqueeze(0)
                given_label = noisy_node_labels[node_idx].unsqueeze(0)
                pseudo_label_dist = propagated_labels[node_idx].unsqueeze(0)
                
                l1 = F.cross_entropy(y_hat, given_label, reduction='none')
                l2 = -(pseudo_label_dist * F.log_softmax(y_hat, dim=1)).sum(dim=1)
                
                # Compute lambda
                lambda_j = self.label_weighting_net(torch.stack([l1.detach(), l2.detach()], dim=1))
                
                # Correct label
                given_onehot = F.one_hot(given_label, self.num_classes).float()
                corrected_label = lambda_j * given_onehot + (1 - lambda_j) * pseudo_label_dist
                
                # Training loss
                node_loss = -(corrected_label * F.log_softmax(y_hat, dim=1)).sum()
                corrected_loss += node_loss
            
            if corrected_loss > 0:
                avg_corrected_loss = corrected_loss / len(left_indices)
                
                # Update GNN
                self.gnn_optimizer.zero_grad()
                avg_corrected_loss.backward(retain_graph=True)
                
                temp_gnn_params = {}
                for name, param in self.gnn_model.named_parameters():
                    if param.grad is not None:
                        temp_gnn_params[name] = param.clone().detach()
                
                self.gnn_optimizer.step()
                
                # Update aggregation network
                clean_indices = self.initial_clean_mask.nonzero(as_tuple=True)[0]
                if len(clean_indices) > 0:
                    try:
                        final_embeddings = self.gnn_model(self.data)
                    except:
                        final_embeddings = self.gnn_model(node_features, edge_connectivity)
                    
                    clean_embeddings = final_embeddings[clean_indices]
                    clean_labels = ground_truth_labels[clean_indices]
                    
                    clean_loss = F.cross_entropy(clean_embeddings, clean_labels)
                    
                    self.network_optimizer.zero_grad()
                    clean_loss.backward()
                    self.network_optimizer.step()
                
                total_loss += avg_corrected_loss.item()
        
        return total_loss, 0.0

    def _display_dataset_statistics(self):
        print("GNN Cleaner Dataset Statistics")
        print(f"Total nodes: {self.data.x.shape[0]}")
        print(f"Node features: {self.data.x.shape[1]}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Total edges: {self.data.edge_index.shape[1] // 2}")
        print(f"Train/Validation/Test split: {self.data.train_mask.sum()}/{self.data.val_mask.sum()}/{self.data.test_mask.sum()}")
        print(f"Initial clean set size: {self.initial_clean_mask.sum()}")
        
        if hasattr(self.data, 'y_original'):
            train_corrupted = (self.data.y[self.data.train_mask] != self.data.y_original[self.data.train_mask]).sum().item()
            train_total = self.data.train_mask.sum().item()
            
            val_clean = (self.data.y[self.data.val_mask] == self.data.y_original[self.data.val_mask]).all().item()
            test_clean = (self.data.y[self.data.test_mask] == self.data.y_original[self.data.test_mask]).all().item()
            
            print(f"Training labels corrupted: {train_corrupted}/{train_total} ({train_corrupted/train_total:.3%})")
            print(f"Validation labels clean: {val_clean}")
            print(f"Test labels clean: {test_clean}")
            
            if train_corrupted > 0:
                actual_noise_rate = train_corrupted / train_total
                print(f"Actual noise rate on training: {actual_noise_rate:.3%}")
            else:
                print("WARNING: No noise detected in training labels.")
        else:
            print("WARNING: No y_original found - cannot verify label corruption.")
        
        print("Sample Label Check")
        train_indices = self.data.train_mask.nonzero(as_tuple=True)[0][:5]
        for i, idx in enumerate(train_indices):
            if hasattr(self.data, 'y_original'):
                original_label = self.data.y_original[idx].item()
                current_label = self.data.y[idx].item()
                print(f"Train node {idx}: Original={original_label}, Current={current_label}, Match={original_label==current_label}")
            else:
                current_label = self.data.y[idx].item()
                print(f"Train node {idx}: Current={current_label}")

    def _build_node_similarity_matrix(self, edge_connectivity, node_feature_embeddings):
        total_nodes = node_feature_embeddings.size(0)
        
        edge_source, edge_target = edge_connectivity.cpu().numpy()
        graph_adjacency = sparse.coo_matrix(
            (np.ones(len(edge_source)), (edge_source, edge_target)), 
            shape=(total_nodes, total_nodes)
        )
        
        similarity_weights = []
        similarity_row_indices = []
        similarity_col_indices = []
        
        for edge_idx in range(len(edge_source)):
            source_node, target_node = edge_source[edge_idx], edge_target[edge_idx]
            if source_node != target_node:
                source_features = node_feature_embeddings[source_node].cpu().numpy()
                target_features = node_feature_embeddings[target_node].cpu().numpy()
                feature_distance = np.linalg.norm(source_features - target_features) + self.similarity_epsilon
                edge_similarity = 1.0 / feature_distance
                
                similarity_weights.append(edge_similarity)
                similarity_row_indices.append(source_node)
                similarity_col_indices.append(target_node)
        
        similarity_matrix = sparse.coo_matrix(
            (similarity_weights, (similarity_row_indices, similarity_col_indices)), 
            shape=(total_nodes, total_nodes)
        )
        
        node_degrees = np.array(similarity_matrix.sum(axis=1)).flatten()
        node_degrees[node_degrees == 0] = 1.0
        degree_normalization = sparse.diags(1.0 / np.sqrt(node_degrees))
        normalized_similarity = degree_normalization @ similarity_matrix @ degree_normalization
        
        return normalized_similarity


    def _identify_clean_samples(self, propagated_label_probabilities, given_node_labels, training_nodes_mask):

        predicted_hard_labels = propagated_label_probabilities.argmax(dim=1)
        
        training_node_indices = training_nodes_mask.nonzero(as_tuple=True)[0]
        identified_clean_mask = torch.zeros_like(training_nodes_mask)
        identified_noisy_mask = torch.zeros_like(training_nodes_mask)
        
        for node_idx in training_node_indices:
            if predicted_hard_labels[node_idx] == given_node_labels[node_idx]:
                identified_clean_mask[node_idx] = True
            else:
                identified_noisy_mask[node_idx] = True
        
        return identified_clean_mask, identified_noisy_mask

    @torch.no_grad()
    def _evaluate_model_performance(self, include_test_metrics=False):

        from sklearn.metrics import precision_score, recall_score
        
        self.gnn_model.eval()
        
        node_features, edge_connectivity = self.data.x.to(self.device), self.data.edge_index.to(self.device)
        
        try:
            model_outputs = self.gnn_model(self.data)
        except:
            model_outputs = self.gnn_model(node_features, edge_connectivity)
        
        predicted_labels = model_outputs.argmax(dim=1)
        ground_truth_labels = self.data.y_original.to(self.device)
        
        train_loss_value = F.cross_entropy(model_outputs[self.data.train_mask], ground_truth_labels[self.data.train_mask]).item()
        validation_loss_value = F.cross_entropy(model_outputs[self.data.val_mask], ground_truth_labels[self.data.val_mask]).item()
        
        train_accuracy = (predicted_labels[self.data.train_mask] == ground_truth_labels[self.data.train_mask]).float().mean().item()
        validation_accuracy = (predicted_labels[self.data.val_mask] == ground_truth_labels[self.data.val_mask]).float().mean().item()
        
        train_f1_score = f1_score(
            ground_truth_labels[self.data.train_mask].cpu().numpy(), 
            predicted_labels[self.data.train_mask].cpu().numpy(), 
            average='macro'
        )
        validation_f1_score = f1_score(
            ground_truth_labels[self.data.val_mask].cpu().numpy(), 
            predicted_labels[self.data.val_mask].cpu().numpy(), 
            average='macro'
        )
        
        evaluation_results = {
            'train_loss': train_loss_value, 
            'val_loss': validation_loss_value,
            'train_acc': train_accuracy, 
            'val_acc': validation_accuracy,
            'train_f1': train_f1_score, 
            'val_f1': validation_f1_score
        }
        
        if include_test_metrics:
            test_loss_value = F.cross_entropy(model_outputs[self.data.test_mask], ground_truth_labels[self.data.test_mask]).item()
            test_accuracy = (predicted_labels[self.data.test_mask] == ground_truth_labels[self.data.test_mask]).float().mean().item()
            test_f1_score = f1_score(
                ground_truth_labels[self.data.test_mask].cpu().numpy(), 
                predicted_labels[self.data.test_mask].cpu().numpy(), 
                average='macro'
            )

            test_precision_score = precision_score(
                ground_truth_labels[self.data.test_mask].cpu().numpy(), 
                predicted_labels[self.data.test_mask].cpu().numpy(), 
                average='macro'
            )
            test_recall_score = recall_score(
                ground_truth_labels[self.data.test_mask].cpu().numpy(), 
                predicted_labels[self.data.test_mask].cpu().numpy(), 
                average='macro'
            )
            evaluation_results.update({
                'test_loss': test_loss_value, 
                'test_acc': test_accuracy, 
                'test_f1': test_f1_score,
                'test_precision': test_precision_score,
                'test_recall': test_recall_score
            })
        
        return evaluation_results

    def execute_full_training(self, enable_debug_output=True):
        per_epochs_oversmoothing = defaultdict(list)
        training_start_time = time.time()
        
        best_validation_loss = float("inf")
        best_model_checkpoint = None
        early_stopping_counter = 0
        
        oversmoothing_calculation_interval = 20
        
        for training_epoch in range(self.max_training_epochs):
            epoch_training_loss, epoch_weighting_loss = self._perform_training_step(training_epoch)
            
            current_metrics = self._evaluate_model_performance(include_test_metrics=False)
            
            train_oversmoothing_metrics = None
            validation_oversmoothing_metrics = None
            
            if training_epoch % oversmoothing_calculation_interval == 0 or training_epoch == self.max_training_epochs - 1:
                try:
                    current_embeddings = self.gnn_model(self.data)
                except:
                    current_embeddings = self.gnn_model(self.data.x.to(self.device), self.data.edge_index.to(self.device))

                train_oversmoothing_metrics = self._compute_oversmoothing_for_node_subset(
                    current_embeddings, self.data.edge_index.to(self.device), self.data.train_mask, self.data.y_original
                )
                validation_oversmoothing_metrics = self._compute_oversmoothing_for_node_subset(
                    current_embeddings, self.data.edge_index.to(self.device), self.data.val_mask, self.data.y_original
                )

                if train_oversmoothing_metrics is not None:
                    self.oversmoothing_training_history['train'].append(train_oversmoothing_metrics)
                    for key, value in train_oversmoothing_metrics.items():
                        per_epochs_oversmoothing[key].append(value)
                if validation_oversmoothing_metrics is not None:
                    self.oversmoothing_training_history['val'].append(validation_oversmoothing_metrics)

            # Early stopping
            if current_metrics['val_loss'] < best_validation_loss:
                best_validation_loss = current_metrics['val_loss']
                best_model_checkpoint = {
                    "gnn_model": self.gnn_model.state_dict(),
                    "weighting_network": self.label_weighting_net.state_dict()
                }
                best_training_performance = {
                    'train_acc': current_metrics['train_acc'],
                    'train_f1': current_metrics['train_f1']
                }
                best_validation_performance = {
                    'val_acc': current_metrics['val_acc'], 
                    'val_f1': current_metrics['val_f1']
                }
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= self.early_stopping_patience:
                if enable_debug_output:
                    print(f"Early stopping triggered at epoch {training_epoch}")
                break

            if enable_debug_output:
                newly_selected_count = self.expanding_clean_sample_mask.sum().item() - self.initial_clean_mask.sum().item()
                
                if training_epoch % oversmoothing_calculation_interval == 0 or training_epoch == self.max_training_epochs - 1:

                    train_energy_direction = train_oversmoothing_metrics['EDir'] if train_oversmoothing_metrics else 0.0
                    train_energy_direction_traditional = train_oversmoothing_metrics['EDir_traditional'] if train_oversmoothing_metrics else 0.0
                    train_energy_projection = train_oversmoothing_metrics['EProj'] if train_oversmoothing_metrics else 0.0
                    train_mean_absolute_deviation = train_oversmoothing_metrics['MAD'] if train_oversmoothing_metrics else 0.0
                    train_numerical_rank = train_oversmoothing_metrics['NumRank'] if train_oversmoothing_metrics else 0.0
                    train_e_rank = train_oversmoothing_metrics['Erank'] if train_oversmoothing_metrics else 0.0
                    
                    val_energy_direction = validation_oversmoothing_metrics['EDir'] if validation_oversmoothing_metrics else 0.0
                    val_energy_direction_traditional = validation_oversmoothing_metrics['EDir_traditional'] if validation_oversmoothing_metrics else 0.0
                    val_energy_projection = validation_oversmoothing_metrics['EProj'] if validation_oversmoothing_metrics else 0.0
                    val_mean_absolute_deviation = validation_oversmoothing_metrics['MAD'] if validation_oversmoothing_metrics else 0.0
                    val_numerical_rank = validation_oversmoothing_metrics['NumRank'] if validation_oversmoothing_metrics else 0.0
                    val_e_rank = validation_oversmoothing_metrics['Erank'] if validation_oversmoothing_metrics else 0.0
                    
                    print(f"Epoch {training_epoch+1:03d} | Train Loss: {epoch_training_loss:.4f}, Val Loss: {current_metrics['val_loss']:.4f} | "
                        f"Train Acc: {current_metrics['train_acc']:.4f}, Val Acc: {current_metrics['val_acc']:.4f} | "
                        f"Train F1: {current_metrics['train_f1']:.4f}, Val F1: {current_metrics['val_f1']:.4f} | "
                        f"Newly Selected: {newly_selected_count}")
                    print(f"Train EDir: {train_energy_direction:.4f}, Val EDir: {val_energy_direction:.4f} | "
                        f"Train EDir_trad: {train_energy_direction_traditional:.4f}, Val EDir_trad: {val_energy_direction_traditional:.4f} | "
                        f"Train EProj: {train_energy_projection:.4f}, Val EProj: {val_energy_projection:.4f} | "
                        f"Train MAD: {train_mean_absolute_deviation:.4f}, Val MAD: {val_mean_absolute_deviation:.4f} | "
                        f"Train NumRank: {train_numerical_rank:.4f}, Val NumRank: {val_numerical_rank:.4f} | "
                        f"Train Erank: {train_e_rank:.4f}, Val Erank: {val_e_rank:.4f}")
                else:

                    print(f"Epoch {training_epoch+1:03d} | Train Loss: {epoch_training_loss:.4f}, Val Loss: {current_metrics['val_loss']:.4f} | "
                        f"Train Acc: {current_metrics['train_acc']:.4f}, Val Acc: {current_metrics['val_acc']:.4f} | "
                        f"Train F1: {current_metrics['train_f1']:.4f}, Val F1: {current_metrics['val_f1']:.4f} | "
                        f"Newly Selected: {newly_selected_count}")
        
        if best_model_checkpoint is not None:
            self.gnn_model.load_state_dict(best_model_checkpoint["gnn_model"])
            self.label_weighting_net.load_state_dict(best_model_checkpoint["weighting_network"])

        final_evaluation_metrics = self._evaluate_model_performance(include_test_metrics=True)
        
        try:
            final_embeddings = self.gnn_model(self.data)
        except:
            final_embeddings = self.gnn_model(self.data.x.to(self.device), self.data.edge_index.to(self.device))

        final_train_oversmoothing = self._compute_oversmoothing_for_node_subset(
            final_embeddings, self.data.edge_index.to(self.device), self.data.train_mask, self.data.y_original
        )
        final_validation_oversmoothing = self._compute_oversmoothing_for_node_subset(
            final_embeddings, self.data.edge_index.to(self.device), self.data.val_mask, self.data.y_original
        )
        final_test_oversmoothing = self._compute_oversmoothing_for_node_subset(
            final_embeddings, self.data.edge_index.to(self.device), self.data.test_mask, self.data.y_original
        )

        if final_test_oversmoothing is not None:
            self.oversmoothing_training_history['test'].append(final_test_oversmoothing)
        
        total_training_time = time.time() - training_start_time
        
        if enable_debug_output:
            print(f"\nGNN Cleaner Training Completed")
            print(f"Final Test Accuracy: {final_evaluation_metrics['test_acc']:.4f}")
            print(f"Final Test F1 Score: {final_evaluation_metrics['test_f1']:.4f}")
            print(f"Final Test Precision: {final_evaluation_metrics['test_precision']:.4f}")
            print(f"Final Test Recall: {final_evaluation_metrics['test_recall']:.4f}")
            print(f"Total training time: {total_training_time:.2f} seconds")
            print("Final Oversmoothing Metrics")
            
            if final_train_oversmoothing is not None:
                print(f"Train: EDir: {final_train_oversmoothing['EDir']:.4f}, EDir_traditional: {final_train_oversmoothing['EDir_traditional']:.4f}, "
                    f"EProj: {final_train_oversmoothing['EProj']:.4f}, MAD: {final_train_oversmoothing['MAD']:.4f}, "
                    f"NumRank: {final_train_oversmoothing['NumRank']:.4f}, Erank: {final_train_oversmoothing['Erank']:.4f}")
            
            if final_validation_oversmoothing is not None:
                print(f"Val: EDir: {final_validation_oversmoothing['EDir']:.4f}, EDir_traditional: {final_validation_oversmoothing['EDir_traditional']:.4f}, "
                    f"EProj: {final_validation_oversmoothing['EProj']:.4f}, MAD: {final_validation_oversmoothing['MAD']:.4f}, "
                    f"NumRank: {final_validation_oversmoothing['NumRank']:.4f}, Erank: {final_validation_oversmoothing['Erank']:.4f}")
            
            if final_test_oversmoothing is not None:
                print(f"Test: EDir: {final_test_oversmoothing['EDir']:.4f}, EDir_traditional: {final_test_oversmoothing['EDir_traditional']:.4f}, "
                    f"EProj: {final_test_oversmoothing['EProj']:.4f}, MAD: {final_test_oversmoothing['MAD']:.4f}, "
                    f"NumRank: {final_test_oversmoothing['NumRank']:.4f}, Erank: {final_test_oversmoothing['Erank']:.4f}")
        
        return {
            'accuracy': final_evaluation_metrics['test_acc'],
            'f1': final_evaluation_metrics['test_f1'],
            'precision': final_evaluation_metrics['test_precision'],
            'recall': final_evaluation_metrics['test_recall'],
            'oversmoothing': final_test_oversmoothing,
            'train_oversmoothing': per_epochs_oversmoothing
        }
    
    def _compute_oversmoothing_for_node_subset(self, node_embeddings, edge_connectivity, node_subset_mask, node_labels=None):

        try:
            subset_node_indices = torch.where(node_subset_mask)[0]
            subset_embeddings = node_embeddings[node_subset_mask]
            
            subset_node_set = set(subset_node_indices.cpu().numpy())
            edge_subset_mask = torch.tensor([
                source_node.item() in subset_node_set and target_node.item() in subset_node_set
                for source_node, target_node in edge_connectivity.t()
            ], device=edge_connectivity.device)
            
            if not edge_subset_mask.any():
                return {
                    'NumRank': float(min(subset_embeddings.shape)),
                    'Erank': float(min(subset_embeddings.shape)),
                    'EDir': 0.0,
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0
                }
            
            subset_edges = edge_connectivity[:, edge_subset_mask]
            node_index_mapping = {original_idx.item(): local_idx for local_idx, original_idx in enumerate(subset_node_indices)}
            
            remapped_edge_connectivity = torch.stack([
                torch.tensor([node_index_mapping[source_node.item()] for source_node in subset_edges[0]], device=edge_connectivity.device),
                torch.tensor([node_index_mapping[target_node.item()] for target_node in subset_edges[1]], device=edge_connectivity.device)
            ])
            
            subset_graph_data = [{
                'X': subset_embeddings,
                'edge_index': remapped_edge_connectivity,
                'edge_weight': None
            }]
            
            return self.oversmoothing_metric_evaluator.compute_all_metrics(
                X=subset_embeddings,
                edge_index=remapped_edge_connectivity,
                graphs_in_class=subset_graph_data
            )
            
        except Exception as computation_error:
            print(f"Warning: Could not compute oversmoothing metrics for node subset: {computation_error}")
            return None

    def get_oversmoothing_training_history(self):
        return self.oversmoothing_training_history