import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, negative_sampling
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, precision_score, recall_score

from model.evaluation import OversmoothingMetrics

class NRGNN:

    def __init__(self, config, device, base_model=None):
        self.device = device
        self.config = config
        self.base_model_class = base_model
        
        # Training parameters from config
        self.learning_rate = config.get('lr', 0.01)
        self.weight_decay = float(config.get('weight_decay', 5e-4))
        self.max_epochs = config.get('epochs', 1000)
        self.patience = config.get('patience', 100)
        
        # NRGNN specific parameters
        nrgnn_params = config.get('nrgnn_params', {})
        self.edge_hidden_dim = nrgnn_params.get('edge_hidden', 16)
        self.num_potential_edges = nrgnn_params.get('n_p', 10)
        self.confidence_threshold = nrgnn_params.get('p_u', 0.7)
        self.reconstruction_weight = nrgnn_params.get('alpha', 0.05)
        self.consistency_weight = nrgnn_params.get('beta', 1.0)
        self.edge_threshold = nrgnn_params.get('t_small', 0.1)
        self.negative_samples_ratio = nrgnn_params.get('n_n', 50)
        self.debug_mode = nrgnn_params.get('debug', True)
        
        # Model components
        self.node_predictor = None
        self.main_model = None
        self.edge_weight_estimator = None
        self.adjacency_estimator = None
        self.optimizer = None
        
        # Training state
        self.best_validation_loss = float('inf')
        self.best_predictor_accuracy = 0.0
        self.early_stopping_counter = 0
        
        # Model weights storage
        self.best_main_weights = None
        self.best_predictor_weights = None
        self.best_predictions = None
        self.best_edge_weights = None
        self.best_predictor_edge_weights = None
        self.best_edge_indices = None
        
        # Graph structure
        self.original_edge_index = None
        self.potential_edge_index = None
        self.confident_edge_index = None
        self.confident_node_indices = []
        self.unlabeled_node_indices = None
        
        # Data
        self.node_features = None
        self.node_labels = None
        
        # Evaluation
        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        self.oversmoothing_metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }

    def compute_accuracy(self, model_output, true_labels):
        if not isinstance(true_labels, torch.Tensor):
            true_labels = torch.LongTensor(true_labels)
        predictions = model_output.max(1)[1].type_as(true_labels)
        correct_predictions = predictions.eq(true_labels).double()
        return correct_predictions.sum() / len(true_labels)

    def convert_sparse_to_torch_tensor(self, sparse_matrix):
        #Convert scipy sparse matrix to PyTorch sparse tensor
        sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_matrix.data)
        shape = torch.Size(sparse_matrix.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def create_edge_weight_estimator(self, input_features, hidden_dim, output_dim):
        #Create GCN-based edge weight estimator
        estimator = nn.Module()
        estimator.first_conv = GCNConv(input_features, hidden_dim, bias=True, add_self_loops=True)
        estimator.second_conv = GCNConv(hidden_dim, output_dim, bias=True, add_self_loops=True)
        
        def forward_fn(node_features, edge_index, edge_weights=None):
            node_features = F.relu(estimator.first_conv(node_features, edge_index, edge_weights))
            node_features = F.dropout(node_features, 0.0, training=estimator.training)
            node_features = estimator.second_conv(node_features, edge_index, edge_weights)
            return node_features
            
        def reset_params_fn():
            estimator.first_conv.reset_parameters()
            estimator.second_conv.reset_parameters()
        
        estimator.forward = forward_fn
        estimator.reset_parameters = reset_params_fn
        return estimator

    def create_gnn_wrapper(self, base_model):
        #Create wrapper for base GNN model
        wrapper = nn.Module()
        wrapper.base_gnn_model = base_model
        
        def forward_fn(node_features, edge_index, edge_weights=None):
            data_obj = type('GraphData', (), {})()
            data_obj.x = node_features
            data_obj.edge_index = edge_index
            data_obj.edge_weight = edge_weights
            return wrapper.base_gnn_model(data_obj)
            
        def reset_params_fn():
            if hasattr(wrapper.base_gnn_model, 'initialize'):
                wrapper.base_gnn_model.initialize()
            else:
                for module in wrapper.base_gnn_model.modules():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
        
        wrapper.forward = forward_fn
        wrapper.reset_parameters = reset_params_fn
        return wrapper

    def compute_estimated_edge_weights(self, edge_index, node_representations):
        #Compute edge weights from node representations
        source_nodes = node_representations[edge_index[0]]
        target_nodes = node_representations[edge_index[1]]
        similarity_scores = torch.sum(source_nodes * target_nodes, dim=1)
        estimated_weights = F.relu(similarity_scores)
        
        #Apply threshold
        weight_mask = estimated_weights >= self.edge_threshold
        estimated_weights = estimated_weights * weight_mask.float()
        
        return estimated_weights

    def compute_reconstruction_loss(self, edge_index, node_representations):
        #Compute edge reconstruction loss
        num_nodes = node_representations.shape[0]
        
        #Generate negative samples
        negative_edges = negative_sampling(
            edge_index, num_nodes=num_nodes, 
            num_neg_samples=self.negative_samples_ratio * num_nodes
        )
        
        negative_edges = negative_edges[:, negative_edges[0] < negative_edges[1]]
        positive_edges = edge_index[:, edge_index[0] < edge_index[1]]
        
        if negative_edges.shape[1] == 0 or positive_edges.shape[1] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        #Compute similarities
        negative_similarities = torch.sum(
            node_representations[negative_edges[0]] * node_representations[negative_edges[1]], 
            dim=1
        )
        positive_similarities = torch.sum(
            node_representations[positive_edges[0]] * node_representations[positive_edges[1]], 
            dim=1
        )
        
        #Compute losses
        negative_loss = F.mse_loss(negative_similarities, torch.zeros_like(negative_similarities), reduction='sum')
        positive_loss = F.mse_loss(positive_similarities, torch.ones_like(positive_similarities), reduction='sum')
        
        total_edges = negative_edges.shape[1] + positive_edges.shape[1]
        reconstruction_loss = (negative_loss + positive_loss) * num_nodes / total_edges
        
        return reconstruction_loss

    def compute_oversmoothing_for_node_set(self, node_embeddings, edge_index, node_indices):
        #Compute oversmoothing metrics
        try:
            if isinstance(node_indices, np.ndarray):
                node_indices = torch.tensor(node_indices, device=self.device)
            elif not isinstance(node_indices, torch.Tensor):
                node_indices = torch.tensor(list(node_indices), device=self.device)
            
            subset_embeddings = node_embeddings[node_indices]
            
            node_set = set(node_indices.cpu().numpy())
            edge_mask = torch.tensor([
                src.item() in node_set and tgt.item() in node_set
                for src, tgt in edge_index.t()
            ], device=edge_index.device)
            
            if not edge_mask.any():
                return {
                    'NumRank': float(min(subset_embeddings.shape)),
                    'Erank': float(min(subset_embeddings.shape)),
                    'EDir': 0.0,
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0
                }
            
            #Extract and remap edges
            subset_edges = edge_index[:, edge_mask]
            node_mapping = {orig_idx.item(): local_idx for local_idx, orig_idx in enumerate(node_indices)}
            
            remapped_edges = torch.stack([
                torch.tensor([node_mapping[src.item()] for src in subset_edges[0]], device=edge_index.device),
                torch.tensor([node_mapping[tgt.item()] for tgt in subset_edges[1]], device=edge_index.device)
            ])
            
            graphs_data = [{
                'X': subset_embeddings,
                'edge_index': remapped_edges,
                'edge_weight': None
            }]
            
            return self.oversmoothing_evaluator.compute_all_metrics(
                X=subset_embeddings,
                edge_index=remapped_edges,
                graphs_in_class=graphs_data
            )
            
        except Exception as e:
            print(f"Warning: Could not compute oversmoothing metrics: {e}")
            return {
                'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0
            }

    def prepare_training_data(self, features, adjacency_matrix, labels, train_indices):
        #Convert adjacency matrix to edge index
        self.original_edge_index, _ = from_scipy_sparse_matrix(adjacency_matrix)
        self.original_edge_index = self.original_edge_index.to(self.device)
        
        #Convert features
        if sp.issparse(features):
            features = self.convert_sparse_to_torch_tensor(features).to_dense()
        else:
            features = torch.FloatTensor(np.array(features))
        self.node_features = features.to(self.device)
        
        #Convert labels
        self.node_labels = torch.LongTensor(np.array(labels)).to(self.device)
        
        #Find unlabeled nodes
        all_node_indices = set(range(self.node_features.shape[0]))
        train_node_set = set(train_indices.tolist() if isinstance(train_indices, np.ndarray) else train_indices)
        unlabeled_nodes = list(all_node_indices - train_node_set)
        self.unlabeled_node_indices = torch.LongTensor(unlabeled_nodes).to(self.device)

    def initialize_model_components(self):

        predictor_base = deepcopy(self.base_model_class)
        main_model_base = deepcopy(self.base_model_class)
        
        self.node_predictor = self.create_gnn_wrapper(predictor_base).to(self.device)
        self.main_model = self.create_gnn_wrapper(main_model_base).to(self.device)
        self.edge_weight_estimator = self.create_edge_weight_estimator(
            self.node_features.shape[1], self.edge_hidden_dim, self.edge_hidden_dim
        ).to(self.device)
        
        self.potential_edge_index = self.generate_potential_edges()
        
        all_parameters = (list(self.main_model.parameters()) + 
                         list(self.edge_weight_estimator.parameters()) + 
                         list(self.node_predictor.parameters()))
        
        self.optimizer = optim.Adam(
            all_parameters, 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )

    def generate_potential_edges(self):
        if self.num_potential_edges <= 0:
            return None
            
        num_nodes = self.node_features.shape[0]
        max_neighbors = min(self.num_potential_edges, num_nodes - 1)
        
        # Normalize features
        normalized_features = F.normalize(self.node_features, p=2, dim=1)
        
        potential_edges = []
        
        for node_idx in range(num_nodes):
            # Compute similarities
            similarities = torch.mm(normalized_features[node_idx:node_idx+1], normalized_features.t()).squeeze()
            
            similarities[node_idx] = -1
            _, top_similar_nodes = similarities.topk(max_neighbors)
            
            # Find existing neighbors
            existing_neighbors = set(self.original_edge_index[1][self.original_edge_index[0] == node_idx].cpu().numpy())
            
            # Add edges to non-existing neighbors
            for similar_node in top_similar_nodes:
                similar_node_idx = similar_node.item()
                if similar_node_idx not in existing_neighbors:
                    potential_edges.append([node_idx, similar_node_idx])
        
        if not potential_edges:
            return None
            
        potential_edges_tensor = torch.tensor(potential_edges, dtype=torch.long, device=self.device).t()
        potential_edges_tensor = to_undirected(potential_edges_tensor, num_nodes=num_nodes)
        
        return potential_edges_tensor

    def identify_confident_edges(self, prediction_probabilities):
        #Identify edges to confident unlabeled nodes
        if len(self.unlabeled_node_indices) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), []

        max_probabilities = prediction_probabilities.max(dim=1)[0]
        confident_mask = max_probabilities[self.unlabeled_node_indices] > self.confidence_threshold
        confident_nodes = self.unlabeled_node_indices[confident_mask]
        
        if len(confident_nodes) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), confident_nodes.cpu().numpy()

        num_unlabeled = len(self.unlabeled_node_indices)
        num_confident = len(confident_nodes)
        
        if num_unlabeled == 0 or num_confident == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), confident_nodes.cpu().numpy()
        
        # Create all pairs between unlabeled and confident nodes
        source_nodes = self.unlabeled_node_indices.repeat(num_confident)
        target_nodes = confident_nodes.repeat(num_unlabeled)
        
        valid_connections_mask = source_nodes != target_nodes
        
        if valid_connections_mask.sum() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), confident_nodes.cpu().numpy()
        
        confident_edge_index = torch.stack([source_nodes[valid_connections_mask], 
                                          target_nodes[valid_connections_mask]], dim=0)
        
        return confident_edge_index, confident_nodes.cpu().numpy()

    def train_single_epoch(self, epoch, train_indices, validation_indices):
        start_time = time.time()
        
        self.main_model.train()
        self.node_predictor.train()
        self.edge_weight_estimator.train()
        
        self.optimizer.zero_grad()
        
        # Get node representations and reconstruction loss
        edge_weights = torch.ones(self.original_edge_index.shape[1], device=self.device, dtype=torch.float32)
        node_representations = self.edge_weight_estimator.forward(self.node_features, self.original_edge_index, edge_weights)
        reconstruction_loss = self.compute_reconstruction_loss(self.original_edge_index, node_representations)
        
        if self.potential_edge_index is not None and self.potential_edge_index.shape[1] > 0:
            potential_weights = self.compute_estimated_edge_weights(self.potential_edge_index, node_representations)
            predictor_edges = torch.cat([self.original_edge_index, self.potential_edge_index], dim=1)
            predictor_weights = torch.cat([
                torch.ones(self.original_edge_index.shape[1], device=self.device), 
                potential_weights
            ], dim=0)
        else:
            predictor_edges = self.original_edge_index
            predictor_weights = torch.ones(self.original_edge_index.shape[1], device=self.device)

        predictor_logits = self.node_predictor.forward(self.node_features, predictor_edges, predictor_weights)
        
        if self.best_predictions is None:
            with torch.no_grad():
                prediction_probs = F.softmax(predictor_logits, dim=1)
                self.best_predictions = prediction_probs.detach()
                self.confident_edge_index, self.confident_node_indices = self.identify_confident_edges(self.best_predictions)
        
        if self.confident_edge_index is not None and self.confident_edge_index.shape[1] > 0:
            confident_weights = self.compute_estimated_edge_weights(self.confident_edge_index, node_representations)
            main_model_weights = torch.cat([predictor_weights, confident_weights], dim=0)
            main_model_edges = torch.cat([predictor_edges, self.confident_edge_index], dim=1)
        else:
            main_model_weights = predictor_weights
            main_model_edges = predictor_edges

        main_model_output = self.main_model.forward(self.node_features, main_model_edges, main_model_weights)
        
        # Compute losses
        predictor_loss = F.cross_entropy(predictor_logits[train_indices], self.node_labels[train_indices])
        main_model_loss = F.cross_entropy(main_model_output[train_indices], self.node_labels[train_indices])
        
        # Consistency loss for confident nodes
        if len(self.confident_node_indices) > 0:
            main_model_probs = F.softmax(main_model_output, dim=1)
            main_model_probs = torch.clamp(main_model_probs, 1e-8, 1 - 1e-8)
            
            consistency_loss = F.kl_div(
                torch.log(main_model_probs[self.confident_node_indices]), 
                self.best_predictions[self.confident_node_indices], 
                reduction='batchmean'
            )
        else:
            consistency_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Total loss
        total_loss = (main_model_loss + predictor_loss + 
                     self.reconstruction_weight * reconstruction_loss + 
                     self.consistency_weight * consistency_loss)

        if torch.isnan(total_loss):
            print(f"NaN detected! Skipping epoch {epoch}")
            return

        total_loss.backward()
        self.optimizer.step()

        self.evaluate_epoch(epoch, train_indices, validation_indices, 
                          predictor_edges, predictor_weights, 
                          main_model_edges, main_model_weights, 
                          total_loss, start_time)

    def evaluate_epoch(self, epoch, train_indices, validation_indices, 
                      predictor_edges, predictor_weights,
                      main_model_edges, main_model_weights, 
                      total_loss, start_time):

        self.main_model.eval()
        self.node_predictor.eval()
        self.edge_weight_estimator.eval()
        
        with torch.no_grad():

            predictor_output = self.node_predictor.forward(self.node_features, predictor_edges, predictor_weights)
            predictor_probabilities = F.softmax(predictor_output, dim=1)

            main_model_output = self.main_model.forward(self.node_features, main_model_edges, main_model_weights.detach())
            
            # Compute metrics
            train_loss = F.cross_entropy(main_model_output[train_indices], self.node_labels[train_indices]).item()
            validation_loss = F.cross_entropy(main_model_output[validation_indices], self.node_labels[validation_indices]).item()
            
            train_accuracy = self.compute_accuracy(main_model_output[train_indices], self.node_labels[train_indices])
            validation_accuracy = self.compute_accuracy(main_model_output[validation_indices], self.node_labels[validation_indices])
            predictor_validation_accuracy = self.compute_accuracy(predictor_probabilities[validation_indices], self.node_labels[validation_indices])
            
            train_f1 = f1_score(self.node_labels[train_indices].cpu(), main_model_output[train_indices].argmax(dim=1).cpu(), average='macro')
            validation_f1 = f1_score(self.node_labels[validation_indices].cpu(), main_model_output[validation_indices].argmax(dim=1).cpu(), average='macro')

            # Compute oversmoothing metrics
            train_oversmoothing = self.compute_oversmoothing_for_node_set(main_model_output, main_model_edges, train_indices)
            validation_oversmoothing = self.compute_oversmoothing_for_node_set(main_model_output, main_model_edges, validation_indices)
            
            if train_oversmoothing is not None:
                self.oversmoothing_metrics_history['train'].append(train_oversmoothing)
            if validation_oversmoothing is not None:
                self.oversmoothing_metrics_history['val'].append(validation_oversmoothing)

            if self.debug_mode and epoch%20 == 0:
                train_metrics = train_oversmoothing if train_oversmoothing else {}
                val_metrics = validation_oversmoothing if validation_oversmoothing else {}
                
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Val Loss: {validation_loss:.4f} | "
                      f"Train Acc: {train_accuracy:.4f}, Val Acc: {validation_accuracy:.4f} | "
                      f"Train F1: {train_f1:.4f}, Val F1: {validation_f1:.4f}")
                print(f"Train DE: {train_metrics.get('EDir', 0.0):.4f}, Val DE: {val_metrics.get('EDir', 0.0):.4f} | "
                      f"Train DE_trad: {train_metrics.get('EDir_traditional', 0.0):.4f}, Val DE_trad: {val_metrics.get('EDir_traditional', 0.0):.4f} | "
                      f"Train EProj: {train_metrics.get('EProj', 0.0):.4f}, Val EProj: {val_metrics.get('EProj', 0.0):.4f} | "
                      f"Train MAD: {train_metrics.get('MAD', 0.0):.4f}, Val MAD: {val_metrics.get('MAD', 0.0):.4f} | "
                      f"Train NumRank: {train_metrics.get('NumRank', 0.0):.4f}, Val NumRank: {val_metrics.get('NumRank', 0.0):.4f} | "
                      f"Train Erank: {train_metrics.get('Erank', 0.0):.4f}, Val Erank: {val_metrics.get('Erank', 0.0):.4f}")
                print(f"  Pred Val Acc: {predictor_validation_accuracy:.4f} | Add Nodes: {len(self.confident_node_indices)} | "
                      f"Time: {time.time() - start_time:.2f}s")

            # Update best models
            if predictor_validation_accuracy > self.best_predictor_accuracy:
                self.best_predictor_accuracy = predictor_validation_accuracy
                self.best_predictor_edge_weights = predictor_weights.detach()
                self.best_predictions = predictor_probabilities.detach()
                self.best_predictor_weights = deepcopy(self.node_predictor.state_dict())
                
                self.confident_edge_index, self.confident_node_indices = self.identify_confident_edges(self.best_predictions)

            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                self.best_edge_weights = main_model_weights.detach()
                self.best_edge_indices = main_model_edges
                self.best_main_weights = deepcopy(self.main_model.state_dict())
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # Check early stopping
            if self.early_stopping_counter >= self.patience:
                if self.debug_mode:
                    print(f"Early stopping at epoch {epoch+1}, best val loss: {self.best_validation_loss:.4f}")
                raise StopIteration

    def load_best_model_weights(self):

        print("Loading best model according to validation performance")
        if self.best_main_weights is not None:
            self.main_model.load_state_dict(self.best_main_weights)
        if self.best_predictor_weights is not None:
            self.node_predictor.load_state_dict(self.best_predictor_weights)

    def fit(self, features, adjacency_matrix, labels, train_indices, validation_indices):

        # Prepare data
        self.prepare_training_data(features, adjacency_matrix, labels, train_indices)
        
        # Initialize models
        self.initialize_model_components()
        
        # Training loop
        start_time = time.time()
        try:
            for epoch in range(self.max_epochs):
                self.train_single_epoch(epoch, train_indices, validation_indices)
        except StopIteration:
            if self.debug_mode:
                print(f"Early stopping at epoch {epoch+1}")

        total_training_time = time.time() - start_time
        print(f"\nTraining completed in {total_training_time:.2f}s")
        
        # Load best models
        self.load_best_model_weights()

    def test(self, test_indices):

        self.main_model.eval()
        self.node_predictor.eval()
        
        with torch.no_grad():

            if self.best_predictor_edge_weights is not None and self.potential_edge_index is not None:
                predictor_edges = torch.cat([self.original_edge_index, self.potential_edge_index], dim=1)
                predictor_output = self.node_predictor.forward(self.node_features, predictor_edges, self.best_predictor_edge_weights)
                predictor_probabilities = F.softmax(predictor_output, dim=1)
                
                predictor_accuracy = self.compute_accuracy(predictor_probabilities[test_indices], self.node_labels[test_indices])
                y_true = self.node_labels[test_indices].cpu().numpy()
                y_pred = predictor_probabilities[test_indices].argmax(dim=1).cpu().numpy()
                predictor_f1 = f1_score(y_true, y_pred, average='macro')
                
                print(f"Predictor Test Acc: {predictor_accuracy:.4f} | Test F1: {predictor_f1:.4f}")

            if self.best_edge_weights is not None and self.best_edge_indices is not None:
                main_model_output = self.main_model.forward(self.node_features, self.best_edge_indices, self.best_edge_weights)
                
                test_loss = F.cross_entropy(main_model_output[test_indices], self.node_labels[test_indices]).item()
                test_accuracy = self.compute_accuracy(main_model_output[test_indices], self.node_labels[test_indices])
                
                y_true = self.node_labels[test_indices].cpu().numpy()
                y_pred = main_model_output[test_indices].argmax(dim=1).cpu().numpy()
                test_f1 = f1_score(y_true, y_pred, average='macro')
                test_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                test_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                
                test_oversmoothing = self.compute_oversmoothing_for_node_set(main_model_output, self.best_edge_indices, test_indices)
                if test_oversmoothing is not None:
                    self.oversmoothing_metrics_history['test'].append(test_oversmoothing)
                
                final_test_metrics = {
                    'test_loss': test_loss,
                    'test_acc': test_accuracy,
                    'test_f1': test_f1,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_oversmoothing': test_oversmoothing if test_oversmoothing is not None else {
                        'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                        'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0
                    }
                }
                
                print(f"Test Loss: {final_test_metrics['test_loss']:.4f} | Test Acc: {final_test_metrics['test_acc']:.4f} | Test F1: {final_test_metrics['test_f1']:.4f}")
                print(f"Test Precision: {final_test_metrics['test_precision']:.4f} | Test Recall: {final_test_metrics['test_recall']:.4f}")
                
                print("Test Oversmoothing Metrics:")
                if test_oversmoothing is not None:
                    print(f"Test: EDir: {test_oversmoothing['EDir']:.4f}, EDir_traditional: {test_oversmoothing['EDir_traditional']:.4f}, "
                        f"EProj: {test_oversmoothing['EProj']:.4f}, MAD: {test_oversmoothing['MAD']:.4f}, "
                        f"NumRank: {test_oversmoothing['NumRank']:.4f}, Erank: {test_oversmoothing['Erank']:.4f}")
                
                return final_test_metrics
        
        return {
            'test_acc': 0.0,
            'test_f1': 0.0,
            'test_precision': 0.0,
            'test_recall': 0.0,
            'test_oversmoothing': {
                'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0
            }
        }

    def get_oversmoothing_metrics_history(self):
        return self.oversmoothing_metrics_history