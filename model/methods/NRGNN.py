import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, negative_sampling, to_scipy_sparse_matrix
from torch_geometric.nn import GCNConv
from collections import defaultdict

from model.evaluation import (OversmoothingMetrics, ClassificationMetrics,
                              compute_oversmoothing_for_mask, evaluate_model,
                              DEFAULT_OVERSMOOTHING)
from model.base import BaseTrainer
from model.registry import register

class NRGNN:

    def __init__(self, config, device, base_model=None):
        self.device = device
        self.config = config
        self.base_model_class = base_model
        
        # Training parameters
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
        self.oversmoothing_every = config.get('oversmoothing_every', 20)

        # Model components
        self.node_predictor = None
        self.main_model = None
        self.edge_weight_estimator = None
        self.adjacency_estimator = None
        self.optimizer = None
        
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
        
        self.original_edge_index = None
        self.potential_edge_index = None
        self.confident_edge_index = None
        self.confident_node_indices = []
        self.unlabeled_node_indices = None

        self.node_features = None
        self.node_labels = None
        
        # Evaluation
        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        self.cls_evaluator = ClassificationMetrics(average='macro')
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
        sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_matrix.data)
        shape = torch.Size(sparse_matrix.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def create_edge_weight_estimator(self, input_features, hidden_dim, output_dim):
        # Create GCN-based edge weight estimator
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
        ).to(self.device)
        
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

    def _indices_to_mask(self, indices, num_nodes):
        """Convert index array to boolean mask."""
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        if isinstance(indices, np.ndarray):
            indices = torch.tensor(indices, device=self.device)
        elif not isinstance(indices, torch.Tensor):
            indices = torch.tensor(list(indices), device=self.device)
        mask[indices] = True
        return mask

    def prepare_training_data(self, features, adjacency_matrix, labels, train_indices):
        self.original_edge_index, _ = from_scipy_sparse_matrix(adjacency_matrix)
        self.original_edge_index = self.original_edge_index.to(self.device)
        
        if sp.issparse(features):
            features = self.convert_sparse_to_torch_tensor(features).to_dense()
        else:
            if sp.issparse(features):
                features = self.convert_sparse_to_torch_tensor(features).to_dense()
            else:
                if isinstance(features, torch.Tensor):
                    features = features.float()
                else:
                    features = torch.FloatTensor(np.array(features))

        self.node_features = features.to(self.device)

        if isinstance(labels, torch.Tensor):
            self.node_labels = labels.long().to(self.device)
        else:
            self.node_labels = torch.LongTensor(np.array(labels)).to(self.device)

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
                self.best_predictions = prediction_probs.detach().to(self.device)
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

        metrics = self.evaluate_epoch(epoch, train_indices, validation_indices, 
                          predictor_edges, predictor_weights, 
                          main_model_edges, main_model_weights, 
                          total_loss, start_time)
        return metrics

    def evaluate_epoch(self, epoch, train_indices, validation_indices, 
                      predictor_edges, predictor_weights,
                      main_model_edges, main_model_weights, 
                      total_loss, start_time):
        logging = False
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
                
            
            if self.debug_mode and epoch % self.oversmoothing_every == 0:

                train_f1 = self.cls_evaluator.compute_f1(main_model_output[train_indices].argmax(dim=1), self.node_labels[train_indices])
                validation_f1 = self.cls_evaluator.compute_f1(main_model_output[validation_indices].argmax(dim=1), self.node_labels[validation_indices])

                # Compute oversmoothing metrics
                train_mask = self._indices_to_mask(train_indices, self.node_features.shape[0])
                val_mask = self._indices_to_mask(validation_indices, self.node_features.shape[0])
                train_oversmoothing = compute_oversmoothing_for_mask(self.oversmoothing_evaluator, main_model_output, main_model_edges, train_mask)
                validation_oversmoothing = compute_oversmoothing_for_mask(self.oversmoothing_evaluator, main_model_output, main_model_edges, val_mask)
                
                if train_oversmoothing is not None:
                    self.oversmoothing_metrics_history['train'].append(train_oversmoothing)
                if validation_oversmoothing is not None:
                    self.oversmoothing_metrics_history['val'].append(validation_oversmoothing)

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

                logging = True

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

            # Early stopping
            if self.early_stopping_counter >= self.patience:
                if self.debug_mode:
                    print(f"Early stopping at epoch {epoch+1}, best val loss: {self.best_validation_loss:.4f}")
                raise StopIteration
        if logging:
            return train_metrics, val_metrics

    def load_best_model_weights(self):

        print("Loading best model according to validation performance")
        if self.best_main_weights is not None:
            self.main_model.load_state_dict(self.best_main_weights)
        if self.best_predictor_weights is not None:
            self.node_predictor.load_state_dict(self.best_predictor_weights)

    def fit(self, features, adjacency_matrix, labels, train_indices, validation_indices):

        self.prepare_training_data(features, adjacency_matrix, labels, train_indices)
        self.train_indices = train_indices
        self.validation_indices = validation_indices

        self.initialize_model_components()
        per_epochs_oversmoothing = defaultdict(list)
        per_epochs_val_oversmoothing = defaultdict(list)

        start_time = time.time()
        try:
            for epoch in range(self.max_epochs):
                results = self.train_single_epoch(epoch, train_indices, validation_indices)
                if results is not None:
                    train_metrics, val_metrics = results
                    for key, value in train_metrics.items():
                        per_epochs_oversmoothing[key].append(value)
                    for key, value in val_metrics.items():
                        per_epochs_val_oversmoothing[key].append(value)
        except StopIteration:
            if self.debug_mode:
                print(f"Early stopping at epoch {epoch+1}")

        total_training_time = time.time() - start_time
        print(f"\nTraining completed in {total_training_time:.2f}s")

        self.load_best_model_weights()
        return per_epochs_oversmoothing, per_epochs_val_oversmoothing

    def test(self, test_indices):
        self.main_model.eval()
        self.node_predictor.eval()

        with torch.no_grad():
            if self.best_edge_weights is not None and self.best_edge_indices is not None:
                num_nodes = self.node_features.shape[0]
                train_mask = self._indices_to_mask(self.train_indices, num_nodes)
                val_mask = self._indices_to_mask(self.validation_indices, num_nodes)
                test_mask = self._indices_to_mask(test_indices, num_nodes)

                def _get_predictions():
                    return self.main_model.forward(
                        self.node_features, self.best_edge_indices, self.best_edge_weights
                    ).argmax(dim=1)

                def _get_embeddings():
                    return self.main_model.forward(
                        self.node_features, self.best_edge_indices, self.best_edge_weights
                    )

                results = evaluate_model(
                    _get_predictions, _get_embeddings, self.node_labels,
                    train_mask, val_mask, test_mask,
                    self.best_edge_indices, self.device
                )

                print(f"Test Acc: {results['test_cls']['accuracy']:.4f} | Test F1: {results['test_cls']['f1']:.4f} | "
                      f"Precision: {results['test_cls']['precision']:.4f}, Recall: {results['test_cls']['recall']:.4f}")
                print(f"Test Oversmoothing: {results['test_oversmoothing']}")

                return results

            _zero_cls = {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
            return {
                'test_cls': dict(_zero_cls),
                'train_cls': dict(_zero_cls),
                'val_cls': dict(_zero_cls),
                'test_oversmoothing': dict(DEFAULT_OVERSMOOTHING),
                'train_oversmoothing_final': dict(DEFAULT_OVERSMOOTHING),
                'val_oversmoothing_final': dict(DEFAULT_OVERSMOOTHING),
            }


@register('nrgnn')
class NRGNNMethodTrainer(BaseTrainer):
    supports_eval_only = False

    def train(self):
        d = self.init_data
        nrgnn_config = {
            'lr': d['lr'],
            'weight_decay': d['weight_decay'],
            'epochs': d['epochs'],
            'patience': d['patience'],
            'nrgnn_params': self.config.get('nrgnn_params', {}),
            'oversmoothing_every': d['oversmoothing_every'],
        }

        self._nrgnn = NRGNN(nrgnn_config, d['device'], base_model=d['backbone_model'])

        adj = to_scipy_sparse_matrix(
            d['data_for_training'].edge_index,
            num_nodes=d['data_for_training'].x.size(0),
        )
        train_idx = d['train_mask'].nonzero(as_tuple=True)[0].cpu().numpy()
        val_idx = d['val_mask'].nonzero(as_tuple=True)[0].cpu().numpy()
        self._test_idx = d['test_mask'].nonzero(as_tuple=True)[0].cpu().numpy()

        train_oversmoothing, val_oversmoothing = self._nrgnn.fit(
            d['data_for_training'].x.to(d['device']),
            adj,
            d['data_for_training'].y.to(d['device']),
            train_idx, val_idx,
        )
        return {
            'train_oversmoothing': dict(train_oversmoothing),
            'val_oversmoothing': dict(val_oversmoothing),
        }

    def evaluate(self):
        return self._nrgnn.test(self._test_idx)
