import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix
import scipy.sparse as sp
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data

from model.evaluation import OversmoothingMetrics


class GNNGuardTrainer:

    def __init__(self, input_features, hidden_channels, num_classes, dropout=0.5, lr=0.01, 
                 weight_decay=5e-4, attention=True, device=None, similarity_threshold=0.5, 
                 num_layers=2, attention_dim=16, data_for_training=None, backbone=None):

        self.device = device
        self.backbone = backbone
        self.model = None
        self.input_features = input_features
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout
        self.learning_rate = lr
        self.weight_decay_coeff = weight_decay if attention else 0
        self.use_attention = attention
        
        # GNNGuard specific parameters
        self.similarity_threshold = similarity_threshold
        self.num_layers = num_layers
        self.attention_dim = attention_dim
        
        self.model_output = None
        self.best_model_state = None
        self.best_output = None
        self.normalized_adjacency = None
        self.node_features = None
        self.node_labels = None
        
        # Oversmoothing evaluation
        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        self.oversmoothing_metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }
        
        self._initialize_model()
        self.data_for_training = data_for_training
    
    def _initialize_model(self):

        self.model = GNNGuardModel(
            input_features=self.input_features,
            hidden_channels=self.hidden_channels,
            num_classes=self.num_classes,
            dropout=self.dropout_rate,
            similarity_threshold=self.similarity_threshold,
            num_layers=self.num_layers,
            attention_dim=self.attention_dim,
            device=self.device,
            backbone=self.backbone
        ).to(self.device)

    def _compute_mask_oversmoothing_metrics(self, node_embeddings, edge_index, node_mask, labels=None):

        try:
            masked_node_indices = torch.where(node_mask)[0]
            masked_embeddings = node_embeddings[node_mask]
            
            # Filter edges
            masked_node_set = set(masked_node_indices.cpu().numpy())
            edge_filter = torch.tensor([
                source.item() in masked_node_set and target.item() in masked_node_set
                for source, target in edge_index.t()
            ], device=edge_index.device)
            
            if not edge_filter.any():

                return {
                    'NumRank': float(min(masked_embeddings.shape)),
                    'Erank': float(min(masked_embeddings.shape)),
                    'EDir': 0.0,
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0
                }
            
            filtered_edges = edge_index[:, edge_filter]
            local_node_mapping = {orig_idx.item(): local_idx 
                                for local_idx, orig_idx in enumerate(masked_node_indices)}
            
            remapped_edge_index = torch.stack([
                torch.tensor([local_node_mapping[src.item()] for src in filtered_edges[0]], 
                           device=edge_index.device),
                torch.tensor([local_node_mapping[tgt.item()] for tgt in filtered_edges[1]], 
                           device=edge_index.device)
            ])
            
            # Prepare data for oversmoothing evaluation
            graph_data_for_class = [{
                'X': masked_embeddings,
                'edge_index': remapped_edge_index,
                'edge_weight': None
            }]
            
            return self.oversmoothing_evaluator.compute_all_metrics(
                X=masked_embeddings,
                edge_index=remapped_edge_index,
                graphs_in_class=graph_data_for_class
            )
            
        except Exception as e:
            print(f"Warning: Could not compute oversmoothing metrics for mask: {e}")
            return None
        
    def prepare_data(self):
        # adjacency
        self.adjacency_matrix = to_scipy_sparse_matrix(
            self.data_for_training.edge_index, num_nodes=self.data_for_training.num_nodes
        )

        # split train/val/test
        if hasattr(self.data_for_training, "train_mask"):
            self.train_indices = self.data_for_training.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            self.val_indices = self.data_for_training.val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            self.test_indices = self.data_for_training.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        else:
            total_nodes = self.data_for_training.num_nodes
            self.train_indices = np.arange(min(140, total_nodes // 5))
            self.val_indices = np.arange(len(self.train_indices),
                                         min(len(self.train_indices) + 500, total_nodes // 2))
            self.test_indices = np.arange(max(len(self.train_indices) + len(self.val_indices), total_nodes // 2),
                                          total_nodes)
    
    def _convert_sparse_matrix_to_torch_tensor(self, sparse_matrix):
        # Convert scipy sparse matrix to PyTorch sparse tensor
        sparse_coo = sparse_matrix.tocoo().astype(np.float32)
        indices_tensor = torch.from_numpy(
            np.vstack((sparse_coo.row, sparse_coo.col)).astype(np.int64)
        ).to(self.device)
        values_tensor = torch.from_numpy(sparse_coo.data).to(self.device)
        shape_tensor = torch.Size(sparse_coo.shape)
        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape_tensor, 
                                     dtype=torch.float32, device=self.device)
    
    def _add_self_loops_to_sparse_tensor(self, adjacency_tensor, loop_weight=1):

        num_nodes = adjacency_tensor.shape[0]
        diagonal_indices = torch.arange(num_nodes, dtype=torch.int64)
        self_loop_indices = torch.stack((diagonal_indices, diagonal_indices), dim=0)
        self_loop_values = torch.ones(num_nodes, dtype=torch.float32)
        identity_tensor = torch.sparse.FloatTensor(
            self_loop_indices, self_loop_values, adjacency_tensor.shape
        ).to(self.device)
        return adjacency_tensor + identity_tensor
    
    def _reset_model_parameters(self):
        if hasattr(self.model, 'gcn_layers'):
            for gcn_layer in self.model.gcn_layers:
                gcn_layer.reset_parameters()
        elif hasattr(self.model, 'initialize'):
            self.model.initialize()

    def train_model(self, node_features=None, node_labels=None, max_epochs=200, verbose=True, patience=5):
        self._reset_model_parameters()

        if node_features is not None:
            self.node_features = node_features.to(self.device)
        if node_labels is not None:
            self.node_labels = node_labels.to(self.device)

        self.normalized_adjacency = self._add_self_loops_to_sparse_tensor(
            self._convert_sparse_matrix_to_torch_tensor(self.adjacency_matrix)
        )

        self._train_with_early_stopping(
            self.node_labels,
            self.train_indices,
            self.val_indices,
            self.test_indices,
            max_epochs,
            patience,
            verbose
        )
    
    def _train_with_early_stopping(self, labels, train_indices, val_indices, test_indices, 
                                  max_epochs, patience, verbose):

        if verbose:
            print('Training GNNGuard model')
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, 
                             weight_decay=self.weight_decay_coeff)
        
        # Early stopping
        best_validation_loss = float('inf')
        patience_counter = patience
        best_model_weights = deepcopy(self.model.state_dict())
        
        for epoch in range(max_epochs):

            self.model.train()
            optimizer.zero_grad()
            
            train_output = self.model(self.node_features, self.normalized_adjacency, 
                                    use_attention=self.use_attention)
            train_loss = F.nll_loss(train_output[train_indices], labels[train_indices])
            train_loss.backward()
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(self.node_features, self.normalized_adjacency, 
                                      use_attention=self.use_attention)
                val_loss = F.nll_loss(val_output[val_indices], labels[val_indices])
                
                train_predictions = train_output[train_indices].max(1)[1].cpu().numpy()
                val_predictions = val_output[val_indices].max(1)[1].cpu().numpy()
                train_true_labels = labels[train_indices].cpu().numpy()
                val_true_labels = labels[val_indices].cpu().numpy()
                
                train_accuracy = accuracy_score(train_true_labels, train_predictions)
                val_accuracy = accuracy_score(val_true_labels, val_predictions)
                train_f1_score = f1_score(train_true_labels, train_predictions, average='macro')
                val_f1_score = f1_score(val_true_labels, val_predictions, average='macro')
                
                # Compute oversmoothing metrics
                train_node_mask = torch.zeros(self.node_features.size(0), dtype=torch.bool, device=self.device)
                val_node_mask = torch.zeros(self.node_features.size(0), dtype=torch.bool, device=self.device)
                train_node_mask[train_indices] = True
                val_node_mask[val_indices] = True
                
                edge_connectivity = self.normalized_adjacency._indices()
                if epoch%20 == 0:
                    train_oversmoothing_metrics = self._compute_mask_oversmoothing_metrics(
                        train_output, edge_connectivity, train_node_mask, labels)
                    val_oversmoothing_metrics = self._compute_mask_oversmoothing_metrics(
                        val_output, edge_connectivity, val_node_mask, labels)
                
                if train_oversmoothing_metrics is not None:
                    self.oversmoothing_metrics_history['train'].append(train_oversmoothing_metrics)
                if val_oversmoothing_metrics is not None:
                    self.oversmoothing_metrics_history['val'].append(val_oversmoothing_metrics)
            
            if verbose:
                self._log_training_progress(epoch, train_loss, val_loss, train_accuracy, val_accuracy,
                                          train_f1_score, val_f1_score, train_oversmoothing_metrics,
                                          val_oversmoothing_metrics)
            
            # Early stopping
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                self.model_output = val_output
                best_model_weights = deepcopy(self.model.state_dict())
                patience_counter = patience
            else:
                patience_counter -= 1
            
            if patience_counter <= 0:
                if verbose:
                    print(f'Early stopping at epoch {epoch}, best_val_loss={best_validation_loss:.4f}')
                break
        
        self.model.load_state_dict(best_model_weights)
        
        if test_indices is not None:
            self._evaluate_final_test_performance(train_indices, val_indices, test_indices, verbose)
    
    def _log_training_progress(self, epoch, train_loss, val_loss, train_acc, val_acc, 
                             train_f1, val_f1, train_oversmoothing, val_oversmoothing):
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
              f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        if epoch%20 == 0:
            train_edir = train_oversmoothing['EDir'] if train_oversmoothing else 0.0
            train_edir_trad = train_oversmoothing['EDir_traditional'] if train_oversmoothing else 0.0
            train_eproj = train_oversmoothing['EProj'] if train_oversmoothing else 0.0
            train_mad = train_oversmoothing['MAD'] if train_oversmoothing else 0.0
            train_numrank = train_oversmoothing['NumRank'] if train_oversmoothing else 0.0
            train_erank = train_oversmoothing['Erank'] if train_oversmoothing else 0.0
            
            val_edir = val_oversmoothing['EDir'] if val_oversmoothing else 0.0
            val_edir_trad = val_oversmoothing['EDir_traditional'] if val_oversmoothing else 0.0
            val_eproj = val_oversmoothing['EProj'] if val_oversmoothing else 0.0
            val_mad = val_oversmoothing['MAD'] if val_oversmoothing else 0.0
            val_numrank = val_oversmoothing['NumRank'] if val_oversmoothing else 0.0
            val_erank = val_oversmoothing['Erank'] if val_oversmoothing else 0.0
            print(f"Train EDir: {train_edir:.4f}, Val EDir: {val_edir:.4f} | "
                  f"Train EDir_trad: {train_edir_trad:.4f}, Val EDir_trad: {val_edir_trad:.4f} | "
                  f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                  f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                  f"Train NumRank: {train_numrank:.4f}, Val NumRank: {val_numrank:.4f} | "
                  f"Train Erank: {train_erank:.4f}, Val Erank: {val_erank:.4f}")
    
    def _evaluate_final_test_performance(self, train_indices, val_indices, test_indices, verbose):

        test_node_mask = torch.zeros(self.node_features.size(0), dtype=torch.bool, device=self.device)
        train_node_mask = torch.zeros(self.node_features.size(0), dtype=torch.bool, device=self.device)
        val_node_mask = torch.zeros(self.node_features.size(0), dtype=torch.bool, device=self.device)
        
        train_node_mask[train_indices] = True
        val_node_mask[val_indices] = True
        test_node_mask[test_indices] = True
        
        self.model.eval()
        with torch.no_grad():
            final_output = self.model(self.node_features, self.normalized_adjacency, 
                                    use_attention=self.use_attention)
            
            test_predictions = final_output[test_indices].max(1)[1].cpu().numpy()
            test_true_labels = self.node_labels[test_indices].cpu().numpy()
            test_accuracy = accuracy_score(test_true_labels, test_predictions)
            test_f1_score = f1_score(test_true_labels, test_predictions, average='macro')
            test_loss = F.nll_loss(final_output[test_indices], self.node_labels[test_indices])
            
            edge_connectivity = self.normalized_adjacency._indices()
            final_train_oversmoothing = self._compute_mask_oversmoothing_metrics(
                final_output, edge_connectivity, train_node_mask, self.node_labels)
            final_val_oversmoothing = self._compute_mask_oversmoothing_metrics(
                final_output, edge_connectivity, val_node_mask, self.node_labels)
            final_test_oversmoothing = self._compute_mask_oversmoothing_metrics(
                final_output, edge_connectivity, test_node_mask, self.node_labels)
            
            if final_test_oversmoothing is not None:
                self.oversmoothing_metrics_history['test'].append(final_test_oversmoothing)
        
        if verbose:
            print(f"\nTraining completed")
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | Test F1: {test_f1_score:.4f}")
            print("Final Oversmoothing Metrics:")
            
            self._log_final_oversmoothing_metrics(final_train_oversmoothing, 
                                                final_val_oversmoothing, 
                                                final_test_oversmoothing)
    
    def _log_final_oversmoothing_metrics(self, train_metrics, val_metrics, test_metrics):

        if train_metrics is not None:
            print(f"Train: EDir: {train_metrics['EDir']:.4f}, EDir_traditional: {train_metrics['EDir_traditional']:.4f}, "
                  f"EProj: {train_metrics['EProj']:.4f}, MAD: {train_metrics['MAD']:.4f}, "
                  f"NumRank: {train_metrics['NumRank']:.4f}, Erank: {train_metrics['Erank']:.4f}")
        
        if val_metrics is not None:
            print(f"Val: EDir: {val_metrics['EDir']:.4f}, EDir_traditional: {val_metrics['EDir_traditional']:.4f}, "
                  f"EProj: {val_metrics['EProj']:.4f}, MAD: {val_metrics['MAD']:.4f}, "
                  f"NumRank: {val_metrics['NumRank']:.4f}, Erank: {val_metrics['Erank']:.4f}")
        
        if test_metrics is not None:
            print(f"Test: EDir: {test_metrics['EDir']:.4f}, EDir_traditional: {test_metrics['EDir_traditional']:.4f}, "
                  f"EProj: {test_metrics['EProj']:.4f}, MAD: {test_metrics['MAD']:.4f}, "
                  f"NumRank: {test_metrics['NumRank']:.4f}, Erank: {test_metrics['Erank']:.4f}")
    
    def evaluate_model(self):
        test_indices = self.test_indices

        self.model.eval()
        with torch.no_grad():
            model_output = self.model(self.node_features, self.normalized_adjacency, 
                                    use_attention=self.use_attention)
            test_loss = F.nll_loss(model_output[test_indices], self.node_labels[test_indices])
            
            test_predictions = model_output[test_indices].max(1)[1].cpu().numpy()
            test_true_labels = self.node_labels[test_indices].cpu().numpy()
            
            test_accuracy = accuracy_score(test_true_labels, test_predictions)
            test_f1_score = f1_score(test_true_labels, test_predictions, average='macro')
            test_precision = precision_score(test_true_labels, test_predictions, average='macro')
            test_recall = recall_score(test_true_labels, test_predictions, average='macro')
            
            # Oversmoothing metrics
            test_node_mask = torch.zeros(self.node_features.size(0), dtype=torch.bool, device=self.device)
            test_node_mask[test_indices] = True
            edge_connectivity = self.normalized_adjacency._indices()
            test_oversmoothing_metrics = self._compute_mask_oversmoothing_metrics(
                model_output, edge_connectivity, test_node_mask, self.node_labels)

        print(f"GNNGuard Training completed!")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1: {test_f1_score:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        if test_oversmoothing_metrics is not None:
            print(f"Test Oversmoothing: EDir: {test_oversmoothing_metrics['EDir']:.4f}, "
                  f"EDir_traditional: {test_oversmoothing_metrics['EDir_traditional']:.4f}, "
                  f"EProj: {test_oversmoothing_metrics['EProj']:.4f}, MAD: {test_oversmoothing_metrics['MAD']:.4f}, "
                  f"NumRank: {test_oversmoothing_metrics['NumRank']:.4f}, Erank: {test_oversmoothing_metrics['Erank']:.4f}")
        
        return {
            'accuracy': test_accuracy,
            'f1': test_f1_score,
            'precision': test_precision,
            'recall': test_recall,
            'oversmoothing': test_oversmoothing_metrics
        }
    
    def get_oversmoothing_metrics_history(self):

        return self.oversmoothing_metrics_history

class GNNGuardModel(nn.Module):
    
    def __init__(self, input_features, hidden_channels, num_classes, dropout=0.5, 
                 similarity_threshold=0.5, num_layers=2, attention_dim=16, device=None,
                 backbone=None):

        super(GNNGuardModel, self).__init__()
        
        self.device = device
        self.dropout_rate = dropout
        self.similarity_threshold = similarity_threshold
        self.num_layers = num_layers
        self.attention_dim = attention_dim
        
        self.attention_gate = Parameter(torch.rand(1))
        self.test_parameter = Parameter(torch.rand(1))
        
        if backbone is not None:
            self.backbone = backbone.to(device)
            self.use_backbone = True
        else:
            self.use_backbone = False

            self.gcn_layers = nn.ModuleList()
            self.gcn_layers.append(GCNConv(input_features, hidden_channels, bias=True))
            if self.num_layers > 1:
                self.gcn_layers.append(GCNConv(hidden_channels, self.attention_dim, bias=True))
            final_input_dim = self.attention_dim if self.num_layers > 1 else hidden_channels
            self.gcn_layers.append(GCNConv(final_input_dim, num_classes, bias=True))
    
    def forward(self, node_features, adjacency_tensor, use_attention=True):
        x = node_features.to_dense() if node_features.is_sparse else node_features
        x = x.to(self.device)

        if self.use_backbone:
            
            data = Data(x=x, edge_index=adjacency_tensor._indices(), edge_weight=adjacency_tensor._values())
            out = self.backbone(data)
            return F.log_softmax(out, dim=1)
        
        for layer_idx, gcn_layer in enumerate(self.gcn_layers[:-1]):
            if use_attention:
                modified_adjacency = self._compute_attention_coefficients(x, adjacency_tensor, layer_idx)
                edge_indices = modified_adjacency._indices()
                edge_weights = modified_adjacency._values()
            else:
                edge_indices = adjacency_tensor._indices()
                edge_weights = adjacency_tensor._values()
            
            x = gcn_layer(x, edge_indices, edge_weight=edge_weights)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)
        
        final_gcn_layer = self.gcn_layers[-1]
        if use_attention:
            modified_adjacency = self._compute_attention_coefficients(x, adjacency_tensor, self.num_layers)
            edge_indices = modified_adjacency._indices()
            edge_weights = modified_adjacency._values()
        else:
            edge_indices = adjacency_tensor._indices()
            edge_weights = adjacency_tensor._values()
        
        x = final_gcn_layer(x, edge_indices, edge_weight=edge_weights)
        return F.log_softmax(x, dim=1)

    
    def _compute_attention_coefficients(self, node_features, adjacency_tensor, layer_index, 
                                      is_lil_matrix=False, debug=False):

        if not is_lil_matrix:
            edge_indices = adjacency_tensor._indices()
        else:
            adjacency_coo = adjacency_tensor.tocoo()
            edge_indices = torch.stack([
                torch.from_numpy(adjacency_coo.row).to(self.device),
                torch.from_numpy(adjacency_coo.col).to(self.device)
            ])
        
        num_nodes = node_features.shape[0]
        
        # Extract edge connectivity
        source_nodes, target_nodes = edge_indices[0].cpu().data.numpy(), edge_indices[1].cpu().data.numpy()
        source_nodes = np.array(source_nodes, dtype=np.int64, copy=True)
        target_nodes = np.array(target_nodes, dtype=np.int64, copy=True)
        
        # Compute cosine similarity
        features_cpu = node_features.cpu().data.numpy()
        similarity_matrix = cosine_similarity(X=features_cpu, Y=features_cpu)
        similarity_matrix = np.array(similarity_matrix, dtype=np.float32, copy=True)
        
        # Extract similarities
        edge_similarities = similarity_matrix[source_nodes, target_nodes].copy()
        
        # Apply similarity threshold
        edge_similarities[edge_similarities < self.similarity_threshold] = 0
        
        # Create sparse attention matrix
        attention_matrix = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        attention_matrix[source_nodes, target_nodes] = edge_similarities
        
        # Remove self-loops if they exist
        if attention_matrix[0, 0] == 1:
            attention_matrix -= sp.diags(attention_matrix.diagonal(), offsets=0, format="lil")
        
        # Normalize attention weights
        normalized_attention_matrix = normalize(attention_matrix, axis=1, norm='l1')
        
        # Add self-connections with adaptive weights
        if normalized_attention_matrix[0, 0] == 0:
            # Compute adaptive self-loop weights
            node_degrees = (normalized_attention_matrix != 0).sum(1).A1
            adaptive_self_weights = 1 / (node_degrees + 1)
            self_loop_matrix = sp.diags(np.array(adaptive_self_weights), offsets=0, format="lil")
            final_attention_matrix = normalized_attention_matrix + self_loop_matrix
        else:
            final_attention_matrix = normalized_attention_matrix
        
        attention_row_indices, attention_col_indices = final_attention_matrix.nonzero()
        attention_edge_weights = final_attention_matrix[attention_row_indices, attention_col_indices]
        attention_edge_weights = np.array(attention_edge_weights, dtype=np.float32, copy=True).flatten()
        
        attention_edge_weights = np.exp(attention_edge_weights)
        attention_weights_tensor = torch.tensor(attention_edge_weights, dtype=torch.float32).to(self.device)
        
        attention_edge_indices = torch.tensor(
            np.vstack((attention_row_indices, attention_col_indices)), dtype=torch.int64
        ).to(self.device)
        
        tensor_shape = (num_nodes, num_nodes)
        attention_weighted_adjacency = torch.sparse.FloatTensor(
            attention_edge_indices, attention_weights_tensor, tensor_shape
        ).to(self.device)
        
        return attention_weighted_adjacency