import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import time
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data as PyGData
from collections import defaultdict
try:
    import networkx as nx
except ImportError:
    nx = None

from model.evaluation import (OversmoothingMetrics, ClassificationMetrics,
                              compute_oversmoothing_for_mask)
from model.base import BaseTrainer
from model.registry import register

class GraphCommunityDefenseTrainer:

    def __init__(self, 
                 graph_data: PyGData,
                 community_detection_method: str = "louvain",
                 num_communities: int | None = None,
                 community_loss_weight: float = 2.0,
                 positive_pair_weight: float = 1.0,
                 negative_pair_weight: float = 2.0,
                 similarity_margin: float = 1.5,
                 negative_samples_per_node: int = 3,
                 device: torch.device | None = None,
                 verbose: bool = True):

        self.graph_data = graph_data
        self.community_method = community_detection_method
        self.target_num_communities = num_communities
        self.community_loss_weight = float(community_loss_weight)
        self.pos_pair_weight = float(positive_pair_weight)
        self.neg_pair_weight = float(negative_pair_weight)
        self.margin_threshold = float(similarity_margin)
        self.neg_samples_count = int(negative_samples_per_node)
        self.verbose = verbose
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        
        self.oversmoothing_evaluator = OversmoothingMetrics(device=self.device)
        self.cls_evaluator = ClassificationMetrics(average='macro')
        
        self._prepare_graph_structure()
        self.community_assignments = self._detect_communities()
        self.num_communities = len(np.unique(self.community_assignments))
        # community prediction layer
        self.community_classifier = nn.Linear(
            self.graph_data.x.shape[1],
            self.num_communities
        ).to(self.device)
        
    def _prepare_graph_structure(self):

        edge_indices = self.graph_data.edge_index.cpu().numpy()
        row_indices, col_indices = edge_indices[0], edge_indices[1]
        num_nodes = self.graph_data.num_nodes
        
        # Create symmetric adjacency matrix
        adjacency_matrix = sp.coo_matrix(
            (np.ones_like(row_indices, dtype=np.float32), (row_indices, col_indices)), 
            shape=(num_nodes, num_nodes)
        )
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        adjacency_matrix.data = np.ones_like(adjacency_matrix.data, dtype=np.float32)
        adjacency_matrix.eliminate_zeros()
        
        self.adjacency_matrix = adjacency_matrix.tocsr()
        self.num_nodes = num_nodes

        assert len(self.graph_data.train_mask) == num_nodes, f"Train mask dim {len(self.graph_data.train_mask)} != N {num_nodes}"
        assert len(self.graph_data.val_mask) == num_nodes, f"Val mask dim {len(self.graph_data.val_mask)} != N {num_nodes}"
        assert len(self.graph_data.test_mask) == num_nodes, f"Test mask dim {len(self.graph_data.test_mask)} != N {num_nodes}"
        
        self.train_node_mask = torch.as_tensor(self.graph_data.train_mask, dtype=torch.bool, device=self.device)
        self.val_node_mask = torch.as_tensor(self.graph_data.val_mask, dtype=torch.bool, device=self.device)
        self.test_node_mask = torch.as_tensor(self.graph_data.test_mask, dtype=torch.bool, device=self.device)

    def _detect_communities(self):
        #Detect communities
        if self.community_method.lower() == "louvain":
            return self._detect_louvain_communities()
        elif self.community_method.lower() == "spectral":
            return self._detect_spectral_communities()
        else:
            raise ValueError("community_method must be 'louvain' or 'spectral'")

    def _detect_louvain_communities(self):
        #Detect communities using Louvain algorithm
        if nx is None:
            raise ImportError("For 'louvain' you need networkx")
        
        community_detector = self._import_community_detector()
        
        if community_detector is None:

            return self._fallback_networkx_community_detection()
        
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        row_idx, col_idx = self.adjacency_matrix.nonzero()
        edges = list(zip(row_idx.tolist(), col_idx.tolist()))
        graph.add_edges_from(edges)

        # Apply Louvain community detection
        node_to_community = community_detector.best_partition(graph, random_state=0)
        community_labels = np.array([node_to_community[i] for i in range(self.num_nodes)], dtype=np.int64)
        
        if self.verbose:
            num_communities = len(set(community_labels.tolist()))
            print(f"Defense: Louvain detected {num_communities} communities")
            
        return community_labels

    def _import_community_detector(self):

        # community.community_louvain
        try:
            import community.community_louvain as community_louvain
            if self.verbose:
                print("Using community.community_louvain")
            return community_louvain
        except ImportError:
            pass
        
        # community module with best_partition
        try:
            import community as community_module
            if hasattr(community_module, 'best_partition'):
                if self.verbose:
                    print("Using community module with best_partition")
                return community_module
        except ImportError:
            pass
        
        # direct community_louvain import
        try:
            import community_louvain
            if self.verbose:
                print("Using direct community_louvain import")
            return community_louvain
        except ImportError:
            pass
            
        return None

    def _fallback_networkx_community_detection(self):

        try:
            import networkx.algorithms.community as nx_community
            if self.verbose:
                print("Using NetworkX community detection as fallback")
            
            graph = nx.Graph()
            graph.add_nodes_from(range(self.num_nodes))
            row_idx, col_idx = self.adjacency_matrix.nonzero()
            edges = list(zip(row_idx.tolist(), col_idx.tolist()))
            graph.add_edges_from(edges)
            
            # Try Louvain if available, otherwise use greedy modularity
            if hasattr(nx_community, 'louvain_communities'):
                communities = nx_community.louvain_communities(graph, seed=0)
            else:
                communities = nx_community.greedy_modularity_communities(graph)
            
            node_to_community = {}
            for comm_id, nodes in enumerate(communities):
                for node in nodes:
                    node_to_community[node] = comm_id
            
            community_labels = np.array([node_to_community.get(i, 0) for i in range(self.num_nodes)], dtype=np.int64)
            
            if self.verbose:
                num_communities = len(set(community_labels.tolist()))
                print(f"Defense: NetworkX community detection found {num_communities} communities")
                
            return community_labels
            
        except (ImportError, AttributeError) as e:
            if self.verbose:
                print(f"NetworkX community detection failed: {e}")
            raise ImportError("Could not import any community detection library")

    def _detect_spectral_communities(self):
        #Detect communities using spectral clustering
        from sklearn.cluster import KMeans
        
        # Compute normalized Laplacian
        degree_vector = np.array(self.adjacency_matrix.sum(1)).flatten()
        degree_inv_sqrt = np.power(np.maximum(degree_vector, 1e-12), -0.5)
        degree_inv_sqrt_matrix = sp.diags(degree_inv_sqrt)
        
        normalized_laplacian = sp.eye(self.num_nodes) - degree_inv_sqrt_matrix @ self.adjacency_matrix @ degree_inv_sqrt_matrix

        if self.target_num_communities is None:
            if hasattr(self.graph_data, "y") and self.graph_data.y.dim() == 1:
                self.target_num_communities = int(self.graph_data.y.max().item()) + 1
            else:
                self.target_num_communities = 8

        from scipy.sparse.linalg import eigsh
        eigenvalues, eigenvectors = eigsh(
            normalized_laplacian.asfptype(), 
            k=self.target_num_communities, 
            which="SM"
        )
        
        normalized_eigenvectors = eigenvectors / (np.linalg.norm(eigenvectors, axis=1, keepdims=True) + 1e-12)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.target_num_communities, n_init=10, random_state=0)
        community_labels = kmeans.fit_predict(normalized_eigenvectors).astype(np.int64)
        
        if self.verbose:
            print(f"Defense: Spectral clustering with k={self.target_num_communities} communities")
            
        return community_labels

    def compute_community_regularization_loss(self, node_embeddings: torch.Tensor, community_labels: np.ndarray) -> torch.Tensor:

        device = node_embeddings.device
        community_labels_tensor = torch.as_tensor(community_labels, dtype=torch.long, device=device)
        
        if self.community_classifier.in_features != node_embeddings.shape[1]:
            self.community_classifier = nn.Linear(node_embeddings.shape[1], self.num_communities).to(device)
        
        community_logits = self.community_classifier(node_embeddings)
        
        loss_fn = nn.CrossEntropyLoss()
        community_loss = loss_fn(community_logits, community_labels_tensor)
        
        return community_loss

    def train_with_community_defense(self,
                                   gnn_model,
                                   training_epochs: int = 200,
                                   early_stopping_patience: int = 20,
                                   learning_rate: float = 0.005,
                                   weight_decay_rate: float = 1e-3,
                                   enable_debug: bool = True,
                                   oversmoothing_every: int = 20,
                                   log_epoch_fn=None):

        training_start_time = time.time()
        per_epochs_oversmoothing = defaultdict(list)
        per_epochs_val_oversmoothing = defaultdict(list)
        
        model = gnn_model.to(self.device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.community_classifier.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay_rate
        )

        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        adjacency_matrix = self.adjacency_matrix
        if hasattr(adjacency_matrix, "tocoo"):
            adjacency_matrix = adjacency_matrix.tocoo()
        edge_index, _ = from_scipy_sparse_matrix(adjacency_matrix)
        edge_index = edge_index.to(self.device)
        
        node_features = self.graph_data.x.to(self.device, dtype=torch.float32)
        node_labels = self.graph_data.y.to(self.device, dtype=torch.long)
        
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        community_labels = self.community_assignments.copy()
        
        for epoch in range(training_epochs):
            model.train()
            optimizer.zero_grad()
            
            batch_data = PyGData(x=node_features, edge_index=edge_index, y=node_labels)
            model_output = model(batch_data)
            
            if isinstance(model_output, tuple) and len(model_output) == 2:
                logits, node_embeddings = model_output
            else:
                logits = model_output
            
            # TODO: no model has "last_hidden", we are using logits which are low-dim
            if hasattr(model, 'last_hidden'):
                node_embeddings = model.last_hidden
            else:
                node_embeddings = F.dropout(logits, p=0.3, training=True)
            
            # Compute losses
            supervised_loss = cross_entropy_loss(logits[self.train_node_mask], node_labels[self.train_node_mask])
            
            # TODO: Is this a duplicate? no model has "last_hidden", we are using logits which are low-dim
            if hasattr(model, 'last_hidden'):
                node_embeddings = model.last_hidden
            else:
                node_embeddings = F.dropout(logits, p=0.3, training=True)

            # community loss softmax multi-class
            adaptive_community_weight = self.community_loss_weight * min(1.0, (epoch + 1) / 50.0)
            community_loss = self.compute_community_regularization_loss(node_embeddings, community_labels) if adaptive_community_weight > 0 else torch.zeros([], device=self.device)

            # Total loss
            total_loss = supervised_loss + adaptive_community_weight * community_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                validation_output = model(batch_data)
                validation_logits = validation_output[0] if (
                    isinstance(validation_output, tuple) and len(validation_output) == 2
                ) else validation_output

                validation_loss = cross_entropy_loss(
                    validation_logits[self.val_node_mask], node_labels[self.val_node_mask]
                ).item()

                train_pred = logits[self.train_node_mask].argmax(dim=-1).cpu().numpy()
                train_true = node_labels[self.train_node_mask].cpu().numpy()
                val_pred = validation_logits[self.val_node_mask].argmax(dim=-1).cpu().numpy()
                val_true = node_labels[self.val_node_mask].cpu().numpy()

                train_acc = self.cls_evaluator.compute_accuracy(train_pred, train_true)
                val_acc = self.cls_evaluator.compute_accuracy(val_pred, val_true)
                train_f1 = self.cls_evaluator.compute_f1(train_pred, train_true)
                val_f1 = self.cls_evaluator.compute_f1(val_pred, val_true)
                train_loss = cross_entropy_loss(
                    logits[self.train_node_mask], node_labels[self.train_node_mask]
                ).item()

                print(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {validation_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                    f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}"
                )

                os_entry = None
                if (epoch + 1) % oversmoothing_every == 0:
                    emb = model.get_embeddings(batch_data).detach()

                    train_metrics = compute_oversmoothing_for_mask(
                        self.oversmoothing_evaluator, emb, edge_index, self.train_node_mask
                    )
                    val_metrics = compute_oversmoothing_for_mask(
                        self.oversmoothing_evaluator, emb, edge_index, self.val_node_mask
                    )

                    os_entry = {'train': dict(train_metrics), 'val': dict(val_metrics)}

                    print(
                        f"Epoch {epoch}: | Oversmoothing | "
                        f"EDir: Train {train_metrics['EDir']:.4f}, Val {val_metrics['EDir']:.4f} | "
                        f"EDir_trad: Train {train_metrics['EDir_traditional']:.4f}, Val {val_metrics['EDir_traditional']:.4f}"
                    )
                    print(
                        f"Epoch {epoch}: | Oversmoothing | "
                        f"EProj: Train {train_metrics['EProj']:.4f}, Val {val_metrics['EProj']:.4f} | "
                        f"MAD: Train {train_metrics['MAD']:.4f}, Val {val_metrics['MAD']:.4f}"
                    )
                    print(
                        f"Epoch {epoch}: | Oversmoothing | "
                        f"NumRank: Train {train_metrics['NumRank']:.4f}, Val {val_metrics['NumRank']:.4f} | "
                        f"Erank: Train {train_metrics['Erank']:.4f}, Val {val_metrics['Erank']:.4f}"
                    )

                    for key, value in train_metrics.items():
                        per_epochs_oversmoothing[key].append(value)
                    for key, value in val_metrics.items():
                        per_epochs_val_oversmoothing[key].append(value)

                # Early stopping
                is_best = validation_loss < best_validation_loss
                if is_best:
                    best_validation_loss = validation_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if log_epoch_fn is not None:
                    log_epoch_fn(epoch, train_loss, validation_loss, train_acc, val_acc,
                                 train_f1=train_f1, val_f1=val_f1,
                                 oversmoothing=os_entry, is_best=is_best)

                if epochs_without_improvement >= early_stopping_patience:
                    if self.verbose and enable_debug:
                        print(f"Defense: Early stopping at epoch {epoch}")
                    break

        training_duration = time.time() - training_start_time
        if enable_debug:
            print(f"\nTraining completed in {training_duration:.2f}s")

        return {
            'train_oversmoothing': dict(per_epochs_oversmoothing),
            'val_oversmoothing': dict(per_epochs_val_oversmoothing),
            'stopped_at_epoch': epoch,
        }


@register('community_defense')
class CommunityDefenseMethodTrainer(BaseTrainer):
    def train(self):
        d = self.init_data
        comm_params = self.config.get('community_defense_params', {})

        defense_trainer = GraphCommunityDefenseTrainer(
            graph_data=d['data_for_training'],
            community_detection_method=comm_params.get('community_method', 'louvain'),
            num_communities=comm_params.get('num_communities', None),
            community_loss_weight=float(comm_params.get('lambda_comm', 2.0)),
            positive_pair_weight=float(comm_params.get('pos_weight', 1.0)),
            negative_pair_weight=float(comm_params.get('neg_weight', 2.0)),
            similarity_margin=float(comm_params.get('margin', 1.5)),
            negative_samples_per_node=int(comm_params.get('num_neg_samples', 3)),
            device=d['device'],
            verbose=True,
        )
        return defense_trainer.train_with_community_defense(
            gnn_model=d['backbone_model'],
            training_epochs=d['epochs'],
            early_stopping_patience=d['patience'],
            learning_rate=d['lr'],
            weight_decay_rate=d['weight_decay'],
            enable_debug=True,
            oversmoothing_every=d['oversmoothing_every'],
            log_epoch_fn=self.log_epoch,
        )