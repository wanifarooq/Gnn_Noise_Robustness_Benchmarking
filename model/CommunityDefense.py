import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import copy
import time
import scipy
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph
from torch_geometric.data import Data as PyGData
from sklearn.metrics import accuracy_score, f1_score
try:
    import networkx as nx
except ImportError:
    nx = None
from sklearn.metrics import f1_score, precision_score, recall_score

from model.evaluation import OversmoothingMetrics

class GraphCommunityDefenseTrainer:

    def __init__(self, 
                 graph_data: PyGData,
                 community_detection_method: str = "louvain",
                 num_communities: int = None,
                 community_loss_weight: float = 2.0,
                 positive_pair_weight: float = 1.0,
                 negative_pair_weight: float = 2.0,
                 similarity_margin: float = 1.5,
                 negative_samples_per_node: int = 3,
                 device: torch.device = None,
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

    def _generate_community_pairs(self, community_labels):
        #Generate positive and negative pairs
        row_indices, col_indices = self.adjacency_matrix.nonzero()

        upper_triangular_mask = row_indices < col_indices
        row_filtered = row_indices[upper_triangular_mask]
        col_filtered = col_indices[upper_triangular_mask]

        # Positive pairs: same community
        same_community_mask = community_labels[row_filtered] == community_labels[col_filtered]
        positive_pairs = np.stack([row_filtered[same_community_mask], col_filtered[same_community_mask]], axis=1) if np.any(same_community_mask) else np.zeros((0, 2), dtype=np.int64)

        # Negative pairs: different communities
        negative_pairs = []
        rng = np.random.default_rng(0)
        
        for node_idx in range(self.num_nodes):
            # Find nodes in different communities
            different_community_nodes = np.where(community_labels != community_labels[node_idx])[0]
            
            if different_community_nodes.size == 0:
                continue
                
            if different_community_nodes.size <= self.neg_samples_count:
                selected_nodes = different_community_nodes
            else:
                selected_nodes = rng.choice(
                    different_community_nodes, 
                    size=self.neg_samples_count, 
                    replace=False
                )
            
            for other_node in selected_nodes:
                if node_idx < other_node:
                    negative_pairs.append((node_idx, other_node))
                elif other_node < node_idx:
                    negative_pairs.append((other_node, node_idx))
        
        if negative_pairs:
            negative_pairs = np.unique(np.array(negative_pairs, dtype=np.int64), axis=0)
        else:
            negative_pairs = np.zeros((0, 2), dtype=np.int64)

        return positive_pairs, negative_pairs

    def compute_community_regularization_loss(self, node_embeddings: torch.Tensor, community_labels: np.ndarray) -> torch.Tensor:

        device = node_embeddings.device
        community_labels_tensor = torch.as_tensor(community_labels, dtype=torch.long, device=device)
        
        if self.community_classifier.in_features != node_embeddings.shape[1]:
            self.community_classifier = nn.Linear(node_embeddings.shape[1], self.num_communities).to(device)
        
        community_logits = self.community_classifier(node_embeddings)
        
        loss_fn = nn.CrossEntropyLoss()
        community_loss = loss_fn(community_logits, community_labels_tensor)
        
        return community_loss

    def _compute_oversmoothing_metrics_for_mask(self, embeddings, edge_index, node_mask, labels=None):

        try:
            masked_node_indices = torch.where(node_mask)[0]
            masked_embeddings = embeddings[node_mask]
            
            mask_set = set(masked_node_indices.cpu().numpy())
            edge_mask = torch.tensor([
                src.item() in mask_set and tgt.item() in mask_set
                for src, tgt in edge_index.t()
            ], device=edge_index.device)
            
            if not edge_mask.any():
                return {
                    'NumRank': float(min(masked_embeddings.shape)),
                    'Erank': float(min(masked_embeddings.shape)),
                    'EDir': 0.0,
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0
                }
            
            masked_edges = edge_index[:, edge_mask]
            node_mapping = {orig_idx.item(): local_idx for local_idx, orig_idx in enumerate(masked_node_indices)}
            
            remapped_edges = torch.stack([
                torch.tensor([node_mapping[src.item()] for src in masked_edges[0]], device=edge_index.device),
                torch.tensor([node_mapping[tgt.item()] for tgt in masked_edges[1]], device=edge_index.device)
            ])
            
            graphs_in_class = [{
                'X': masked_embeddings,
                'edge_index': remapped_edges,
                'edge_weight': None
            }]
            
            return self.oversmoothing_evaluator.compute_all_metrics(
                X=masked_embeddings,
                edge_index=remapped_edges,
                graphs_in_class=graphs_in_class
            )
            
        except Exception as e:
            print(f"Warning: Could not compute oversmoothing metrics for mask: {e}")
            return None

    def train_with_community_defense(self,
                                   gnn_model,
                                   training_epochs: int = 200,
                                   early_stopping_patience: int = 20,
                                   learning_rate: float = 0.005,
                                   weight_decay_rate: float = 1e-3,
                                   enable_debug: bool = True):

        training_start_time = time.time()
        
        model = gnn_model.to(self.device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.community_classifier.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay_rate
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
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
        best_model_state = None
        epochs_without_improvement = 0
        community_labels = self.community_assignments.copy()
        
        for epoch in range(1, training_epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            batch_data = PyGData(x=node_features, edge_index=edge_index, y=node_labels)
            model_output = model(batch_data)
            
            if isinstance(model_output, tuple) and len(model_output) == 2:
                logits, node_embeddings = model_output
            else:
                logits = model_output
            
            if hasattr(model, 'last_hidden'):
                node_embeddings = model.last_hidden
            else:
                node_embeddings = F.dropout(logits, p=0.3, training=True)
            
            # Compute losses
            supervised_loss = cross_entropy_loss(logits[self.train_node_mask], node_labels[self.train_node_mask])
            
            if hasattr(model, 'last_hidden'):
                node_embeddings = model.last_hidden
            else:
                node_embeddings = F.dropout(logits, p=0.3, training=True)

            # community loss softmax multi-class
            adaptive_community_weight = self.community_loss_weight * min(1.0, epoch / 50.0)
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
                lr_scheduler.step(validation_loss)

                train_pred = logits[self.train_node_mask].argmax(dim=-1).cpu().numpy()
                train_true = node_labels[self.train_node_mask].cpu().numpy()
                val_pred = validation_logits[self.val_node_mask].argmax(dim=-1).cpu().numpy()
                val_true = node_labels[self.val_node_mask].cpu().numpy()

                train_acc = accuracy_score(train_true, train_pred)
                val_acc = accuracy_score(val_true, val_pred)
                train_f1 = f1_score(train_true, train_pred, average="macro")
                val_f1 = f1_score(val_true, val_pred, average="macro")
                train_loss = cross_entropy_loss(
                    logits[self.train_node_mask], node_labels[self.train_node_mask]
                ).item()

                print(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {validation_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                    f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}"
                )

                if epoch % 20 == 0:
                    emb = node_embeddings.detach()

                    train_metrics = self._compute_oversmoothing_metrics_for_mask(
                        emb, edge_index, self.train_node_mask, node_labels
                    )
                    val_metrics = self._compute_oversmoothing_metrics_for_mask(
                        emb, edge_index, self.val_node_mask, node_labels
                    )

                    if train_metrics is not None and val_metrics is not None:
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

                # Early stopping
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    if self.verbose and enable_debug:
                        print(f"Defense: Early stopping at epoch {epoch}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        model.eval()
        with torch.no_grad():
            batch_data = PyGData(x=node_features, edge_index=edge_index, y=node_labels)
            final_output = model(batch_data)
            final_logits = final_output[0] if (isinstance(final_output, tuple) and len(final_output) == 2) else final_output
            
            predictions = final_logits.argmax(dim=1).cpu().numpy()
            true_labels = node_labels.cpu().numpy()

            test_mask_np = self.test_node_mask.cpu().numpy()
            test_true_labels = true_labels[test_mask_np]
            test_predictions = predictions[test_mask_np]

            test_accuracy = (test_predictions == test_true_labels).mean()
            test_f1_score = f1_score(test_true_labels, test_predictions, average="macro")
            test_precision = precision_score(test_true_labels, test_predictions, average="macro", zero_division=0)
            test_recall = recall_score(test_true_labels, test_predictions, average="macro", zero_division=0)

            test_oversmoothing_metrics = self._compute_oversmoothing_metrics_for_mask(
                final_logits, edge_index, self.test_node_mask, node_labels
            )
            
            if test_oversmoothing_metrics is not None:
                oversmoothing_results = {
                    'EDir': test_oversmoothing_metrics['EDir'],
                    'EDir_traditional': test_oversmoothing_metrics['EDir_traditional'],
                    'EProj': test_oversmoothing_metrics['EProj'],
                    'MAD': test_oversmoothing_metrics['MAD'],
                    'NumRank': test_oversmoothing_metrics['NumRank'],
                    'Erank': test_oversmoothing_metrics['Erank']
                }
            else:
                oversmoothing_results = {
                    'EDir': 0.0,
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0,
                    'NumRank': 0.0,
                    'Erank': 0.0
                }
            
            final_results = {
                'accuracy': test_accuracy,
                'f1': test_f1_score,
                'precision': test_precision,
                'recall': test_recall,
                'oversmoothing': oversmoothing_results
            }
            
            training_duration = time.time() - training_start_time
            if enable_debug:
                print(f"\nTraining completed in {training_duration:.2f}s")
                print(f"Test Acc: {final_results['accuracy']:.4f} | Test F1: {final_results['f1']:.4f}")
                print(f"Test Precision: {final_results['precision']:.4f} | Test Recall: {final_results['recall']:.4f}")
                print("Oversmoothing metrics:")
                for metric_name, metric_value in oversmoothing_results.items():
                    print(f"  {metric_name}: {metric_value:.4f}")
        
        return final_results