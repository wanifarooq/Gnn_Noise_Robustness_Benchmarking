import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix
import scipy.sparse as sp
import numpy as np
from torch_geometric.data import Data
from model.base import BaseTrainer
from model.registry import register


class GNNGuardModel(nn.Module):
    """GNNGuard defence wrapper (Zhang & Zitnik, NeurIPS 2020).

    GNNGuard is originally an *adversarial-structure-attack* defence: it
    reweights edges by the cosine similarity of the incident node
    representations, prunes dissimilar edges (similarity < P0), row-normalises
    the resulting attention coefficients and adds an adaptive self-loop term
    (N_hat / (N_hat + 1) ≈ 1 / (deg + 1)), then passes these edge weights into
    message passing.  A layer-wise graph-memory term blends the attention of
    successive layers via a learnable ``beta``:

        omega^k = beta * omega^{k-1} + (1 - beta) * alpha_hat^k

    In this benchmark it is applied (off-label) as a *label-noise* baseline.
    The defining mechanism is run for real and wired through the shared
    backbone, whose GCN/GAT/GATv2/GPS layers accept ``edge_weight``.
    """

    def __init__(self, input_features, hidden_channels, num_classes, dropout=0.5,
                 similarity_threshold=0.5, num_layers=2, attention_dim=16, device=None,
                 backbone=None):

        super(GNNGuardModel, self).__init__()

        self.device = device
        self.dropout_rate = dropout
        self.similarity_threshold = similarity_threshold
        self.num_layers = num_layers
        self.attention_dim = attention_dim

        # Learnable layer-wise graph-memory coefficient (beta).  Stored as a
        # raw parameter and squashed through a sigmoid so beta stays in (0, 1).
        self.attention_gate = Parameter(torch.rand(1))

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

    def _guard_edge_weights(self, x, adjacency_tensor):
        """Compute GNNGuard edge weights via per-layer cosine attention + memory.

        Runs the defining GNNGuard mechanism: at each of the K layers it
        recomputes the cosine-similarity attention coefficients (with P0
        pruning, row-normalisation and the adaptive self-loop term) from the
        node representations propagated through the previous layer's attention,
        and blends them with the learnable graph-memory term ``beta``.

        Returns ``(edge_index, edge_weight)`` describing the final reweighted
        sparse adjacency, suitable for passing to the backbone via
        ``data.edge_weight``.
        """
        beta = torch.sigmoid(self.attention_gate).reshape(())

        # Layer 0 attention from the raw input features.
        omega = self._compute_attention_coefficients(x, adjacency_tensor, 0)
        rep = x

        for layer_idx in range(1, self.num_layers):
            # Propagate node representations one hop through the current
            # attention-weighted adjacency, then recompute attention.
            rep = torch.sparse.mm(omega, rep)
            alpha_k = self._compute_attention_coefficients(rep, adjacency_tensor, layer_idx)
            # Graph-memory term: omega^k = beta * omega^{k-1} + (1 - beta) * alpha_hat^k.
            # Sum the two sparse tensors (with beta-scaled values) and coalesce so
            # overlapping indices are merged before the next propagation step.
            omega_scaled = torch.sparse_coo_tensor(
                omega.indices(), omega.values() * beta, omega.shape,
            )
            alpha_scaled = torch.sparse_coo_tensor(
                alpha_k.indices(), alpha_k.values() * (1.0 - beta), alpha_k.shape,
            )
            omega = (omega_scaled + alpha_scaled).coalesce()

        return omega.indices(), omega.values()

    def forward(self, node_features, adjacency_tensor, use_attention=True):
        x = node_features.to_dense() if node_features.is_sparse else node_features
        x = x.to(self.device)

        if self.use_backbone:
            if use_attention:
                edge_index, edge_weight = self._guard_edge_weights(x, adjacency_tensor)
            else:
                edge_index = adjacency_tensor._indices()
                edge_weight = adjacency_tensor._values()
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
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

    def get_embeddings(self, node_features, adjacency_tensor, use_attention=True):
        """Return hidden representations before the final projection.

        With backbone (standard path): returns hidden_channels dim.
        Without backbone (legacy fallback): returns attention_dim when num_layers > 1.
        """
        x = node_features.to_dense() if node_features.is_sparse else node_features
        x = x.to(self.device)

        if self.use_backbone:
            if use_attention:
                edge_index, edge_weight = self._guard_edge_weights(x, adjacency_tensor)
            else:
                edge_index = adjacency_tensor._indices()
                edge_weight = adjacency_tensor._values()
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            return self.backbone.get_embeddings(data)

        last_hidden_idx = len(self.gcn_layers) - 2
        for layer_idx, gcn_layer in enumerate(self.gcn_layers[:-1]):
            if use_attention:
                modified_adjacency = self._compute_attention_coefficients(x, adjacency_tensor, layer_idx)
                edge_indices = modified_adjacency._indices()
                edge_weights = modified_adjacency._values()
            else:
                edge_indices = adjacency_tensor._indices()
                edge_weights = adjacency_tensor._values()

            x = gcn_layer(x, edge_indices, edge_weight=edge_weights)
            if layer_idx < last_hidden_idx:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, training=self.training)
        return x

    
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
        
        # G-2 Fix: Robust self-loop removal (check trace/diagonal sum)
        if attention_matrix.diagonal().sum() > 0:
            attention_matrix -= sp.diags(attention_matrix.diagonal(), offsets=0, format="lil")
        
        # Normalize attention weights
        normalized_attention_matrix = normalize(attention_matrix, axis=1, norm='l1')
        
        # Add self-connections with adaptive weights
        if normalized_attention_matrix.diagonal().sum() == 0:
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
        
        # G-1 Fix: Removed np.exp(). Weights are already L1-normalized probabilities.
        # Applying exp() destroys the normalization and biases toward high-degree nodes.
        attention_weights_tensor = torch.tensor(attention_edge_weights, dtype=torch.float32).to(self.device)
        
        attention_edge_indices = torch.tensor(
            np.vstack((attention_row_indices, attention_col_indices)), dtype=torch.int64
        ).to(self.device)
        
        tensor_shape = (num_nodes, num_nodes)
        attention_weighted_adjacency = torch.sparse_coo_tensor(
            attention_edge_indices, attention_weights_tensor, tensor_shape,
            dtype=torch.float32, device=self.device,
        ).coalesce()

        return attention_weighted_adjacency


@register('gnnguard')
class GNNGuardMethodTrainer(BaseTrainer):
    def train(self):
        from methods.registry import get_helper
        from training.training_loop import TrainingLoop

        d = self.init_data
        self._helper = get_helper('gnnguard')
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
            self._helper = get_helper('gnnguard')
            d = self.init_data
            # Setup state for eval-only mode
            self._loop_state = self._helper.setup(
                d['backbone_model'], d['data_for_training'],
                self.config, d['device'], d,
            )
        else:
            self._loop_state = self._loop.state
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
        from model.evaluation import evaluate_model as shared_evaluate_model
        d = self.init_data
        state = self._get_state()
        data = d['data_for_training']

        def get_predictions():
            return self._helper.get_predictions(state, data)

        def get_embeddings():
            return self._helper.get_embeddings(state, data)

        return shared_evaluate_model(
            get_predictions, get_embeddings, data.y,
            data.train_mask, data.val_mask, data.test_mask,
            data.edge_index, d['device'],
        )
