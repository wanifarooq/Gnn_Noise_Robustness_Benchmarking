"""GNNGuard method helper — attention-based edge reweighting for robustness.

Wraps the backbone in a GNNGuardModel that computes cosine-similarity-based
attention coefficients over the adjacency matrix.  Training uses NLL loss
(model outputs log_softmax).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch_geometric.utils import to_scipy_sparse_matrix

from methods.base_helper import MethodHelper
from methods.registry import register_helper
from model.methods.GNNGuard import GNNGuardModel


@register_helper('gnnguard')
class GNNGuardHelper(MethodHelper):
    """GNNGuard: cosine-similarity attention over edges."""

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _sparse_scipy_to_torch(sparse_matrix, device):
        """Convert a scipy sparse matrix to a PyTorch sparse COO tensor."""
        coo = sparse_matrix.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((coo.row, coo.col)).astype(np.int64)
        ).to(device)
        values = torch.from_numpy(coo.data).to(device)
        return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape),
                                       dtype=torch.float32, device=device)

    @staticmethod
    def _add_self_loops(adj_tensor, device):
        """Add identity self-loops to a sparse adjacency tensor."""
        n = adj_tensor.shape[0]
        diag_idx = torch.arange(n, dtype=torch.int64)
        loop_indices = torch.stack((diag_idx, diag_idx), dim=0)
        loop_values = torch.ones(n, dtype=torch.float32)
        identity = torch.sparse.FloatTensor(
            loop_indices, loop_values, adj_tensor.shape
        ).to(device)
        return adj_tensor + identity

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        gnnguard_params = config.get('gnnguard_params', {})

        lr = float(training_cfg.get('lr', 0.01))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))
        attention = bool(gnnguard_params.get('attention', True))

        # If attention is disabled, GNNGuard paper sets weight_decay to 0
        if not attention:
            weight_decay = 0.0

        num_classes = init_data.get('num_classes', int(data.y.max().item()) + 1)

        gnnguard_model = GNNGuardModel(
            input_features=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            num_classes=num_classes,
            dropout=config['model'].get('dropout', 0.5),
            similarity_threshold=float(gnnguard_params.get('P0', 0.5)),
            num_layers=int(gnnguard_params.get('K', 2)),
            attention_dim=int(gnnguard_params.get('D2', 16)),
            device=device,
            backbone=backbone_model,
        ).to(device)

        # Build adjacency: scipy -> torch sparse + self-loops
        adj_scipy = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        adj_torch = self._sparse_scipy_to_torch(adj_scipy, device)
        normalized_adj = self._add_self_loops(adj_torch, device)

        node_features = data.x.to(device)
        node_labels = data.y.to(device)

        train_indices = data.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        val_indices = data.val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        test_indices = data.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()

        optimizer = torch.optim.Adam(
            gnnguard_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        return {
            'models': [gnnguard_model],
            'optimizers': [optimizer],
            'backbone': backbone_model,
            'gnnguard_model': gnnguard_model,
            'optimizer': optimizer,
            'device': device,
            'node_features': node_features,
            'node_labels': node_labels,
            'normalized_adjacency': normalized_adj,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'use_attention': attention,
        }

    # ── Train step ─────────────────────────────────────────────────────────

    def train_step(self, state, data, epoch):
        model = state['gnnguard_model']
        optimizer = state['optimizer']
        features = state['node_features']
        adj = state['normalized_adjacency']
        labels = state['node_labels']
        train_idx = state['train_indices']
        use_attn = state['use_attention']

        model.train()
        optimizer.zero_grad(set_to_none=True)

        output = model(features, adj, use_attention=use_attn)
        loss = F.nll_loss(output[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        return {'train_loss': loss.item()}

    # ── Validation ─────────────────────────────────────────────────────────

    def compute_val_loss(self, state, data):
        model = state['gnnguard_model']
        features = state['node_features']
        adj = state['normalized_adjacency']
        labels = state['node_labels']
        val_idx = state['val_indices']
        use_attn = state['use_attention']

        model.eval()
        with torch.no_grad():
            output = model(features, adj, use_attention=use_attn)
            return F.nll_loss(output[val_idx], labels[val_idx]).item()

    # ── Predictions / embeddings ───────────────────────────────────────────

    def get_predictions(self, state, data):
        model = state['gnnguard_model']
        features = state['node_features']
        adj = state['normalized_adjacency']
        use_attn = state['use_attention']

        model.eval()
        with torch.no_grad():
            return model(features, adj, use_attention=use_attn).argmax(dim=1)

    def get_embeddings(self, state, data):
        model = state['gnnguard_model']
        features = state['node_features']
        adj = state['normalized_adjacency']
        use_attn = state['use_attention']

        model.eval()
        with torch.no_grad():
            return model.get_embeddings(features, adj, use_attention=use_attn)

    # ── Checkpointing ─────────────────────────────────────────────────────

    def get_checkpoint_state(self, state):
        return {
            'backbone': deepcopy(state['backbone'].state_dict()),
            'gnnguard_model': deepcopy(state['gnnguard_model'].state_dict()),
        }

    def load_checkpoint_state(self, state, checkpoint):
        state['gnnguard_model'].load_state_dict(checkpoint['gnnguard_model'])

    # ── Profiling ──────────────────────────────────────────────────────────

    def get_inference_forward_fn(self, state, data):
        model = state['gnnguard_model']
        features = state['node_features']
        adj = state['normalized_adjacency']
        use_attn = state['use_attention']

        return lambda: model(features, adj, use_attention=use_attn)

    def get_training_step_fn(self, state, data):
        model = state['gnnguard_model']
        features = state['node_features']
        adj = state['normalized_adjacency']
        labels = state['node_labels']
        train_idx = state['train_indices']
        use_attn = state['use_attention']

        def step_fn():
            out = model(features, adj, use_attention=use_attn)
            return F.nll_loss(out[train_idx], labels[train_idx])

        return step_fn
