"""RTGNN method helper — robust training with dual GNN branches.

Wraps the RTGNN class's per-epoch logic into the shared
TrainingLoop + MethodHelper system.

RTGNN uses a DualBranchGNNModel (two GNN branches), a GraphStructureEstimator
for adaptive edge weights, AdaptiveCoTeachingLoss for noise handling, and
IntraViewRegularization for consistency between branches.
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import from_scipy_sparse_matrix

from methods.base_helper import MethodHelper
from methods.registry import register_helper
from model.methods.RTGNN import RTGNN, RTGNNTrainingConfig


@register_helper('rtgnn')
class RTGNNHelper(MethodHelper):
    """MethodHelper for the RTGNN robustness method."""

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup(self, backbone_model, data, config, device, init_data):
        local_config = deepcopy(config)
        local_config.setdefault('training', {})['oversmoothing_every'] = (
            init_data.get('oversmoothing_every', 20)
        )
        rtgnn_config = RTGNNTrainingConfig(local_config)

        rtgnn = RTGNN(
            training_config=rtgnn_config,
            device=device,
            gnn_backbone=config['model']['name'].lower(),
            data_for_training=data,
        ).to(device)

        # Extract tensors (replicating train_model's pre-loop setup)
        node_features = rtgnn.node_features.to(device)
        node_labels = torch.as_tensor(
            rtgnn.node_labels, dtype=torch.long, device=device,
        )
        train_indices = rtgnn.train_node_indices
        val_indices = rtgnn.val_node_indices

        edge_indices, _ = from_scipy_sparse_matrix(rtgnn.adjacency_matrix)
        edge_indices = edge_indices.to(device)

        knn_edge_indices = rtgnn._generate_knn_edge_connections(
            node_features, edge_indices, train_indices,
        )

        optimizer = optim.Adam(
            rtgnn.parameters(),
            lr=rtgnn_config.lr,
            weight_decay=rtgnn_config.weight_decay,
        )

        # R-1 Implementation: Initialize noise transition matrix if not already present
        if not hasattr(rtgnn, 'transition_matrix'):
            rtgnn.transition_matrix = torch.eye(num_classes, device=device)

        return {
            'models': [rtgnn.dual_branch_predictor, rtgnn.structure_estimator],
            'optimizers': [optimizer],
            'rtgnn': rtgnn,
            'optimizer': optimizer,
            'node_features': node_features,
            'node_labels': node_labels,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'edge_indices': edge_indices,
            'knn_edge_indices': knn_edge_indices,
            'device': device,
            'num_classes': num_classes,
        }

    # ── Per-epoch training ─────────────────────────────────────────────────

    def train_step(self, state, data, epoch):
        rtgnn = state['rtgnn']
        optimizer = state['optimizer']
        device = state['device']
        node_features = state['node_features']
        node_labels = state['node_labels']
        train_indices = state['train_indices']
        val_indices = state['val_indices']
        edge_indices = state['edge_indices']
        knn_edge_indices = state['knn_edge_indices']

        rtgnn.train()
        optimizer.zero_grad(set_to_none=True)

        # R-1 Implementation: Every few epochs, update transition matrix using validation set
        if epoch % 5 == 0:
            rtgnn.eval()
            with torch.no_grad():
                out1, out2 = rtgnn.dual_branch_predictor(
                    node_features, getattr(rtgnn, '_current_edges', edge_indices),
                    getattr(rtgnn, '_current_weights', torch.ones(edge_indices.size(1), device=device))
                )
                val_preds = (out1[val_indices] + out2[val_indices]).argmax(dim=1)
                rtgnn.transition_matrix = rtgnn.estimate_noise_transition_matrix(
                    val_preds, node_labels[val_indices]
                )
            rtgnn.train()

        # Structure estimator forward
        node_reps, recon_loss = rtgnn.structure_estimator(
            node_features, edge_indices,
        )

        # Combine original + KNN edges
        if knn_edge_indices.size(1) > 0:
            combined_edges = torch.cat(
                [edge_indices, knn_edge_indices], dim=1,
            )
            base_weights = torch.cat([
                torch.ones(edge_indices.size(1)),
                torch.zeros(knn_edge_indices.size(1)),
            ]).to(device)
        else:
            combined_edges = edge_indices
            base_weights = torch.ones(edge_indices.size(1)).to(device)

        # Adaptive edge weights
        adaptive_weights = rtgnn.structure_estimator.compute_adaptive_edge_weights(
            combined_edges, node_reps, base_weights,
        )

        # Filter edges with positive weights
        valid_mask = adaptive_weights > 0
        final_edges = combined_edges[:, valid_mask]
        final_weights = adaptive_weights[valid_mask]

        # Store for get_predictions / get_embeddings / checkpoint
        rtgnn._current_edges = final_edges
        rtgnn._current_weights = final_weights

        # Dual branch forward
        out1, out2 = rtgnn.dual_branch_predictor(
            node_features, final_edges, final_weights,
        )

        # Losses
        main_loss = rtgnn.adaptive_loss_function(
            out1[train_indices], out2[train_indices],
            node_labels[train_indices], epoch,
        )
        pseudo_loss = rtgnn._compute_pseudo_labeling_loss(
            out1, out2, train_indices,
        )
        intraview_loss = rtgnn.intraview_regularizer(
            out1, out2, final_edges, final_weights, train_indices,
        )

        # R-4 Fix: Apply co_lambda only once to the intraview_loss term
        total_loss = (
            main_loss
            + rtgnn.training_config.alpha * recon_loss
            + pseudo_loss
            + rtgnn.training_config.co_lambda * intraview_loss
        )

        total_loss.backward()
        optimizer.step()

        return {'train_loss': total_loss.item()}

    # ── Validation ─────────────────────────────────────────────────────────

    def compute_val_loss(self, state, data):
        rtgnn = state['rtgnn']
        node_features = state['node_features']
        node_labels = state['node_labels']
        val_indices = state['val_indices']

        rtgnn.eval()
        with torch.no_grad():
            edges = rtgnn._current_edges
            weights = rtgnn._current_weights
            out1, out2 = rtgnn.dual_branch_predictor(
                node_features, edges, weights,
            )
            val_loss1 = F.cross_entropy(out1[val_indices], node_labels[val_indices])
            val_loss2 = F.cross_entropy(out2[val_indices], node_labels[val_indices])
            return ((val_loss1 + val_loss2) / 2).item()

    # ── Predictions / Embeddings ───────────────────────────────────────────

    def _is_train_graph(self, state, data):
        """Check if data is the same graph used during setup (train subgraph)."""
        return data.num_nodes == state['node_features'].size(0)

    def get_predictions(self, state, data):
        rtgnn = state['rtgnn']

        rtgnn.eval()
        with torch.no_grad():
            if not self._is_train_graph(state, data):
                # Inductive: different subgraph — use backbone with its own edges
                backbone = rtgnn.dual_branch_predictor.first_branch
                return backbone(data).argmax(dim=1)
            node_features = state['node_features']
            out1, out2 = rtgnn.dual_branch_predictor(
                node_features, rtgnn._current_edges, rtgnn._current_weights,
            )
            return ((out1 + out2) / 2).argmax(dim=1)

    def get_embeddings(self, state, data):
        rtgnn = state['rtgnn']

        rtgnn.eval()
        with torch.no_grad():
            if not self._is_train_graph(state, data):
                backbone = rtgnn.dual_branch_predictor.first_branch
                return backbone.get_embeddings(data)
            node_features = state['node_features']
            return rtgnn.dual_branch_predictor.get_embeddings(
                node_features, rtgnn._current_edges, rtgnn._current_weights,
            )

    # ── Checkpointing ──────────────────────────────────────────────────────

    def get_checkpoint_state(self, state):
        rtgnn = state['rtgnn']
        result = {
            'dual_branch_predictor': deepcopy(
                rtgnn.dual_branch_predictor.state_dict(),
            ),
            'structure_estimator': deepcopy(
                rtgnn.structure_estimator.state_dict(),
            ),
        }
        if hasattr(rtgnn, '_current_edges') and rtgnn._current_edges is not None:
            result['edges'] = rtgnn._current_edges.clone()
            result['weights'] = rtgnn._current_weights.clone()
        return result

    def load_checkpoint_state(self, state, checkpoint):
        rtgnn = state['rtgnn']
        rtgnn.dual_branch_predictor.load_state_dict(
            checkpoint['dual_branch_predictor'],
        )
        rtgnn.structure_estimator.load_state_dict(
            checkpoint['structure_estimator'],
        )
        rtgnn.best_model_state = {
            'model': checkpoint['dual_branch_predictor'],
            'edges': checkpoint.get('edges'),
            'weights': checkpoint.get('weights'),
        }
        rtgnn._current_edges = checkpoint.get('edges')
        rtgnn._current_weights = checkpoint.get('weights')
