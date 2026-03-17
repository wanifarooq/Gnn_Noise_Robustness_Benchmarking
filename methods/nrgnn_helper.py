"""NRGNN method helper — robust GNN with edge augmentation.

Wraps the NRGNN class's per-epoch logic into the shared
TrainingLoop + MethodHelper system.

NRGNN uses three models (main_model + node_predictor + edge_weight_estimator),
a single optimizer, potential/confident edge augmentation, and a multi-component
loss (main + predictor + reconstruction + consistency).
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix

from methods.base_helper import MethodHelper
from methods.registry import register_helper
from model.methods.NRGNN import NRGNN


@register_helper('nrgnn')
class NRGNNHelper(MethodHelper):
    """MethodHelper for the NRGNN robustness method."""

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        nrgnn_config = {
            'lr': float(training_cfg.get('lr', 0.01)),
            'weight_decay': float(training_cfg.get('weight_decay', 5e-4)),
            'epochs': int(training_cfg.get('epochs', 1000)),
            'patience': int(training_cfg.get('patience', 100)),
            'nrgnn_params': config.get('nrgnn_params', {}),
            'oversmoothing_every': int(training_cfg.get('oversmoothing_every', 20)),
        }

        nrgnn = NRGNN(nrgnn_config, device, base_model=backbone_model)

        adj = to_scipy_sparse_matrix(
            data.edge_index, num_nodes=data.x.size(0),
        )
        train_idx = data.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        val_idx = data.val_mask.nonzero(as_tuple=True)[0].cpu().numpy()

        nrgnn.prepare_training_data(
            data.x.to(device), adj, data.y.to(device), train_idx,
        )
        nrgnn.train_indices = train_idx
        nrgnn.validation_indices = val_idx
        nrgnn.initialize_model_components()

        return {
            'models': [nrgnn.main_model, nrgnn.node_predictor],
            'optimizers': [nrgnn.optimizer],
            'nrgnn': nrgnn,
            'device': device,
            'train_indices': train_idx,
            'val_indices': val_idx,
        }

    # ── Per-epoch training ─────────────────────────────────────────────────

    def train_step(self, state, data, epoch):
        nrgnn = state['nrgnn']
        device = state['device']
        train_indices = state['train_indices']
        val_indices = state['val_indices']

        nrgnn.main_model.train()
        nrgnn.node_predictor.train()
        nrgnn.edge_weight_estimator.train()
        nrgnn.optimizer.zero_grad(set_to_none=True)

        # Edge weight estimator forward
        edge_weights = torch.ones(
            nrgnn.original_edge_index.shape[1], device=device, dtype=torch.float32,
        )
        node_reps = nrgnn.edge_weight_estimator.forward(
            nrgnn.node_features, nrgnn.original_edge_index, edge_weights,
        )
        recon_loss = nrgnn.compute_reconstruction_loss(
            nrgnn.original_edge_index, node_reps,
        )

        # Potential edges
        if (nrgnn.potential_edge_index is not None
                and nrgnn.potential_edge_index.shape[1] > 0):
            potential_weights = nrgnn.compute_estimated_edge_weights(
                nrgnn.potential_edge_index, node_reps,
            )
            predictor_edges = torch.cat(
                [nrgnn.original_edge_index, nrgnn.potential_edge_index], dim=1,
            )
            predictor_weights = torch.cat([
                torch.ones(nrgnn.original_edge_index.shape[1], device=device),
                potential_weights,
            ])
        else:
            predictor_edges = nrgnn.original_edge_index
            predictor_weights = torch.ones(
                nrgnn.original_edge_index.shape[1], device=device,
            )

        # Node predictor forward
        predictor_logits = nrgnn.node_predictor.forward(
            nrgnn.node_features, predictor_edges, predictor_weights,
        )

        # Initialize best_predictions on first epoch
        if nrgnn.best_predictions is None:
            with torch.no_grad():
                pred_probs = F.softmax(predictor_logits, dim=1)
                nrgnn.best_predictions = pred_probs.detach().to(device)
                nrgnn.confident_edge_index, nrgnn.confident_node_indices = (
                    nrgnn.identify_confident_edges(nrgnn.best_predictions)
                )

        # Confident edges
        if (nrgnn.confident_edge_index is not None
                and nrgnn.confident_edge_index.shape[1] > 0):
            confident_weights = nrgnn.compute_estimated_edge_weights(
                nrgnn.confident_edge_index, node_reps,
            )
            main_edges = torch.cat(
                [predictor_edges, nrgnn.confident_edge_index], dim=1,
            )
            main_weights = torch.cat([predictor_weights, confident_weights])
        else:
            main_edges = predictor_edges
            main_weights = predictor_weights

        # Main model forward
        main_output = nrgnn.main_model.forward(
            nrgnn.node_features, main_edges, main_weights,
        )

        # Losses
        predictor_loss = F.cross_entropy(
            predictor_logits[train_indices], nrgnn.node_labels[train_indices],
        )
        main_loss = F.cross_entropy(
            main_output[train_indices], nrgnn.node_labels[train_indices],
        )

        if len(nrgnn.confident_node_indices) > 0:
            main_probs = torch.clamp(
                F.softmax(main_output, dim=1), 1e-8, 1 - 1e-8,
            )
            consistency_loss = F.kl_div(
                torch.log(main_probs[nrgnn.confident_node_indices]),
                nrgnn.best_predictions[nrgnn.confident_node_indices],
                reduction='batchmean',
            )
        else:
            consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = (
            main_loss + predictor_loss
            + nrgnn.reconstruction_weight * recon_loss
            + nrgnn.consistency_weight * consistency_loss
        )

        if not torch.isnan(total_loss):
            total_loss.backward()
            nrgnn.optimizer.step()

        # ── Internal state update (required for next epoch) ────────────────
        nrgnn.main_model.eval()
        nrgnn.node_predictor.eval()
        nrgnn.edge_weight_estimator.eval()

        with torch.no_grad():
            nrgnn._current_edge_indices = main_edges
            nrgnn._current_edge_weights = main_weights.detach()

            pred_output = nrgnn.node_predictor.forward(
                nrgnn.node_features, predictor_edges, predictor_weights,
            )
            pred_probs = F.softmax(pred_output, dim=1)
            pred_val_acc = nrgnn.compute_accuracy(
                pred_probs[val_indices], nrgnn.node_labels[val_indices],
            )
            if pred_val_acc > nrgnn.best_predictor_accuracy:
                nrgnn.best_predictor_accuracy = pred_val_acc
                nrgnn.best_predictor_edge_weights = predictor_weights.detach()
                nrgnn.best_predictions = pred_probs.detach()
                nrgnn.confident_edge_index, nrgnn.confident_node_indices = (
                    nrgnn.identify_confident_edges(nrgnn.best_predictions)
                )

        return {'train_loss': total_loss.item() if not torch.isnan(total_loss) else 0.0}

    # ── Validation ─────────────────────────────────────────────────────────

    def compute_val_loss(self, state, data):
        nrgnn = state['nrgnn']
        val_indices = state['val_indices']
        nrgnn.main_model.eval()
        with torch.no_grad():
            edges = nrgnn._current_edge_indices
            weights = nrgnn._current_edge_weights
            if edges is None:
                edges = nrgnn.original_edge_index
                weights = torch.ones(
                    edges.shape[1], device=state['device'],
                )
            out = nrgnn.main_model.forward(nrgnn.node_features, edges, weights)
            return F.cross_entropy(out[val_indices], nrgnn.node_labels[val_indices]).item()

    # ── Predictions / Embeddings ───────────────────────────────────────────

    def _get_edges_and_weights(self, nrgnn, device):
        edges = nrgnn._current_edge_indices
        weights = nrgnn._current_edge_weights
        if edges is None:
            edges = nrgnn.original_edge_index
            weights = torch.ones(edges.shape[1], device=device)
        return edges, weights

    def get_predictions(self, state, data):
        nrgnn = state['nrgnn']
        nrgnn.main_model.eval()
        with torch.no_grad():
            edges, weights = self._get_edges_and_weights(nrgnn, state['device'])
            return nrgnn.main_model.forward(
                nrgnn.node_features, edges, weights,
            ).argmax(dim=1)

    def get_embeddings(self, state, data):
        nrgnn = state['nrgnn']
        nrgnn.main_model.eval()
        with torch.no_grad():
            edges, weights = self._get_edges_and_weights(nrgnn, state['device'])
            return nrgnn.main_model.get_embeddings(
                nrgnn.node_features, edges, weights,
            )

    # ── Checkpointing ──────────────────────────────────────────────────────

    def get_checkpoint_state(self, state):
        nrgnn = state['nrgnn']
        result = {
            'main_model': deepcopy(nrgnn.main_model.state_dict()),
            'node_predictor': deepcopy(nrgnn.node_predictor.state_dict()),
            'edge_weight_estimator': deepcopy(nrgnn.edge_weight_estimator.state_dict()),
        }
        if nrgnn._current_edge_indices is not None:
            result['edge_indices'] = nrgnn._current_edge_indices.clone()
            result['edge_weights'] = nrgnn._current_edge_weights.clone()
        return result

    def load_checkpoint_state(self, state, checkpoint):
        nrgnn = state['nrgnn']
        nrgnn.main_model.load_state_dict(checkpoint['main_model'])
        nrgnn.node_predictor.load_state_dict(checkpoint['node_predictor'])
        if 'edge_weight_estimator' in checkpoint:
            nrgnn.edge_weight_estimator.load_state_dict(checkpoint['edge_weight_estimator'])
        nrgnn.best_edge_indices = checkpoint.get('edge_indices')
        nrgnn.best_edge_weights = checkpoint.get('edge_weights')
        nrgnn._current_edge_indices = checkpoint.get('edge_indices')
        nrgnn._current_edge_weights = checkpoint.get('edge_weights')
