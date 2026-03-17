"""CR-GNN method helper — contrastive-regularized GNN for noisy labels.

Uses double graph augmentation (edge dropout + feature masking), a contrastive
projection head, a classification head, and an optional adapter layer.  Loss
combines contrastive loss, dynamic cross-entropy, and cross-space consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch_geometric.utils import dropout_edge, mask_feature
from torch_geometric.data import Data

from methods.base_helper import MethodHelper
from methods.registry import register_helper
from model.methods.CR_GNN import (
    ContrastiveProjectionHead,
    NodeClassificationHead,
    contrastive_loss_original_style,
    dynamic_cross_entropy_loss_corrected,
    compute_cross_space_consistency_fixed,
)


@register_helper('cr_gnn')
class CRGNNHelper(MethodHelper):
    """CR-GNN: contrastive regularization with dual augmentation."""

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        cr_params = config.get('cr_gnn_params', {})

        lr = float(cr_params.get('lr', training_cfg.get('lr', 0.001)))
        weight_decay = float(cr_params.get('weight_decay',
                                           training_cfg.get('weight_decay', 5e-4)))
        hidden_channels = int(config['model'].get('hidden_channels', 64))

        alpha = float(cr_params.get('alpha', 1.0))
        beta = float(cr_params.get('beta', 1.0))
        tau = float(cr_params.get('tau', 0.5))
        T = float(cr_params.get('T', 0.5))
        p_threshold = float(cr_params.get('p', 0.5))
        pr = float(cr_params.get('pr', 0.3))

        num_classes = init_data.get('num_classes', int(data.y.max().item()) + 1)

        backbone_model.to(device)

        # Determine if adapter is needed (backbone output dim != hidden_channels)
        with torch.no_grad():
            sample_out = backbone_model(data)
        if sample_out.size(1) != hidden_channels:
            adapter = nn.Linear(sample_out.size(1), hidden_channels).to(device)
        else:
            adapter = nn.Identity().to(device)

        proj_head = ContrastiveProjectionHead(hidden_channels, hidden_channels).to(device)
        class_head = NodeClassificationHead(hidden_channels, num_classes).to(device)

        # Single optimizer over all parameters
        params = (list(backbone_model.parameters()) +
                  list(adapter.parameters()) +
                  list(proj_head.parameters()) +
                  list(class_head.parameters()))
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        clean_labels = getattr(data, 'y_original', data.y)

        return {
            'models': [backbone_model, adapter, proj_head, class_head],
            'optimizers': [optimizer],
            'backbone': backbone_model,
            'adapter': adapter,
            'proj_head': proj_head,
            'class_head': class_head,
            'optimizer': optimizer,
            'device': device,
            'alpha': alpha,
            'beta': beta,
            'tau': tau,
            'T': T,
            'p_threshold': p_threshold,
            'pr': pr,
            'clean_labels': clean_labels,
        }

    # ── Train step ─────────────────────────────────────────────────────────

    def train_step(self, state, data, epoch):
        backbone = state['backbone']
        adapter = state['adapter']
        proj_head = state['proj_head']
        class_head = state['class_head']
        optimizer = state['optimizer']
        device = state['device']
        alpha = state['alpha']
        beta = state['beta']
        tau = state['tau']
        T = state['T']
        p_threshold = state['p_threshold']
        pr = state['pr']

        backbone.train()
        adapter.train()
        proj_head.train()
        class_head.train()
        optimizer.zero_grad(set_to_none=True)

        x = data.x
        edge_index = data.edge_index
        labels = data.y
        train_mask = data.train_mask

        # Double augmentation
        edge_idx1, _ = dropout_edge(edge_index, p=pr, training=True)
        edge_idx2, _ = dropout_edge(edge_index, p=pr, training=True)
        x1, _ = mask_feature(x, p=pr)
        x2, _ = mask_feature(x, p=pr)

        # Forward both views
        h1 = adapter(backbone(Data(x=x1, edge_index=edge_idx1)))
        h2 = adapter(backbone(Data(x=x2, edge_index=edge_idx2)))

        # Contrastive loss on projections
        z1 = proj_head(h1)
        z2 = proj_head(h2)
        loss_con = contrastive_loss_original_style(z1, z2, tau)

        # Classification predictions
        p1 = class_head(h1)
        p2 = class_head(h2)

        # Dynamic cross-entropy loss
        if train_mask.sum() > 0:
            loss_sup = dynamic_cross_entropy_loss_corrected(
                p1[train_mask], p2[train_mask], labels[train_mask]
            )
        else:
            loss_sup = torch.tensor(0.0, device=device, requires_grad=True)

        # Cross-space consistency
        loss_ccon = torch.tensor(0.0, device=device)
        if beta > 0:
            try:
                loss_ccon = compute_cross_space_consistency_fixed(
                    z1, z2, p1, p2, T, p_threshold
                )
            except Exception:
                loss_ccon = torch.tensor(0.0, device=device)

        total_loss = alpha * loss_con + loss_sup + beta * loss_ccon

        if torch.isfinite(total_loss):
            total_loss.backward()
            optimizer.step()

        return {'train_loss': total_loss.item()}

    # ── Validation ─────────────────────────────────────────────────────────

    def compute_val_loss(self, state, data):
        backbone = state['backbone']
        adapter = state['adapter']
        class_head = state['class_head']

        backbone.eval()
        adapter.eval()
        class_head.eval()

        with torch.no_grad():
            h = adapter(backbone(Data(x=data.x, edge_index=data.edge_index)))
            preds = class_head(h)
            val_idx = data.val_mask.nonzero(as_tuple=True)[0]
            return F.nll_loss(preds[val_idx], data.y[val_idx]).item()

    # ── Predictions / embeddings ───────────────────────────────────────────

    def get_predictions(self, state, data):
        backbone = state['backbone']
        adapter = state['adapter']
        class_head = state['class_head']

        backbone.eval()
        adapter.eval()
        class_head.eval()

        with torch.no_grad():
            h = adapter(backbone(Data(x=data.x, edge_index=data.edge_index)))
            return class_head(h).exp().argmax(dim=1)

    def get_embeddings(self, state, data):
        backbone = state['backbone']
        backbone.eval()
        with torch.no_grad():
            return backbone.get_embeddings(data)

    # ── Checkpointing ─────────────────────────────────────────────────────

    def get_checkpoint_state(self, state):
        return {
            'backbone': deepcopy(state['backbone'].state_dict()),
            'adapter': deepcopy(state['adapter'].state_dict()),
            'proj_head': deepcopy(state['proj_head'].state_dict()),
            'class_head': deepcopy(state['class_head'].state_dict()),
        }

    def load_checkpoint_state(self, state, checkpoint):
        state['backbone'].load_state_dict(checkpoint['backbone'])
        state['adapter'].load_state_dict(checkpoint['adapter'])
        state['proj_head'].load_state_dict(checkpoint['proj_head'])
        state['class_head'].load_state_dict(checkpoint['class_head'])

    # ── Profiling ──────────────────────────────────────────────────────────

    def get_inference_forward_fn(self, state, data):
        backbone = state['backbone']
        adapter = state['adapter']
        class_head = state['class_head']

        def fwd():
            h = adapter(backbone(Data(x=data.x, edge_index=data.edge_index)))
            return class_head(h)

        return fwd

    def get_training_step_fn(self, state, data):
        backbone = state['backbone']
        adapter = state['adapter']
        proj_head = state['proj_head']
        class_head = state['class_head']
        tau = state['tau']
        alpha = state['alpha']
        pr = state['pr']

        def step_fn():
            edge_idx1, _ = dropout_edge(data.edge_index, p=pr, training=True)
            edge_idx2, _ = dropout_edge(data.edge_index, p=pr, training=True)
            x1, _ = mask_feature(data.x, p=pr)
            x2, _ = mask_feature(data.x, p=pr)

            h1 = adapter(backbone(Data(x=x1, edge_index=edge_idx1)))
            h2 = adapter(backbone(Data(x=x2, edge_index=edge_idx2)))

            z1 = proj_head(h1)
            z2 = proj_head(h2)
            loss_con = contrastive_loss_original_style(z1, z2, tau)

            p1 = class_head(h1)
            loss_sup = F.nll_loss(p1[data.train_mask], data.y[data.train_mask])

            return alpha * loss_con + loss_sup

        return step_fn
