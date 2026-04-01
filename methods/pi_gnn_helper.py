"""PI-GNN method helper — pairwise interaction GNN with MI regularization.

Wraps the backbone in a PiGnnModel (backbone + GraphLinkDecoder) and trains
a separate mutual-information (MI) model.  Two optimizers: one for the main
model (classification + context-aware link regularization) and one for the
MI model (link reconstruction).
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from methods.base_helper import MethodHelper
from methods.registry import register_helper
from model.methods.PI_GNN import PiGnnModel, GraphLinkDecoder


@register_helper('pi_gnn')
class PiGnnHelper(MethodHelper):
    """PI-GNN: context-aware MI regularization for graph robustness."""

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        pi_params = config.get('pi_gnn_params', {})

        lr = float(training_cfg.get('lr', 0.01))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))
        mi_start_epoch = int(pi_params.get('start_epoch', 200))
        use_self_mi = bool(pi_params.get('miself', False))
        norm_factor = pi_params.get('norm', None)
        vanilla = bool(pi_params.get('vanilla', False))

        num_classes = init_data.get('num_classes', int(data.y.max().item()) + 1)

        backbone_model.to(device)

        # Main model: backbone + link decoder
        link_decoder = GraphLinkDecoder()
        main_model = PiGnnModel(
            backbone_gnn=backbone_model,
            supplementary_decoder=link_decoder,
        ).to(device)

        # MI model: fresh backbone copy + link decoder
        get_model = init_data.get('get_model')
        model_cfg = config.get('model', {})
        mi_backbone = get_model(
            model_name=model_cfg.get('name', 'GCN'),
            in_channels=data.num_features,
            hidden_channels=model_cfg.get('hidden_channels', 64),
            out_channels=num_classes,
            n_layers=model_cfg.get('n_layers', 2),
            dropout=model_cfg.get('dropout', 0.5),
            mlp_layers=model_cfg.get('mlp_layers', 2),
            train_eps=model_cfg.get('train_eps', True),
            heads=model_cfg.get('heads', 8),
            self_loop=model_cfg.get('self_loop', True),
        )
        mi_link_decoder = GraphLinkDecoder()
        mi_model = PiGnnModel(
            backbone_gnn=mi_backbone,
            supplementary_decoder=mi_link_decoder,
        ).to(device)

        # Two optimizers
        main_optimizer = torch.optim.Adam(
            main_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        mi_optimizer = torch.optim.Adam(
            mi_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Adjacency target for link prediction
        edges = data.edge_index.t().cpu().numpy()
        edge_weights = np.ones(edges.shape[0])
        adj_csr = sp.csr_matrix(
            (edge_weights, (edges[:, 0], edges[:, 1])),
            shape=(data.num_nodes, data.num_nodes),
        )
        sym_adj = (adj_csr + adj_csr.T) / 2
        adj_with_loops = sym_adj + sp.eye(sym_adj.shape[0])
        adjacency_target = torch.FloatTensor(adj_with_loops.toarray()).to(device)

        positive_edge_weight = torch.tensor(
            [float(data.num_nodes ** 2 - len(edge_weights)) / len(edge_weights)],
            device=device,
        )

        if norm_factor is not None and norm_factor != 10000:
            loss_norm = norm_factor
        else:
            loss_norm = data.num_nodes ** 2 / float(
                (data.num_nodes ** 2 - len(edge_weights)) * 2
            )

        return {
            'models': [main_model, mi_model],
            'optimizers': [main_optimizer, mi_optimizer],
            'backbone': backbone_model,
            'main_model': main_model,
            'mi_model': mi_model,
            'main_optimizer': main_optimizer,
            'mi_optimizer': mi_optimizer,
            'device': device,
            'adjacency_target': adjacency_target,
            'positive_edge_weight': positive_edge_weight,
            'loss_norm': loss_norm,
            'mi_start_epoch': mi_start_epoch,
            'use_self_mi': use_self_mi,
            'vanilla': vanilla,
        }

    # ── Train step ─────────────────────────────────────────────────────────

    def train_step(self, state, data, epoch):
        main_model = state['main_model']
        mi_model = state['mi_model']
        main_opt = state['main_optimizer']
        mi_opt = state['mi_optimizer']
        adj_target = state['adjacency_target']
        pos_weight = state['positive_edge_weight']
        loss_norm = state['loss_norm']
        mi_start = state['mi_start_epoch']
        use_self_mi = state['use_self_mi']
        vanilla = state['vanilla']
        device = state['device']

        main_model.train()
        mi_model.train()
        main_opt.zero_grad(set_to_none=True)
        mi_opt.zero_grad(set_to_none=True)

        main_cls, main_link = main_model(data)
        mi_cls, mi_link = mi_model(data)

        train_labels = data.y[data.train_mask]

        # MI reconstruction loss (computed now, backward later)
        mi_recon_loss = loss_norm * F.binary_cross_entropy_with_logits(
            mi_link, adj_target, pos_weight=pos_weight
        )

        # Classification loss
        cls_loss = F.nll_loss(main_cls[data.train_mask], train_labels)

        # Context-aware regularization
        ctx_loss = 0
        if not vanilla:
            if epoch > mi_start:
                # Importance-weighted link loss
                importance = torch.zeros_like(mi_link).view(-1).to(device)
                pos_mask = adj_target.view(-1).bool()
                neg_mask = ~pos_mask

                if use_self_mi:
                    sig_pred = torch.sigmoid(main_link).view(-1)
                else:
                    sig_pred = torch.sigmoid(mi_link).view(-1)

                importance[pos_mask] = sig_pred[pos_mask]
                importance[neg_mask] = 1 - sig_pred[neg_mask]
                importance = importance.view(adj_target.size(0), adj_target.size(1))

                weighted = F.binary_cross_entropy_with_logits(
                    main_link, adj_target,
                    pos_weight=pos_weight, reduction='none',
                ) * importance
                ctx_loss = loss_norm * weighted.mean()
            else:
                ctx_loss = loss_norm * F.binary_cross_entropy_with_logits(
                    main_link, adj_target, pos_weight=pos_weight
                )

        # Combined backward: MI gets gradients from both mi_recon_loss and ctx_loss
        total_loss = cls_loss + ctx_loss + mi_recon_loss
        total_loss.backward()
        main_opt.step()
        mi_opt.step()

        return {'train_loss': total_loss.item()}

    # ── Validation ─────────────────────────────────────────────────────────

    def compute_val_loss(self, state, data):
        model = state['main_model']
        model.eval()
        with torch.no_grad():
            cls_out, _ = model(data)
            val_idx = data.val_mask.nonzero(as_tuple=True)[0]
            return F.nll_loss(cls_out[val_idx], data.y[val_idx]).item()

    # ── Predictions / embeddings ───────────────────────────────────────────

    def get_predictions(self, state, data):
        model = state['main_model']
        model.eval()
        with torch.no_grad():
            cls_out, _ = model(data)
            return cls_out.argmax(dim=1)

    def get_embeddings(self, state, data):
        backbone = state['main_model'].backbone_gnn
        backbone.eval()
        with torch.no_grad():
            return backbone.get_embeddings(data)

    # ── Checkpointing ─────────────────────────────────────────────────────

    def get_checkpoint_state(self, state):
        return {
            'main_model': deepcopy(state['main_model'].state_dict()),
            'mi_model': deepcopy(state['mi_model'].state_dict()),
        }

    def load_checkpoint_state(self, state, checkpoint):
        state['main_model'].load_state_dict(checkpoint['main_model'])
        state['mi_model'].load_state_dict(checkpoint['mi_model'])

    # ── Profiling ──────────────────────────────────────────────────────────

    def get_inference_forward_fn(self, state, data):
        model = state['main_model']
        return lambda: model(data)

    def get_training_step_fn(self, state, data):
        main_model = state['main_model']
        adj_target = state['adjacency_target']
        pos_weight = state['positive_edge_weight']
        loss_norm = state['loss_norm']

        def step_fn():
            cls_out, link_out = main_model(data)
            cls_loss = F.nll_loss(cls_out[data.train_mask], data.y[data.train_mask])
            link_loss = loss_norm * F.binary_cross_entropy_with_logits(
                link_out, adj_target, pos_weight=pos_weight
            )
            return cls_loss + link_loss

        return step_fn
