"""GNN_Cleaner method helper — label propagation + learnable label correction.

Faithful implementation of "GNN Cleaner: Label Cleaner for Graph-Structured
Data" (Xia et al., IEEE TKDE 2023).

Per epoch:
    1. Propagate labels from a trusted (clean) set over a node-representation
       similarity graph built with SPARSE ops (no dense N x N materialisation,
       no python edge loops). Similarity is computed on node representations
       (model.get_embeddings / features) — NOT on the classifier logits.
    2. Clean-sample selection: nodes whose propagated label agrees with the
       given (noisy) label form the supervised set D_select; the remaining
       training nodes D_left are relabelled.
    3. Learnable interpolation corrector (Eq):
           y_corrected = w * y_propagated + (1 - w) * y_given
       where w is produced by a small network from the (CE_given, CE_prop)
       pair, trained jointly with the GNN in a single optimizer step.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base_helper import MethodHelper
from methods.registry import register_helper


@register_helper('gnn_cleaner')
class GNNCleanerHelper(MethodHelper):
    """MethodHelper for the GNN_Cleaner robustness method."""

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        gc_params = config.get('gnn_cleaner_params', {})

        model_lr = float(training_cfg.get('lr', 0.01))
        net_lr = float(gc_params.get('net_learning_rate', model_lr))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))
        label_prop_iterations = int(gc_params.get('label_propagation_iterations', 50))
        similarity_epsilon = float(gc_params.get('similarity_epsilon', 1e-8))

        num_classes = int(data.y.max().item()) + 1
        backbone_model = backbone_model.to(device)

        # Learnable interpolation corrector: maps (CE_given, CE_prop) -> w in (0,1).
        label_weighting_net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        ).to(device)

        gnn_optimizer = torch.optim.Adam(
            backbone_model.parameters(), lr=model_lr, weight_decay=weight_decay,
        )
        net_optimizer = torch.optim.Adam(
            label_weighting_net.parameters(), lr=net_lr,
        )

        # Trusted (clean) set used as the propagation source.
        initial_clean_mask, expanding_clean_mask = self._initialize_clean_set(
            data, device, gc_params.get('clean_set_ratio', 0.1)
        )

        # Symmetric-normalised sparse adjacency used for propagation. The edge
        # weights are recomputed each epoch from node representations, but the
        # COO structure (indices, degree bookkeeping) is built once here.
        edge_index = data.edge_index.to(device)

        return {
            'models': [backbone_model],
            'optimizers': [gnn_optimizer, net_optimizer],
            'backbone': backbone_model,
            'weighting_net': label_weighting_net,
            'gnn_optimizer': gnn_optimizer,
            'net_optimizer': net_optimizer,
            'initial_clean_mask': initial_clean_mask,
            'expanding_clean_mask': expanding_clean_mask,
            'num_classes': num_classes,
            'device': device,
            'label_prop_iterations': label_prop_iterations,
            'similarity_epsilon': similarity_epsilon,
            'edge_index': edge_index,
        }

    # ── Per-epoch training ─────────────────────────────────────────────────

    def train_step(self, state, data, epoch):
        model = state['backbone']
        weighting_net = state['weighting_net']
        gnn_optimizer = state['gnn_optimizer']
        net_optimizer = state['net_optimizer']
        device = state['device']
        num_classes = state['num_classes']
        label_prop_iters = state['label_prop_iterations']
        sim_eps = state['similarity_epsilon']
        edge_index = state['edge_index']

        model.train()
        weighting_net.train()

        # data.y holds noisy train+val labels (clean only on test). We NEVER
        # read y_original — propagation seeds from the trusted set's noisy labels.
        noisy_labels = data.y.to(device)

        # ── Build the propagation operator from NODE REPRESENTATIONS ───────
        # Similarity is computed on embeddings (the paper propagates over node
        # representation similarity), NOT on classifier logits. Sparse ops only.
        with torch.no_grad():
            embeddings = model.get_embeddings(data).detach()
            prop_edge_index, prop_edge_weight = self._build_sparse_operator(
                edge_index, embeddings, data.num_nodes, sim_eps, device,
            )

            propagated = self._label_propagation(
                prop_edge_index, prop_edge_weight, noisy_labels,
                state['expanding_clean_mask'], data.num_nodes, num_classes,
                device, label_prop_iters,
            )
            propagated_label = propagated.argmax(dim=1)

        # ── Clean-sample selection (propagated == given) ───────────────────
        train_mask = data.train_mask.to(device)
        agree = propagated_label == noisy_labels
        D_select = train_mask & agree            # supervised CE on given labels
        D_left = train_mask & (~agree)           # relabel via corrector

        # Grow the trusted set with the newly-agreeing nodes.
        state['expanding_clean_mask'] = (
            state['expanding_clean_mask'] | D_select
        ).detach()

        # ── Single forward / single optimizer step ─────────────────────────
        gnn_optimizer.zero_grad(set_to_none=True)
        net_optimizer.zero_grad(set_to_none=True)

        out = model(data)                        # logits [N, C]
        log_p = F.log_softmax(out, dim=1)

        loss = out.new_zeros(())

        # Supervised CE on agreeing (clean) nodes.
        if D_select.any():
            sel_idx = D_select.nonzero(as_tuple=True)[0]
            loss = loss + F.cross_entropy(out[sel_idx], noisy_labels[sel_idx])

        # Learnable label correction on disagreeing nodes:
        #   y_corrected = w * y_propagated + (1 - w) * y_given
        if D_left.any():
            left_idx = D_left.nonzero(as_tuple=True)[0]
            given_oh = F.one_hot(noisy_labels[left_idx], num_classes).float()
            prop_dist = propagated[left_idx]                       # soft propagated dist

            ce_given = F.nll_loss(log_p[left_idx], noisy_labels[left_idx],
                                  reduction='none')                # [n_left]
            ce_prop = -(prop_dist * log_p[left_idx]).sum(dim=1)    # [n_left]

            # w from the learnable net; detached losses as features (the net is
            # supervised through the corrected-label CE below).
            net_in = torch.stack([ce_given.detach(), ce_prop.detach()], dim=1)
            w = weighting_net(net_in)                              # [n_left, 1]

            y_corrected = w * prop_dist + (1.0 - w) * given_oh     # [n_left, C]
            corr_loss = -(y_corrected * log_p[left_idx]).sum(dim=1).mean()
            loss = loss + corr_loss

        loss.backward()
        gnn_optimizer.step()
        net_optimizer.step()

        return {'train_loss': float(loss.item())}

    # ── Validation ─────────────────────────────────────────────────────────

    def compute_val_loss(self, state, data):
        model = state['backbone']
        model.eval()
        with torch.no_grad():
            out = model(data)
            val_idx = data.val_mask.nonzero(as_tuple=True)[0]
            return F.cross_entropy(out[val_idx], data.y[val_idx]).item()

    # ── Checkpointing ──────────────────────────────────────────────────────

    def get_checkpoint_state(self, state):
        return {
            'backbone': deepcopy(state['backbone'].state_dict()),
            'weighting_net': deepcopy(state['weighting_net'].state_dict()),
            'initial_clean_mask': state['initial_clean_mask'].clone(),
            'expanding_clean_mask': state['expanding_clean_mask'].clone(),
        }

    def load_checkpoint_state(self, state, checkpoint):
        if 'backbone' in checkpoint:
            state['backbone'].load_state_dict(checkpoint['backbone'])
        if 'weighting_net' in checkpoint:
            state['weighting_net'].load_state_dict(checkpoint['weighting_net'])
        if 'initial_clean_mask' in checkpoint:
            state['initial_clean_mask'] = checkpoint['initial_clean_mask']
        if 'expanding_clean_mask' in checkpoint:
            state['expanding_clean_mask'] = checkpoint['expanding_clean_mask']

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _initialize_clean_set(data, device, clean_set_ratio=0.1):
        """Initialize the trusted set by random sampling from training nodes.

        No privileged access to y_original — in a real-world setting we don't
        know which labels are clean.  We randomly select a subset of training
        nodes as the initial trusted set.  The expanding clean set is refined
        during training via label propagation and consistency checks.
        """
        train_indices = data.train_mask.nonzero(as_tuple=True)[0]
        needed = max(1, int(train_indices.size(0) * clean_set_ratio))

        perm = torch.randperm(train_indices.size(0), device=device)
        clean_train = train_indices[perm[:needed]]

        initial_clean_mask = torch.zeros(data.y.size(0), dtype=torch.bool, device=device)
        initial_clean_mask[clean_train] = True
        expanding_clean_mask = initial_clean_mask.clone()

        return initial_clean_mask, expanding_clean_mask

    @staticmethod
    def _build_sparse_operator(edge_index, embeddings, num_nodes, eps, device):
        """Build a symmetric-normalised sparse propagation operator.

        Edge weights are representation-similarity based: w_ij = 1 / (||h_i - h_j|| + eps).
        Self-loops are dropped. Returns (edge_index, edge_weight) of the
        normalised operator D^-1/2 W D^-1/2 — no dense N x N matrix and no
        python per-edge loop.
        """
        src, dst = edge_index[0], edge_index[1]
        non_self = src != dst
        src, dst = src[non_self], dst[non_self]

        dist = (embeddings[src] - embeddings[dst]).norm(dim=1) + eps
        w = 1.0 / dist                                            # [E]

        # Degree per row, then symmetric normalisation.
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, src, w)
        deg_inv_sqrt = deg.clamp(min=eps).pow(-0.5)
        norm_w = deg_inv_sqrt[src] * w * deg_inv_sqrt[dst]

        return torch.stack([src, dst], dim=0), norm_w

    @staticmethod
    def _label_propagation(edge_index, edge_weight, noisy_labels, clean_mask,
                           num_nodes, num_classes, device, iterations):
        """Iterative label propagation from the trusted set via sparse spmm.

        No access to y_original — propagation seeds from the given (noisy)
        labels of the trusted set, clamped each iteration.
        """
        src, dst = edge_index[0], edge_index[1]

        seed = torch.zeros(num_nodes, num_classes, device=device)
        clean_idx = clean_mask.nonzero(as_tuple=True)[0]
        if clean_idx.numel() > 0:
            seed[clean_idx] = F.one_hot(
                noisy_labels[clean_idx], num_classes
            ).float()

        label_probs = seed.clone()
        for _ in range(iterations):
            # Sparse message passing: aggregate dst -> src with edge weights.
            msg = label_probs[dst] * edge_weight.unsqueeze(1)     # [E, C]
            label_probs = torch.zeros(num_nodes, num_classes, device=device)
            label_probs.index_add_(0, src, msg)
            # Row-normalize back to a valid distribution. Without this, the
            # sub-stochastic operator makes rows decay toward ~0 away from the
            # seed set, so argmax becomes arbitrary and the corrected-label loss
            # degenerates (this was destroying gnn_cleaner's training signal).
            label_probs = label_probs / label_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
            # Clamp trusted nodes back to their seed one-hot.
            if clean_idx.numel() > 0:
                label_probs[clean_idx] = seed[clean_idx]

        return label_probs
