"""GNN_Cleaner method helper — label propagation + dual-optimizer cleaning.

Wraps the GNNCleanerTrainer's per-epoch logic into the shared
TrainingLoop + MethodHelper system.

GNN_Cleaner uses dual optimizers (GNN model + label weighting network),
an expanding clean-sample set identified via label propagation, and
sample-wise label correction for nodes where the pseudo-label disagrees
with the given (noisy) label.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

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

        # Label weighting network (same architecture as GNNCleanerTrainer)
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

        # Initialize clean set
        initial_clean_mask, expanding_clean_mask = self._initialize_clean_set(
            data, device, gc_params.get('clean_set_ratio', 0.1)
        )

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

        model.train()
        weighting_net.train()

        noisy_labels = data.y.to(device)
        ground_truth = data.y_original.to(device) if hasattr(data, 'y_original') else data.y.to(device)

        # Forward pass
        node_embeddings = model(data)

        # Build similarity matrix from embeddings
        similarity_matrix = self._build_similarity_matrix(
            data.edge_index, node_embeddings.detach(), sim_eps
        )

        # Label propagation from expanding clean set
        propagated_labels = self._label_propagation(
            similarity_matrix, ground_truth,
            state['expanding_clean_mask'], data, num_classes, device,
            label_prop_iters,
        )

        # Sample selection: D_select (pseudo==given), D_left (pseudo!=given)
        train_indices = data.train_mask.nonzero(as_tuple=True)[0]
        D_select = torch.zeros_like(data.train_mask, dtype=torch.bool)
        D_left = torch.zeros_like(data.train_mask, dtype=torch.bool)

        for idx in train_indices:
            pseudo = propagated_labels[idx].argmax().item()
            given = noisy_labels[idx].item()
            if pseudo == given:
                D_select[idx] = True
            else:
                D_left[idx] = True

        # Update expanding clean set
        state['expanding_clean_mask'] = (
            state['expanding_clean_mask'] | D_select
        ).detach()

        total_loss = 0.0

        # Phase 1: Train on selected (consistent) nodes
        if D_select.sum() > 0:
            selected_emb = node_embeddings[D_select]
            selected_labels = ground_truth[D_select]
            select_loss = F.cross_entropy(selected_emb, selected_labels)

            gnn_optimizer.zero_grad(set_to_none=True)
            select_loss.backward(retain_graph=True)
            gnn_optimizer.step()
            total_loss += select_loss.item()

        # Phase 2: Label correction for inconsistent nodes
        if D_left.sum() > 0:
            updated_emb = model(data)
            left_indices = D_left.nonzero(as_tuple=True)[0]

            corrected_loss = torch.tensor(0.0, device=device)
            for idx in left_indices:
                y_hat = updated_emb[idx].unsqueeze(0)
                given_label = noisy_labels[idx].unsqueeze(0)
                pseudo_dist = propagated_labels[idx].unsqueeze(0)

                l1 = F.cross_entropy(y_hat, given_label, reduction='none')
                l2 = -(pseudo_dist * F.log_softmax(y_hat, dim=1)).sum(dim=1)

                lam = weighting_net(torch.stack([l1.detach(), l2.detach()], dim=1))
                given_onehot = F.one_hot(given_label, num_classes).float()
                corrected = lam * given_onehot + (1 - lam) * pseudo_dist
                node_loss = -(corrected * F.log_softmax(y_hat, dim=1)).sum()
                corrected_loss = corrected_loss + node_loss

            if corrected_loss > 0:
                avg_corr_loss = corrected_loss / len(left_indices)

                gnn_optimizer.zero_grad(set_to_none=True)
                avg_corr_loss.backward(retain_graph=True)
                gnn_optimizer.step()

                # Update weighting net on clean set
                clean_indices = state['initial_clean_mask'].nonzero(as_tuple=True)[0]
                if len(clean_indices) > 0:
                    final_emb = model(data)
                    clean_emb = final_emb[clean_indices]
                    clean_labels = ground_truth[clean_indices]
                    clean_loss = F.cross_entropy(clean_emb, clean_labels)

                    net_optimizer.zero_grad(set_to_none=True)
                    clean_loss.backward()
                    net_optimizer.step()

                total_loss += avg_corr_loss.item()

        return {'train_loss': total_loss}

    # ── Validation ─────────────────────────────────────────────────────────

    def compute_val_loss(self, state, data):
        model = state['backbone']
        model.eval()
        with torch.no_grad():
            out = model(data)
            ground_truth = data.y_original if hasattr(data, 'y_original') else data.y
            val_idx = data.val_mask.nonzero(as_tuple=True)[0]
            return F.cross_entropy(out[val_idx], ground_truth[val_idx]).item()

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
        """Identify initially clean training nodes (matching GNNCleanerTrainer)."""
        if not hasattr(data, 'y_original'):
            raise ValueError("GNN_Cleaner requires y_original to identify clean samples")

        train_indices = data.train_mask.nonzero(as_tuple=True)[0]
        clean_train = []
        for idx in train_indices:
            if data.y[idx] == data.y_original[idx]:
                clean_train.append(idx.item())

        clean_train = torch.tensor(clean_train, device=device)

        needed = int(train_indices.size(0) * clean_set_ratio)
        if len(clean_train) < needed:
            remaining = train_indices[~torch.isin(train_indices, clean_train)]
            extra = needed - len(clean_train)
            if len(remaining) > 0:
                additional = remaining[torch.randperm(len(remaining))[:extra]]
                clean_train = torch.cat([clean_train, additional])

        max_clean = max(1, needed)
        clean_train = clean_train[:max_clean]

        initial_clean_mask = torch.zeros(data.y.size(0), dtype=torch.bool, device=device)
        initial_clean_mask[clean_train] = True
        expanding_clean_mask = initial_clean_mask.clone()

        return initial_clean_mask, expanding_clean_mask

    @staticmethod
    def _build_similarity_matrix(edge_index, embeddings, eps):
        """Build normalised similarity matrix from node embeddings."""
        num_nodes = embeddings.size(0)
        src, tgt = edge_index.cpu().numpy()

        weights, rows, cols = [], [], []
        emb_np = embeddings.cpu().numpy()

        for i in range(len(src)):
            s, t = int(src[i]), int(tgt[i])
            if s != t:
                dist = np.linalg.norm(emb_np[s] - emb_np[t]) + eps
                weights.append(1.0 / dist)
                rows.append(s)
                cols.append(t)

        sim = sparse.coo_matrix((weights, (rows, cols)), shape=(num_nodes, num_nodes))
        degrees = np.array(sim.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1.0
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
        return D_inv_sqrt @ sim @ D_inv_sqrt

    @staticmethod
    def _label_propagation(sim_matrix, ground_truth, clean_mask, data,
                           num_classes, device, iterations):
        """Run label propagation from clean nodes."""
        num_nodes = ground_truth.size(0)
        label_probs = torch.zeros(num_nodes, num_classes, device=device)

        clean_indices = clean_mask.nonzero(as_tuple=True)[0]
        for idx in clean_indices:
            if hasattr(data, 'y_original'):
                lbl = data.y_original[idx].item()
            else:
                lbl = ground_truth[idx].item()
            label_probs[idx, lbl] = 1.0

        sim_tensor = torch.from_numpy(sim_matrix.toarray()).float().to(device)

        for _ in range(iterations):
            label_probs = torch.matmul(sim_tensor, label_probs)
            # Reset clean node labels
            for idx in clean_indices:
                if hasattr(data, 'y_original'):
                    lbl = data.y_original[idx].item()
                else:
                    lbl = ground_truth[idx].item()
                label_probs[idx] = 0.0
                label_probs[idx, lbl] = 1.0

        return label_probs
