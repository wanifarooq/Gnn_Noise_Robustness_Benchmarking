"""UnionNET method helper — label-noise robust training via KNN support sets.

Builds per-node support sets from k-nearest neighbors in the graph, then
combines a reweighted cross-entropy loss, a standard cross-entropy loss, and
a KL divergence regularizer to handle noisy labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base_helper import MethodHelper
from methods.registry import register_helper


@register_helper('unionnet')
class UnionNETHelper(MethodHelper):
    """UnionNET: KNN support-set reweighting for noisy-label robustness."""

    def supports_batched_training(self):
        return True

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        unet_params = config.get('unionnet_params', {})

        lr = float(training_cfg.get('lr', 0.01))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))

        k = int(unet_params.get('k', 5))
        alpha = float(unet_params.get('alpha', 0.5))
        beta = float(unet_params.get('beta', 1.0))
        feat_norm = bool(unet_params.get('feat_norm', True))

        num_classes = init_data.get('num_classes', int(data.y.max().item()) + 1)

        backbone_model.to(device)

        # Optionally normalize features in-place on the data object
        if feat_norm:
            row_sum = data.x.sum(dim=1, keepdim=True).clamp(min=1.0)
            data.x = data.x / row_sum

        # Build sparse adjacency (with self-loops)
        num_nodes = data.num_nodes
        adj = torch.sparse_coo_tensor(
            data.edge_index,
            torch.ones(data.edge_index.size(1), device=device),
            [num_nodes, num_nodes],
        ).coalesce()
        identity = torch.eye(num_nodes, device=device).to_sparse()
        adjacency = (adj + identity).coalesce()

        optimizer = torch.optim.Adam(
            backbone_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        return {
            'models': [backbone_model],
            'optimizers': [optimizer],
            'backbone': backbone_model,
            'optimizer': optimizer,
            'device': device,
            'adjacency': adjacency,
            'k': k,
            'alpha': alpha,
            'beta': beta,
            'num_classes': num_classes,
        }

    # ── Train step ─────────────────────────────────────────────────────────

    def train_step(self, state, data, epoch):
        model = state['backbone']
        optimizer = state['optimizer']
        device = state['device']
        k = state['k']
        alpha = state['alpha']
        beta = state['beta']
        num_classes = state['num_classes']

        model.train()
        optimizer.zero_grad(set_to_none=True)

        out = model(data)

        train_mask = data.train_mask
        labels = data.y  # already noisy on train nodes

        # ── Build support sets (KNN among graph neighbours) ────────────
        features = data.x
        edge_index = data.edge_index
        n_nodes = features.size(0)
        feat_dim = features.size(1)

        support_features = torch.zeros(n_nodes, k, feat_dim, device=device)
        support_labels = torch.zeros(n_nodes, k, dtype=torch.long, device=device)

        for node_idx in range(n_nodes):
            if not train_mask[node_idx]:
                continue
            neighbor_nodes = edge_index[1][edge_index[0] == node_idx]
            if len(neighbor_nodes) >= k:
                anchor = features[node_idx].unsqueeze(0)
                sims = torch.mm(features[neighbor_nodes], anchor.T).squeeze()
                _, top_k = torch.topk(sims, k=k)
                support_features[node_idx] = features[neighbor_nodes[top_k]]
                support_labels[node_idx] = labels[neighbor_nodes[top_k]]

        # ── Aggregate labels from support set ──────────────────────────
        train_features = features[train_mask]
        train_labels = labels[train_mask]

        sup_feat = support_features[train_mask]   # [n_train, k, feat_dim]
        sup_lab = support_labels[train_mask]       # [n_train, k]
        n_train = train_features.size(0)

        class_probs = torch.zeros(n_train, num_classes, device=device)
        for i in range(n_train):
            if sup_feat[i].sum() != 0:
                sims = torch.exp(
                    torch.mm(sup_feat[i], train_features[i:i + 1].T)
                ).squeeze()
                weights = sims / sims.sum()
                for j, c in enumerate(sup_lab[i]):
                    class_probs[i, c] += weights[j]

        # ── Loss components ────────────────────────────────────────────
        # Reweighted CE
        confidence = class_probs[range(n_train), train_labels]
        per_sample_ce = F.cross_entropy(out[train_mask], train_labels, reduction='none')
        reweighted_loss = (confidence * per_sample_ce).mean()

        # Standard CE
        standard_loss = F.cross_entropy(out[train_mask], train_labels)

        # KL divergence
        one_hot = F.one_hot(train_labels, num_classes=num_classes).float()
        log_probs = F.log_softmax(out[train_mask], dim=1)
        kl_loss = F.kl_div(log_probs, one_hot, reduction='batchmean')

        loss = alpha * reweighted_loss + (1 - alpha) * standard_loss + beta * kl_loss
        loss.backward()
        optimizer.step()

        return {'train_loss': loss.item()}
