"""UnionNET method helper — unified robust training against label noise.

Faithful implementation of "Unified Robust Training for Graph Neural Networks
against Label Noise" (Li, Yin, Chen, PAKDD 2021, arXiv:2103.03414).

For each labelled node v a support set S(v) is formed from its k most similar
nodes, where proximity is measured by INNER PRODUCT in the GNN embedding space
h = model.get_embeddings(data) (Eq 2/3).  From the support set we estimate a
class distribution P(y | v, S) by softmax-weighted label voting, which gives:

    - a sample weight  w = P(y_given | v, S)        -> reweighted CE  J_r (Eq 4)
    - a corrected label y_c = argmax_y P(y | v, S)
      with weight       p_c = max_y P(y | v, S)      -> correction loss J_c (Eq 5/6)

A KL regularizer J_p (Eq 7) keeps the batch-mean predicted distribution close to
the global class prior.  The final objective (Eq 8) is

    J_f = (1 - alpha) * J_r + alpha * J_c + beta * J_p
"""

import torch
import torch.nn.functional as F

from methods.base_helper import MethodHelper
from methods.registry import register_helper


@register_helper('unionnet')
class UnionNETHelper(MethodHelper):
    """UnionNET: support-set label correction for noisy-label robustness."""

    def supports_batched_training(self):
        # Algorithm is inherently full-graph (support sets over all labelled
        # nodes via an embedding similarity matrix). Train full-batch.
        return False

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        unet_params = config.get('unionnet_params', {})

        lr = float(training_cfg.get('lr', 0.01))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))

        k = int(unet_params.get('k', 5))
        alpha = float(unet_params.get('alpha', 0.5))
        beta = float(unet_params.get('beta', 1.0))

        num_classes = init_data.get('num_classes', int(data.y.max().item()) + 1)

        backbone_model.to(device)

        # Global class prior over the (noisy) training labels — target of the
        # KL regularizer J_p (Eq 7). Computed once; does not change.
        train_mask = data.train_mask
        train_labels = data.y[train_mask]
        prior = torch.bincount(train_labels, minlength=num_classes).float()
        prior = (prior + 1e-8) / prior.sum().clamp(min=1.0)
        prior = prior.to(device)

        optimizer = torch.optim.Adam(
            backbone_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        return {
            'models': [backbone_model],
            'optimizers': [optimizer],
            'backbone': backbone_model,
            'optimizer': optimizer,
            'device': device,
            'k': k,
            'alpha': alpha,
            'beta': beta,
            'num_classes': num_classes,
            'class_prior': prior,
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
        prior = state['class_prior']

        model.train()
        optimizer.zero_grad(set_to_none=True)

        out = model(data)  # logits [N, C]

        train_mask = data.train_mask
        labels = data.y  # noisy on train nodes
        train_idx = train_mask.nonzero(as_tuple=True)[0]
        train_labels = labels[train_idx]
        n_train = train_idx.size(0)

        # ── Support-set class distribution P(y | v, S) in embedding space ──
        # Proximity by inner product of GNN embeddings h (paper Eq 2/3),
        # NOT raw features. Detached: the support set defines targets/weights,
        # the gradient flows through `out` (Eq 4/5).
        with torch.no_grad():
            h = model.get_embeddings(data)              # [N, d]
            # L2-normalize -> cosine similarity. Raw inner products blow up in
            # magnitude on wide backbones (hidden=512), so the softmax saturates
            # to a single high-norm neighbour and the support set degenerates.
            h = F.normalize(h, p=2, dim=1)
            h_train = h[train_idx]                       # [n_train, d]

            # Similarity matrix over labelled nodes (cosine, in [-1, 1]).
            sim = h_train @ h_train.t()                  # [n_train, n_train]
            # Exclude self when picking neighbours.
            self_mask = torch.eye(n_train, dtype=torch.bool, device=device)
            sim_masked = sim.masked_fill(self_mask, float('-inf'))

            kk = min(k, n_train - 1) if n_train > 1 else 1
            if n_train > 1:
                top_sim, top_idx = torch.topk(sim_masked, k=kk, dim=1)  # [n_train, kk]
                # Softmax-weight the support set by similarity (Eq 3).
                weights = F.softmax(top_sim, dim=1)                      # [n_train, kk]
                neigh_labels = train_labels[top_idx]                     # [n_train, kk]
                # P(y | v, S): scatter-add softmax weights into class bins.
                class_probs = torch.zeros(n_train, num_classes, device=device)
                class_probs.scatter_add_(1, neigh_labels, weights)
            else:
                class_probs = F.one_hot(train_labels, num_classes).float()

            # Reweighting term w = P(y_given | v, S)  (Eq 4).
            sample_weight = class_probs[torch.arange(n_train, device=device), train_labels]
            # Corrected label y_c and its confidence p_c (Eq 5/6).
            p_c, y_c = class_probs.max(dim=1)

        log_p = F.log_softmax(out[train_idx], dim=1)     # [n_train, C]

        # ── J_r: reweighted cross-entropy (Eq 4) ──────────────────────────
        ce_given = F.nll_loss(log_p, train_labels, reduction='none')
        J_r = (sample_weight * ce_given).mean()

        # ── J_c: correction loss on inferred labels (Eq 5/6) ──────────────
        ce_corrected = F.nll_loss(log_p, y_c, reduction='none')
        J_c = (p_c * ce_corrected).mean()

        # ── J_p: KL(prior || mean predicted distribution) (Eq 7) ──────────
        mean_pred = F.softmax(out[train_idx], dim=1).mean(dim=0)         # [C]
        mean_pred = mean_pred.clamp(min=1e-8)
        J_p = (prior * (prior.clamp(min=1e-8).log() - mean_pred.log())).sum()

        # ── J_f (Eq 8) ────────────────────────────────────────────────────
        loss = (1.0 - alpha) * J_r + alpha * J_c + beta * J_p
        loss.backward()
        optimizer.step()

        return {'train_loss': loss.item()}
