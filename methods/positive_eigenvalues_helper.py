"""Positive Eigenvalues method helper.

Implements the "negative-eigenvalue removal" strategy from
Wani et al., "Energy Guided Smoothness to Improve Robustness in Graph
Classification" (arXiv:2412.08419).  After each optimizer step, every square
weight matrix W of the backbone is eigendecomposed (W = Phi mu Phi^-1) and its
negative eigenvalues are clipped to zero (W+ = Phi [mu]+ Phi^-1).  Negative
eigenvalues induce edge-gradient *expansion* (sharpening); removing them biases
the layer toward smoothing, which reduces Dirichlet energy and improves
robustness to label noise.

Training is full-batch on the shared graph (one optimizer step per epoch), so
"after every optimizer step" == "after every epoch".  This keeps the method on
the same full-batch footing as the other benchmark methods (no internal
NeighborLoader), so the comparison is not confounded by the sampling regime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base_helper import MethodHelper
from methods.registry import register_helper


# ---------------------------------------------------------------------------
# Negative-eigenvalue removal (paper Eq. 16-17)
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_matrix_with_positive_eigenvalues(W):
    """Return W with its negative eigenvalues clipped to zero.

    W = Phi diag(mu) Phi^-1  ->  W+ = Phi diag(relu(mu)) Phi^-1.
    W must be square.  Eigenvalues of a general real matrix may be complex; we
    clip the real part to >= 0 (dropping the negative-eigenvalue directions that
    sharpen) and take the real part of the reconstruction.
    """
    try:
        eigvals, eigvecs = torch.linalg.eig(W)
    except Exception:
        return W  # leave untouched if decomposition fails

    clipped = torch.clamp(eigvals.real, min=0.0).to(eigvals.dtype)
    try:
        W_plus = (eigvecs @ torch.diag(clipped) @ torch.linalg.inv(eigvecs)).real
    except Exception:
        return W  # singular eigenvector matrix -> skip
    return W_plus.to(dtype=W.dtype)


@torch.no_grad()
def apply_positive_eigenvalue_constraint(model):
    """Clip negative eigenvalues of every square 2-D weight matrix in the model.

    Eigendecomposition is only defined for square matrices, so non-square
    projections (e.g. in_features != hidden, hidden != num_classes) are skipped;
    the constraint acts on the square hidden->hidden weight matrices where the
    smoothing/sharpening inductive bias lives.
    """
    for p in model.parameters():
        if p.dim() == 2 and p.size(0) == p.size(1) and p.size(0) > 1:
            p.data.copy_(reconstruct_matrix_with_positive_eigenvalues(p.data))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

@register_helper('positive_eigenvalues')
class PositiveEigenvaluesHelper(MethodHelper):
    """Full-batch training with the negative-eigenvalue-removal constraint."""

    def supports_batched_training(self):
        # The eigenvalue constraint is applied inside train_step after the step,
        # so we own the gradient flow and keep training full-batch.
        return False

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        lr = float(training_cfg.get('lr', 0.01))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))

        backbone_model.to(device)
        optimizer = torch.optim.Adam(
            backbone_model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        return {
            'models': [backbone_model],
            'optimizers': [optimizer],
            'backbone': backbone_model,
            'optimizer': optimizer,
            'criterion': nn.CrossEntropyLoss(),
            'device': device,
        }

    def train_step(self, state, data, epoch):
        model = state['backbone']
        optimizer = state['optimizer']
        criterion = state['criterion']

        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(data)

        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        # Negative-eigenvalue removal right after the parameter update,
        # excluded from the gradient computation (paper Eq. 16-17).
        apply_positive_eigenvalue_constraint(model)

        return {'train_loss': loss.item()}
