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


def _is_constrainable_square(p):
    """A 2-D parameter the eigenvalue constraint can act on (square, > 1x1)."""
    return p.dim() == 2 and p.size(0) == p.size(1) and p.size(0) > 1


def has_square_weight(model):
    """True iff the model already owns a square 2-D weight (e.g. a hidden->hidden
    conv in a >=3-layer net). Used to decide whether PE must synthesise one."""
    return any(_is_constrainable_square(p) for p in model.parameters())


@torch.no_grad()
def apply_positive_eigenvalue_constraint(model):
    """Clip negative eigenvalues of every square 2-D weight matrix in the model.

    Eigendecomposition is only defined for square matrices, so non-square
    projections (e.g. in_features != hidden, hidden != num_classes) are skipped;
    the constraint acts on the square hidden->hidden weight matrices where the
    smoothing/sharpening inductive bias lives.
    """
    for p in model.parameters():
        if _is_constrainable_square(p):
            p.data.copy_(reconstruct_matrix_with_positive_eigenvalues(p.data))


class PESquareHead(nn.Module):
    """Backbone + an appended square readout, used ONLY by positive_eigenvalues.

    The negative-eigenvalue constraint needs a square weight to act on. Shallow
    backbones (e.g. a 2-layer GCN) have only rectangular weights, so the
    constraint is a no-op and PE collapses to `standard`. When that happens we
    append a square ``out_dim x out_dim`` linear on top of the *full, unchanged*
    backbone forward — this is the only placement that adds a square operator
    without dropping a graph hop or altering the backbone's representation, so
    PE stays directly comparable to `standard` in depth.

    The layer is identity-initialised, so at epoch 0 the wrapped model is exactly
    the bare backbone; the constraint then shapes its spectrum during training.
    Deeper backbones already have square hidden->hidden weights, so they are used
    directly and this wrapper is never built (see setup()).
    """

    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.square = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.square.weight)  # identity -> no-op at initialisation

    def forward(self, data):
        return self.square(self.backbone(data))

    def get_embeddings(self, data):
        # Delegate: the smoothing head acts on logits, not the embedding, so the
        # representation other code reads is the backbone's untouched embedding.
        return self.backbone.get_embeddings(data)

    def initialize(self):
        if hasattr(self.backbone, 'initialize'):
            self.backbone.initialize()
        nn.init.eye_(self.square.weight)


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

        # PE needs a square weight to constrain. If the backbone already owns one
        # (any >=3-layer net has square hidden->hidden convs) use it as-is. Only
        # when there is none (shallow backbones, e.g. 2-layer GCN, whose weights
        # are all rectangular) do we append a square out_dim x out_dim readout so
        # the constraint is active instead of a silent no-op. This is PE-only.
        if not has_square_weight(backbone_model):
            num_classes = init_data.get('num_classes', int(data.y.max().item()) + 1)
            backbone_model = PESquareHead(backbone_model, num_classes).to(device)
            print(f"[positive_eigenvalues] backbone has no square weight; "
                  f"appended a square {num_classes}x{num_classes} readout so the "
                  f"negative-eigenvalue constraint is active.")

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
