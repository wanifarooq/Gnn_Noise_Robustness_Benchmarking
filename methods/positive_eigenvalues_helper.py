"""Positive Eigenvalues method helper — SVD-constrained mini-batch training.

After every optimizer step the final square Linear layer's weight matrix is
reconstructed to keep only positive singular values.  Training uses
NeighborLoader for mini-batch sampling.

Mirrors the logic in model/methods/Positive_Eigenvalues.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from methods.base_helper import MethodHelper
from methods.registry import register_helper


# ---------------------------------------------------------------------------
# SVD constraint utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_matrix_with_positive_singular_values(weight_matrix, eps=1e-8):
    """Reconstruct *weight_matrix* keeping only positive singular values."""
    U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    positive_mask = S > eps
    if positive_mask.sum() == 0:
        return (
            torch.eye(
                weight_matrix.size(0), weight_matrix.size(1),
                device=weight_matrix.device,
            ) * 0.01
        )
    S_pos = S[positive_mask]
    U_pos = U[:, positive_mask]
    Vh_pos = Vh[positive_mask, :]
    return U_pos @ torch.diag(S_pos) @ Vh_pos


@torch.no_grad()
def apply_positive_eigenvalue_constraint(model):
    """Find the last square Linear layer and enforce positive singular values."""
    for _name, module in reversed(list(model.named_modules())):
        if (
            isinstance(module, nn.Linear)
            and module.weight.dim() == 2
            and module.weight.size(0) == module.weight.size(1)
        ):
            module.weight.data = reconstruct_matrix_with_positive_singular_values(
                module.weight.data,
            )
            break


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

@register_helper('positive_eigenvalues')
class PositiveEigenvaluesHelper(MethodHelper):
    """Mini-batch training with positive-eigenvalue SVD constraint."""

    def supports_batched_training(self):
        # Returns False — batching is handled internally in train_step via NeighborLoaders.
        # Returning True would bypass train_step and lose the eigenvalue constraint.
        return False

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        lr = float(training_cfg.get('lr', 0.01))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))
        pe_params = config.get('positive_eigenvalues_params', {})
        batch_size = int(pe_params.get('batch_size', 32))

        backbone_model.to(device)
        data = data.to(device)

        optimizer = torch.optim.Adam(
            backbone_model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        # --- NeighborLoaders ---
        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        val_idx = data.val_mask.nonzero(as_tuple=True)[0]

        train_loader = NeighborLoader(
            data,
            num_neighbors=[15, 10],
            batch_size=batch_size,
            input_nodes=train_idx,
            shuffle=True,
        )
        val_loader = NeighborLoader(
            data,
            num_neighbors=[15, 10],
            batch_size=batch_size,
            input_nodes=val_idx,
            shuffle=False,
        )

        return {
            'models': [backbone_model],
            'optimizers': [optimizer],
            'backbone': backbone_model,
            'optimizer': optimizer,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'device': device,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, state, data, epoch):
        model = state['backbone']
        optimizer = state['optimizer']
        device = state['device']
        train_loader = state['train_loader']

        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(batch)

            # Only seed (target) nodes carry valid masks
            batch_train_mask = batch.train_mask[:batch.batch_size]
            target_idx = batch_train_mask.nonzero(as_tuple=True)[0]
            if len(target_idx) == 0:
                continue

            loss = F.cross_entropy(logits[target_idx], batch.y[target_idx])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        # PE-1 Fix: Apply positive eigenvalue constraint only once at the end of the epoch.
        # This significantly reduces FLOPs while maintaining the robustness benefit.
        apply_positive_eigenvalue_constraint(model)

        avg_loss = total_loss / max(batch_count, 1)
        return {'train_loss': avg_loss}

    # ------------------------------------------------------------------
    # Validation (mini-batch)
    # ------------------------------------------------------------------

    def compute_val_loss(self, state, data):
        model = state['backbone']
        device = state['device']
        val_loader = state['val_loader']

        model.eval()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)

                batch_val_mask = batch.val_mask[:batch.batch_size]
                target_idx = batch_val_mask.nonzero(as_tuple=True)[0]
                if len(target_idx) == 0:
                    continue

                loss = F.cross_entropy(logits[target_idx], batch.y[target_idx])
                total_loss += loss.item()
                batch_count += 1

        return total_loss / max(batch_count, 1)

    # ------------------------------------------------------------------
    # Predictions & embeddings — full-graph forward (not mini-batch)
    # ------------------------------------------------------------------

    def get_predictions(self, state, data):
        model = state['backbone']
        model.eval()
        with torch.no_grad():
            return model(data).argmax(dim=1)

    def get_embeddings(self, state, data):
        model = state['backbone']
        model.eval()
        with torch.no_grad():
            return model.get_embeddings(data)
