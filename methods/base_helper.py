"""MethodHelper ABC — per-method logic called by the shared TrainingLoop.

Each robustness method implements a MethodHelper subclass that provides only
the method-specific operations.  The shared training loop (training/training_loop.py)
handles epoch iteration, early stopping, metric computation, oversmoothing,
logging, and checkpointing.

Lifecycle (called by TrainingLoop):
    1. setup()           — build models, optimizers, any method state
    2. pre_train()       — optional multi-phase warmup (e.g. noise detection)
    3. for each epoch:
         a. train_step()         — one training step (owns gradient flow)
         b. compute_val_loss()   — scalar for early stopping
         c. get_predictions()    — integer class predictions for metrics
         d. get_embeddings()     — hidden representations for oversmoothing
    4. get_checkpoint_state() / load_checkpoint_state()  — serialisation
    5. get_predictions() / get_embeddings()  — final evaluation

Batched mode (activated by ``training.batch_size``):
    Methods that support mini-batch training override
    ``supports_batched_training()`` → True and implement
    ``train_step_batched()`` / ``compute_val_loss_batched()``.

    Batched **inference** (``get_predictions_batched`` /
    ``get_embeddings_batched``) is available to ALL methods via the default
    implementation which iterates a NeighborLoader and scatters results
    back to a full-graph tensor.
"""

from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MethodHelper(ABC):
    """Abstract base for method-specific logic.

    The ``state`` dict is created by ``setup()`` and threaded through every
    subsequent call.  Methods can store anything they need: models, optimizers,
    edge indices, adjacency matrices, sklearn classifiers, etc.

    Convention for ``state`` keys:
        - 'models'      : list[nn.Module]  — toggled train/eval by the loop
        - 'optimizers'  : list[Optimizer]  — for reference / profiling
        - 'backbone'    : nn.Module        — the original backbone (may or may not be used)
        - (method-specific keys are free-form)
    """

    # ── Phase 0: Setup ────────────────────────────────────────────────────

    @abstractmethod
    def setup(self, backbone_model: nn.Module, data, config: dict,
              device: torch.device, init_data: dict) -> dict:
        """Create method-specific models, optimizers, and state.

        Args:
            backbone_model: The shared GNN backbone (GCN/GIN/GAT/GATv2/GPS).
            data: PyG Data object (with noisy train labels, clean val/test).
            config: Full experiment configuration dict.
            device: Torch device.
            init_data: Full init_data dict from initialize_experiment (for
                       access to masks, noise info, num_classes, etc.).

        Returns:
            State dict that will be passed to all subsequent hooks.
            Must include at minimum:
                'models': list[nn.Module]
                'optimizers': list[torch.optim.Optimizer]
        """

    # ── Phase 1: Pre-training (optional) ──────────────────────────────────

    def pre_train(self, state: dict, data, config: dict) -> None:
        """Optional pre-training phase.

        Override for multi-phase methods (e.g. GraphCleaner: train disposable
        model -> detect noise -> modify train_mask before main loop).

        May mutate ``data`` (e.g. modify train_mask, labels) or ``state``.
        """

    # ── Phase 2: Per-epoch hooks ──────────────────────────────────────────

    @abstractmethod
    def train_step(self, state: dict, data, epoch: int) -> dict:
        """Execute one training step (full-batch).

        This method OWNS the full gradient flow: zero_grad, forward, loss
        computation, backward, optimizer step.  The shared loop does not
        interpose on gradients because some methods need multiple backward
        passes, dual optimizers, or retain_graph.

        For mini-batch methods, iterate over data loaders internally.

        Args:
            state: Method state dict from setup().
            data: PyG Data object.
            epoch: Current epoch index (0-based).

        Returns:
            Dict with at least:
                'train_loss': float — training loss for logging
            Optionally:
                'val_loss': float — if computed during training step
                (any other method-specific diagnostics)
        """

    def compute_val_loss(self, state: dict, data) -> float:
        """Compute validation loss for early stopping.

        Default: cross-entropy on val_mask using the primary model.
        Override for methods with custom validation (e.g. mini-batch, custom
        loss functions).

        Args:
            state: Method state dict.
            data: PyG Data object.

        Returns:
            Scalar validation loss.
        """
        model = state['models'][0]
        model.eval()
        with torch.no_grad():
            out = model(data)
            val_idx = data.val_mask.nonzero(as_tuple=True)[0]
            return F.cross_entropy(out[val_idx], data.y[val_idx]).item()

    def get_predictions(self, state: dict, data) -> torch.Tensor:
        """Return integer class predictions for all nodes.

        Default: primary model forward -> argmax.
        Override for methods with custom inference paths (e.g. ERASE uses
        LogisticRegression, NRGNN uses augmented edges).

        Args:
            state: Method state dict.
            data: PyG Data object.

        Returns:
            Tensor[num_nodes] of integer class labels.
        """
        model = state['models'][0]
        model.eval()
        with torch.no_grad():
            return model(data).argmax(dim=1)

    def get_embeddings(self, state: dict, data) -> torch.Tensor:
        """Return hidden embeddings for oversmoothing metrics.

        Default: primary model get_embeddings().
        Override if the method uses a wrapper model or non-standard embedding
        extraction.

        Args:
            state: Method state dict.
            data: PyG Data object.

        Returns:
            Tensor[num_nodes, hidden_dim].
        """
        model = state['models'][0]
        model.eval()
        with torch.no_grad():
            return model.get_embeddings(data)

    # ── Phase 2b: Batched hooks ───────────────────────────────────────────

    def supports_batched_training(self) -> bool:
        """Whether this method can train in mini-batch mode.

        Methods whose algorithms are inherently full-graph (e.g. NRGNN edge
        augmentation, RTGNN structure estimation) return False.  They can
        still benefit from batched *inference* via
        ``get_predictions_batched`` / ``get_embeddings_batched``.
        """
        return False

    def train_step_batched(self, state: dict, loaders, data, epoch: int) -> dict:
        """Mini-batch training step.

        Only called when ``supports_batched_training()`` is True and
        ``training.batch_size`` is set.

        Default implementation iterates ``loaders.train_loader`` and calls the
        primary model with CE loss on seed (target) nodes.  Override for
        non-trivial training logic.

        Args:
            state: Method state dict.
            loaders: ``GraphLoaders`` from ``util.graph_sampling``.
            data: Full PyG Data object (for reference / masks).
            epoch: Current epoch (0-based).

        Returns:
            Dict with at least ``'train_loss': float``.
        """
        from util.graph_sampling import get_seed_indices

        model = state['models'][0]
        optimizer = state['optimizers'][0]
        device = state.get('device', next(model.parameters()).device)

        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loaders.train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            out = model(batch)
            n_seed = get_seed_indices(batch, loaders.sampler_type)

            # Loss on seed/target nodes only
            seed_mask = batch.train_mask[:n_seed]
            idx = seed_mask.nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue

            loss = F.cross_entropy(out[idx], batch.y[idx])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {'train_loss': total_loss / max(n_batches, 1)}

    def compute_val_loss_batched(self, state: dict, loaders, data) -> float:
        """Batched validation loss.  Default: CE over val_loader."""
        from util.graph_sampling import get_seed_indices

        model = state['models'][0]
        device = state.get('device', next(model.parameters()).device)

        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            loader = loaders.val_loader or loaders.inference_loader
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                n_seed = get_seed_indices(batch, loaders.sampler_type)

                seed_mask = batch.val_mask[:n_seed]
                idx = seed_mask.nonzero(as_tuple=True)[0]
                if len(idx) == 0:
                    continue

                loss = F.cross_entropy(out[idx], batch.y[idx])
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def get_predictions_batched(self, state: dict, loaders, data) -> torch.Tensor:
        """Batched inference — collect predictions for ALL nodes.

        Iterates ``loaders.inference_loader`` and scatters argmax results
        back to a full-graph tensor.  Works for any method whose primary
        model accepts a PyG Data / batch object.

        Override if the method has a custom inference pipeline (e.g. ERASE
        with LogisticRegression, or NRGNN with augmented edges).
        """
        from util.graph_sampling import get_seed_indices, get_global_ids

        model = state['models'][0]
        device = state.get('device', next(model.parameters()).device)

        model.eval()
        all_preds = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

        with torch.no_grad():
            for batch in loaders.inference_loader:
                batch = batch.to(device)
                out = model(batch)
                n_seed = get_seed_indices(batch, loaders.sampler_type)
                global_ids = get_global_ids(batch, loaders.sampler_type)

                if global_ids is not None:
                    all_preds[global_ids] = out[:n_seed].argmax(dim=1)

        return all_preds

    def get_embeddings_batched(self, state: dict, loaders, data) -> torch.Tensor:
        """Batched inference — collect hidden embeddings for ALL nodes.

        Iterates ``loaders.inference_loader`` and scatters embeddings
        back to a full-graph tensor.
        """
        from util.graph_sampling import get_seed_indices, get_global_ids

        model = state['models'][0]
        device = state.get('device', next(model.parameters()).device)

        model.eval()
        # Determine embedding dim from first batch
        all_embeds = None

        with torch.no_grad():
            for batch in loaders.inference_loader:
                batch = batch.to(device)
                emb = model.get_embeddings(batch)
                n_seed = get_seed_indices(batch, loaders.sampler_type)
                global_ids = get_global_ids(batch, loaders.sampler_type)

                if all_embeds is None:
                    all_embeds = torch.zeros(
                        data.num_nodes, emb.size(1), device=device,
                    )

                if global_ids is not None:
                    all_embeds[global_ids] = emb[:n_seed]

        if all_embeds is None:
            # Fallback: empty tensor (should never happen with valid data)
            all_embeds = torch.zeros(data.num_nodes, 1, device=device)

        return all_embeds

    # ── Phase 3: Checkpointing ────────────────────────────────────────────

    def get_checkpoint_state(self, state: dict) -> dict:
        """Serialize all state needed for checkpoint save/restore.

        Default: save state_dict of all models in state['models'].
        Override to include additional state (edge indices, sklearn models, etc.).

        Returns:
            Dict suitable for torch.save().
        """
        return {
            f'model_{i}': deepcopy(m.state_dict())
            for i, m in enumerate(state['models'])
        }

    def load_checkpoint_state(self, state: dict, checkpoint: dict) -> None:
        """Restore state from a checkpoint dict.

        Default: load state_dicts into state['models'].
        Override to restore additional state.
        """
        for i, m in enumerate(state['models']):
            key = f'model_{i}'
            if key in checkpoint:
                m.load_state_dict(checkpoint[key])

    # ── Phase 4: Profiling ────────────────────────────────────────────────

    def get_inference_forward_fn(self, state: dict, data):
        """Return a zero-arg callable for FLOPs profiling (inference).

        Default: primary model forward pass.
        Override for multi-model inference pipelines.

        Returns:
            Callable that performs one inference forward pass.
        """
        model = state['models'][0]
        return lambda: model(data)

    def get_training_step_fn(self, state: dict, data):
        """Return a zero-arg callable for FLOPs profiling (one training step).

        Default: primary model forward + CE loss.
        Override for methods with complex training steps.

        Returns:
            Callable that returns a scalar loss tensor.
        """
        model = state['models'][0]

        def step_fn():
            out = model(data)
            return F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        return step_fn
