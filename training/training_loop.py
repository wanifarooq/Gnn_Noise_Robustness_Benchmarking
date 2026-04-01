"""Shared training loop — single implementation used by all methods.

Handles epoch iteration, early stopping, metric computation (classification +
oversmoothing), epoch logging, and checkpoint management.  Method-specific
logic is delegated to a MethodHelper instance via hooks.

Supports three modes:
    - **Full-batch transductive** (default): entire graph, mask-based splits.
    - **Mini-batch transductive**: data loaders + mask-based splits.
    - **Inductive**: disjoint train/val/test subgraphs (optional batching).
"""

from collections import defaultdict

import torch

from training.early_stopping import EarlyStopping
from evaluation.metrics import (
    ClassificationMetrics,
    OversmoothingMetrics,
    compute_oversmoothing_for_mask,
)


class TrainingLoop:
    """Unified training loop that delegates method-specific logic to a helper.

    Args:
        helper: MethodHelper instance providing method-specific hooks.
        log_epoch_fn: Callable matching BaseTrainer.log_epoch signature.
            Used to record per-epoch metrics, save checkpoints, etc.
        verbose: If True, print epoch metrics at oversmoothing checkpoints.
    """

    def __init__(self, helper, log_epoch_fn=None, verbose=True):
        self.helper = helper
        self.log_epoch_fn = log_epoch_fn
        self.verbose = verbose

    def run(self, backbone_model, data, config, device, init_data) -> dict:
        """Execute the full training loop.

        Args:
            backbone_model: GNN backbone model.
            data: PyG Data object (noisy train labels, clean val/test).
                In inductive mode this is the *train* subgraph.
            config: Full experiment configuration dict.
            device: Torch device.
            init_data: Full init_data dict from initialize_experiment.

        Returns:
            Dict with keys:
                'train_oversmoothing': dict[str, list[float]]
                'val_oversmoothing': dict[str, list[float]]
                'stopped_at_epoch': int (0-based index of last epoch)
        """
        # ── Extract training parameters ───────────────────────────────────
        training_cfg = config.get('training', {})
        total_epochs = int(training_cfg.get('epochs', 200))
        patience = int(training_cfg.get('patience', 20))
        oversmoothing_every = int(training_cfg.get('oversmoothing_every', 20))

        # ── Mode detection ────────────────────────────────────────────────
        mode = training_cfg.get('mode', 'transductive').lower()
        is_inductive = mode == 'inductive'

        # In inductive mode, separate subgraphs for each split
        train_data = init_data.get('train_subgraph', data) if is_inductive else data
        val_data = init_data.get('val_subgraph', data) if is_inductive else data
        # full_data keeps the original graph (needed for global pred mapping)
        full_data = init_data.get('data_for_training', data)

        # ── Setup ─────────────────────────────────────────────────────────
        state = self.helper.setup(backbone_model, train_data, config, device, init_data)
        self._state = state  # Available immediately for checkpoint during training
        self.helper.pre_train(state, train_data, config)

        # ── Batching ──────────────────────────────────────────────────────
        batch_size = training_cfg.get('batch_size')
        use_batched_training = (
            batch_size is not None
            and self.helper.supports_batched_training()
        )
        use_batched_eval = batch_size is not None

        loaders = None
        val_loaders = None
        if use_batched_training or use_batched_eval:
            from util.graph_sampling import create_loaders
            loaders = create_loaders(train_data, config, device)
            if is_inductive and val_data is not train_data:
                val_loaders = create_loaders(val_data, config, device)
            else:
                val_loaders = loaders

        if batch_size is not None and not self.helper.supports_batched_training():
            method_name = type(self.helper).__name__
            print(
                f"[WARNING] {method_name} does not support batched training. "
                f"Using full-batch training with batched evaluation."
            )

        # ── Metric evaluators ─────────────────────────────────────────────
        cls_evaluator = ClassificationMetrics(average='macro')
        oversmoothing_evaluator = OversmoothingMetrics(device=device)

        # ── Early stopping ────────────────────────────────────────────────
        warmup = min(50, total_epochs // 3)
        early_stopping = EarlyStopping(patience=patience, warmup_epochs=warmup)

        # ── Oversmoothing accumulators ────────────────────────────────────
        per_epochs_oversmoothing = defaultdict(list)
        per_epochs_val_oversmoothing = defaultdict(list)

        # ── Epoch loop ────────────────────────────────────────────────────
        last_epoch = 0
        for epoch in range(total_epochs):
            last_epoch = epoch

            # 1. Method-specific training step
            if use_batched_training:
                train_out = self.helper.train_step_batched(
                    state, loaders, train_data, epoch,
                )
            else:
                train_out = self.helper.train_step(state, train_data, epoch)
            train_loss = train_out.get('train_loss', 0.0)

            # 2. Validation loss
            if 'val_loss' in train_out:
                val_loss = train_out['val_loss']
            elif use_batched_eval and val_loaders and val_loaders.is_batched:
                val_loss = self.helper.compute_val_loss_batched(
                    state, val_loaders, val_data,
                )
            else:
                val_loss = self.helper.compute_val_loss(state, val_data)

            # 3. Predictions for classification metrics
            if use_batched_eval and loaders and loaders.is_batched:
                train_pred = self.helper.get_predictions_batched(
                    state, loaders, train_data,
                )
            else:
                train_pred = self.helper.get_predictions(state, train_data)

            if is_inductive:
                # Separate forward on val subgraph
                if use_batched_eval and val_loaders and val_loaders.is_batched:
                    val_pred = self.helper.get_predictions_batched(
                        state, val_loaders, val_data,
                    )
                else:
                    val_pred = self.helper.get_predictions(state, val_data)

                # In inductive mode all nodes in each subgraph are the split
                train_acc = cls_evaluator.compute_accuracy(train_pred, train_data.y)
                val_acc = cls_evaluator.compute_accuracy(val_pred, val_data.y)
                train_f1 = cls_evaluator.compute_f1(train_pred, train_data.y)
                val_f1 = cls_evaluator.compute_f1(val_pred, val_data.y)

                # Build global predictions tensor for log_epoch
                pred = torch.zeros(
                    full_data.num_nodes, dtype=torch.long, device=device,
                )
                pred[train_data.original_node_ids] = train_pred
                pred[val_data.original_node_ids] = val_pred
            else:
                pred = train_pred  # single prediction tensor for whole graph
                train_idx = data.train_mask.nonzero(as_tuple=True)[0]
                val_idx = data.val_mask.nonzero(as_tuple=True)[0]

                train_acc = cls_evaluator.compute_accuracy(pred[train_idx], data.y[train_idx])
                val_acc = cls_evaluator.compute_accuracy(pred[val_idx], data.y[val_idx])
                train_f1 = cls_evaluator.compute_f1(pred[train_idx], data.y[train_idx])
                val_f1 = cls_evaluator.compute_f1(pred[val_idx], data.y[val_idx])

            # 4. Early stopping (based on validation accuracy)
            should_stop = early_stopping.step(val_acc, epoch)
            is_best = early_stopping.is_best

            # 5. Oversmoothing metrics (periodic)
            os_entry = None
            if epoch % oversmoothing_every == 0 or epoch == total_epochs - 1:
                if use_batched_eval and loaders and loaders.is_batched:
                    train_embeddings = self.helper.get_embeddings_batched(
                        state, loaders, train_data,
                    )
                else:
                    train_embeddings = self.helper.get_embeddings(state, train_data)

                if is_inductive:
                    # Oversmoothing on train subgraph (all nodes)
                    all_true = torch.ones(
                        train_data.num_nodes, dtype=torch.bool, device=device,
                    )
                    train_oversmoothing = compute_oversmoothing_for_mask(
                        oversmoothing_evaluator, train_embeddings,
                        train_data.edge_index, all_true,
                    )

                    # Val subgraph
                    if use_batched_eval and val_loaders and val_loaders.is_batched:
                        val_embeddings = self.helper.get_embeddings_batched(
                            state, val_loaders, val_data,
                        )
                    else:
                        val_embeddings = self.helper.get_embeddings(state, val_data)

                    all_true_val = torch.ones(
                        val_data.num_nodes, dtype=torch.bool, device=device,
                    )
                    val_oversmoothing = compute_oversmoothing_for_mask(
                        oversmoothing_evaluator, val_embeddings,
                        val_data.edge_index, all_true_val,
                    )
                else:
                    train_oversmoothing = compute_oversmoothing_for_mask(
                        oversmoothing_evaluator, train_embeddings,
                        data.edge_index, data.train_mask,
                    )
                    val_oversmoothing = compute_oversmoothing_for_mask(
                        oversmoothing_evaluator, train_embeddings,
                        data.edge_index, data.val_mask,
                    )

                for key, value in train_oversmoothing.items():
                    per_epochs_oversmoothing[key].append(value)
                for key, value in val_oversmoothing.items():
                    per_epochs_val_oversmoothing[key].append(value)

                os_entry = {
                    'train': dict(train_oversmoothing),
                    'val': dict(val_oversmoothing),
                }

                if self.verbose:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                        f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}"
                    )
                    print(
                        f"Train DE: {train_oversmoothing['EDir']:.4f}, "
                        f"Val DE: {val_oversmoothing['EDir']:.4f} | "
                        f"Train MAD: {train_oversmoothing['MAD']:.4f}, "
                        f"Val MAD: {val_oversmoothing['MAD']:.4f} | "
                        f"Train NumRank: {train_oversmoothing['NumRank']:.4f}, "
                        f"Val NumRank: {val_oversmoothing['NumRank']:.4f}"
                    )

            # 6. Log epoch (checkpointing, noise-split metrics, etc.)
            if self.log_epoch_fn is not None:
                self.log_epoch_fn(
                    epoch, train_loss, val_loss, train_acc, val_acc,
                    train_f1=train_f1, val_f1=val_f1,
                    oversmoothing=os_entry, is_best=is_best,
                    train_predictions=pred,
                )

            # 7. Check early stopping
            if should_stop:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        return {
            'train_oversmoothing': dict(per_epochs_oversmoothing),
            'val_oversmoothing': dict(per_epochs_val_oversmoothing),
            'stopped_at_epoch': last_epoch,
        }

    @property
    def state(self) -> dict:
        """Access the method state after run() completes.

        Used by the trainer to delegate evaluate/checkpoint/profiling calls
        to the helper using the trained state.
        """
        if not hasattr(self, '_state'):
            raise RuntimeError("TrainingLoop.state accessed before run()")
        return self._state
