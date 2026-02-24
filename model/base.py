"""BaseTrainer ABC — unified interface for all 13 GNN robustness trainers."""

from abc import ABC, abstractmethod

import torch
import numpy as np

from model.evaluation import evaluate_model


class BaseTrainer(ABC):
    """Common interface for experiment trainers.

    Subclasses implement ``train()`` which trains and returns oversmoothing
    dicts.  The default ``run()`` orchestrates train → evaluate → result.
    """

    def __init__(self, init_data: dict, config: dict):
        self.init_data = init_data
        self.config = config

    # ── template ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Train + evaluate. Return standardised result dict.

        Subclasses that still override ``run()`` bypass this template —
        nothing breaks.  Once migrated, they only override ``train()``
        (and optionally ``evaluate()``).
        """
        train_out = self.train()
        eval_result = self.evaluate()
        return self._make_result(
            eval_result,
            train_out.get('train_oversmoothing', {}),
            train_out.get('val_oversmoothing'),
            reduce=train_out.get('reduce', True),
        )

    def train(self) -> dict:
        """Train the model. Return dict with at least
        ``train_oversmoothing`` and ``val_oversmoothing`` keys."""
        raise NotImplementedError

    def evaluate(self) -> dict:
        """Evaluate after training. Default: backbone_model(data) on test split."""
        d = self.init_data
        model = d['backbone_model']
        data = d['data_for_training']
        device = d['device']

        model.eval()
        with torch.no_grad():
            def get_predictions():
                return model(data).argmax(dim=1)

            def get_embeddings():
                return model(data)

            return evaluate_model(
                get_predictions, get_embeddings, data.y,
                data.train_mask, data.val_mask, data.test_mask,
                data.edge_index, device,
            )

    # ── checkpoint ───────────────────────────────────────────────────────

    def get_checkpoint_state(self) -> dict:
        """Return serialisable state dict for saving to disk."""
        return {'backbone': self.init_data['backbone_model'].state_dict()}

    def load_checkpoint_state(self, state: dict) -> None:
        """Restore model weights from a checkpoint state dict."""
        self.init_data['backbone_model'].load_state_dict(state['backbone'])

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _reduce_oversmoothing(oversmoothing_dict: dict) -> dict:
        """Average per-epoch lists into scalar values."""
        return {
            k: float(np.mean(v)) if isinstance(v, (list, tuple, np.ndarray)) else float(v)
            for k, v in oversmoothing_dict.items()
        }

    def _make_result(self, result_dict: dict, train_oversmoothing: dict,
                     val_oversmoothing: dict | None = None,
                     *, reduce: bool = True) -> dict:
        """Assemble the standardised result dict returned by every trainer.

        Parameters
        ----------
        result_dict : dict
            Must contain keys: accuracy, f1, precision, recall, oversmoothing.
        train_oversmoothing : dict
            Per-epoch oversmoothing metrics collected during training.
        val_oversmoothing : dict or None
            Per-epoch validation oversmoothing metrics collected during training.
        reduce : bool
            If *True* (default), average the per-epoch lists via
            ``_reduce_oversmoothing``.  Set to *False* for models (e.g. GCOD)
            that already return reduced values.
        """
        return {
            'accuracy': result_dict['accuracy'],
            'f1': result_dict['f1'],
            'precision': result_dict['precision'],
            'recall': result_dict['recall'],
            'oversmoothing': result_dict['oversmoothing'],
            'train_oversmoothing': (
                self._reduce_oversmoothing(train_oversmoothing)
                if reduce else train_oversmoothing
            ),
            'val_oversmoothing': (
                self._reduce_oversmoothing(val_oversmoothing)
                if reduce else val_oversmoothing
            ) if val_oversmoothing else {},
            'flops_info': self.init_data['flops_info'],
        }
