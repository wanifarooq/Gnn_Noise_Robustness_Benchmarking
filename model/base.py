"""BaseTrainer ABC — unified interface for all 13 GNN robustness trainers."""

from abc import ABC, abstractmethod

import numpy as np


class BaseTrainer(ABC):
    """Common interface for experiment trainers.

    Subclasses implement ``run()`` which trains + evaluates and returns
    the standardised 7-key result dict via ``_make_result()``.
    """

    def __init__(self, init_data: dict, config: dict):
        self.init_data = init_data
        self.config = config

    @abstractmethod
    def run(self) -> dict:
        """Train + evaluate. Return standardised 7-key result dict."""

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
