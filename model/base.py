"""BaseTrainer ABC — unified interface for all 13 GNN robustness trainers."""

import json
import os
import time
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import numpy as np

from model.evaluation import evaluate_model, ZERO_CLS
from sweep_utils import json_serializer


class BaseTrainer(ABC):
    """Common interface for experiment trainers.

    Subclasses implement ``train()`` which trains and returns oversmoothing
    dicts.  The default ``run()`` orchestrates train → evaluate → result.
    """

    #: Override to ``False`` in subclasses whose ``evaluate()`` depends on
    #: internal state created during ``train()`` (e.g. extra models, learned
    #: edges).  When *False*, the eval-only checkpoint path is blocked with a
    #: clear error rather than an opaque ``AttributeError``.
    supports_eval_only: bool = True

    #: (PROBABLY) Approximate ratio of training-step FLOPS to inference FLOPS.
    #: Default: 3 (1× forward + ~2× backward per step).  Override in
    #: subclasses whose training loop is heavier (extra losses, auxiliary
    #: models, iterative steps, etc.).
    TRAINING_FLOPS_MULTIPLIER: int = 3

    def __init__(self, init_data: dict, config: dict):
        self.init_data = init_data
        self.config = config
        self.epoch_log: list = []
        self.best_epoch: int | None = None
        self.best_val_loss: float = float('inf')

    # ── epoch logging ───────────────────────────────────────────────────

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc,
                  *, train_f1=None, val_f1=None, oversmoothing=None,
                  is_best=False):
        """Record one epoch's metrics and optionally save a checkpoint."""
        entry = {
            'epoch': epoch,
            'train_loss': float(train_loss) if train_loss is not None else None,
            'val_loss': float(val_loss) if val_loss is not None else None,
            'train_acc': float(train_acc) if train_acc is not None else None,
            'val_acc': float(val_acc) if val_acc is not None else None,
            'train_f1': float(train_f1) if train_f1 is not None else None,
            'val_f1': float(val_f1) if val_f1 is not None else None,
            'oversmoothing': oversmoothing,
        }
        self.epoch_log.append(entry)

        if is_best:
            self.best_epoch = epoch
            self.best_val_loss = float(val_loss) if val_loss is not None else 0.0

        run_dir = self.init_data.get('run_dir')
        checkpoint_every = self.config.get('training', {}).get('checkpoint_every_epoch', True)
        should_save = (checkpoint_every or is_best) and run_dir and self.supports_eval_only
        if should_save:
            vl = float(val_loss) if val_loss is not None else 0.0
            fname = f"epoch_{epoch:03d}_valloss_{vl:.4f}.pt"
            state = self.get_checkpoint_state()
            torch.save(state, os.path.join(run_dir, fname))

    def save_training_log(self, run_id, config, duration, stopped_at_epoch,
                          final_result):
        """Write training_log.json to the run directory."""
        run_dir = self.init_data.get('run_dir')
        if not run_dir:
            return

        best_ckpt = None
        if self.best_epoch is not None and self.supports_eval_only:
            best_ckpt = f"epoch_{self.best_epoch:03d}_valloss_{self.best_val_loss:.4f}.pt"

        training_params = {
            'method': self.init_data.get('method'),
            'lr': self.init_data.get('lr'),
            'weight_decay': self.init_data.get('weight_decay'),
            'epochs': self.init_data.get('epochs'),
            'patience': self.init_data.get('patience'),
            'oversmoothing_every': self.init_data.get('oversmoothing_every'),
        }

        log = {
            'run_id': run_id,
            'config': config,
            'training_params': training_params,
            'duration_seconds': round(duration, 4),
            'stopped_at_epoch': stopped_at_epoch,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss if self.best_epoch is not None else None,
            'best_checkpoint': best_ckpt,
            'epoch_log': self.epoch_log,
            'final_result': final_result,
        }

        path = os.path.join(run_dir, 'training_log.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, default=json_serializer)

    # ── template ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Train + evaluate. Return standardised result dict."""
        t0 = time.perf_counter()
        train_out = self.train()
        duration = time.perf_counter() - t0

        stopped_at_epoch = train_out.get('stopped_at_epoch')

        flops_result = self.profile_flops()

        t0_eval = time.perf_counter()
        eval_result = self.evaluate()
        time_inference = time.perf_counter() - t0_eval

        epochs = self.init_data.get('epochs', 0)
        self.init_data['compute_info'] = {
            'flops_inference': flops_result['total_flops'],
            'flops_training_total': flops_result['total_flops'] * self.TRAINING_FLOPS_MULTIPLIER * epochs,
            'time_training_total': round(duration, 4),
            'time_inference': round(time_inference, 4),
        }
        result = self._make_result(
            eval_result,
            train_out.get('train_oversmoothing', {}),
            train_out.get('val_oversmoothing'),
            reduce=train_out.get('reduce', True),
        )

        self.save_training_log(
            run_id=self.init_data.get('_run_id', 1),
            config=self.init_data.get('_config', {}),
            duration=duration,
            stopped_at_epoch=stopped_at_epoch,
            final_result=result,
        )

        return result

    @abstractmethod
    def train(self) -> dict:
        """Train the model. Return dict with at least
        ``train_oversmoothing`` and ``val_oversmoothing`` keys."""

    def evaluate(self) -> dict:
        """Evaluate after training.

        Uses backbone_model(data) for classification predictions (out_channels dim)
        and backbone_model.get_embeddings(data) for oversmoothing metrics (hidden_channels dim).
        Override in subclasses that use wrapper models or non-standard evaluation.
        """
        d = self.init_data
        model = d['backbone_model']
        data = d['data_for_training']
        device = d['device']

        model.eval()
        with torch.no_grad():
            def get_predictions():
                return model(data).argmax(dim=1)

            def get_embeddings():
                return model.get_embeddings(data)

            return evaluate_model(
                get_predictions, get_embeddings, data.y,
                data.train_mask, data.val_mask, data.test_mask,
                data.edge_index, device,
            )

    # ── flops ─────────────────────────────────────────────────────────────

    def profile_flops(self) -> dict:
        """Profile FLOPS for one inference forward pass.

        Override in subclasses that use models beyond the backbone (e.g.
        adapter heads, dual branches).
        """
        from util.profiling import profile_model_flops
        d = self.init_data
        return profile_model_flops(d['backbone_model'], d['data_for_training'],
                                   d['device'])

    # ── checkpoint ───────────────────────────────────────────────────────

    def get_checkpoint_state(self) -> dict:
        """Return serialisable snapshot of model weights for saving to disk."""
        return {'backbone': deepcopy(self.init_data['backbone_model'].state_dict())}

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
            Must contain keys: test_cls, train_cls, val_cls, test_oversmoothing.
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
            'test_cls': result_dict.get('test_cls', dict(ZERO_CLS)),
            'train_cls': result_dict.get('train_cls', dict(ZERO_CLS)),
            'val_cls': result_dict.get('val_cls', dict(ZERO_CLS)),
            'test_oversmoothing': result_dict.get('test_oversmoothing', {}),
            'train_oversmoothing': (
                self._reduce_oversmoothing(train_oversmoothing)
                if reduce else train_oversmoothing
            ),
            'val_oversmoothing': (
                self._reduce_oversmoothing(val_oversmoothing)
                if reduce else val_oversmoothing
            ) if val_oversmoothing else {},
            'compute_info': self.init_data['compute_info'],
        }
