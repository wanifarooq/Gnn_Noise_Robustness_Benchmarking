"""Unified evaluation metrics — classification and oversmoothing.

This module re-exports the existing metrics from model.evaluation so that
the new training loop can import from the canonical location (evaluation.metrics)
while the old code continues to work unchanged.

During Phase 6 cleanup, the actual implementations will be moved here and
model.evaluation will become a thin backward-compat re-export layer.
"""

from model.evaluation import (
    ClassificationMetrics,
    OversmoothingMetrics,
    OVERSMOOTHING_KEYS,
    DEFAULT_OVERSMOOTHING,
    ZERO_CLS,
    get_noise_split_indices,
    compute_train_noise_split_cls,
    compute_val_noise_split_cls,
    compute_oversmoothing_for_mask,
    evaluate_model,
    normalize_metrics,
)

__all__ = [
    'ClassificationMetrics',
    'OversmoothingMetrics',
    'OVERSMOOTHING_KEYS',
    'DEFAULT_OVERSMOOTHING',
    'ZERO_CLS',
    'get_noise_split_indices',
    'compute_train_noise_split_cls',
    'compute_val_noise_split_cls',
    'compute_oversmoothing_for_mask',
    'evaluate_model',
    'normalize_metrics',
]
