"""
Smoke tests for all 13 training methods.

Runs each method end-to-end on Cora with 5 epochs, uniform noise at 0.2.
Expensive params are reduced to keep total runtime under ~5 minutes on CPU.
"""
import copy
import math
import sys
import os
import pytest
import torch

# Ensure project root is on sys.path so `from util import ...` works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.experiment import run_experiment

# ── base config (mirrors config.yaml with reduced expensive params) ──────────

BASE_CONFIG = {
    'seed': 42,
    'device': 'cpu',

    'dataset': {
        'name': 'cora',
        'root': 'data',
    },

    'noise': {
        'type': 'uniform',
        'rate': 0.2,
        'seed': 42,
    },

    'model': {
        'name': 'gcn',
        'hidden_channels': 16,
        'n_layers': 2,
        'dropout': 0.5,
        'self_loop': True,
        'mlp_layers': 1,
        'train_eps': False,
        'heads': 8,
        'use_pe': False,
        'pe_dim': 8,
    },

    'training': {
        'method': 'standard',       # overridden per test
        'lr': 0.001,
        'weight_decay': 5e-4,
        'epochs': 5,
        'patience': 10,
    },

    # ── method-specific params (reduced for speed) ───────────────────────

    'positive_eigenvalues_params': {
        'batch_size': 32,
    },

    'gcod_params': {
        'batch_size': 32,
        'uncertainty_lr': 1.0,
    },

    'nrgnn_params': {
        'edge_hidden': 16,
        'n_p': 2,          # reduced from 10
        'p_u': 0.7,
        'alpha': 0.05,
        'beta': 1.0,
        't_small': 0.1,
        'n_n': 5,          # reduced from 50
    },

    'pi_gnn_params': {
        'start_epoch': 1,   # must be < epochs to exercise MI path
        'miself': False,
        'norm': None,
        'vanilla': False,
    },

    'cr_gnn_params': {
        'T': 2,
        'tau': 0.6,
        'p': 0.9,
        'alpha': 0.2,
        'beta': 0.9,
        'pr': 0.3,
    },

    'community_defense_params': {
        'community_method': 'louvain',
        'num_communities': None,
        'lambda_comm': 1.0,
        'pos_weight': 2.0,
        'neg_weight': 2.0,
        'margin': 1.5,
        'num_neg_samples': 3,
    },

    'rtgnn_params': {
        'edge_hidden': 16,
        'co_lambda': 0.1,
        'alpha': 0.3,
        'th': 0.8,
        'K': 3,            # reduced from 50
        'tau': 0.05,
        'n_neg': 10,        # reduced from 100
    },

    'graphcleaner_params': {
        'k': 5,
        'sample_rate': 0.5,
        'max_iter_classifier': 50,   # reduced from 5000
        'held_split': 'valid',
    },

    'unionnet_params': {
        'k': 10,
        'alpha': 0.5,
        'beta': 1,
        'feat_norm': True,
    },

    'gnn_cleaner_params': {
        'label_propagation_iterations': 5,  # reduced from 50
        'similarity_epsilon': 1e-8,
    },

    'erase_params': {
        'n_embedding': 32,          # reduced from 512
        'n_heads': 8,
        'use_layer_norm': False,
        'use_residual': False,
        'use_residual_linear': False,
        'gam1': 1.0,
        'gam2': 2.0,
        'eps': 0.05,
        'alpha': 0.6,
        'beta': 0.6,
        'T': 3,
    },

    'gnnguard_params': {
        'P0': 0.5,
        'K': 2,
        'D2': 16,
        'attention': True,
    },
}

ALL_METHODS = [
    'standard',
    'positive_eigenvalues',
    'gcod',
    'nrgnn',
    'pi_gnn',
    'cr_gnn',
    'community_defense',
    'rtgnn',
    'graphcleaner',
    'unionnet',
    'gnn_cleaner',
    'erase',
    'gnnguard',
]

OVERSMOOTHING_KEYS = {'NumRank', 'Erank', 'EDir', 'EDir_traditional', 'EProj', 'MAD'}


def _make_config(method):
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg['training']['method'] = method
    return cfg


@pytest.mark.parametrize('method', ALL_METHODS)
def test_model_smoke(method):
    """Run one method end-to-end and validate the result dict."""
    config = _make_config(method)
    result = run_experiment(config, run_id=1)

    # ── required top-level keys ──────────────────────────────────────────
    for key in ('accuracy', 'f1', 'precision', 'recall',
                'oversmoothing', 'train_oversmoothing', 'compute_info'):
        assert key in result, f"Missing key '{key}' in result for method '{method}'"

    # ── classification metrics in [0, 1] and finite ──────────────────────
    for metric in ('accuracy', 'f1', 'precision', 'recall'):
        val = result[metric]
        assert isinstance(val, (int, float)), (
            f"{metric} should be numeric, got {type(val)} for method '{method}'"
        )
        assert math.isfinite(val), (
            f"{metric} is not finite ({val}) for method '{method}'"
        )
        assert 0.0 <= val <= 1.0, (
            f"{metric} out of [0,1] range ({val}) for method '{method}'"
        )

    # ── oversmoothing dict has the 6 standard keys, all finite ───────────
    os_dict = result['oversmoothing']
    assert isinstance(os_dict, dict), (
        f"oversmoothing should be dict, got {type(os_dict)} for method '{method}'"
    )
    for os_key in OVERSMOOTHING_KEYS:
        assert os_key in os_dict, (
            f"Missing oversmoothing key '{os_key}' for method '{method}'"
        )
        assert math.isfinite(os_dict[os_key]), (
            f"oversmoothing['{os_key}'] is not finite ({os_dict[os_key]}) for method '{method}'"
        )

    # ── train_oversmoothing is a dict ────────────────────────────────────
    assert isinstance(result['train_oversmoothing'], dict), (
        f"train_oversmoothing should be dict for method '{method}'"
    )

    # ── val_oversmoothing is a dict ──────────────────────────────────────
    assert 'val_oversmoothing' in result, (
        f"Missing key 'val_oversmoothing' in result for method '{method}'"
    )
    assert isinstance(result['val_oversmoothing'], dict), (
        f"val_oversmoothing should be dict for method '{method}'"
    )

    # ── compute_info has 4 compute metrics with valid values ──────────────
    ci = result['compute_info']
    for ckey in ('flops_inference', 'flops_training_total',
                 'time_training_total', 'time_inference'):
        assert ckey in ci, (
            f"compute_info missing '{ckey}' for method '{method}'"
        )
    assert ci['flops_inference'] > 0, (
        f"flops_inference should be positive for method '{method}'"
    )
    assert ci['flops_training_total'] > 0, (
        f"flops_training_total should be positive for method '{method}'"
    )
    assert ci['time_training_total'] > 0, (
        f"time_training_total should be positive for method '{method}'"
    )
    assert ci['time_inference'] > 0, (
        f"time_inference should be positive for method '{method}'"
    )


# ── Checkpoint round-trip ────────────────────────────────────────────────────

@pytest.mark.parametrize('method', ['standard', 'positive_eigenvalues'])
def test_checkpoint_roundtrip(method, tmp_path):
    """Train with checkpoint -> verify file -> eval-only -> verify results."""
    config = _make_config(method)
    ckpt = str(tmp_path / f"{method}.pt")

    # Normal run — saves checkpoint
    result_train = run_experiment(config, run_id=1, checkpoint_path=ckpt)
    assert os.path.exists(ckpt), "Checkpoint file not created"

    # Checkpoint contains backbone key
    state = torch.load(ckpt, weights_only=False)
    assert 'backbone' in state

    # Eval-only run — loads checkpoint
    result_eval = run_experiment(config, run_id=1, checkpoint_path=ckpt,
                                eval_only=True)

    # Result structure is valid
    for key in ('accuracy', 'f1', 'precision', 'recall',
                'oversmoothing', 'train_oversmoothing', 'compute_info'):
        assert key in result_eval, f"Missing key '{key}' in eval-only result"

    # Eval-only should have zero training metrics and positive inference time
    assert result_eval['compute_info']['time_training_total'] == 0.0
    assert result_eval['compute_info']['flops_training_total'] == 0
    assert result_eval['compute_info']['time_inference'] > 0

    # Classification metrics should match (same weights, same data)
    assert result_eval['accuracy'] == pytest.approx(result_train['accuracy'], abs=1e-5)
    assert result_eval['f1'] == pytest.approx(result_train['f1'], abs=1e-5)


def test_eval_only_blocked_for_unsupported_method(tmp_path):
    """eval_only raises NotImplementedError for methods that don't support it."""
    config = _make_config('cr_gnn')
    ckpt = str(tmp_path / "cr_gnn.pt")

    # Normal run — saves checkpoint
    run_experiment(config, run_id=1, checkpoint_path=ckpt)
    assert os.path.exists(ckpt)

    # eval_only should raise for cr_gnn (supports_eval_only = False)
    with pytest.raises(NotImplementedError, match="does not support eval_only"):
        run_experiment(config, run_id=1, checkpoint_path=ckpt, eval_only=True)
