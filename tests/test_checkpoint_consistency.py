"""
Checkpoint consistency tests for all 13 training methods.

Three parametrised test suites:
  A. Inference roundtrip  — train → save → eval_only → metrics match
  B. State roundtrip      — save → load → save → tensors match
  C. Resume viability     — load → train again → no crash
"""
import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.experiment import initialize_experiment, run_experiment
from model.registry import discover_trainers, get_trainer

# ── shared config (mirrors test_smoke.py, epochs=2 for speed) ────────────────

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
        'method': 'standard',
        'lr': 0.001,
        'weight_decay': 5e-4,
        'epochs': 2,
        'patience': 10,
        'oversmoothing_every': 2,
    },

    'positive_eigenvalues_params': {'batch_size': 32},
    'gcod_params': {'batch_size': 32, 'uncertainty_lr': 1.0},
    'nrgnn_params': {
        'edge_hidden': 16, 'n_p': 2, 'p_u': 0.7,
        'alpha': 0.05, 'beta': 1.0, 't_small': 0.1, 'n_n': 5,
    },
    'pi_gnn_params': {'start_epoch': 1, 'miself': False, 'norm': None, 'vanilla': False},
    'cr_gnn_params': {'T': 2, 'tau': 0.6, 'p': 0.9, 'alpha': 0.2, 'beta': 0.9, 'pr': 0.3},
    'community_defense_params': {
        'community_method': 'louvain', 'num_communities': None,
        'lambda_comm': 1.0, 'pos_weight': 2.0, 'neg_weight': 2.0,
        'margin': 1.5, 'num_neg_samples': 3,
    },
    'rtgnn_params': {
        'edge_hidden': 16, 'co_lambda': 0.1, 'alpha': 0.3,
        'th': 0.8, 'K': 3, 'tau': 0.05, 'n_neg': 10,
    },
    'graphcleaner_params': {
        'k': 5, 'sample_rate': 0.5, 'max_iter_classifier': 50, 'held_split': 'valid',
    },
    'unionnet_params': {'k': 10, 'alpha': 0.5, 'beta': 1, 'feat_norm': True},
    'gnn_cleaner_params': {'label_propagation_iterations': 5, 'similarity_epsilon': 1e-8},
    'erase_params': {
        'n_embedding': 32, 'n_heads': 8,
        'use_layer_norm': False, 'use_residual': False, 'use_residual_linear': False,
        'gam1': 1.0, 'gam2': 2.0, 'eps': 0.05,
        'alpha': 0.6, 'beta': 0.6, 'T': 3,
    },
    'gnnguard_params': {'P0': 0.5, 'K': 2, 'D2': 16, 'attention': True},
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

# Minimum required checkpoint keys per method (some methods include additional
# optional keys like edge_indices/edge_weights or edges/weights).
REQUIRED_KEYS = {
    'standard':             {'backbone'},
    'positive_eigenvalues': {'backbone'},
    'pi_gnn':               {'backbone'},
    'community_defense':    {'backbone'},
    'graphcleaner':         {'backbone'},
    'unionnet':             {'backbone'},
    'gnn_cleaner':          {'backbone'},
    'gcod':                 {'backbone'},
    'cr_gnn':               {'backbone', 'adapter', 'proj_head', 'class_head'},
    'gnnguard':             {'backbone', 'gnnguard_model'},
    'nrgnn':                {'main_model', 'node_predictor'},
    'rtgnn':                {'dual_branch_predictor', 'structure_estimator'},
    'erase':                {'trained_model'},
}


def _make_config(method):
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg['training']['method'] = method
    return cfg


# ── helpers ──────────────────────────────────────────────────────────────────

def _assert_states_equal(state_1, state_2, path=""):
    """Recursively compare checkpoint dicts. Tensors checked with torch.equal."""
    assert set(state_1.keys()) == set(state_2.keys()), (
        f"Key mismatch at {path}: {set(state_1.keys())} vs {set(state_2.keys())}"
    )
    for key in state_1:
        v1, v2 = state_1[key], state_2[key]
        current_path = f"{path}.{key}" if path else key
        if isinstance(v1, torch.Tensor):
            assert isinstance(v2, torch.Tensor), f"{current_path}: type mismatch"
            assert v1.shape == v2.shape, f"{current_path}: shape {v1.shape} vs {v2.shape}"
            assert torch.equal(v1, v2), f"{current_path}: tensor values differ"
        elif isinstance(v1, dict):
            _assert_states_equal(v1, v2, current_path)
        # skip non-tensor, non-dict values (e.g. metadata)


# ── Test A: inference roundtrip ──────────────────────────────────────────────

@pytest.mark.parametrize('method', ALL_METHODS)
def test_inference_roundtrip(method, tmp_path):
    """Train → save checkpoint → eval_only → metrics must match."""
    config = _make_config(method)
    ckpt = str(tmp_path / f"{method}.pt")
    run_dir = str(tmp_path / f"run_{method}")

    result_train = run_experiment(config, run_id=1, checkpoint_path=ckpt, run_dir=run_dir)

    # Checkpoint file exists
    assert os.path.exists(ckpt), f"Checkpoint not created for {method}"

    # Checkpoint has expected keys
    state = torch.load(ckpt, weights_only=False)
    required = REQUIRED_KEYS[method]
    for key in required:
        assert key in state, f"Missing key '{key}' in checkpoint for {method}"

    # Eval-only run
    result_eval = run_experiment(config, run_id=1, checkpoint_path=ckpt, eval_only=True)

    # Structure valid
    for key in ('test_cls', 'train_cls', 'val_cls', 'test_oversmoothing', 'compute_info'):
        assert key in result_eval, f"Missing key '{key}' in eval-only result for {method}"

    # Eval-only has zero training cost
    assert result_eval['compute_info']['time_training_total'] == 0.0, (
        f"{method}: expected time_training_total==0.0"
    )
    assert result_eval['compute_info']['flops_training_total'] == 0, (
        f"{method}: expected flops_training_total==0"
    )

    # Classification metrics match
    for metric in ('accuracy', 'f1', 'precision', 'recall'):
        train_val = result_train['test_cls'][metric]
        eval_val = result_eval['test_cls'][metric]
        assert eval_val == pytest.approx(train_val, abs=1e-5), (
            f"{method}: test_cls.{metric} mismatch: train={train_val}, eval={eval_val}"
        )


# ── Test B: state roundtrip ─────────────────────────────────────────────────

@pytest.mark.parametrize('method', ALL_METHODS)
def test_state_roundtrip(method):
    """save → load → save → all tensor values identical."""
    discover_trainers()
    config = _make_config(method)
    init_data = initialize_experiment(config, run_id=1)
    init_data['run_dir'] = None
    init_data['_run_id'] = 1
    init_data['_config'] = config
    trainer = get_trainer(method, init_data, config)

    trainer.train()

    state_1 = trainer.get_checkpoint_state()
    trainer.load_checkpoint_state(state_1)
    state_2 = trainer.get_checkpoint_state()

    _assert_states_equal(state_1, state_2, method)


# ── Test C: resume viability ────────────────────────────────────────────────

@pytest.mark.parametrize('method', ALL_METHODS)
def test_resume_viability(method):
    """load checkpoint → train again (1 epoch) → no crash."""
    discover_trainers()
    config = _make_config(method)
    init_data = initialize_experiment(config, run_id=1)
    init_data['run_dir'] = None
    init_data['_run_id'] = 1
    init_data['_config'] = config
    trainer = get_trainer(method, init_data, config)

    train_out = trainer.train()

    state = trainer.get_checkpoint_state()
    trainer.load_checkpoint_state(state)

    # Reduce to 1 epoch for the resume run
    init_data['epochs'] = 1

    train_out_2 = trainer.train()

    assert 'stopped_at_epoch' in train_out_2, (
        f"{method}: second train() missing 'stopped_at_epoch'"
    )
    assert 'train_oversmoothing' in train_out_2, (
        f"{method}: second train() missing 'train_oversmoothing'"
    )
