"""
Unit tests for the vectorized evaluation functions and oversmoothing utilities.

Covers:
  - Phase 1: Vectorized edge metrics (_compute_edir_average, _compute_dirichlet_energy_traditional,
             _compute_mad, compute_oversmoothing_for_mask edge filtering/remapping)
  - Phase 2: val_oversmoothing is present in _make_result output
  - Phase 3: oversmoothing_every is propagated from experiment init_data
"""
import math
import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.base import BaseTrainer
from model.evaluation import (
    OversmoothingMetrics,
    compute_oversmoothing_for_mask,
    OVERSMOOTHING_KEYS,
    DEFAULT_OVERSMOOTHING,
    ZERO_CLS,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_graph():
    """4-node line graph: 0-1-2-3, with random 3-dim features."""
    torch.manual_seed(0)
    X = torch.randn(4, 3)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    return X, edge_index


@pytest.fixture
def evaluator():
    return OversmoothingMetrics(device='cpu')


# ── Phase 1: Vectorized metric tests ────────────────────────────────────────

class TestEdirAverage:
    """Test _compute_edir_average (vectorized Dirichlet energy)."""

    def test_returns_float(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        result = evaluator._compute_edir_average(X, edge_index)
        assert isinstance(result, float)

    def test_nonnegative(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        result = evaluator._compute_edir_average(X, edge_index)
        assert result >= 0.0

    def test_zero_for_identical_features(self, evaluator, simple_graph):
        _, edge_index = simple_graph
        X_const = torch.ones(4, 3)
        result = evaluator._compute_edir_average(X_const, edge_index)
        assert result == pytest.approx(0.0, abs=1e-8)

    def test_empty_edges(self, evaluator):
        X = torch.randn(4, 3)
        empty_edges = torch.zeros(2, 0, dtype=torch.long)
        result = evaluator._compute_edir_average(X, empty_edges)
        assert result == 0.0

    def test_with_edge_weight(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        weights = torch.ones(edge_index.size(1)) * 2.0
        result_weighted = evaluator._compute_edir_average(X, edge_index, weights)
        result_unweighted = evaluator._compute_edir_average(X, edge_index)
        assert result_weighted == pytest.approx(result_unweighted * 2.0, rel=1e-5)

    def test_manual_computation(self, evaluator):
        """Verify against hand-computed value for a 2-node graph."""
        X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        # diff for edge (0,1) = [1,-1], sq = [1,1], sum = 2
        # diff for edge (1,0) = [-1,1], sq = [1,1], sum = 2
        # total / 2.0 = (2+2)/2 = 2.0
        result = evaluator._compute_edir_average(X, edge_index)
        assert result == pytest.approx(2.0, abs=1e-6)


class TestDirichletEnergyTraditional:
    """Test _compute_dirichlet_energy_traditional (vectorized)."""

    def test_returns_float(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        result = evaluator._compute_dirichlet_energy_traditional(X, edge_index)
        assert isinstance(result, float)

    def test_nonnegative(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        result = evaluator._compute_dirichlet_energy_traditional(X, edge_index)
        assert result >= 0.0

    def test_finite(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        result = evaluator._compute_dirichlet_energy_traditional(X, edge_index)
        assert math.isfinite(result)


class TestMAD:
    """Test _compute_mad (vectorized Mean Angular Distance)."""

    def test_returns_float(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        result = evaluator._compute_mad(X, edge_index)
        assert isinstance(result, float)

    def test_in_range(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        result = evaluator._compute_mad(X, edge_index)
        # MAD should be in [0, 2] (1 - cos_sim, where cos_sim in [-1,1])
        assert 0.0 <= result <= 2.0

    def test_zero_for_parallel_features(self, evaluator):
        """All nodes have same direction -> cosine similarity = 1 -> MAD = 0."""
        X = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        result = evaluator._compute_mad(X, edge_index)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_empty_edges(self, evaluator):
        X = torch.randn(4, 3)
        empty_edges = torch.zeros(2, 0, dtype=torch.long)
        result = evaluator._compute_mad(X, empty_edges)
        assert result == 0.0

    def test_zero_norm_handling(self, evaluator):
        """Nodes with zero-norm should not cause errors."""
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        result = evaluator._compute_mad(X, edge_index)
        assert math.isfinite(result)


class TestComputeOversmoothingForMask:
    """Test edge filtering and remapping in compute_oversmoothing_for_mask."""

    def test_returns_all_keys(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        mask = torch.tensor([True, True, True, True])
        result = compute_oversmoothing_for_mask(evaluator, X, edge_index, mask)
        for key in OVERSMOOTHING_KEYS:
            assert key in result, f"Missing key '{key}'"

    def test_all_values_finite(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        mask = torch.tensor([True, True, True, True])
        result = compute_oversmoothing_for_mask(evaluator, X, edge_index, mask)
        for key, val in result.items():
            assert math.isfinite(val), f"{key} is not finite: {val}"

    def test_subset_mask_filters_edges(self, evaluator, simple_graph):
        """Only nodes 0,1 are in mask -> edges (0,1) and (1,0) should remain."""
        X, edge_index = simple_graph
        mask = torch.tensor([True, True, False, False])
        result = compute_oversmoothing_for_mask(evaluator, X, edge_index, mask)
        assert isinstance(result, dict)
        for key in OVERSMOOTHING_KEYS:
            assert key in result

    def test_single_node_mask(self, evaluator, simple_graph):
        """Single node in mask -> no internal edges -> fallback values."""
        X, edge_index = simple_graph
        mask = torch.tensor([True, False, False, False])
        result = compute_oversmoothing_for_mask(evaluator, X, edge_index, mask)
        # With a single node and no edges, EDir should be 0
        assert result['EDir'] == 0.0

    def test_no_edges_between_masked_nodes(self, evaluator):
        """Two disconnected masked nodes."""
        X = torch.randn(4, 3)
        # Only edges between nodes 0-1 and 2-3
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        # Mask only nodes 0 and 2 (which are not connected)
        mask = torch.tensor([True, False, True, False])
        result = compute_oversmoothing_for_mask(evaluator, X, edge_index, mask)
        assert result['EDir'] == 0.0
        assert result['MAD'] == 0.0

    def test_remapping_produces_valid_indices(self, evaluator):
        """Ensure remapped edge indices are within bounds of masked node count."""
        X = torch.randn(10, 5)
        # Create a complete graph on nodes 3,5,7
        edge_index = torch.tensor([
            [3, 3, 5, 5, 7, 7],
            [5, 7, 3, 7, 3, 5],
        ])
        mask = torch.zeros(10, dtype=torch.bool)
        mask[3] = True
        mask[5] = True
        mask[7] = True
        result = compute_oversmoothing_for_mask(evaluator, X, edge_index, mask)
        # Should compute without errors (remapped indices 0,1,2)
        assert isinstance(result, dict)
        for key in OVERSMOOTHING_KEYS:
            assert math.isfinite(result[key])


class TestComputeAllMetrics:
    """Test the full compute_all_metrics pipeline."""

    def test_all_keys_present(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        result = evaluator.compute_all_metrics(X, edge_index)
        for key in OVERSMOOTHING_KEYS:
            assert key in result

    def test_with_edge_weight(self, evaluator, simple_graph):
        X, edge_index = simple_graph
        weights = torch.ones(edge_index.size(1))
        result = evaluator.compute_all_metrics(X, edge_index, edge_weight=weights)
        for key in OVERSMOOTHING_KEYS:
            assert math.isfinite(result[key])


# ── Shared DummyTrainer for BaseTrainer tests ────────────────────────────────

_ZERO_COMPUTE_INFO = {
    'flops_inference': 0, 'flops_training_total': 0,
    'time_training_total': 0.0, 'time_inference': 0.0,
}

class DummyTrainer(BaseTrainer):
    """Minimal concrete BaseTrainer for unit tests."""
    def train(self):
        pass


def _make_dummy(init_data=None, config=None):
    """Create a DummyTrainer instance via the real constructor."""
    defaults = {
        'compute_info': dict(_ZERO_COMPUTE_INFO),
        'backbone_model': nn.Linear(3, 2),
    }
    if init_data is not None:
        defaults.update(init_data)
    return DummyTrainer(
        init_data=defaults,
        config=config or {},
    )


# ── Phase 4: log_epoch and training log ──────────────────────────────────────

class TestLogEpoch:
    """Test BaseTrainer.log_epoch() epoch logging."""

    def test_log_epoch_populates_epoch_log(self):
        dummy = _make_dummy()
        dummy.log_epoch(1, 1.5, 1.2, 0.3, 0.35, train_f1=0.2, val_f1=0.25)
        dummy.log_epoch(2, 1.0, 0.9, 0.5, 0.55, train_f1=0.4, val_f1=0.45)
        assert len(dummy.epoch_log) == 2
        assert dummy.epoch_log[0]['epoch'] == 1
        assert dummy.epoch_log[0]['train_loss'] == 1.5
        assert dummy.epoch_log[1]['val_acc'] == 0.55

    def test_log_epoch_tracks_best(self):
        dummy = _make_dummy()
        dummy.log_epoch(1, 1.5, 1.2, 0.3, 0.35, is_best=True)
        assert dummy.best_epoch == 1
        assert dummy.best_val_loss == 1.2

        dummy.log_epoch(2, 1.0, 0.8, 0.5, 0.55, is_best=True)
        assert dummy.best_epoch == 2
        assert dummy.best_val_loss == 0.8

    def test_log_epoch_not_best_keeps_previous(self):
        dummy = _make_dummy()
        dummy.log_epoch(1, 1.5, 1.2, 0.3, 0.35, is_best=True)
        dummy.log_epoch(2, 1.0, 1.5, 0.5, 0.55, is_best=False)
        assert dummy.best_epoch == 1
        assert dummy.best_val_loss == 1.2

    def test_log_epoch_oversmoothing_null(self):
        dummy = _make_dummy()
        dummy.log_epoch(1, 1.5, 1.2, 0.3, 0.35)
        assert dummy.epoch_log[0]['oversmoothing'] is None

    def test_log_epoch_oversmoothing_present(self):
        dummy = _make_dummy()
        os_data = {'train': {'EDir': 1.0}, 'val': {'EDir': 0.8}}
        dummy.log_epoch(1, 1.5, 1.2, 0.3, 0.35, oversmoothing=os_data)
        assert dummy.epoch_log[0]['oversmoothing'] == os_data

    def test_log_epoch_saves_checkpoint_on_best(self, tmp_path):
        model = nn.Linear(3, 2)
        dummy = _make_dummy({
            'backbone_model': model,
            'run_dir': str(tmp_path),
            'compute_info': dict(_ZERO_COMPUTE_INFO),
        })
        dummy.log_epoch(5, 1.0, 0.5, 0.6, 0.65, is_best=True)
        expected_file = tmp_path / "epoch_005_valloss_0.5000.pt"
        assert expected_file.exists()

    def test_log_epoch_checkpoint_saved_every_epoch(self, tmp_path):
        model = nn.Linear(3, 2)
        dummy = _make_dummy({
            'backbone_model': model,
            'run_dir': str(tmp_path),
            'compute_info': dict(_ZERO_COMPUTE_INFO),
        })
        dummy.log_epoch(5, 1.0, 0.5, 0.6, 0.65, is_best=False)
        assert len(list(tmp_path.glob("epoch_*.pt"))) == 1

    def test_save_training_log(self, tmp_path):
        import json
        dummy = _make_dummy({
            'run_dir': str(tmp_path),
            'method': 'standard',
            'lr': 0.01,
            'weight_decay': 5e-4,
            'epochs': 100,
            'patience': 20,
            'oversmoothing_every': 20,
            'compute_info': dict(_ZERO_COMPUTE_INFO),
        })
        dummy.epoch_log.append({'epoch': 1, 'train_loss': 1.5, 'val_loss': 1.2})
        dummy.best_epoch = 1
        dummy.best_val_loss = 1.2
        dummy.save_training_log(
            run_id=1, config={'seed': 42}, duration=5.0,
            stopped_at_epoch=10, final_result={'test_cls': {}}
        )
        log_file = tmp_path / 'training_log.json'
        assert log_file.exists()
        with open(log_file) as f:
            log = json.load(f)
        assert log['run_id'] == 1
        assert log['duration_seconds'] == 5.0
        assert log['stopped_at_epoch'] == 10
        assert log['best_epoch'] == 1
        assert len(log['epoch_log']) == 1
        assert 'training_params' in log


# ── Phase 2: val_oversmoothing in _make_result ──────────────────────────────

class TestMakeResult:
    """Test that BaseTrainer._make_result includes val_oversmoothing."""

    def test_val_oversmoothing_present(self):
        dummy = _make_dummy()

        result_dict = {
            'test_cls': {'accuracy': 0.8, 'f1': 0.7, 'precision': 0.75, 'recall': 0.72},
            'train_cls': {'accuracy': 0.9, 'f1': 0.85, 'precision': 0.88, 'recall': 0.82},
            'val_cls': {'accuracy': 0.85, 'f1': 0.80, 'precision': 0.83, 'recall': 0.78},
            'test_oversmoothing': dict(DEFAULT_OVERSMOOTHING),
        }
        train_os = {'EDir': [1.0, 2.0], 'MAD': [0.5, 0.6]}
        val_os = {'EDir': [1.1, 2.1], 'MAD': [0.51, 0.61]}

        out = dummy._make_result(result_dict, train_os, val_os)
        assert 'val_oversmoothing' in out
        assert isinstance(out['val_oversmoothing'], dict)
        assert out['val_oversmoothing']['EDir'] == pytest.approx(1.6, rel=1e-5)

        # test_cls / train_cls / val_cls pass through
        assert out['test_cls'] == result_dict['test_cls']
        assert out['train_cls'] == result_dict['train_cls']
        assert out['val_cls'] == result_dict['val_cls']

    def test_val_oversmoothing_none(self):
        dummy = _make_dummy()

        result_dict = {
            'test_oversmoothing': dict(DEFAULT_OVERSMOOTHING),
        }
        train_os = {'EDir': [1.0], 'MAD': [0.5]}

        out = dummy._make_result(result_dict, train_os)
        assert 'val_oversmoothing' in out
        assert out['val_oversmoothing'] == {}

        # test_cls / train_cls / val_cls default to zero-filled dict when absent
        assert out['test_cls'] == ZERO_CLS
        assert out['train_cls'] == ZERO_CLS
        assert out['val_cls'] == ZERO_CLS

    def test_val_oversmoothing_no_reduce(self):
        dummy = _make_dummy()

        test_cls = {'accuracy': 0.8, 'f1': 0.7, 'precision': 0.75, 'recall': 0.72}
        train_cls = {'accuracy': 0.9, 'f1': 0.85, 'precision': 0.88, 'recall': 0.82}
        val_cls = {'accuracy': 0.85, 'f1': 0.80, 'precision': 0.83, 'recall': 0.78}
        result_dict = {
            'test_cls': test_cls,
            'train_cls': train_cls,
            'val_cls': val_cls,
            'test_oversmoothing': dict(DEFAULT_OVERSMOOTHING),
        }
        train_os = {'EDir': [1.0, 2.0]}
        val_os = {'EDir': [1.1, 2.1]}

        out = dummy._make_result(result_dict, train_os, val_os, reduce=False)
        assert out['val_oversmoothing'] == val_os
        assert out['test_cls'] == test_cls
        assert out['train_cls'] == train_cls
        assert out['val_cls'] == val_cls


# ── Phase 2b: checkpoint state ────────────────────────────────────────────────

class TestCheckpoint:
    """Test BaseTrainer checkpoint helpers."""

    def test_get_checkpoint_state_has_backbone(self):
        dummy = _make_dummy({'backbone_model': nn.Linear(3, 2)})

        state = dummy.get_checkpoint_state()
        assert 'backbone' in state
        assert isinstance(state['backbone'], dict)

    def test_load_checkpoint_state_restores_weights(self, tmp_path):
        model = nn.Linear(3, 2)
        dummy = _make_dummy({'backbone_model': model})

        # Save checkpoint to disk (mimics real usage)
        state = dummy.get_checkpoint_state()
        ckpt_file = tmp_path / "ckpt.pt"
        torch.save(state, ckpt_file)
        original_weight = model.weight.data.clone()

        # Corrupt weights
        model.weight.data.fill_(99.0)
        assert not torch.equal(model.weight.data, original_weight)

        # Restore from checkpoint loaded from disk
        loaded = torch.load(ckpt_file, weights_only=False)
        dummy.load_checkpoint_state(loaded)
        assert torch.equal(model.weight.data, original_weight)

    def test_get_checkpoint_state_is_snapshot(self, tmp_path):
        """get_checkpoint_state returns an independent copy of the weights."""
        model = nn.Linear(3, 2)
        dummy = _make_dummy({'backbone_model': model})

        state = dummy.get_checkpoint_state()
        original_weight = state['backbone']['weight'].clone()

        # Mutating the model should NOT change the snapshot
        model.weight.data.fill_(99.0)
        assert torch.equal(state['backbone']['weight'], original_weight)


# ── Phase 3: oversmoothing_every propagation ─────────────────────────────────

class TestOversmoothingEvery:
    """Test that oversmoothing_every is passed through init_data."""

    def test_default_value(self):
        """When not set in config, defaults to 20."""
        from util.experiment import initialize_experiment

        config = {
            'seed': 42,
            'device': 'cpu',
            'dataset': {'name': 'cora', 'root': 'data'},
            'noise': {'type': 'clean', 'rate': 0, 'seed': 42},
            'model': {
                'name': 'gcn', 'hidden_channels': 16,
                'n_layers': 2, 'dropout': 0.5,
                'self_loop': True, 'mlp_layers': 1,
                'train_eps': False, 'heads': 8,
            },
            'training': {
                'method': 'standard',
                'lr': 0.01, 'weight_decay': 5e-4,
                'epochs': 5, 'patience': 10,
            },
        }
        init_data = initialize_experiment(config, run_id=1)
        assert init_data['oversmoothing_every'] == 20

    def test_custom_value(self):
        """When oversmoothing_every is set, it propagates."""
        from util.experiment import initialize_experiment

        config = {
            'seed': 42,
            'device': 'cpu',
            'dataset': {'name': 'cora', 'root': 'data'},
            'noise': {'type': 'clean', 'rate': 0, 'seed': 42},
            'model': {
                'name': 'gcn', 'hidden_channels': 16,
                'n_layers': 2, 'dropout': 0.5,
                'self_loop': True, 'mlp_layers': 1,
                'train_eps': False, 'heads': 8,
            },
            'training': {
                'method': 'standard',
                'lr': 0.01, 'weight_decay': 5e-4,
                'epochs': 5, 'patience': 10,
                'oversmoothing_every': 10,
            },
        }
        init_data = initialize_experiment(config, run_id=1)
        assert init_data['oversmoothing_every'] == 10
