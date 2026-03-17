"""GraphCleaner method helper — two-phase noise detection + clean training.

GraphCleaner is a two-phase method:
    Phase 1 (pre_train): Train the backbone on noisy data, use the trained model
        to estimate a noise transition matrix, generate negative samples, build
        detection features, train a binary classifier to identify noisy nodes,
        and remove detected noisy nodes from train_mask.
    Phase 2 (train_step): Standard cross-entropy training on the cleaned
        training set (data.train_mask has been narrowed by pre_train).

The noise detection phase reuses GraphCleanerNoiseDetector from the original
implementation.  The clean training phase is identical to StandardHelper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base_helper import MethodHelper
from methods.registry import register_helper
from model.methods.GraphCleaner import GraphCleanerNoiseDetector


@register_helper('graphcleaner')
class GraphCleanerHelper(MethodHelper):
    """Two-phase training: noise detection then standard CE on clean data."""

    def supports_batched_training(self):
        return True

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        lr = float(training_cfg.get('lr', 0.01))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))

        backbone_model.to(device)
        optimizer = torch.optim.Adam(
            backbone_model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        return {
            'models': [backbone_model],
            'optimizers': [optimizer],
            'backbone': backbone_model,
            'optimizer': optimizer,
            'criterion': nn.CrossEntropyLoss(),
            'device': device,
            'config': config,
            'init_data': init_data,
        }

    # ── Pre-training: noise detection ──────────────────────────────────────

    def pre_train(self, state, data, config):
        """Run GraphCleaner noise detection and narrow data.train_mask."""
        init_data = state['init_data']
        device = state['device']
        backbone = state['backbone']

        # Noise detection needs both train and val nodes (for val_loss early
        # stopping and noise transition matrix estimation).  In inductive mode
        # `data` is only the train subgraph, so use the full graph instead.
        detection_data = init_data.get('data_for_training', data)

        # Ensure oversmoothing_every is available in training config
        # (mirrors GraphCleanerMethodTrainer.train)
        config.setdefault('training', {})['oversmoothing_every'] = (
            init_data.get('oversmoothing_every', 20)
        )

        detector = GraphCleanerNoiseDetector(
            configuration_params=config,
            computation_device=device,
            random_seed=init_data.get('seed', 42),
        )

        clean_train_mask, _cleaned_data = detector.clean_training_data(
            graph_data=detection_data,
            neural_network_model=backbone,
            num_classes=init_data['num_classes'],
        )

        # Narrow the training mask to exclude detected noisy nodes.
        # The backbone has been trained during detection; the main training
        # loop will continue training it on the cleaned data (matching the
        # original GraphCleanerMethodTrainer behaviour).
        # In inductive mode, propagate the cleaned mask to the train subgraph.
        if data is not detection_data:
            # Map full-graph clean_train_mask back to train subgraph
            # The train subgraph's train_mask is all True (all nodes are train)
            # clean_train_mask is on the full graph — extract train-node entries
            full_train_mask = detection_data.train_mask
            # clean_train_mask[i] is True if node i is clean AND was a train node
            # For the subgraph, node indices correspond to the original train nodes
            clean_of_train = clean_train_mask[full_train_mask]
            data.train_mask = clean_of_train
        else:
            data.train_mask = clean_train_mask

        # Reinitialise the optimizer so that momentum / adaptive state from
        # the detection phase does not leak into the clean training phase.
        training_cfg = config.get('training', {})
        lr = float(training_cfg.get('lr', 0.01))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))
        optimizer = torch.optim.Adam(
            backbone.parameters(), lr=lr, weight_decay=weight_decay,
        )
        state['optimizer'] = optimizer
        state['optimizers'] = [optimizer]

    # ── Training step (standard CE on cleaned data) ────────────────────────

    def train_step(self, state, data, epoch):
        model = state['backbone']
        optimizer = state['optimizer']
        criterion = state['criterion']

        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(data)

        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        return {'train_loss': loss.item()}
