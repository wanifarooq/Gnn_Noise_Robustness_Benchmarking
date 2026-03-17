"""Standard method helper — vanilla cross-entropy training.

This is the baseline method.  It validates that the shared TrainingLoop +
MethodHelper interface works correctly by reproducing the exact same behavior
as the original model/methods/Standard.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base_helper import MethodHelper
from methods.registry import register_helper


@register_helper('standard')
class StandardHelper(MethodHelper):
    """Vanilla supervised training with cross-entropy loss."""

    def supports_batched_training(self):
        return True

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        lr = float(training_cfg.get('lr', 0.01))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))

        backbone_model.to(device)
        optimizer = torch.optim.Adam(
            backbone_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        return {
            'models': [backbone_model],
            'optimizers': [optimizer],
            'backbone': backbone_model,
            'optimizer': optimizer,
            'criterion': nn.CrossEntropyLoss(),
            'device': device,
        }

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
