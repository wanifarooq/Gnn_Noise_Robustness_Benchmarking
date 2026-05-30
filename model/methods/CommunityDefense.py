"""CommunityDefense — custom community-structure-regularized baseline (no source paper).

This is NOT an implementation of any published "community defense" method. It is a
custom baseline: standard supervised CE on the (noisy) train labels plus a bounded
auxiliary community-label cross-entropy over all nodes. Community labels come from
noise-independent structure (Louvain/spectral) and never read ``data.y_original``.
The actual training logic lives in ``methods/community_defense_helper.py``; the
legacy contrastive-style config keys (pos_weight/neg_weight/margin/num_neg_samples)
are vestigial and unused.
"""

from model.base import BaseTrainer
from model.registry import register


@register('community_defense')
class CommunityDefenseMethodTrainer(BaseTrainer):
    def train(self):
        from methods.registry import get_helper
        from training.training_loop import TrainingLoop

        d = self.init_data
        helper = get_helper('community_defense')
        loop = TrainingLoop(helper, log_epoch_fn=self.log_epoch)
        result = loop.run(
            d['backbone_model'], d['data_for_training'],
            self.config, d['device'], d,
        )
        self._loop = loop
        return result
