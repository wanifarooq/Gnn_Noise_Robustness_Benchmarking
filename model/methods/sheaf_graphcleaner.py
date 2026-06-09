from model.base import BaseTrainer
from model.registry import register


@register('sheaf_graphcleaner')
class Sheaf_graphcleaner(BaseTrainer):
    def train(self):
        from methods.registry import get_helper
        from training.training_loop import TrainingLoop
        
        d = self.init_data
        helper = get_helper('sheaf_graphcleaner')
        loop = TrainingLoop(helper, log_epoch_fn=self.log_epoch)
        result = loop.run(
            d['backbone_model'], d['data_for_training'],
            self.config, d['device'], d,
        )
        self._loop = loop
        return result