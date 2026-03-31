import torch
import torch.nn.functional as F
from model.evaluation import evaluate_model
from model.base import BaseTrainer
from model.registry import register


class ContrastiveProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.first_layer = torch.nn.Linear(input_dim, output_dim)
        self.second_layer = torch.nn.Linear(output_dim, output_dim)

    def forward(self, embeddings):
        embeddings = F.relu(self.first_layer(embeddings))
        return self.second_layer(embeddings)


class NodeClassificationHead(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = torch.nn.Linear(input_dim, num_classes)

    def forward(self, embeddings):
        return F.log_softmax(self.classifier(embeddings), dim=-1)


def contrastive_loss_original_style(z1, z2, tau):

    N = z1.size(0)
    
    # Positive similarities
    pos1 = torch.exp(F.cosine_similarity(z1, z2, dim=1) / tau)
    pos2 = torch.exp(F.cosine_similarity(z2, z1, dim=1) / tau)
    
    # Negative similarities
    neg_matrix_1 = torch.exp(torch.mm(z1, z1.t()) / tau)  
    neg_matrix_2 = torch.exp(torch.mm(z2, z2.t()) / tau)

    neg1 = torch.sum(neg_matrix_1, dim=1) - torch.diag(neg_matrix_1)
    neg2 = torch.sum(neg_matrix_2, dim=1) - torch.diag(neg_matrix_2)
    
    loss1 = -torch.log(pos1 / (pos1 + neg1 + 1e-8))
    loss2 = -torch.log(pos2 / (pos2 + neg2 + 1e-8))
    
    return (torch.sum(loss1) + torch.sum(loss2)) / (2 * N)


def dynamic_cross_entropy_loss_corrected(p1, p2, labels):
    # C-1 Fix: Use pseudo-labels for samples where both views agree (consistency).
    # Also handle empty masks to avoid gradient issues.
    if len(labels) == 0:
        return (p1.sum() * 0.0)

    labels = labels.long()
    # Predictions (proxy for pseudo-labels)
    pred1 = p1.argmax(dim=1)
    pred2 = p2.argmax(dim=1)
    consistent_mask = (pred1 == pred2)

    # For inconsistent samples, we use noisy labels (p1/p2 use log_softmax already)
    loss = F.nll_loss(p1, labels, reduction='none') + F.nll_loss(p2, labels, reduction='none')

    # C-1 Implementation: For consistent samples, trust the model's agreement over the label
    if consistent_mask.any():
        pseudo_labels = pred1[consistent_mask]
        loss[consistent_mask] = (
            F.nll_loss(p1[consistent_mask], pseudo_labels, reduction='none') +
            F.nll_loss(p2[consistent_mask], pseudo_labels, reduction='none')
        )

    return loss.mean() / 2.0


def compute_cross_space_consistency_fixed(z1, z2, p1, p2, T, p_threshold):
    # C-2 Fix: Use normalized similarities and avoid collapsing the matrix.
    # Prediction similarity (Softmax-based)
    p1_soft = F.softmax(p1 / T, dim=1)
    p2_soft = F.softmax(p2 / T, dim=1)
    pm = torch.mm(p1_soft, p2_soft.t())
    
    # Projection similarity (L2-normalized cosine)
    z1_norm = F.normalize(z1, p=2, dim=1)
    z2_norm = F.normalize(z2, p=2, dim=1)
    zm = torch.mm(z1_norm, z2_norm.t())
    
    # Apply thresholding to probability similarity to filter out low-confidence relations
    pm = torch.where(pm > p_threshold, pm, torch.zeros_like(pm))
    
    # MSE between the feature relationship space and prediction relationship space
    return F.mse_loss(zm, pm)


@register('cr_gnn')
class CRGNNMethodTrainer(BaseTrainer):
    def train(self):
        from methods.registry import get_helper
        from training.training_loop import TrainingLoop

        d = self.init_data
        self._helper = get_helper('cr_gnn')
        self._loop = TrainingLoop(self._helper, log_epoch_fn=self.log_epoch)
        return self._loop.run(
            d['backbone_model'], d['data_for_training'],
            self.config, d['device'], d,
        )

    def _get_state(self):
        """Get helper state — works both during and after training."""
        if hasattr(self, '_loop') and hasattr(self._loop, '_state'):
            return self._loop.state
        return self._loop_state

    def get_checkpoint_state(self) -> dict:
        return self._helper.get_checkpoint_state(self._get_state())

    def load_checkpoint_state(self, state):
        if not hasattr(self, '_helper'):
            from methods.registry import get_helper
            self._helper = get_helper('cr_gnn')
            d = self.init_data
            self._loop_state = self._helper.setup(
                d['backbone_model'], d['data_for_training'],
                self.config, d['device'], d,
            )
        else:
            self._loop_state = self._get_state()
        self._helper.load_checkpoint_state(self._loop_state, state)

    def profile_flops(self):
        from util.profiling import profile_model_flops
        d = self.init_data
        state = self._get_state()
        fwd = self._helper.get_inference_forward_fn(state, d['data_for_training'])
        return profile_model_flops(state['models'][0], d['data_for_training'],
                                   d['device'], forward_fn=fwd)

    def profile_training_step(self):
        from util.profiling import profile_training_step_flops
        d = self.init_data
        state = self._get_state()
        step_fn = self._helper.get_training_step_fn(state, d['data_for_training'])
        models = state['models']
        return profile_training_step_flops(models, d['device'], step_fn)

    def evaluate(self):
        d = self.init_data
        state = self._get_state()
        data = d['data_for_training']
        clean_labels = getattr(data, 'y_original', data.y)

        def get_predictions():
            return self._helper.get_predictions(state, data)

        def get_embeddings():
            return self._helper.get_embeddings(state, data)

        return evaluate_model(
            get_predictions, get_embeddings, clean_labels,
            data.train_mask, data.val_mask, data.test_mask,
            data.edge_index, d['device'],
        )
