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
    #Dynamic cross-entropy loss
    if len(labels) == 0:
        return torch.tensor(0.0, device=p1.device, requires_grad=True)
    
    labels = labels.long()
    pseudo_labels1 = p1.argmax(dim=1)
    pseudo_labels2 = p2.argmax(dim=1)
    consistent_mask = (pseudo_labels1 == pseudo_labels2)
    
    if consistent_mask.sum() > 0:

        loss = F.nll_loss(p1[consistent_mask], labels[consistent_mask])
    else:
        loss = torch.tensor(0.0, device=p1.device, requires_grad=True)
    
    return loss


def compute_cross_space_consistency_fixed(z1, z2, p1, p2, T, p_threshold):

    # Similarity matrix
    z1_expanded = z1.unsqueeze(1)
    z2_expanded = z2.unsqueeze(0)
    zm = torch.exp(F.cosine_similarity(z1_expanded, z2_expanded, dim=2) / T)
    zm = zm.mean(dim=1)
    
    # Similarity matrix
    p1_expanded = p1.unsqueeze(1)
    p2_expanded = p2.unsqueeze(0)
    pm = torch.exp(F.cosine_similarity(p1_expanded, p2_expanded, dim=2) / T)
    pm = pm.mean(dim=1)
    
    # Apply thresholding
    pm = torch.where(pm > p_threshold, pm, torch.zeros_like(pm))
    
    # MSE loss
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
