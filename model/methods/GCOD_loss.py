import torch
import torch.nn as nn
import torch.nn.functional as F

from model.evaluation import ClassificationMetrics
from model.base import BaseTrainer
from model.registry import register

def evaluate_ce_only(model, data_loader, device='cuda', mask_name='val'):
    model.eval()
    total_ce_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_true_labels = []
    _cls_eval = ClassificationMetrics(average='macro')

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)

            if mask_name == 'train':
                mask = batch.train_mask[:batch.batch_size]
            elif mask_name == 'val':
                mask = batch.val_mask[:batch.batch_size]
            else:
                mask = batch.test_mask[:batch.batch_size]

            target_nodes = mask.nonzero(as_tuple=True)[0]
            if len(target_nodes) == 0:
                continue

            labels = batch.y[target_nodes]
            ce_loss = F.cross_entropy(outputs[target_nodes], labels)
            total_ce_loss += ce_loss.item() * len(target_nodes)

            preds = outputs[target_nodes].argmax(dim=1)
            total_samples += len(target_nodes)

            all_predictions.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    avg_ce_loss = total_ce_loss / total_samples if total_samples > 0 else float('inf')
    if total_samples > 0:
        epoch_metrics = _cls_eval.compute_all_metrics(all_predictions, all_true_labels)
        accuracy = epoch_metrics['accuracy']
        f1 = epoch_metrics['f1']
    else:
        accuracy = 0.0
        f1 = 0.0

    return {'ce_loss': avg_ce_loss, 'accuracy': accuracy, 'f1': f1}

class GraphCentroidOutlierDiscounting(nn.Module):
    
    def __init__(self, num_classes, device, num_samples, embedding_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim if embedding_dim is not None else num_classes
        
        self.uncertainty_params = nn.Parameter(torch.zeros(num_samples, dtype=torch.float32))
        
        # Class centroids for soft label computation
        self.class_centroids = nn.Parameter(torch.randn(num_classes, self.embedding_dim) * 0.1)
        
        self.register_buffer('class_counts', torch.zeros(num_classes))

    def update_class_centroids(self, node_embeddings, node_labels, global_indices):
        # Update class centroids based on current embeddings
        with torch.no_grad():
            for class_id in range(self.num_classes):
                class_mask = (node_labels == class_id)
                if class_mask.sum() > 0:
                    class_embeddings = node_embeddings[class_mask]
                    current_centroid = class_embeddings.mean(dim=0)
                    
                    # Update with momentum
                    momentum = min(0.9, self.class_counts[class_id] / (self.class_counts[class_id] + 1))
                    self.class_centroids[class_id] = momentum * self.class_centroids[class_id] + (1 - momentum) * current_centroid
                    self.class_counts[class_id] += class_mask.sum().float()
        
    def compute_soft_labels(self, node_embeddings, true_labels_onehot):
        # Compute soft labels based on similarity to class centroids
        # Normalize embeddings and centroids
        normalized_embeddings = F.normalize(node_embeddings, p=2, dim=1)
        normalized_centroids = F.normalize(self.class_centroids, p=2, dim=1)
        
        # Compute similarities
        similarities = torch.mm(normalized_embeddings, normalized_centroids.t())
        
        # Temperature scaling
        temperature = 1.0
        soft_labels = F.softmax(similarities / temperature, dim=1)
        
        interpolation_weight = 0.7
        soft_labels = interpolation_weight * true_labels_onehot + (1 - interpolation_weight) * soft_labels
        
        return soft_labels
    
    def compute_loss_l1(self, batch_indices, model_predictions, true_labels_onehot, 
                        node_embeddings, training_accuracy):

        batch_uncertainty = self.uncertainty_params[batch_indices]
        
        # Compute soft labels
        soft_labels = self.compute_soft_labels(node_embeddings, true_labels_onehot)
        
        # Create diagonal matrix from uncertainty parameters
        uncertainty_diag = batch_uncertainty.unsqueeze(1) * true_labels_onehot

        modified_predictions = model_predictions + training_accuracy * uncertainty_diag

        loss_l1 = F.cross_entropy(modified_predictions, soft_labels)
        
        return loss_l1
    
    def compute_loss_l2(self, batch_indices, model_predictions, true_labels_onehot):

        batch_uncertainty = self.uncertainty_params[batch_indices]

        predicted_onehot = F.one_hot(
            model_predictions.argmax(dim=1),
            num_classes=self.num_classes
        ).float()
        
        # Diagonal uncertainty matrix
        uncertainty_diag = batch_uncertainty.unsqueeze(1) * true_labels_onehot
        
        diff = predicted_onehot + uncertainty_diag - true_labels_onehot
        loss_l2 = (diff.pow(2).sum(dim=1).mean()) / self.num_classes
        
        return loss_l2
        
    def compute_loss_l3(self, batch_indices, model_predictions, true_labels_onehot, 
                        training_accuracy):

        batch_uncertainty = self.uncertainty_params[batch_indices]
        
        # Compute alignment between predictions and true labels
        alignment = torch.sum(model_predictions * true_labels_onehot, dim=1)
        
        # First distribution
        p_logits = alignment
        p_probs = torch.sigmoid(p_logits)
        
        # Second distribution
        clamped_uncertainty = torch.clamp(batch_uncertainty, min=1e-8, max=1-1e-8)
        q_logits = -torch.log(clamped_uncertainty)
        q_probs = torch.sigmoid(q_logits)
        
        # KL divergence
        eps = 1e-8
        kl_div = (p_probs * torch.log((p_probs + eps) / (q_probs + eps)) + 
                  (1 - p_probs) * torch.log((1 - p_probs + eps) / (1 - q_probs + eps)))
        
        kl_divergence = kl_div.mean()
        
        loss_l3 = (1 - training_accuracy) * kl_divergence
        
        loss_l3 = torch.clamp(loss_l3, min=0.0)
        
        return loss_l3
    
    def forward(self, batch_indices, model_predictions, true_labels_onehot, 
                node_embeddings, training_accuracy, true_labels=None):
        
        if true_labels is not None:
            self.update_class_centroids(node_embeddings, true_labels, batch_indices)
        
        loss_l1 = self.compute_loss_l1(
            batch_indices, model_predictions, true_labels_onehot, 
            node_embeddings, training_accuracy
        )
        
        loss_l2 = self.compute_loss_l2(
            batch_indices, model_predictions, true_labels_onehot
        )
        
        loss_l3 = self.compute_loss_l3(
            batch_indices, model_predictions, true_labels_onehot, training_accuracy
        )
        
        total_loss = loss_l1 + loss_l3
        
        return total_loss, loss_l1, loss_l2, loss_l3


@register('gcod')
class GCODMethodTrainer(BaseTrainer):
    def train(self):
        from methods.registry import get_helper
        from training.training_loop import TrainingLoop

        d = self.init_data
        self._helper = get_helper('gcod')
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
            self._helper = get_helper('gcod')
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
        from model.evaluation import evaluate_model
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