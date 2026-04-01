import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    """GCOD loss — matches the reference ncodLoss implementation.

    Uses stored per-sample embeddings from the previous epoch to compute
    class centroids, similarity-based soft targets, and three loss components:
        L1: Soft-label cross-entropy with uncertainty-adjusted predictions
        L2: MSE between (predicted_onehot + uncertainty) and true labels
        L3: Batch-wide KL divergence between alignment and uncertainty
    """

    def __init__(self, num_classes, device, num_samples, embedding_dim=None,
                 sample_labels=None, train_mask=None, momentum=0.9, temperature=1.0,
                 kl_start_epoch=2):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim if embedding_dim is not None else num_classes
        self.momentum = momentum        # Improvement 2: Centroid momentum
        self.temperature = temperature  # Improvement 3: Softmax temperature
        self.kl_start_epoch = kl_start_epoch # Configurable delay

        # Uncertainty parameter — shape (N, 1), init with very small values
        self.u = nn.Parameter(torch.empty(num_samples, 1, dtype=torch.float32))
        nn.init.normal_(self.u, mean=1e-8, std=1e-9)

        # Buffers for centroid computation (not trainable)
        # Improvement 1: Store L2-normalised features for scale-invariant similarity
        self.register_buffer(
            'prevSimilarity', torch.zeros(num_samples, self.embedding_dim))
        self.register_buffer(
            'masterVector', torch.zeros(num_classes, self.embedding_dim))

        # Pre-compute per-class bins of training node indices
        if sample_labels is not None and train_mask is not None:
            train_idx = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            labels_np = (sample_labels.cpu().numpy()
                         if isinstance(sample_labels, torch.Tensor)
                         else np.asarray(sample_labels))
            self.bins = []
            for i in range(num_classes):
                mask = labels_np[train_idx] == i
                self.bins.append(train_idx[mask].copy())
        else:
            self.bins = None

    def recompute_centroids(self):
        """Recompute class centroids from stored embeddings (call once per epoch)."""
        if self.bins is None:
            return

        with torch.no_grad():
            for i in range(self.num_classes):
                if len(self.bins[i]) == 0:
                    continue
                class_u = self.u.detach()[self.bins[i]]
                bottomK = max(int(0.5 * len(class_u)), 1)
                important_idx = torch.topk(
                    class_u.view(-1), bottomK, largest=False, dim=0,
                )[1]
                
                # Improvement 1: Centroids are now averages of L2-normalised features
                computed_centroid = torch.mean(
                    self.prevSimilarity[self.bins[i]][important_idx], dim=0,
                )
                computed_centroid = F.normalize(computed_centroid, p=2, dim=0)

                # Improvement 2: Momentum-based update to prevent jitter
                # new = momentum * old + (1-momentum) * current
                self.masterVector[i] = (self.momentum * self.masterVector[i]) + \
                                      (1 - self.momentum) * computed_centroid
                # Ensure the master vector itself stays normalised
                self.masterVector[i] = F.normalize(self.masterVector[i], p=2, dim=0)

    def compute_loss_l1(self, u_masked, model_logits, label_onehot,
                        embeddings_detached, training_accuracy):
        """L1: soft-label CE with uncertainty-adjusted predictions."""
        eps = 1e-4

        # Normalized centroid matrix
        mv_norm = self.masterVector.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        mv_t = self.masterVector.div(mv_norm).t()

        # Similarity between normalized embeddings and centroids
        out_norm = embeddings_detached.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        out_normalized = embeddings_detached.div(out_norm)
        similarity = torch.mm(out_normalized, mv_t)
        similarity = similarity * label_onehot
        similarity = similarity * (similarity > 0.0).float()

        # Prediction: softmax + uncertainty offset (u detached for L1)
        prediction = F.softmax(model_logits, dim=1)
        prediction = torch.clamp(
            prediction + training_accuracy * u_masked.detach(),
            min=eps, max=1.0,
        )
       


        # similarity_weighted = 0.8 * similarity + 0.2 * label_onehot
        loss = torch.mean(-torch.sum(similarity * torch.log(prediction), dim=1))
        return loss

    def compute_loss_l2(self, u_masked, model_logits, label_onehot):
        """L2: MSE between (predicted_onehot + u) and true labels."""
        label_one_hot = self._soft_to_hard(model_logits.detach())
        mse_loss = F.mse_loss(
            label_one_hot + u_masked, label_onehot, reduction='sum',
        ) / len(label_onehot)
        return mse_loss

    def compute_loss_l3(self, batch_indices, model_logits, label_onehot,
                        training_accuracy):
        """L3: batch-wide categorical KL between alignment and uncertainty."""
        # Improvement 3: Use Temperature scaling to soften the alignment distribution
        alignment = torch.sum(model_logits * label_onehot, dim=1) / self.temperature
        u_detached = self.u[batch_indices].detach().view(-1)

        kl_loss = F.kl_div(
            F.log_softmax(alignment, dim=0),
            F.softmax(-torch.log(u_detached.clamp(min=1e-8)), dim=0),
            reduction='batchmean',
        )

        loss = (1 - training_accuracy) * kl_loss
        return loss

    def forward(self, batch_indices, model_logits, label_onehot,
                embeddings_detached, training_accuracy, epoch):
        """Compute GCOD loss components.

        Args:
            batch_indices: Global node indices for this batch.
            model_logits: Model output logits (with gradient).
            label_onehot: One-hot encoded labels.
            embeddings_detached: Detached node embeddings from backbone.
            training_accuracy: Current epoch training accuracy.
            epoch: Current training epoch.
        """
        # Improvement 1: Store L2-normalised features for scale-invariant similarity
        with torch.no_grad():
            self.prevSimilarity[batch_indices] = F.normalize(embeddings_detached, p=2, dim=1)

        # u masked by label — only true-class position (used in L1 and L2)
        u = self.u[batch_indices]       # (batch, 1)
        u_masked = u * label_onehot     # (batch, C)

        # Restore missing L1 call
        loss_l1 = self.compute_loss_l1(
            u_masked, model_logits, label_onehot,
            embeddings_detached, training_accuracy,
        )

        loss_l2 = self.compute_loss_l2(u_masked, model_logits, label_onehot)

        # GCOD Improvement: Implement configurable delay for L3 KL loss.
        # Uncertainty u is initialized very small (1e-8), which makes -log(u) huge.
        # We wait kl_start_epoch epochs for the model to stabilize before turning on KL.
        if epoch >= self.kl_start_epoch:
            loss_l3 = self.compute_loss_l3(
                batch_indices, model_logits, label_onehot, training_accuracy,
            )
        else:
            loss_l3 = torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = loss_l1 + loss_l2 + loss_l3
        return total_loss, loss_l1, loss_l2, loss_l3

    def _soft_to_hard(self, x):
        """Convert logits to one-hot of predicted class."""
        with torch.no_grad():
            return torch.zeros(
                len(x), self.num_classes, device=self.device,
            ).scatter_(1, x.argmax(dim=1).view(-1, 1), 1)


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
