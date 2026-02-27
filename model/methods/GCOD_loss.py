import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch_geometric.loader import NeighborLoader
from collections import defaultdict

from model.evaluation import (OversmoothingMetrics, ClassificationMetrics,
                              compute_oversmoothing_for_mask)
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


class GCODTrainer:
    
    def __init__(self, model, data, noisy_indices=None, device='cuda', 
                 learning_rate=0.01, weight_decay=5e-4, uncertainty_lr=0.01,
                 total_epochs=500, patience=100, batch_size=32, debug=True,
                 oversmoothing_every=20):
        
        self.model = model.to(device)
        self.data = data.to(device)
        self.noisy_indices = noisy_indices if noisy_indices is not None else []
        self.device = device
        self.debug = debug

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.uncertainty_lr = uncertainty_lr
        self.total_epochs = total_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.oversmoothing_every = oversmoothing_every

        self.num_classes = int(data.y.max().item()) + 1

        if hasattr(model, 'out_channels'):
            embedding_dim = model.out_channels
        else:
            embedding_dim = self.num_classes
        
        # Initialize GCOD loss
        self.gcod_loss_fn = GraphCentroidOutlierDiscounting(
            num_classes=self.num_classes,
            device=device,
            num_samples=data.num_nodes,
            embedding_dim=embedding_dim
        ).to(device)

        self.model_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )

        self.uncertainty_optimizer = torch.optim.Adam(
            [self.gcod_loss_fn.uncertainty_params], 
            lr=self.uncertainty_lr
        )

        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        self.cls_evaluator = ClassificationMetrics(average='macro')
        self._setup_data_loaders()
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
        self.training_accuracy = 0.1
    
    def _setup_data_loaders(self):
        train_indices = self.data.train_mask.nonzero(as_tuple=True)[0]
        val_indices = self.data.val_mask.nonzero(as_tuple=True)[0]
        test_indices = self.data.test_mask.nonzero(as_tuple=True)[0]
        
        self.train_loader = NeighborLoader(
            self.data,
            num_neighbors=[15, 10],
            batch_size=self.batch_size,
            input_nodes=train_indices,
            shuffle=True
        )
        
        self.val_loader = NeighborLoader(
            self.data,
            num_neighbors=[15, 10],
            batch_size=self.batch_size,
            input_nodes=val_indices,
            shuffle=False
        )
        
        self.test_loader = NeighborLoader(
            self.data,
            num_neighbors=[15, 10],
            batch_size=self.batch_size,
            input_nodes=test_indices,
            shuffle=False
        )
    
    def _compute_epoch_training_accuracy(self):

        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.train_loader:
                batch = batch.to(self.device)
                model_outputs = self.model(batch)
                
                batch_train_mask = batch.train_mask[:batch.batch_size]
                target_nodes = batch_train_mask.nonzero(as_tuple=True)[0]
                
                if len(target_nodes) > 0:
                    predictions = model_outputs[target_nodes].argmax(dim=1)
                    total_correct += (predictions == batch.y[target_nodes]).sum().item()
                    total_samples += len(target_nodes)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return max(0.01, min(0.99, accuracy))
    
    def train_single_epoch(self, epoch):
        self.model.train()
        self.gcod_loss_fn.train()
        
        current_acc = self._compute_epoch_training_accuracy()
        smooth_factor = 0.9 if epoch > 0 else 0.5
        self.training_accuracy = smooth_factor * self.training_accuracy + (1 - smooth_factor) * current_acc

        total_loss = 0
        total_l1 = 0
        total_l2 = 0
        total_l3 = 0
        num_batches = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            model_outputs = self.model(batch)
            node_embeddings = model_outputs

            batch_train_mask = batch.train_mask[:batch.batch_size]
            target_nodes = batch_train_mask.nonzero(as_tuple=True)[0]
            
            if len(target_nodes) == 0:
                continue
            
            original_indices = batch.n_id[target_nodes]
            true_labels = batch.y[target_nodes]
            true_labels_onehot = F.one_hot(true_labels, num_classes=self.num_classes).float()
            
            # Compute GCOD loss components
            gcod_total_loss, loss_l1, loss_l2, loss_l3 = self.gcod_loss_fn(
                batch_indices=original_indices,
                model_predictions=model_outputs[target_nodes],
                true_labels_onehot=true_labels_onehot,
                node_embeddings=node_embeddings[target_nodes],
                training_accuracy=self.training_accuracy,
                true_labels=true_labels
            )
            
            self.model_optimizer.zero_grad()
            model_loss = loss_l1 + loss_l3
            model_loss.backward(retain_graph=True)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.model_optimizer.step()

            self.uncertainty_optimizer.zero_grad()
            loss_l2.backward()
            
            torch.nn.utils.clip_grad_norm_([self.gcod_loss_fn.uncertainty_params], max_norm=1.0)
            self.uncertainty_optimizer.step()

            with torch.no_grad():
                self.gcod_loss_fn.uncertainty_params.clamp_(0, 1)

            total_loss += gcod_total_loss.item()
            total_l1 += loss_l1.item()
            total_l2 += loss_l2.item()
            total_l3 += loss_l3.item()
            num_batches += 1
        
        if num_batches == 0:
            return 0, 0, 0, 0
        
        return (total_loss / num_batches, total_l1 / num_batches, 
                total_l2 / num_batches, total_l3 / num_batches)

    def evaluate_model(self, data_loader, mask_name):
        self.model.eval()
        self.gcod_loss_fn.eval()
        
        total_loss = 0
        total_ce_loss = 0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_true_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                model_outputs = self.model(batch)
                
                if mask_name == 'train':
                    batch_mask = batch.train_mask[:batch.batch_size]
                elif mask_name == 'val':
                    batch_mask = batch.val_mask[:batch.batch_size]
                else:
                    batch_mask = batch.test_mask[:batch.batch_size]
                
                target_nodes = batch_mask.nonzero(as_tuple=True)[0]
                if len(target_nodes) == 0:
                    continue
                
                original_indices = batch.n_id[target_nodes]
                true_labels = batch.y[target_nodes]
                true_labels_onehot = F.one_hot(true_labels, num_classes=self.num_classes).float()
                
                # GCOD loss
                gcod_loss, loss_l1, _, loss_l3 = self.gcod_loss_fn(
                    batch_indices=original_indices,
                    model_predictions=model_outputs[target_nodes],
                    true_labels_onehot=true_labels_onehot,
                    node_embeddings=model_outputs[target_nodes],
                    training_accuracy=self.training_accuracy,
                    true_labels=true_labels
                )
                total_loss += gcod_loss.item()
                
                ce_loss = F.cross_entropy(model_outputs[target_nodes], true_labels)
                total_ce_loss += ce_loss.item()
                
                predictions = model_outputs[target_nodes].argmax(dim=1)
                total_correct += (predictions == true_labels).sum().item()
                total_samples += len(target_nodes)
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(true_labels.cpu().numpy())
                
                num_batches += 1
        
        if num_batches == 0 or total_samples == 0:
            return {'loss': float('inf'), 'ce_loss': float('inf'), 'accuracy': 0.0, 'f1': 0.0}
        
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        epoch_cls_metrics = self.cls_evaluator.compute_all_metrics(all_predictions, all_true_labels)
        accuracy = epoch_cls_metrics['accuracy']
        f1 = epoch_cls_metrics['f1']
        
        return {'loss': avg_loss, 'ce_loss': avg_ce_loss, 'accuracy': accuracy, 'f1': f1}

    def compute_oversmoothing_metrics(self):
        try:
            self.model.eval()
            with torch.no_grad():
                full_embeddings = self.model.get_embeddings(self.data)
            return {
                'train': compute_oversmoothing_for_mask(
                    self.oversmoothing_evaluator, full_embeddings, self.data.edge_index, self.data.train_mask),
                'val': compute_oversmoothing_for_mask(
                    self.oversmoothing_evaluator, full_embeddings, self.data.edge_index, self.data.val_mask),
                'test': compute_oversmoothing_for_mask(
                    self.oversmoothing_evaluator, full_embeddings, self.data.edge_index, self.data.test_mask),
            }
        except Exception as e:
            print(f"Warning: Could not compute oversmoothing metrics: {e}")
            return {'train': {}, 'val': {}, 'test': {}}
        
    def train_full_model(self, log_epoch_fn=None):
        per_epochs_train_oversmoothing = defaultdict(list)
        per_epochs_val_oversmoothing = defaultdict(list)
        if self.debug:
            print(f"Starting GCOD training for {self.total_epochs} epochs")
            print(f"Training samples: {self.data.train_mask.sum().item()}")
            print(f"Noisy samples: {len(self.noisy_indices)}")
        
        for epoch in range(self.total_epochs):
            train_loss, l1_loss, l2_loss, l3_loss = self.train_single_epoch(epoch)
            
            train_metrics = self.evaluate_model(self.train_loader, 'train')
            val_ce_metrics = evaluate_ce_only(self.model, self.val_loader, self.device, mask_name='val')
            val_loss_ce = val_ce_metrics['ce_loss']

            oversmoothing_metrics = self.compute_oversmoothing_metrics()

            os_entry = None
            if self.debug and (epoch + 1) % self.oversmoothing_every == 0:
                with torch.no_grad():
                    u_mean = self.gcod_loss_fn.uncertainty_params.mean().item()
                    u_std = self.gcod_loss_fn.uncertainty_params.std().item()
                    u_min = self.gcod_loss_fn.uncertainty_params.min().item()
                    u_max = self.gcod_loss_fn.uncertainty_params.max().item()

                print(f"Epoch {epoch:03d} | Train Loss: {train_metrics['loss']:.4f}, Val CE Loss: {val_ce_metrics['ce_loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_ce_metrics['accuracy']:.4f} | "
                    f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_ce_metrics['f1']:.4f}")
                print(f"Training Accuracy: {self.training_accuracy:.4f} | "
                    f"L1: {l1_loss:.4f}, L2: {l2_loss:.4f}, L3: {l3_loss:.4f}")
                print(f"Uncertainty Stats - Mean: {u_mean:.4f}, Std: {u_std:.4f}, "
                    f"Min: {u_min:.4f}, Max: {u_max:.4f}")

                train_os = oversmoothing_metrics.get('train', {})
                val_os = oversmoothing_metrics.get('val', {})
                for key, value in train_os.items():
                    per_epochs_train_oversmoothing[key].append(value)
                for key, value in val_os.items():
                    per_epochs_val_oversmoothing[key].append(value)

                train_edir = train_os.get('EDir', 0.0)
                train_edir_traditional = train_os.get('EDir_traditional', 0.0)
                train_eproj = train_os.get('EProj', 0.0)
                train_mad = train_os.get('MAD', 0.0)
                train_num_rank = train_os.get('NumRank', 0.0)
                train_effective_rank = train_os.get('Erank', 0.0)

                val_edir = val_os.get('EDir', 0.0)
                val_edir_traditional = val_os.get('EDir_traditional', 0.0)
                val_eproj = val_os.get('EProj', 0.0)
                val_mad = val_os.get('MAD', 0.0)
                val_num_rank = val_os.get('NumRank', 0.0)
                val_effective_rank = val_os.get('Erank', 0.0)

                print(f"Train DE: {train_edir:.4f}, Val DE: {val_edir:.4f} | "
                    f"Train DE_trad: {train_edir_traditional:.4f}, Val DE_trad: {val_edir_traditional:.4f} | "
                    f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                    f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                    f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                    f"Train Erank: {train_effective_rank:.4f}, Val Erank: {val_effective_rank:.4f}")
                os_entry = {'train': dict(train_os), 'val': dict(val_os)}

            # Early stopping
            is_best = val_loss_ce < self.best_val_loss
            if log_epoch_fn is not None:
                log_epoch_fn(epoch, train_metrics['loss'], val_loss_ce,
                             train_metrics['accuracy'], val_ce_metrics['accuracy'],
                             train_f1=train_metrics['f1'], val_f1=val_ce_metrics['f1'],
                             oversmoothing=os_entry, is_best=is_best)
            if is_best:
                self.best_val_loss = val_loss_ce
                self.best_model_state = deepcopy(self.model.state_dict())
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                if self.debug:
                    print(f"Early stopping triggered at epoch {epoch}")
                break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            'train_oversmoothing': dict(per_epochs_train_oversmoothing),
            'val_oversmoothing': dict(per_epochs_val_oversmoothing),
            'reduce': False,
            'stopped_at_epoch': epoch,
        }



@register('gcod')
class GCODMethodTrainer(BaseTrainer):
    def train(self):
        d = self.init_data
        gcod_params = self.config.get('gcod_params', {})

        trainer = GCODTrainer(
            model=d['backbone_model'],
            data=d['data_for_training'],
            noisy_indices=d['global_noisy_indices'],
            device=d['device'],
            learning_rate=d['lr'],
            weight_decay=d['weight_decay'],
            uncertainty_lr=float(gcod_params.get('uncertainty_lr', 1.0)),
            total_epochs=d['epochs'],
            patience=d['patience'],
            batch_size=int(gcod_params.get('batch_size', 32)),
            debug=True,
            oversmoothing_every=d['oversmoothing_every'],
        )
        return trainer.train_full_model(log_epoch_fn=self.log_epoch)