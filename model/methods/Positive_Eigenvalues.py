import torch
import torch.nn.functional as F
from copy import deepcopy
from torch_geometric.loader import NeighborLoader
from collections import defaultdict

from model.evaluation import (OversmoothingMetrics, ClassificationMetrics,
                              compute_oversmoothing_for_mask)
from model.base import BaseTrainer
from model.registry import register

class PositiveEigenvaluesTrainer:
    # Positive eigenvalues constraint method
    
    def __init__(self, model, data, device='cuda', learning_rate=0.01, weight_decay=5e-4, oversmoothing_every=20):

        self.model = model
        self.data = data
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.oversmoothing_every = oversmoothing_every
        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        self.cls_evaluator = ClassificationMetrics(average='macro')
        
        self.model.to(device)
        self.data = data.to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
    
    @staticmethod
    def reconstruct_matrix_with_positive_singular_values(weight_matrix, eps=1e-8):
        # Reconstruct weight matrix keeping only positive singular values.

        with torch.no_grad():
            # Compute SVD
            U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
            
            # Keep only positive singular values
            positive_mask = S > eps
            if positive_mask.sum() == 0:
                return torch.eye(weight_matrix.size(0), weight_matrix.size(1), device=weight_matrix.device) * 0.01
            
            S_pos = S[positive_mask]
            U_pos = U[:, positive_mask]
            Vh_pos = Vh[positive_mask, :]
            
            # Reconstruct matrix
            reconstructed = U_pos @ torch.diag(S_pos) @ Vh_pos
            return reconstructed
    
    def apply_positive_eigenvalue_constraint_to_final_projection(self):

        with torch.no_grad():

            model_layers = list(self.model.named_modules())
            
            for module_name, module in reversed(model_layers):
                if (hasattr(module, 'weight') and 
                    isinstance(module, torch.nn.Linear) and 
                    module.weight.dim() == 2):

                    if module.weight.size(0) == module.weight.size(1):
                        original_weight = module.weight.data.clone()
                        constrained_weight = self.reconstruct_matrix_with_positive_singular_values(original_weight)
                        module.weight.data = constrained_weight
                        break
    
    def train_with_positive_eigenvalue_constraint(self, max_epochs=200, batch_size=32,
                                                patience=20, noisy_indices=None,
                                                log_epoch_fn=None):

        
        per_epochs_oversmoothing = defaultdict(list)
        per_epochs_val_oversmoothing = defaultdict(list)
        train_loader, val_loader, test_loader = self.create_data_loaders(batch_size)
        
        best_validation_loss = float('inf')
        best_model_weights = None
        epochs_without_improvement = 0
        
        for epoch in range(max_epochs):

            train_metrics = self.train_single_epoch(train_loader)

            val_metrics = self.evaluate_on_split(val_loader, 'val')

            # Early stopping tracking
            is_best = val_metrics['loss'] < best_validation_loss
            if is_best:
                best_validation_loss = val_metrics['loss']
                best_model_weights = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Oversmoothing computation
            os_entry = None
            if (epoch + 1) % self.oversmoothing_every == 0:

                self.model.eval()
                with torch.no_grad():
                    full_embeddings = self.model.get_embeddings(self.data)

                train_oversmoothing = compute_oversmoothing_for_mask(
                    self.oversmoothing_evaluator, full_embeddings, self.data.edge_index, self.data.train_mask
                )

                val_oversmoothing = compute_oversmoothing_for_mask(
                    self.oversmoothing_evaluator, full_embeddings, self.data.edge_index, self.data.val_mask
                )

                os_entry = {'train': dict(train_oversmoothing), 'val': dict(val_oversmoothing)}

                metrics = {
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_acc': val_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'val_f1': val_metrics['f1']
                }

                print(f"Epoch {epoch:03d} | Train Loss: {metrics['train_loss']:.4f}, Val Loss: {metrics['val_loss']:.4f} | "
                    f"Train Acc: {metrics['train_acc']:.4f}, Val Acc: {metrics['val_acc']:.4f} | "
                    f"Train F1: {metrics['train_f1']:.4f}, Val F1: {metrics['val_f1']:.4f}")


                for key, value in train_oversmoothing.items():
                    per_epochs_oversmoothing[key].append(value)
                for key, value in val_oversmoothing.items():
                    per_epochs_val_oversmoothing[key].append(value)
                train_edir = train_oversmoothing.get('EDir', 0.0) if train_oversmoothing else 0.0
                train_edir_traditional = train_oversmoothing.get('EDir_traditional', 0.0) if train_oversmoothing else 0.0
                train_eproj = train_oversmoothing.get('EProj', 0.0) if train_oversmoothing else 0.0
                train_mad = train_oversmoothing.get('MAD', 0.0) if train_oversmoothing else 0.0
                train_num_rank = train_oversmoothing.get('NumRank', 0.0) if train_oversmoothing else 0.0
                train_effective_rank = train_oversmoothing.get('Erank', 0.0) if train_oversmoothing else 0.0

                val_edir = val_oversmoothing.get('EDir', 0.0) if val_oversmoothing else 0.0
                val_edir_traditional = val_oversmoothing.get('EDir_traditional', 0.0) if val_oversmoothing else 0.0
                val_eproj = val_oversmoothing.get('EProj', 0.0) if val_oversmoothing else 0.0
                val_mad = val_oversmoothing.get('MAD', 0.0) if val_oversmoothing else 0.0
                val_num_rank = val_oversmoothing.get('NumRank', 0.0) if val_oversmoothing else 0.0
                val_effective_rank = val_oversmoothing.get('Erank', 0.0) if val_oversmoothing else 0.0

                print(f"Train DE: {train_edir:.4f}, Val DE: {val_edir:.4f} | "
                    f"Train DE_trad: {train_edir_traditional:.4f}, Val DE_trad: {val_edir_traditional:.4f} | "
                    f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                    f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                    f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                    f"Train Erank: {train_effective_rank:.4f}, Val Erank: {val_effective_rank:.4f}")

            elif (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")

            if log_epoch_fn is not None:
                log_epoch_fn(epoch, train_metrics['loss'], val_metrics['loss'],
                             train_metrics['accuracy'], val_metrics['accuracy'],
                             train_f1=train_metrics['f1'], val_f1=val_metrics.get('f1'),
                             oversmoothing=os_entry, is_best=is_best)

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                self.model.load_state_dict(best_model_weights)
                break
        
        print("\nPositive Eigenvalues Training completed")

        return {
            'train_oversmoothing': dict(per_epochs_oversmoothing),
            'val_oversmoothing': dict(per_epochs_val_oversmoothing),
            'stopped_at_epoch': epoch,
        }
    
    def create_data_loaders(self, batch_size=32):

        train_indices = self.data.train_mask.nonzero(as_tuple=True)[0]
        val_indices = self.data.val_mask.nonzero(as_tuple=True)[0]
        test_indices = self.data.test_mask.nonzero(as_tuple=True)[0]
        
        train_loader = NeighborLoader(
            self.data,
            num_neighbors=[15, 10],
            batch_size=batch_size,
            input_nodes=train_indices,
            shuffle=True
        )
        
        val_loader = NeighborLoader(
            self.data,
            num_neighbors=[15, 10],
            batch_size=batch_size,
            input_nodes=val_indices,
            shuffle=False
        )
        
        test_loader = NeighborLoader(
            self.data,
            num_neighbors=[15, 10],
            batch_size=batch_size,
            input_nodes=test_indices,
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_single_epoch(self, train_loader):

        # The positive eigenvalue constraint is applied after each parameter update

        self.model.train()
        
        epoch_loss = 0
        batch_count = 0
        all_predictions = []
        all_true_labels = []
        epoch_embeddings_sample = []
        
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            output_logits = self.model(batch)
            
            batch_train_mask = batch.train_mask[:batch.batch_size]
            target_node_indices = batch_train_mask.nonzero(as_tuple=True)[0]
            
            if len(target_node_indices) == 0:
                continue

            classification_loss = F.cross_entropy(
                output_logits[target_node_indices], 
                batch.y[target_node_indices]
            )

            classification_loss.backward()
            self.optimizer.step()

            self.apply_positive_eigenvalue_constraint_to_final_projection()
            
            with torch.no_grad():
                predictions = output_logits[target_node_indices].argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(batch.y[target_node_indices].cpu().numpy())
            
            epoch_loss += classification_loss.item()
            batch_count += 1
            
            if len(epoch_embeddings_sample) < 3:
                with torch.no_grad():

                    embeddings_for_analysis = output_logits.detach()
                    epoch_embeddings_sample.append({
                        'embeddings': embeddings_for_analysis[:batch.batch_size].cpu(),
                        'edge_index': batch.edge_index.cpu(),
                        'mask': batch_train_mask.cpu()
                    })
        
        epoch_cls_metrics = self.cls_evaluator.compute_all_metrics(all_predictions, all_true_labels)
        
        avg_metrics = {
            'loss': epoch_loss / batch_count,
            'accuracy': epoch_cls_metrics['accuracy'],
            'f1': epoch_cls_metrics['f1'],
            'embeddings_sample': epoch_embeddings_sample
        }
        
        return avg_metrics
    
    def evaluate_on_split(self, data_loader, split_name='val'):

        self.model.eval()
        
        total_loss = 0
        batch_count = 0
        all_predictions = []
        all_true_labels = []
        embeddings_samples = []
        
        mask_attr = f'{split_name}_mask'
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                output_logits = self.model(batch)
                
                batch_mask = getattr(batch, mask_attr)[:batch.batch_size]
                target_indices = batch_mask.nonzero(as_tuple=True)[0]
                
                if len(target_indices) == 0:
                    continue
                
                classification_loss = F.cross_entropy(
                    output_logits[target_indices], 
                    batch.y[target_indices]
                )
                
                predictions = output_logits[target_indices].argmax(dim=1)
                true_labels = batch.y[target_indices]
                
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(true_labels.cpu().numpy())
                
                total_loss += classification_loss.item()
                batch_count += 1
                
                if len(embeddings_samples) < 3:
                    embeddings_samples.append({
                        'embeddings': output_logits[:batch.batch_size].cpu(),
                        'edge_index': batch.edge_index.cpu(),
                        'mask': batch_mask.cpu()
                    })
        
        epoch_cls_metrics = self.cls_evaluator.compute_all_metrics(all_predictions, all_true_labels)
        
        evaluation_results = {
            'loss': total_loss / batch_count,
            'accuracy': epoch_cls_metrics['accuracy'],
            'f1': epoch_cls_metrics['f1'],
            'precision': epoch_cls_metrics['precision'],
            'recall': epoch_cls_metrics['recall'],
            'embeddings_samples': embeddings_samples
        }
        
        return evaluation_results



@register('positive_eigenvalues')
class PositiveEigenvaluesMethodTrainer(BaseTrainer):
    def train(self):
        d = self.init_data
        pe_params = self.config.get('positive_eigenvalues_params', {})

        trainer = PositiveEigenvaluesTrainer(
            model=d['backbone_model'],
            data=d['data_for_training'],
            device=d['device'],
            learning_rate=d['lr'],
            weight_decay=d['weight_decay'],
            oversmoothing_every=d['oversmoothing_every'],
        )
        return trainer.train_with_positive_eigenvalue_constraint(
            max_epochs=d['epochs'],
            batch_size=int(pe_params.get('batch_size', 32)),
            patience=d['patience'],
            noisy_indices=d['global_noisy_indices'],
            log_epoch_fn=self.log_epoch,
        )
