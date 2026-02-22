import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score 
from copy import deepcopy
from torch_geometric.loader import NeighborLoader
from collections import defaultdict

from model.evaluation import OversmoothingMetrics

class PositiveEigenvaluesTrainer:
    # Positive eigenvalues constraint method
    
    def __init__(self, model, data, device='cuda', learning_rate=0.01, weight_decay=5e-4):

        self.model = model
        self.data = data
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        
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
    
    def compute_oversmoothing_metrics_for_subset(self, embeddings, edge_index, node_mask):

        try:
            masked_node_indices = torch.where(node_mask)[0]
            masked_embeddings = embeddings[masked_node_indices]
            
            node_index_set = set(masked_node_indices.cpu().numpy())
            
            edge_mask = torch.tensor([
                source.item() in node_index_set and target.item() in node_index_set
                for source, target in edge_index.t()
            ], device=edge_index.device)
            
            if not edge_mask.any():
                return {
                    'NumRank': float(min(masked_embeddings.shape)),
                    'Erank': float(min(masked_embeddings.shape)),
                    'EDir': 0.0,
                    'EDir_traditional': 0.0,
                    'EProj': 0.0,
                    'MAD': 0.0
                }
            
            filtered_edges = edge_index[:, edge_mask]
            node_remapping = {
                original_idx.item(): local_idx 
                for local_idx, original_idx in enumerate(masked_node_indices)
            }
            
            remapped_edge_index = torch.stack([
                torch.tensor([node_remapping[src.item()] for src in filtered_edges[0]], 
                        device=edge_index.device),
                torch.tensor([node_remapping[tgt.item()] for tgt in filtered_edges[1]], 
                        device=edge_index.device)
            ])
            
            graph_data_list = [{
                'X': masked_embeddings,
                'edge_index': remapped_edge_index,
                'edge_weight': None
            }]
            
            all_metrics = self.oversmoothing_evaluator.compute_all_metrics(
                X=masked_embeddings,
                edge_index=remapped_edge_index,
                graphs_in_class=graph_data_list
            )

            default_metrics = {
                'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0
            }
            
            if all_metrics:
                default_metrics.update(all_metrics)
            
            return default_metrics
            
        except Exception as e:
            print(f"Warning: Could not compute oversmoothing metrics: {e}")
            return {
                'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0
            }

    def train_with_positive_eigenvalue_constraint(self, max_epochs=200, batch_size=32, 
                                                patience=20, noisy_indices=None):

        
        per_epochs_oversmoothing = defaultdict(list)
        train_loader, val_loader, test_loader = self.create_data_loaders(batch_size)
        
        best_validation_loss = float('inf')
        best_model_weights = None
        epochs_without_improvement = 0
        
        oversmoothing_history = {
            'train': [],
            'val': [],
            'test': []
        }

        for epoch in range(1, max_epochs + 1):

            train_metrics = self.train_single_epoch(train_loader)

            val_metrics = self.evaluate_on_split(val_loader, 'val')
            
            # Early stopping
            if val_metrics['loss'] < best_validation_loss:
                best_validation_loss = val_metrics['loss']
                best_model_weights = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                self.model.load_state_dict(best_model_weights)
                break
            
            if epoch % 20 == 0:

                self.model.eval()
                with torch.no_grad():
                    full_embeddings = self.model(self.data)
                
                train_oversmoothing = self.compute_oversmoothing_metrics_for_subset(
                    full_embeddings, self.data.edge_index, self.data.train_mask
                )

                val_oversmoothing = self.compute_oversmoothing_metrics_for_subset(
                    full_embeddings, self.data.edge_index, self.data.val_mask
                )
                
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
            
            elif epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        test_metrics = self.evaluate_on_split(test_loader, 'test')
        
        self.model.eval()
        with torch.no_grad():
            full_embeddings = self.model(self.data)

        test_oversmoothing = self.compute_oversmoothing_metrics_for_subset(
            full_embeddings, self.data.edge_index, self.data.test_mask
        )
        
        print(f"\nPositive Eigenvalues Training completed")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        
        if test_oversmoothing:
            print("Final Test Oversmoothing Metrics:")
            print(f"Test DE: {test_oversmoothing.get('EDir', 0.0):.4f}, "
                f"Test DE_trad: {test_oversmoothing.get('EDir_traditional', 0.0):.4f}, "
                f"Test EProj: {test_oversmoothing.get('EProj', 0.0):.4f}, "
                f"Test MAD: {test_oversmoothing.get('MAD', 0.0):.4f}, "
                f"Test NumRank: {test_oversmoothing.get('NumRank', 0.0):.4f}, "
                f"Test Erank: {test_oversmoothing.get('Erank', 0.0):.4f}")
        
        return {
            'accuracy': test_metrics['accuracy'],
            'f1': test_metrics['f1'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'oversmoothing': test_oversmoothing or {
                'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0
            },
            'train_oversmoothing' : per_epochs_oversmoothing
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
        epoch_accuracy = 0
        epoch_f1 = 0
        batch_count = 0
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
                batch_accuracy = (predictions == batch.y[target_node_indices]).sum().item() / len(target_node_indices)
                batch_f1 = f1_score(
                    batch.y[target_node_indices].cpu(), 
                    predictions.cpu(), 
                    average='macro'
                )
            
            epoch_loss += classification_loss.item()
            epoch_accuracy += batch_accuracy
            epoch_f1 += batch_f1
            batch_count += 1
            
            if len(epoch_embeddings_sample) < 3:
                with torch.no_grad():

                    embeddings_for_analysis = output_logits.detach()
                    epoch_embeddings_sample.append({
                        'embeddings': embeddings_for_analysis[:batch.batch_size].cpu(),
                        'edge_index': batch.edge_index.cpu(),
                        'mask': batch_train_mask.cpu()
                    })
        
        avg_metrics = {
            'loss': epoch_loss / batch_count,
            'accuracy': epoch_accuracy / batch_count,
            'f1': epoch_f1 / batch_count,
            'embeddings_sample': epoch_embeddings_sample
        }
        
        return avg_metrics
    
    def evaluate_on_split(self, data_loader, split_name='val'):

        self.model.eval()
        
        total_loss = 0
        total_accuracy = 0
        total_f1 = 0
        total_precision = 0
        total_recall = 0
        batch_count = 0
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
                
                batch_accuracy = (predictions == true_labels).sum().item() / len(target_indices)
                batch_f1 = f1_score(true_labels.cpu(), predictions.cpu(), average='macro')
                batch_precision = precision_score(true_labels.cpu(), predictions.cpu(), average='macro')
                batch_recall = recall_score(true_labels.cpu(), predictions.cpu(), average='macro')
                
                total_loss += classification_loss.item()
                total_accuracy += batch_accuracy
                total_f1 += batch_f1
                total_precision += batch_precision
                total_recall += batch_recall
                batch_count += 1
                
                if len(embeddings_samples) < 3:
                    embeddings_samples.append({
                        'embeddings': output_logits[:batch.batch_size].cpu(),
                        'edge_index': batch.edge_index.cpu(),
                        'mask': batch_mask.cpu()
                    })
        
        evaluation_results = {
            'loss': total_loss / batch_count,
            'accuracy': total_accuracy / batch_count,
            'f1': total_f1 / batch_count,
            'precision': total_precision / batch_count,
            'recall': total_recall / batch_count,
            'embeddings_samples': embeddings_samples
        }
        
        return evaluation_results
    