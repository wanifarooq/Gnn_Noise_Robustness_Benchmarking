import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score

from model.evaluation import OversmoothingMetrics

def dirichlet_energy(x, edge_index):
    row, col = edge_index
    diff = x[row] - x[col]
    return (diff ** 2).sum(dim=1).mean()

def _compute_oversmoothing_for_mask(oversmoothing_evaluator, embeddings, edge_index, mask, labels=None):
    try:
        mask_indices = torch.where(mask)[0]
        mask_embeddings = embeddings[mask]
        
        mask_set = set(mask_indices.cpu().numpy())
        edge_mask = torch.tensor([
            src.item() in mask_set and tgt.item() in mask_set
            for src, tgt in edge_index.t()
        ], device=edge_index.device)
        
        if not edge_mask.any():
            return {
                'NumRank': float(min(mask_embeddings.shape)),
                'Erank': float(min(mask_embeddings.shape)),
                'EDir': 0.0,
                'EDir_traditional': 0.0,
                'EProj': 0.0,
                'MAD': 0.0
            }
        
        masked_edges = edge_index[:, edge_mask]
        node_mapping = {orig_idx.item(): local_idx for local_idx, orig_idx in enumerate(mask_indices)}
        
        remapped_edges = torch.stack([
            torch.tensor([node_mapping[src.item()] for src in masked_edges[0]], device=edge_index.device),
            torch.tensor([node_mapping[tgt.item()] for tgt in masked_edges[1]], device=edge_index.device)
        ])
        
        graphs_in_class = [{
            'X': mask_embeddings,
            'edge_index': remapped_edges,
            'edge_weight': None
        }]
        
        return oversmoothing_evaluator.compute_all_metrics(
            X=mask_embeddings,
            edge_index=remapped_edges,
            graphs_in_class=graphs_in_class
        )
        
    except Exception as e:
        print(f"Warning: Could not compute oversmoothing metrics for mask: {e}")
        return {
            'NumRank': 0.0,
            'Erank': 0.0,
            'EDir': 0.0,
            'EDir_traditional': 0.0,
            'EProj': 0.0,
            'MAD': 0.0
        }

def train_with_standard_loss(
    model, data, noisy_indices, device,
    total_epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    patience=20,
    debug=True
):

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    oversmoothing_evaluator = OversmoothingMetrics(device=device)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        loss_train = criterion(out[train_idx], data.y[train_idx])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_idx = data.val_mask.nonzero(as_tuple=True)[0]
            test_idx = data.test_mask.nonzero(as_tuple=True)[0]

            val_loss = criterion(out[val_idx], data.y[val_idx])

            pred = out.argmax(dim=1)
            train_acc = (pred[train_idx] == data.y[train_idx]).sum().item() / len(train_idx)
            val_acc = (pred[val_idx] == data.y[val_idx]).sum().item() / len(val_idx)
            train_f1 = f1_score(data.y[train_idx].cpu(), pred[train_idx].cpu(), average='macro')
            val_f1 = f1_score(data.y[val_idx].cpu(), pred[val_idx].cpu(), average='macro')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break

        if debug and epoch % 20 == 0:

            train_oversmoothing = _compute_oversmoothing_for_mask(
                oversmoothing_evaluator, out, data.edge_index, data.train_mask, data.y
            )
            val_oversmoothing = _compute_oversmoothing_for_mask(
                oversmoothing_evaluator, out, data.edge_index, data.val_mask, data.y
            )

            train_de = train_oversmoothing['EDir']
            train_de_traditional = train_oversmoothing['EDir_traditional']
            train_eproj = train_oversmoothing['EProj']
            train_mad = train_oversmoothing['MAD']
            train_num_rank = train_oversmoothing['NumRank']
            train_eff_rank = train_oversmoothing['Erank']

            val_de = val_oversmoothing['EDir']
            val_de_traditional = val_oversmoothing['EDir_traditional']
            val_eproj = val_oversmoothing['EProj']
            val_mad = val_oversmoothing['MAD']
            val_num_rank = val_oversmoothing['NumRank']
            val_eff_rank = val_oversmoothing['Erank']

            print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
            print(f"Train DE: {train_de:.4f}, Val DE: {val_de:.4f} | "
                  f"Train DE_trad: {train_de_traditional:.4f}, Val DE_trad: {val_de_traditional:.4f} | "
                  f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                  f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                  f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                  f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")
    model.eval()
    with torch.no_grad():
        test_idx = data.test_mask.nonzero(as_tuple=True)[0]
        test_loss = criterion(out[test_idx], data.y[test_idx])
        pred = out.argmax(dim=1)
        test_acc = (pred[test_idx] == data.y[test_idx]).sum().item() / len(test_idx)
        test_f1 = f1_score(data.y[test_idx].cpu(), pred[test_idx].cpu(), average='macro')
        test_precision = precision_score(data.y[test_idx].cpu(), pred[test_idx].cpu(), average='macro')
        test_recall = recall_score(data.y[test_idx].cpu(), pred[test_idx].cpu(), average='macro')

        final_train_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.train_mask, data.y
        )
        final_val_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.val_mask, data.y
        )
        test_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.test_mask, data.y
        )

    expected_keys = ['NumRank', 'Erank', 'EDir', 'EDir_traditional', 'EProj', 'MAD']

    def normalize_metrics(d):
        if d is None:
            return {k: 0.0 for k in expected_keys}
        return {k: d.get(k, 0.0) for k in expected_keys}

    final_train_oversmoothing = normalize_metrics(final_train_oversmoothing)
    final_val_oversmoothing = normalize_metrics(final_val_oversmoothing)
    final_test_oversmoothing = normalize_metrics(test_oversmoothing)

    if debug:
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | "
              f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        print("Final Oversmoothing Metrics:")
        print(f"Train: {final_train_oversmoothing}")
        print(f"Val: {final_val_oversmoothing}")
        print(f"Test: {final_test_oversmoothing}")

    results = {
        'accuracy': torch.tensor(test_acc),
        'f1': torch.tensor(test_f1),
        'precision': torch.tensor(test_precision),
        'recall': torch.tensor(test_recall),
        'oversmoothing': final_test_oversmoothing
    }

    return results

class NCODLossModule(nn.Module):

    def __init__(self, sample_labels, device, num_samples=50000, num_classes=100, 
                 consistency_ratio=0, balance_ratio=0, feature_dim=64, total_epochs=100):
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device
        self.num_samples = num_samples
        self.total_epochs = total_epochs
        self.consistency_ratio = consistency_ratio
        self.balance_ratio = balance_ratio
        

        self.uncertainty_params = nn.Parameter(torch.empty(num_samples, 1, dtype=torch.float32))
        self._initialize_uncertainty_params(mean=1e-8, std=1e-9)
        
        self.is_first_forward = True
        self.previous_features = torch.rand((num_samples, feature_dim), device=device)
        self.class_prototypes = torch.rand((num_classes, feature_dim), device=device)
        self.sample_predictions = torch.zeros((num_samples, 1), device=device)
        self.sample_weights = torch.zeros((num_samples, 1), device=device)

        self.sample_labels = sample_labels
        self.class_sample_indices = [np.where(self.sample_labels == i)[0] for i in range(num_classes)]
        self.shuffled_class_indices = copy.deepcopy(self.class_sample_indices)
        for class_indices in self.shuffled_class_indices:
            random.shuffle(class_indices)

    def _initialize_uncertainty_params(self, mean=1e-8, std=1e-9):
        #Initialize uncertainty parameters
        torch.nn.init.normal_(self.uncertainty_params, mean=mean, std=std)

    def forward(self, sample_indices, model_outputs, ground_truth_labels, feature_representations, 
                training_flag, current_epoch, training_accuracy=None):

        if len(model_outputs) > len(sample_indices):
            primary_output, augmented_output = torch.chunk(model_outputs, 2)
            primary_features, augmented_features = torch.chunk(feature_representations, 2)
        else:
            primary_output = model_outputs
            primary_features = feature_representations

        eps = 1e-4
        batch_uncertainty = self.uncertainty_params[sample_indices]
        batch_weights = self.sample_weights[sample_indices]

        if training_flag == 0:
            if self.is_first_forward:
                # Dynamic percentage based on epoch
                selection_percentage = math.ceil((50 - (50 / self.total_epochs) * current_epoch) + 50)
                self._update_class_prototypes(selection_percentage=selection_percentage)
            self._prepare_prototype_matrix()
            self.is_first_forward = True

        self.previous_features[sample_indices] = primary_features.detach()
        
        # Compute similarity-based loss
        similarity_loss = self._compute_similarity_loss(
            sample_indices, primary_output, ground_truth_labels, 
            primary_features, batch_uncertainty, batch_weights, eps
        )
        
        # Compute MSE reconstruction loss
        mse_loss = self._compute_reconstruction_loss(
            primary_output, ground_truth_labels, batch_uncertainty, batch_weights,
            sample_indices
        )
        
        # Compute KL divergence loss
        kl_loss = self._compute_kl_divergence_loss(primary_output, ground_truth_labels, sample_indices)
        
        total_loss = similarity_loss + mse_loss + kl_loss

        if self.balance_ratio > 0:
            balance_loss = self._compute_balance_loss(primary_output, eps)
            total_loss += self.balance_ratio * balance_loss

        if len(model_outputs) > len(sample_indices) and self.consistency_ratio > 0:
            consistency_loss = self._compute_consistency_loss(primary_output, augmented_output)
            total_loss += self.consistency_ratio * torch.mean(consistency_loss)

        return total_loss

    def _update_class_prototypes(self, selection_percentage=100):

        for class_idx in range(len(self.class_sample_indices)):
            class_uncertainties = self.uncertainty_params.detach()[self.class_sample_indices[class_idx]]
            num_samples_to_select = int((len(class_uncertainties) / 100) * selection_percentage)
            
            _, selected_indices = torch.topk(class_uncertainties, num_samples_to_select, 
                                           largest=False, dim=0)
            
            selected_indices_cpu = selected_indices.view(-1).cpu().numpy()
            
            selected_sample_indices = self.class_sample_indices[class_idx][selected_indices_cpu]
            self.class_prototypes[class_idx] = torch.mean(
                self.previous_features[selected_sample_indices], dim=0
            )

    def _prepare_prototype_matrix(self):
        prototype_norms = self.class_prototypes.norm(p=2, dim=1, keepdim=True)
        normalized_prototypes = self.class_prototypes.div(prototype_norms)
        self.normalized_prototypes_transpose = torch.transpose(normalized_prototypes, 0, 1)

    def _compute_similarity_loss(self, sample_indices, model_outputs, labels, features, 
                               uncertainty, weights, eps):
        #Compute similarity-based contrastive loss
        predictions = F.softmax(model_outputs, dim=1)
        
        feature_norms = features.detach().norm(p=2, dim=1, keepdim=True)
        normalized_features = features.detach().div(feature_norms)
        
        # Compute similarity
        similarities = torch.mm(normalized_features, self.normalized_prototypes_transpose)
        masked_similarities = similarities * labels
        
        similarity_mask = (masked_similarities > 0.000).type(torch.float32)
        filtered_similarities = masked_similarities * similarity_mask
        
        # Apply uncertainty weighting to labels
        weighted_uncertainty = uncertainty * labels
        adjusted_predictions = torch.clamp(
            (predictions + ((1 - weights) * weighted_uncertainty.detach())), 
            min=eps, max=1.0
        )
        
        # Compute negative log-likelihood
        loss = torch.mean(-torch.sum(filtered_similarities * torch.log(adjusted_predictions), dim=1))
        
        return loss

    def _compute_reconstruction_loss(self, model_outputs, labels, uncertainty, weights, sample_indices):
        hard_predictions = self._convert_soft_to_hard_labels(model_outputs.detach())
        reconstruction_target = hard_predictions + (weights * uncertainty)
        mse_loss = F.mse_loss(reconstruction_target, labels, reduction='sum') / len(labels)
        
        self.sample_predictions[sample_indices] = torch.sum(
            hard_predictions * labels, dim=1
        ).view(-1, 1)
        
        return mse_loss

    def _compute_kl_divergence_loss(self, model_outputs, labels, sample_indices):
        # Compute KL divergence loss
        class_predictions = torch.sum(model_outputs * labels, dim=1)
        
        kl_loss = F.kl_div(
            F.log_softmax(class_predictions), 
            F.softmax(-self.uncertainty_params[sample_indices].detach().view(-1))
        )
        
        return kl_loss

    def _compute_balance_loss(self, model_outputs, eps):

        avg_predictions = torch.mean(F.softmax(model_outputs, dim=1), dim=0)
        uniform_prior = torch.ones_like(avg_predictions) / self.num_classes
        
        clamped_predictions = torch.clamp(avg_predictions, min=eps, max=1.0)
        balance_kl = torch.mean(-(uniform_prior * torch.log(clamped_predictions)).sum(dim=0))
        
        return balance_kl

    def _compute_consistency_loss(self, primary_output, augmented_output):
        #Compute consistency loss
        primary_probs = F.softmax(primary_output, dim=1).detach()
        augmented_log_probs = F.log_softmax(augmented_output, dim=1)
        
        consistency_kl = F.kl_div(augmented_log_probs, primary_probs, reduction='none')
        return torch.sum(consistency_kl, dim=1)

    def _convert_soft_to_hard_labels(self, predictions):
        with torch.no_grad():
            batch_size = len(predictions)
            hard_labels = torch.zeros(batch_size, self.num_classes).to(self.device)
            max_indices = predictions.argmax(dim=1).view(-1, 1)
            hard_labels.scatter_(1, max_indices, 1)
            return hard_labels


class NCODTrainer:

    def __init__(self, model, data, noisy_indices, device, num_classes=None, 
                 learning_rate=0.01, weight_decay=5e-4, total_epochs=200, 
                 dirichlet_lambda=0.1, patience=20, debug=True):
        
        self.model = model.to(device)
        self.data = data.to(device)
        self.noisy_indices = noisy_indices
        self.device = device
        self.debug = debug
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.total_epochs = total_epochs
        self.dirichlet_lambda = dirichlet_lambda
        self.patience = patience
        
        if num_classes is None:
            self.num_classes = int(data.y.max().item()) + 1
        else:
            self.num_classes = num_classes
            
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Initialize NCOD loss function
        sample_labels = data.y.cpu().numpy()
        self.ncod_loss_fn = NCODLossModule(
            sample_labels=sample_labels,
            device=device,
            num_samples=data.num_nodes,
            num_classes=self.num_classes,
            consistency_ratio=1.5,
            balance_ratio=1.2,
            feature_dim=self.num_classes,
            total_epochs=total_epochs
        ).to(device)
        
        try:
            self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
            self.compute_oversmoothing = _compute_oversmoothing_for_mask
        except ImportError:
            print("Warning: Oversmoothing metrics not available")
            self.oversmoothing_evaluator = None
            self.compute_oversmoothing = None
        
        # Initialize Dirichlet energy
        try:
            self.compute_dirichlet_energy = dirichlet_energy
        except ImportError:
            print("Warning: Dirichlet energy computation not available")
            self.compute_dirichlet_energy = lambda x, edge_index: torch.tensor(0.0)
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0

    def train_single_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        
        model_outputs = self.model(self.data)
        
        train_indices = self.data.train_mask.nonzero(as_tuple=True)[0]
        train_labels_onehot = F.one_hot(
            self.data.y[train_indices], 
            num_classes=self.num_classes
        ).float()
        
        # Compute NCOD loss
        ncod_loss = self.ncod_loss_fn(
            sample_indices=train_indices,
            model_outputs=model_outputs[train_indices],
            ground_truth_labels=train_labels_onehot,
            feature_representations=model_outputs[train_indices],
            training_flag=0,
            current_epoch=epoch,
            training_accuracy=None
        )
        
        # Add Dirichlet energy regularization
        dirichlet_loss = self.compute_dirichlet_energy(model_outputs, self.data.edge_index)
        total_loss = ncod_loss + self.dirichlet_lambda * dirichlet_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), ncod_loss.item(), dirichlet_loss.item()

    def evaluate(self, epoch):
        self.model.eval()
        
        with torch.no_grad():
            model_outputs = self.model(self.data)
            
            val_indices = self.data.val_mask.nonzero(as_tuple=True)[0]
            val_labels_onehot = F.one_hot(
                self.data.y[val_indices], 
                num_classes=self.num_classes
            ).float()
            
            val_ncod_loss = self.ncod_loss_fn(
                sample_indices=val_indices,
                model_outputs=model_outputs[val_indices],
                ground_truth_labels=val_labels_onehot,
                feature_representations=model_outputs[val_indices],
                training_flag=0,
                current_epoch=epoch,
                training_accuracy=None
            )
            
            val_dirichlet_loss = self.compute_dirichlet_energy(model_outputs, self.data.edge_index)
            val_total_loss = val_ncod_loss + self.dirichlet_lambda * val_dirichlet_loss
            
            predictions = model_outputs.argmax(dim=1)
            train_indices = self.data.train_mask.nonzero(as_tuple=True)[0]
            
            train_acc = (predictions[train_indices] == self.data.y[train_indices]).sum().item() / len(train_indices)
            val_acc = (predictions[val_indices] == self.data.y[val_indices]).sum().item() / len(val_indices)
            
            train_f1 = f1_score(self.data.y[train_indices].cpu(), predictions[train_indices].cpu(), average='macro')
            val_f1 = f1_score(self.data.y[val_indices].cpu(), predictions[val_indices].cpu(), average='macro')
            
            train_oversmoothing = {}
            val_oversmoothing = {}
            
            if self.compute_oversmoothing is not None:
                train_oversmoothing = self.compute_oversmoothing(
                    self.oversmoothing_evaluator, model_outputs, self.data.edge_index, 
                    self.data.train_mask, self.data.y
                )
                val_oversmoothing = self.compute_oversmoothing(
                    self.oversmoothing_evaluator, model_outputs, self.data.edge_index, 
                    self.data.val_mask, self.data.y
                )
            
            return {
                'val_loss': val_total_loss.item(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'train_oversmoothing': train_oversmoothing,
                'val_oversmoothing': val_oversmoothing
            }

    def train_full(self):
        if self.debug:
            print(f"Starting NCOD training for {self.total_epochs} epochs...")
            print(f"Training samples: {self.data.train_mask.sum().item()}")
            print(f"Noisy samples: {len(self.noisy_indices)}")
        
        for epoch in range(1, self.total_epochs + 1):

            train_loss, ncod_loss, dirichlet_loss = self.train_single_epoch(epoch)

            eval_metrics = self.evaluate(epoch)
            
            # Early stopping
            if eval_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = eval_metrics['val_loss']
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.patience:
                if self.debug:
                    print(f"Early stopping at epoch {epoch}")
                self.model.load_state_dict(self.best_model_state)
                break
            
            if self.debug and epoch % 20 == 0:
                self._print_debug_info(epoch, eval_metrics)
        
        return self._final_evaluation()

    def _print_debug_info(self, epoch, metrics):
        print(f"Epoch {epoch:03d} | Train Acc: {metrics['train_acc']:.4f}, Val Acc: {metrics['val_acc']:.4f} | "
              f"Train F1: {metrics['train_f1']:.4f}, Val F1: {metrics['val_f1']:.4f}")
        
        if metrics['train_oversmoothing'] and metrics['val_oversmoothing']:
            train_os = metrics['train_oversmoothing']
            val_os = metrics['val_oversmoothing']
            
            print(f"Train DE: {train_os.get('EDir', 0):.4f}, Val DE: {val_os.get('EDir', 0):.4f} | "
                  f"Train DE_trad: {train_os.get('EDir_traditional', 0):.4f}, Val DE_trad: {val_os.get('EDir_traditional', 0):.4f} | "
                  f"Train EProj: {train_os.get('EProj', 0):.4f}, Val EProj: {val_os.get('EProj', 0):.4f} | "
                  f"Train MAD: {train_os.get('MAD', 0):.4f}, Val MAD: {val_os.get('MAD', 0):.4f} | "
                  f"Train NumRank: {train_os.get('NumRank', 0):.4f}, Val NumRank: {val_os.get('NumRank', 0):.4f} | "
                  f"Train Erank: {train_os.get('Erank', 0):.4f}, Val Erank: {val_os.get('Erank', 0):.4f}")

    def _final_evaluation(self):
        self.model.eval()
        
        with torch.no_grad():
            model_outputs = self.model(self.data)
            
            test_indices = self.data.test_mask.nonzero(as_tuple=True)[0]
            test_labels_onehot = F.one_hot(
                self.data.y[test_indices], 
                num_classes=self.num_classes
            ).float()
            
            test_ncod_loss = self.ncod_loss_fn(
                sample_indices=test_indices,
                model_outputs=model_outputs[test_indices],
                ground_truth_labels=test_labels_onehot,
                feature_representations=model_outputs[test_indices],
                training_flag=0,
                current_epoch=self.total_epochs,
                training_accuracy=None
            )
            
            test_dirichlet_loss = self.compute_dirichlet_energy(model_outputs, self.data.edge_index)
            test_total_loss = test_ncod_loss + self.dirichlet_lambda * test_dirichlet_loss
            
            predictions = model_outputs.argmax(dim=1)
            train_indices = self.data.train_mask.nonzero(as_tuple=True)[0]
            val_indices = self.data.val_mask.nonzero(as_tuple=True)[0]
            
            test_acc = (predictions[test_indices] == self.data.y[test_indices]).sum().item() / len(test_indices)
            test_f1 = f1_score(self.data.y[test_indices].cpu(), predictions[test_indices].cpu(), average='macro')
            test_precision = precision_score(self.data.y[test_indices].cpu(), predictions[test_indices].cpu(), average='macro')
            test_recall = recall_score(self.data.y[test_indices].cpu(), predictions[test_indices].cpu(), average='macro')
            
            final_train_oversmoothing = {}
            final_val_oversmoothing = {}
            test_oversmoothing = {}
            
            if self.compute_oversmoothing is not None:
                final_train_oversmoothing = self.compute_oversmoothing(
                    self.oversmoothing_evaluator, model_outputs, self.data.edge_index, 
                    self.data.train_mask, self.data.y
                )
                final_val_oversmoothing = self.compute_oversmoothing(
                    self.oversmoothing_evaluator, model_outputs, self.data.edge_index, 
                    self.data.val_mask, self.data.y
                )
                test_oversmoothing = self.compute_oversmoothing(
                    self.oversmoothing_evaluator, model_outputs, self.data.edge_index, 
                    self.data.test_mask, self.data.y
                )
            
            if self.debug:
                print(f"Test Loss: {test_total_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
                print("Final Oversmoothing Metrics:")
                
                if final_train_oversmoothing:
                    print(f"Train: EDir: {final_train_oversmoothing.get('EDir', 0):.4f}, "
                          f"EDir_traditional: {final_train_oversmoothing.get('EDir_traditional', 0):.4f}, "
                          f"EProj: {final_train_oversmoothing.get('EProj', 0):.4f}, "
                          f"MAD: {final_train_oversmoothing.get('MAD', 0):.4f}, "
                          f"NumRank: {final_train_oversmoothing.get('NumRank', 0):.4f}, "
                          f"Erank: {final_train_oversmoothing.get('Erank', 0):.4f}")
                
                if final_val_oversmoothing:
                    print(f"Val: EDir: {final_val_oversmoothing.get('EDir', 0):.4f}, "
                          f"EDir_traditional: {final_val_oversmoothing.get('EDir_traditional', 0):.4f}, "
                          f"EProj: {final_val_oversmoothing.get('EProj', 0):.4f}, "
                          f"MAD: {final_val_oversmoothing.get('MAD', 0):.4f}, "
                          f"NumRank: {final_val_oversmoothing.get('NumRank', 0):.4f}, "
                          f"Erank: {final_val_oversmoothing.get('Erank', 0):.4f}")
                
                if test_oversmoothing:
                    print(f"Test: EDir: {test_oversmoothing.get('EDir', 0):.4f}, "
                          f"EDir_traditional: {test_oversmoothing.get('EDir_traditional', 0):.4f}, "
                          f"EProj: {test_oversmoothing.get('EProj', 0):.4f}, "
                          f"MAD: {test_oversmoothing.get('MAD', 0):.4f}, "
                          f"NumRank: {test_oversmoothing.get('NumRank', 0):.4f}, "
                          f"Erank: {test_oversmoothing.get('Erank', 0):.4f}")
            
            return {
                'accuracy': torch.tensor(test_acc),
                'f1': torch.tensor(test_f1),
                'precision': torch.tensor(test_precision),
                'recall': torch.tensor(test_recall),
                'test_loss': test_total_loss.item(),
                'oversmoothing': test_oversmoothing if test_oversmoothing else {
                    'EDir': 0, 'EDir_traditional': 0, 'EProj': 0, 
                    'MAD': 0, 'NumRank': 0, 'Erank': 0
                }
            }
