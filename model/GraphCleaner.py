import os
import os
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, 
    matthews_corrcoef, roc_auc_score
)
from torch_geometric.utils import to_scipy_sparse_matrix
from cleanlab.count import estimate_latent, compute_confident_joint

from model.evaluation import OversmoothingMetrics

class GraphCleanerDetector:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.k = config.get('k', 3)
        self.sample_rate = config.get('sample_rate', 0.5)
        self.max_iter_classifier = config.get('max_iter_classifier', 3000)

        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        
    def ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)

    def _compute_oversmoothing_for_mask(self, embeddings, edge_index, mask, labels=None):
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
            
            return self.oversmoothing_evaluator.compute_all_metrics(
                X=mask_embeddings,
                edge_index=remapped_edges,
                graphs_in_class=graphs_in_class
            )
            
        except Exception as e:
            print(f"Warning: Could not compute oversmoothing metrics for mask: {e}")
            return None
        
    def to_softmax(self, log_probs):
        if isinstance(log_probs, np.ndarray):
            e = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        elif isinstance(log_probs, torch.Tensor):
            return F.softmax(log_probs, dim=1).cpu().numpy()
        else:
            return F.softmax(torch.tensor(log_probs), dim=1).cpu().numpy()

    def compute_dirichlet_energy(self, data, predictions, mask):

        if predictions.dim() > 1 and predictions.size(1) > 1:
            probs = F.softmax(predictions, dim=1)
        else:
            probs = predictions
            
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        energy = 0.0
        edge_count = 0

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            
            if mask[src] or mask[dst]:
                diff = torch.norm(probs[src] - probs[dst], p=2) ** 2
                energy += diff.item()
                edge_count += 1
        
        return energy / edge_count if edge_count > 0 else 0.0

    def compute_metrics(self, data, model, out):

        metrics = {}

        train_pred = out[data.train_mask].argmax(dim=-1).cpu()
        train_true = data.y[data.train_mask].cpu()
        metrics['train_loss'] = F.cross_entropy(out[data.train_mask], data.y_noisy[data.train_mask]).item()
        metrics['train_acc'] = accuracy_score(train_true, train_pred)
        metrics['train_f1'] = f1_score(train_true, train_pred, average='micro')

        val_pred = out[data.val_mask].argmax(dim=-1).cpu()
        val_true = data.y[data.val_mask].cpu()
        metrics['val_loss'] = F.cross_entropy(out[data.val_mask], data.y_noisy[data.val_mask]).item()
        metrics['val_acc'] = accuracy_score(val_true, val_pred)
        metrics['val_f1'] = f1_score(val_true, val_pred, average='micro')
        
        train_oversmoothing = self._compute_oversmoothing_for_mask(out, data.edge_index, data.train_mask, data.y)
        val_oversmoothing = self._compute_oversmoothing_for_mask(out, data.edge_index, data.val_mask, data.y)
        
        metrics['train_oversmoothing'] = train_oversmoothing
        metrics['val_oversmoothing'] = val_oversmoothing
        
        return metrics

    def train_base_gnn(self, data, model, n_epochs=200):
        print("Training base GNN for GraphCleaner")
        start_time = time.time()
        
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=self.config.get('lr', 0.001),
                                   weight_decay=self.config.get('weight_decay', 5e-4))
        
        all_sm_vectors = []
        best_sm_vectors = []
        best_model_state = None
        
        data = data.to(self.device)
        model = model.to(self.device)
        best_val_loss = float('inf')
        best_model_state = None
        best_f1 = 0
        patience = self.config.get('patience', 10)
        patience_counter = 0

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()

            out = model(data)
            loss = F.cross_entropy(out[data.train_mask], data.y_noisy[data.train_mask])

            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(data)
                metrics = self.compute_metrics(data, model, out)
                
                train_oversmoothing = metrics.get('train_oversmoothing', {})
                val_oversmoothing = metrics.get('val_oversmoothing', {})
                
                train_de = train_oversmoothing.get('EDir', 0.0)
                train_de_traditional = train_oversmoothing.get('EDir_traditional', 0.0)
                train_eproj = train_oversmoothing.get('EProj', 0.0)
                train_mad = train_oversmoothing.get('MAD', 0.0)
                train_num_rank = train_oversmoothing.get('NumRank', 0.0)
                train_eff_rank = train_oversmoothing.get('Erank', 0.0)
                
                val_de = val_oversmoothing.get('EDir', 0.0)
                val_de_traditional = val_oversmoothing.get('EDir_traditional', 0.0)
                val_eproj = val_oversmoothing.get('EProj', 0.0)
                val_mad = val_oversmoothing.get('MAD', 0.0)
                val_num_rank = val_oversmoothing.get('NumRank', 0.0)
                val_eff_rank = val_oversmoothing.get('Erank', 0.0)
                
                print(f"Epoch {epoch:03d} | Train Loss: {metrics['train_loss']:.4f}, Val Loss: {metrics['val_loss']:.4f} | "
                    f"Train Acc: {metrics['train_acc']:.4f}, Val Acc: {metrics['val_acc']:.4f} | "
                    f"Train F1: {metrics['train_f1']:.4f}, Val F1: {metrics['val_f1']:.4f}")
                print(f"Train DE: {train_de:.4f}, Val DE: {val_de:.4f} | "
                    f"Train DE_trad: {train_de_traditional:.4f}, Val DE_trad: {val_de_traditional:.4f} | "
                    f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                    f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                    f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                    f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")

                if metrics['val_loss'] < best_val_loss:
                    best_val_loss = metrics['val_loss']
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_sm_vectors = out.cpu().detach().numpy()
                    best_f1 = metrics['val_f1']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            if (epoch + 1) % 20 == 0:
                all_sm_vectors.append(out.cpu().detach().numpy())

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        with torch.no_grad():
            out = model(data)
            final_metrics = {}
            test_pred = out[data.test_mask].argmax(dim=-1).cpu()
            test_true = data.y[data.test_mask].cpu()
            final_metrics['test_loss'] = F.cross_entropy(out[data.test_mask], data.y_noisy[data.test_mask]).item()
            final_metrics['test_acc'] = accuracy_score(test_true, test_pred)
            final_metrics['test_f1'] = f1_score(test_true, test_pred, average='micro')
            
            final_train_oversmoothing = self._compute_oversmoothing_for_mask(out, data.edge_index, data.train_mask, data.y)
            final_val_oversmoothing = self._compute_oversmoothing_for_mask(out, data.edge_index, data.val_mask, data.y)
            final_test_oversmoothing = self._compute_oversmoothing_for_mask(out, data.edge_index, data.test_mask, data.y)

        total_time = time.time() - start_time

        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Test Loss: {final_metrics['test_loss']:.4f} | Test Acc: {final_metrics['test_acc']:.4f} | Test F1: {final_metrics['test_f1']:.4f}")
        print("Final Oversmoothing Metrics:")

        if final_train_oversmoothing is not None:
            print(f"Train: EDir: {final_train_oversmoothing['EDir']:.4f}, EDir_traditional: {final_train_oversmoothing['EDir_traditional']:.4f}, "
                f"EProj: {final_train_oversmoothing['EProj']:.4f}, MAD: {final_train_oversmoothing['MAD']:.4f}, "
                f"NumRank: {final_train_oversmoothing['NumRank']:.4f}, Erank: {final_train_oversmoothing['Erank']:.4f}")

        if final_val_oversmoothing is not None:
            print(f"Val: EDir: {final_val_oversmoothing['EDir']:.4f}, EDir_traditional: {final_val_oversmoothing['EDir_traditional']:.4f}, "
                f"EProj: {final_val_oversmoothing['EProj']:.4f}, MAD: {final_val_oversmoothing['MAD']:.4f}, "
                f"NumRank: {final_val_oversmoothing['NumRank']:.4f}, Erank: {final_val_oversmoothing['Erank']:.4f}")

        if final_test_oversmoothing is not None:
            print(f"Test: EDir: {final_test_oversmoothing['EDir']:.4f}, EDir_traditional: {final_test_oversmoothing['EDir_traditional']:.4f}, "
                f"EProj: {final_test_oversmoothing['EProj']:.4f}, MAD: {final_test_oversmoothing['MAD']:.4f}, "
                f"NumRank: {final_test_oversmoothing['NumRank']:.4f}, Erank: {final_test_oversmoothing['Erank']:.4f}")

        return np.array(all_sm_vectors), best_sm_vectors, model
    
    def confident_joint_estimation(self, noisy_labels, pred_probs, num_classes):

        print(f"Using {len(noisy_labels)} samples for confident joint estimation.")
        
        confident_joint = compute_confident_joint(noisy_labels, pred_probs)
        py, noise_matrix, inv_noise_matrix = estimate_latent(confident_joint, noisy_labels)
        
        print(f"Estimated prior probabilities py: {py}")
        print("Estimated noise transition matrix:")
        print(noise_matrix)
        
        return py, noise_matrix, inv_noise_matrix
    
    def negative_sampling(self, data_orig, noise_matrix, n_classes, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        import random

        data = copy.deepcopy(data_orig)
        train_idx = np.argwhere(data.held_mask.cpu().numpy() == True).flatten()
        train_y = data.y[data.held_mask].cpu().numpy()

        valid_subidx = set(range(len(train_y)))
        for c in range(n_classes):
            if c >= noise_matrix.shape[1] or np.isnan(noise_matrix[0, c]) or np.max(noise_matrix[:, c]) == 1:
                print(f"Class {c} is invalid!")
                valid = set(np.where(train_y != c)[0])
                valid_subidx = valid_subidx & valid
        train_idx = train_idx[list(valid_subidx)]

        tr_sample_idx = random.sample(list(train_idx), int(np.round(sample_rate * len(train_idx))))
        for idx in tr_sample_idx:
            y = int(data.y[idx])
            while y == int(data.y[idx]):
                y = np.random.choice(range(n_classes), p=noise_matrix[:, y])
            data.y[idx] = y

        return data, tr_sample_idx

    
    def generate_features(self, k, data, noisy_data, sample_idx, all_sm_vectors, n_classes):
        print("Generating new features")

        y = np.zeros(data.num_nodes)
        y[sample_idx] = 1

        A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        n = A.shape[0]

        S = spdiags(np.squeeze((1e-10 + np.array(A.sum(1)))**-0.5), 0, n, n) @ A @ spdiags(np.squeeze((1e-10 + np.array(A.sum(0)))**-0.5), 0, n, n)
        S2 = S @ S
        S3 = S @ S2
        S2.setdiag(np.zeros(n))
        S3.setdiag(np.zeros(n))
        print("S matrices calculated!")

        ymat = y[:, np.newaxis]
        L0 = np.eye(n_classes)[data.y.cpu().numpy()]
        L_corr = ymat * np.eye(n_classes)[noisy_data.y.cpu().numpy()] + (1 - ymat) * L0
        L1, L2, L3 = S @ L0, S2 @ L0, S3 @ L0

        P0 = self.to_softmax(torch.tensor(all_sm_vectors[-1]))
        P1, P2, P3 = S @ P0, S2 @ P0, S3 @ P0

        features_list = [np.sum(L_corr * P0, axis=1, keepdims=True),
                        np.sum(L_corr * P1, axis=1, keepdims=True),
                        np.sum(L_corr * P2, axis=1, keepdims=True),
                        np.sum(L_corr * L1, axis=1, keepdims=True),
                        np.sum(L_corr * L2, axis=1, keepdims=True),
                        np.sum(L_corr * L3, axis=1, keepdims=True)]

        if k >= 4:
            S4 = S @ S3
            S4.setdiag(np.zeros(n))
            L4 = S4 @ L0
            P4 = S4 @ P0
            features_list += [np.sum(L_corr * P4, axis=1, keepdims=True),
                              np.sum(L_corr * L4, axis=1, keepdims=True)]

        if k >= 5:
            S5 = S @ S4
            S5.setdiag(np.zeros(n))
            L5 = S5 @ L0
            P5 = S5 @ P0
            features_list += [np.sum(L_corr * P5, axis=1, keepdims=True),
                              np.sum(L_corr * L5, axis=1, keepdims=True)]

        feat = np.hstack(features_list)
        return feat, y

    
    def detect_noise(self, data, model, num_classes):
        print("Starting GraphCleaner Detection")

        held_split = self.config.get('held_split', 'valid')
        if held_split == 'train':
            data.held_mask = data.train_mask
        elif held_split == 'valid':
            data.held_mask = data.val_mask
        else:
            data.held_mask = data.test_mask

        all_sm_vectors, best_sm_vectors, trained_model = self.train_base_gnn(
            data, model, self.config.get('total_epochs', 200)
        )
        
        val_noisy_labels = data.y_noisy[data.val_mask].cpu().numpy()
        val_pred_probs = self.to_softmax(torch.tensor(best_sm_vectors))[data.val_mask.cpu().numpy()]
        
        py, noise_matrix, inv_noise_matrix = self.confident_joint_estimation(
            val_noisy_labels, val_pred_probs, num_classes
        )
        
        self.visualize_noise_matrix(noise_matrix, num_classes, "Learned")
        
        noisy_data, sample_idx = self.negative_sampling(data, noise_matrix, num_classes)
        print(f"{len(sample_idx)} negative samples generated")

        features, binary_labels = self.generate_features(
            self.k, data, noisy_data, sample_idx, all_sm_vectors, num_classes
        )

        print("Training binary noise detector")

        val_mask_cpu = data.val_mask.cpu().numpy()
        test_mask_cpu = data.test_mask.cpu().numpy()
        
        X_train = features[val_mask_cpu].reshape(features[val_mask_cpu].shape[0], -1)
        y_train = binary_labels[val_mask_cpu]

        X_test = features[test_mask_cpu].reshape(features[test_mask_cpu].shape[0], -1)
        
        classifier = LogisticRegression(max_iter=self.max_iter_classifier, random_state=42)
        classifier.fit(X_train, y_train)

        test_probs = classifier.predict_proba(X_test)[:, 1]
        test_predictions = test_probs > 0.5
        
        return test_predictions, test_probs, classifier

    def visualize_noise_matrix(self, noise_matrix, num_classes, title_prefix=""):

        plt.figure(figsize=(8, 6))
        plt.imshow(noise_matrix.T, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f'{title_prefix} Noise Transition Matrix')
        
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, f'{noise_matrix.T[i,j]:.2f}', 
                        ha='center', va='center', color='black')
        
        plt.xlabel('True Label')
        plt.ylabel('Noisy Label')
        plt.tight_layout()
        plt.show()
    
    def evaluate_detection(self, predictions, ground_truth, probs):

        acc = accuracy_score(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions)
        recall = recall_score(ground_truth, predictions)
        mcc = matthews_corrcoef(ground_truth, predictions)
        auc = roc_auc_score(ground_truth, probs)
        
        print(f'Detection Accuracy: {acc:.4f}')
        print(f'Detection F1 Score: {f1:.4f}')
        print(f'Detection Precision: {precision:.4f}')
        print(f'Detection Recall: {recall:.4f}')
        print(f'Detection MCC: {mcc:.4f}')
        print(f'Detection AUC: {auc:.4f}')
        print(f"Samples detected as noisy: {np.sum(predictions)}")
        print(f"Actual noisy samples: {np.sum(ground_truth)}")
        
        return {
            'accuracy': acc, 'f1': f1, 'precision': precision, 
            'recall': recall, 'mcc': mcc, 'auc': auc
        }

def get_noisy_ground_truth(data, noisy_indices):

    ground_truth = torch.zeros(data.num_nodes, dtype=torch.bool)
    if isinstance(noisy_indices, torch.Tensor):
        ground_truth[noisy_indices] = True
    else:
        ground_truth[torch.tensor(noisy_indices)] = True
    return ground_truth