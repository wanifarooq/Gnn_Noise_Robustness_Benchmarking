import os
import os
import copy
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


class GraphCleanerDetector:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.k = config.get('k', 3)
        self.sample_rate = config.get('sample_rate', 0.5)
        self.max_iter_classifier = config.get('max_iter_classifier', 3000)
        
    def ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)
        
    def to_softmax(self, log_probs):
        if isinstance(log_probs, np.ndarray):
            return np.exp(log_probs)
        elif isinstance(log_probs, torch.Tensor):
            return F.softmax(log_probs, dim=1).cpu().numpy()
        else:
            return F.softmax(torch.tensor(log_probs), dim=1).cpu().numpy()
    
    def train_base_gnn(self, data, model, n_epochs=200):
        print("Training base GNN for GraphCleaner...")
        
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=self.config.get('lr', 0.001),
                                   weight_decay=self.config.get('weight_decay', 5e-4))
        
        all_sm_vectors = []
        best_sm_vectors = []
        best_f1 = 0
        best_model_state = None
        
        data = data.to(self.device)
        model = model.to(self.device)
        
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            
            out = model(data)
            loss = F.cross_entropy(out[data.train_mask], data.y_noisy[data.train_mask])
            
            if epoch % 50 == 0:
                print(f"Epoch[{epoch + 1}] Loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                eval_out = model(data)
                y_pred = eval_out[data.val_mask].argmax(dim=-1).cpu()
                y_true = data.y[data.val_mask].cpu()
                f1 = f1_score(y_true, y_pred, average='micro')
                
                if f1 > best_f1:
                    print(f"New Best Validation F1: {f1:.4f}")
                    best_f1 = f1
                    best_sm_vectors = eval_out.cpu().detach().numpy()
                    best_model_state = copy.deepcopy(model.state_dict())
            
            if (epoch + 1) % 20 == 0:
                all_sm_vectors.append(eval_out.cpu().detach().numpy())
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        return np.array(all_sm_vectors), best_sm_vectors, model
    
    def confident_joint_estimation(self, noisy_labels, pred_probs, num_classes):

        print(f"Using {len(noisy_labels)} samples for confident joint estimation.")
        
        confident_joint = compute_confident_joint(noisy_labels, pred_probs)
        py, noise_matrix, inv_noise_matrix = estimate_latent(confident_joint, noisy_labels)
        
        print(f"Estimated prior probabilities py: {py}")
        print("Estimated noise transition matrix:")
        print(noise_matrix)
        
        return py, noise_matrix, inv_noise_matrix

    
    def negative_sampling(self, data, noise_matrix, num_classes):
        data_copy = copy.deepcopy(data)
        
        val_idx = data.val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        val_y = data.y_noisy[data.val_mask].cpu().numpy()

        valid_idx = set(range(len(val_y)))
        for c in range(num_classes):
            if c >= noise_matrix.shape[1] or np.isnan(noise_matrix[0, c]) or np.max(noise_matrix[:, c]) == 1:
                print(f"Class {c} is invalid!")
                valid_idx &= set(np.where(val_y != c)[0])
        
        valid_idx = np.array(list(valid_idx))
        val_idx = val_idx[valid_idx]

        sample_count = int(np.round(self.sample_rate * len(val_idx)))
        sample_idx = np.random.choice(val_idx, sample_count, replace=False)

        for idx in sample_idx:
            current_label = int(data_copy.y_noisy[idx])
            if current_label < noise_matrix.shape[1]:
                new_label = current_label
                while new_label == current_label:
                    new_label = np.random.choice(range(noise_matrix.shape[1]), 
                                               p=noise_matrix[:, current_label])
                data_copy.y_noisy[idx] = new_label
        
        return data_copy, sample_idx
    
    def generate_features(self, data, noisy_data, sample_idx, all_sm_vectors, num_classes):
        print("Generating features for noise detection...")
        
        y = np.zeros(data.num_nodes)
        y[sample_idx] = 1

        A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        n = A.shape[0]

        D_inv_sqrt = spdiags(np.power(np.array(A.sum(1)) + 1e-10, -0.5).flatten(), 0, n, n)
        S = D_inv_sqrt @ A @ D_inv_sqrt
        S2 = S @ S
        S3 = S @ S2
        S2.setdiag(np.zeros(n))
        S3.setdiag(np.zeros(n))
        print("Adjacency matrices calculated!")
        
        ymat = y[:, None]
        L0 = np.eye(num_classes)[data.y.cpu().numpy()]
        L_corr = ymat * np.eye(num_classes)[noisy_data.y_noisy.cpu().numpy()] + (1 - ymat) * L0
        L1, L2, L3 = S @ L0, S2 @ L0, S3 @ L0
        print("Label matrices calculated!")
        
        P0 = self.to_softmax(torch.tensor(all_sm_vectors[-1]))
        P1, P2, P3 = S @ P0, S2 @ P0, S3 @ P0
        print("Prediction matrices calculated!")

        if self.k == 3:
            features = np.hstack([
                np.sum(L_corr * P, axis=1, keepdims=True) for P in [P0, P1, P2, P3]
            ] + [
                np.sum(L_corr * L, axis=1, keepdims=True) for L in [L1, L2, L3]
            ])
        else:
            raise NotImplementedError(f"Feature generation only implemented for k=3, got k={self.k}")
        
        return features, y
    
    def detect_noise(self, data, model, num_classes):
        print("Starting GraphCleaner Detection")

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
            data, noisy_data, sample_idx, all_sm_vectors, num_classes
        )
        
        print("Training binary noise detector...")

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