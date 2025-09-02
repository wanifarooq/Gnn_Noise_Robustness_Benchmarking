import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import f1_score

from model.evaluation import OversmoothingMetrics

class Net(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, l1, l2):
        losses = torch.stack([l1, l2], dim=1)
        return self.net(losses).squeeze(-1)

class GNNCleanerTrainer:

    def __init__(self, config, data, device, num_classes, gnn_model=None):
        self.config = config
        self.data = data
        self.device = device
        self.num_classes = num_classes
        
        self.epochs = config.get('epochs', 200)
        self.lr = config.get('lr', 0.01)
        self.net_lr = config.get('net_lr', 0.001)
        self.weight_decay = config.get('weight_decay', 5e-4)

        self.lp_iters = config.get('lp_iters', 50)
        self.epsilon = config.get('epsilon', 1e-8)

        if gnn_model is not None:
            self.model = gnn_model.to(device)
        else:
            raise ValueError("GNN Cleaner requires a GNN model")
        
        self.net = Net(input_dim=2, hidden_dim=32).to(device)
        
        self.model_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr, weight_decay=self.weight_decay
        )
        
        self.net_optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.net_lr
        )
        
        self.clean_mask = self.data.val_mask.clone()
        self.expanding_clean_mask = self.clean_mask.clone()
        
        self.results = {'train': -1, 'val': -1, 'test': -1}
        self._print_stats()
        self.patience = config.get('patience', 10)

        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        self.oversmoothing_history = {
            'train': [],
            'val': [],
            'test': []
        }

    def _print_stats(self):
        print("GNN Cleaner Dataset Statistics")
        print(f"Nodes: {self.data.x.shape[0]}")
        print(f"Features: {self.data.x.shape[1]}")
        print(f"Classes: {self.num_classes}")
        print(f"Edges: {self.data.edge_index.shape[1] // 2}")
        print(f"Train/Val/Test: {self.data.train_mask.sum()}/{self.data.val_mask.sum()}/{self.data.test_mask.sum()}")
        print(f"Clean set size: {self.clean_mask.sum()}")
        
        if hasattr(self.data, 'y_noisy'):
            train_noise = (self.data.y[self.data.train_mask] != self.data.y_noisy[self.data.train_mask]).sum().item()
            if train_noise > 0:
                actual_rate = train_noise / self.data.train_mask.sum().item()
                print(f"Label noise rate: {actual_rate:.3f} ({train_noise} labels changed)")

    def compute_similarity_matrix(self, edge_index, node_features):
        num_nodes = node_features.size(0)
        
        row, col = edge_index.cpu().numpy()
        adjacency = sparse.coo_matrix(
            (np.ones(len(row)), (row, col)), 
            shape=(num_nodes, num_nodes)
        )
        
        W_data = []
        W_row = []
        W_col = []
        
        for edge_idx in range(len(row)):
            i, j = row[edge_idx], col[edge_idx]
            if i != j:
                feat_i = node_features[i].cpu().numpy()
                feat_j = node_features[j].cpu().numpy()
                distance = np.linalg.norm(feat_i - feat_j) + self.epsilon
                similarity = 1.0 / distance
                
                W_data.append(similarity)
                W_row.append(i)
                W_col.append(j)
        
        W = sparse.coo_matrix((W_data, (W_row, W_col)), shape=(num_nodes, num_nodes))
        
        degrees = np.array(W.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1.0
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
        S = D_inv_sqrt @ W @ D_inv_sqrt
        
        return S

    def label_propagation(self, similarity_matrix, initial_labels, clean_mask, num_iterations=50):
        num_nodes = initial_labels.size(0)
        
        Y = torch.zeros(num_nodes, self.num_classes, device=self.device)
        
        clean_indices = clean_mask.nonzero(as_tuple=True)[0]
        for idx in clean_indices:
            label = initial_labels[idx].item()
            Y[idx, label] = 1.0
        
        S = torch.from_numpy(similarity_matrix.toarray()).float().to(self.device)
        
        for k in range(num_iterations):
            Y = torch.matmul(S, Y)
            
            for idx in clean_indices:
                label = initial_labels[idx].item()
                Y[idx] = 0.0
                Y[idx, label] = 1.0
        
        return Y

    def select_clean_samples(self, pseudo_labels, given_labels, train_mask):
        pseudo_hard = pseudo_labels.argmax(dim=1)
        
        train_indices = train_mask.nonzero(as_tuple=True)[0]
        selected_mask = torch.zeros_like(train_mask)
        left_mask = torch.zeros_like(train_mask)
        
        for idx in train_indices:
            if pseudo_hard[idx] == given_labels[idx]:
                selected_mask[idx] = True
            else:
                left_mask[idx] = True
        
        return selected_mask, left_mask

    def train_step(self, epoch):
        self.model.train()
        self.net.train()
        
        x = self.data.x.to(self.device)
        edge_index = self.data.edge_index.to(self.device)
        given_labels = self.data.y.to(self.device)
        true_labels  = self.data.y_original.to(self.device)

        try:
            node_features = self.model(self.data)
        except:
            node_features = self.model(x, edge_index)

        S = self.compute_similarity_matrix(edge_index, node_features.detach())

        pseudo_labels = self.label_propagation(
            S, true_labels, self.expanding_clean_mask, self.lp_iters
        )

        selected_mask, left_mask = self.select_clean_samples(
            pseudo_labels, given_labels, self.data.train_mask
        )

        self.expanding_clean_mask = (self.expanding_clean_mask | selected_mask).detach()
        
        total_loss = 0.0
        net_loss_val = 0.0
        
        if selected_mask.sum() > 0:
            selected_logits = node_features[selected_mask]
            selected_labels = given_labels[selected_mask]
            
            loss_selected = F.cross_entropy(selected_logits, selected_labels)
            total_loss += loss_selected

        if left_mask.sum() > 0:
            left_indices = left_mask.nonzero(as_tuple=True)[0]
            
            left_logits = node_features[left_indices]
            left_given_labels = given_labels[left_indices]
            left_pseudo_labels = pseudo_labels[left_indices]
            
            l1 = F.cross_entropy(left_logits, left_given_labels, reduction='none')
            
            l2 = -(left_pseudo_labels * F.log_softmax(left_logits, dim=1)).sum(dim=1)
            
            lambda_weights = self.net(l1.detach(), l2.detach())
            
            left_given_onehot = F.one_hot(left_given_labels, self.num_classes).float()
            corrected_labels = (lambda_weights.unsqueeze(1) * left_given_onehot + 
                             (1 - lambda_weights.unsqueeze(1)) * left_pseudo_labels)
            
            train_loss = -(corrected_labels * F.log_softmax(left_logits, dim=1)).sum(dim=1).mean()
            total_loss += train_loss
        
        if total_loss != 0.0:
            self.model_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            
            model_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    model_grads.append(param.grad.clone())
                else:
                    model_grads.append(torch.zeros_like(param))
            
            self.model_optimizer.step()
            
            if left_mask.sum() > 0:
                clean_indices = self.clean_mask.nonzero(as_tuple=True)[0]
                if len(clean_indices) > 0:
                    try:
                        updated_logits = self.model(self.data)
                    except:
                        updated_logits = self.model(x, edge_index)
                    
                    clean_logits = updated_logits[clean_indices]
                    clean_labels = true_labels[clean_indices]
                    net_loss = F.cross_entropy(clean_logits, clean_labels)
                    net_loss_val = net_loss.item()
                    
                    self.net_optimizer.zero_grad()
                    net_loss.backward()
                    self.net_optimizer.step()
        
        return total_loss.item() if total_loss != 0.0 else 0.0, net_loss_val

    @torch.no_grad()
    def evaluate(self, include_test=False):
        self.model.eval()
        
        x, edge_index = self.data.x.to(self.device), self.data.edge_index.to(self.device)
        
        try:
            logits = self.model(self.data)
        except:
            logits = self.model(x, edge_index)
        
        preds = logits.argmax(dim=1)
        
        labels = self.data.y_original.to(self.device)
        
        train_loss = F.cross_entropy(logits[self.data.train_mask], labels[self.data.train_mask]).item()
        val_loss = F.cross_entropy(logits[self.data.val_mask], labels[self.data.val_mask]).item()
        
        train_acc = (preds[self.data.train_mask] == labels[self.data.train_mask]).float().mean().item()
        val_acc = (preds[self.data.val_mask] == labels[self.data.val_mask]).float().mean().item()
        
        train_f1 = f1_score(
            labels[self.data.train_mask].cpu().numpy(), 
            preds[self.data.train_mask].cpu().numpy(), 
            average='macro'
        )
        val_f1 = f1_score(
            labels[self.data.val_mask].cpu().numpy(), 
            preds[self.data.val_mask].cpu().numpy(), 
            average='macro'
        )
        
        result = {
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_acc': train_acc, 'val_acc': val_acc,
            'train_f1': train_f1, 'val_f1': val_f1
        }
        
        if include_test:
            test_loss = F.cross_entropy(logits[self.data.test_mask], labels[self.data.test_mask]).item()
            test_acc = (preds[self.data.test_mask] == labels[self.data.test_mask]).float().mean().item()
            test_f1 = f1_score(
                labels[self.data.test_mask].cpu().numpy(), 
                preds[self.data.test_mask].cpu().numpy(), 
                average='macro'
            )
            result.update({'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1})
        
        return result

    def train(self, debug=True):
        start_time = time.time()
        
        best_val_loss = float("inf")
        best_state = None
        counter = 0
        
        for epoch in range(self.epochs):
            train_loss, net_loss = self.train_step(epoch)
            
            metrics = self.evaluate(include_test=False)
            
            try:
                logits_for_oversmoothing = self.model(self.data)
            except:
                logits_for_oversmoothing = self.model(self.data.x.to(self.device), self.data.edge_index.to(self.device))

            train_oversmoothing = self._compute_oversmoothing_for_mask(
                logits_for_oversmoothing, self.data.edge_index.to(self.device), self.data.train_mask, self.data.y_original
            )
            val_oversmoothing = self._compute_oversmoothing_for_mask(
                logits_for_oversmoothing, self.data.edge_index.to(self.device), self.data.val_mask, self.data.y_original
            )

            if train_oversmoothing is not None:
                self.oversmoothing_history['train'].append(train_oversmoothing)
            if val_oversmoothing is not None:
                self.oversmoothing_history['val'].append(val_oversmoothing)
            
            if metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
                best_state = {
                    "model": self.model.state_dict(),
                    "net": self.net.state_dict()
                }
                best_train_metrics = {
                    'train_acc': metrics['train_acc'],
                    'train_f1': metrics['train_f1']
                }
                best_val_metrics = {
                    'val_acc': metrics['val_acc'], 
                    'val_f1': metrics['val_f1']
                }
                counter = 0
            else:
                counter += 1
            
            if counter >= self.patience:
                if debug:
                    print(f"Early stopping at epoch {epoch}")
                break

            if debug:
                selected_count = self.expanding_clean_mask.sum().item() - self.clean_mask.sum().item()
                
                train_de = train_oversmoothing['EDir'] if train_oversmoothing else 0.0
                train_de_traditional = train_oversmoothing['EDir_traditional'] if train_oversmoothing else 0.0
                train_eproj = train_oversmoothing['EProj'] if train_oversmoothing else 0.0
                train_mad = train_oversmoothing['MAD'] if train_oversmoothing else 0.0
                train_num_rank = train_oversmoothing['NumRank'] if train_oversmoothing else 0.0
                train_eff_rank = train_oversmoothing['Erank'] if train_oversmoothing else 0.0
                
                val_de = val_oversmoothing['EDir'] if val_oversmoothing else 0.0
                val_de_traditional = val_oversmoothing['EDir_traditional'] if val_oversmoothing else 0.0
                val_eproj = val_oversmoothing['EProj'] if val_oversmoothing else 0.0
                val_mad = val_oversmoothing['MAD'] if val_oversmoothing else 0.0
                val_num_rank = val_oversmoothing['NumRank'] if val_oversmoothing else 0.0
                val_eff_rank = val_oversmoothing['Erank'] if val_oversmoothing else 0.0
                
                print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f}, Val Loss: {metrics['val_loss']:.4f} | "
                    f"Train Acc: {metrics['train_acc']:.4f}, Val Acc: {metrics['val_acc']:.4f} | "
                    f"Train F1: {metrics['train_f1']:.4f}, Val F1: {metrics['val_f1']:.4f} | "
                    f"Selected: {selected_count}")
                print(f"Train DE: {train_de:.4f}, Val DE: {val_de:.4f} | "
                    f"Train DE_trad: {train_de_traditional:.4f}, Val DE_trad: {val_de_traditional:.4f} | "
                    f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                    f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                    f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                    f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")
        
        if best_state is not None:
            self.model.load_state_dict(best_state["model"])
            self.net.load_state_dict(best_state["net"])
        
        final_metrics = self.evaluate(include_test=True)
        
        try:
            final_logits = self.model(self.data)
        except:
            final_logits = self.model(self.data.x.to(self.device), self.data.edge_index.to(self.device))

        final_train_oversmoothing = self._compute_oversmoothing_for_mask(
            final_logits, self.data.edge_index.to(self.device), self.data.train_mask, self.data.y_original
        )
        final_val_oversmoothing = self._compute_oversmoothing_for_mask(
            final_logits, self.data.edge_index.to(self.device), self.data.val_mask, self.data.y_original
        )
        final_test_oversmoothing = self._compute_oversmoothing_for_mask(
            final_logits, self.data.edge_index.to(self.device), self.data.test_mask, self.data.y_original
        )

        if final_test_oversmoothing is not None:
            self.oversmoothing_history['test'].append(final_test_oversmoothing)
        
        self.results = {
            'train': best_train_metrics['train_acc'],
            'val': best_val_metrics['val_acc'], 
            'test': final_metrics['test_acc']
        }
        
        if debug:
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
        
        return self.results
    
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

    def get_oversmoothing_history(self):
        return self.oversmoothing_history