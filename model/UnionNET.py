import time
from copy import deepcopy
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

def normalize_features(features):
    row_sum = torch.sum(features, dim=1, keepdim=True)
    row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
    return features / row_sum

def construct_support_set(features, train_mask, labels, edge_index, k):
    device = features.device
    n_nodes = features.size(0)
    feat_dim = features.size(1)
    
    support_set = torch.zeros(n_nodes, k, feat_dim, device=device)
    support_labels = torch.zeros(n_nodes, k, dtype=torch.long, device=device)
    
    for i in range(n_nodes):
        if train_mask[i]:
            neighbors = edge_index[1][edge_index[0] == i]
            
            if len(neighbors) >= k:
                anchor = features[i].unsqueeze(0)
                similarities = torch.mm(features[neighbors], anchor.T).squeeze()
                _, topk_indices = torch.topk(similarities, k=k)
                
                support_set[i] = features[neighbors[topk_indices]]
                support_labels[i] = labels[neighbors[topk_indices]]
    
    return support_set, support_labels

def label_aggregation(support_features, support_labels, node_features, n_classes):
    device = node_features.device
    n_nodes = node_features.size(0)
    class_prob = torch.zeros(n_nodes, n_classes, device=device)
    
    for i in range(n_nodes):
        if torch.sum(support_features[i]) != 0:
            similarities = torch.exp(torch.mm(support_features[i], node_features[i:i+1].T)).squeeze()
            weights = similarities / torch.sum(similarities)
            
            for j, label in enumerate(support_labels[i]):
                class_prob[i, label] += weights[j]
    
    return class_prob

def create_adjacency_matrix(edge_index, n_nodes, device):
    adj = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.shape[1], device=device),
        [n_nodes, n_nodes]
    ).coalesce()

    eye = torch.eye(n_nodes, device=device).to_sparse()
    return (adj + eye).coalesce()

def kl_divergence_loss(logits, labels_one_hot):
    log_probs = F.log_softmax(logits, dim=1)
    probs = F.softmax(logits, dim=1)
    kl_loss = F.kl_div(log_probs, labels_one_hot.float(), reduction='batchmean')
    return kl_loss

class UnionNET:
    
    def __init__(self, model, data, dataset, config):
        self.model = model.to(data.x.device)
        self.data = data
        self.device = data.x.device
        
        self.n_epochs = config.get('n_epochs', 200)
        self.lr = config.get('lr', 0.01)
        self.weight_decay = config.get('weight_decay', 5e-4)
        self.patience = config.get('patience', 100)
        
        self.k = config.get('k', 5)
        self.alpha = config.get('alpha', 0.5)
        self.beta = config.get('beta', 1.0)
        
        self.n_nodes = data.x.shape[0]
        self.n_features = data.x.shape[1]
        self.n_classes = dataset

        self.features = data.x.to(torch.float)
        if config.get('feat_norm', True):
            self.features = normalize_features(self.features)

        self.clean_labels = data.y
        self.noisy_labels = data.y_noisy

        self.edge_index = data.edge_index
        self.adj = create_adjacency_matrix(data.edge_index, self.n_nodes, self.device)
 
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.best_loss = float('inf')
        self.wait = 0
        self.best_weights = None
        self.results = {'train': -1, 'val': -1, 'test': -1}
        
        self._print_stats()
    
    def _print_stats(self):
        print("UnionNET Dataset Statistics")
        print(f"Nodes: {self.n_nodes}")
        print(f"Features: {self.n_features}")
        print(f"Classes: {self.n_classes}")
        print(f"Edges: {self.edge_index.shape[1] // 2}")
        print(f"Train/Val/Test: {self.train_mask.sum()}/{self.val_mask.sum()}/{self.test_mask.sum()}")
        
        n_changed = (self.clean_labels != self.noisy_labels).sum().item()
        if n_changed > 0:
            actual_rate = n_changed / len(self.clean_labels)
            print(f"Label noise rate: {actual_rate:.3f} ({n_changed} labels changed)")
    
    def _forward(self, mask=None, use_clean=False):
        if mask != 'train':
            self.model.eval()
        else:
            self.model.train()
        
        context = torch.no_grad() if mask != 'train' else torch.enable_grad()
        with context:
            output = self.model(self.data)

            if mask is not None:
                labels = self.clean_labels if use_clean else self.noisy_labels
                mask_tensor = getattr(self, f'{mask}_mask')
                
                loss = F.cross_entropy(output[mask_tensor], labels[mask_tensor])
                y_true = labels[mask_tensor].cpu().numpy()
                y_pred = output[mask_tensor].detach().cpu().numpy().argmax(1)
                acc = accuracy_score(y_true, y_pred)
                return output, loss, acc
        
        return output

    
    def _compute_unionnet_loss(self, output):
        support_features, support_labels = construct_support_set(
            self.features, self.train_mask, self.noisy_labels,
            self.edge_index, self.k
        )

        train_features = self.features[self.train_mask]
        train_noisy_labels = self.noisy_labels[self.train_mask]

        class_probs = label_aggregation(
            support_features[self.train_mask], support_labels[self.train_mask], 
            train_features, self.n_classes
        )

        weights = class_probs[range(len(train_noisy_labels)), train_noisy_labels]
        reweight_loss = F.cross_entropy(output[self.train_mask], train_noisy_labels, reduction='none')
        reweight_loss = torch.mean(weights * reweight_loss)

        correction_loss = F.cross_entropy(output[self.train_mask], train_noisy_labels)

        one_hot_labels = F.one_hot(train_noisy_labels, num_classes=self.n_classes).to(self.device)
        kl_loss = kl_divergence_loss(output[self.train_mask], one_hot_labels)

        return self.alpha * reweight_loss + (1 - self.alpha) * correction_loss + self.beta * kl_loss

    def train(self, debug=True):
        start_time = time.time()
        
        for epoch in range(self.n_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            output = self.model(self.data)
            loss_train = self._compute_unionnet_loss(output)
            loss_train.backward()
            self.optimizer.step()
            
            train_labels = self.noisy_labels[self.train_mask].cpu().numpy()
            train_preds = output[self.train_mask].detach().cpu().numpy().argmax(1)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='macro')
            
            _, val_loss, _ = self._forward('val')
            val_labels = self.noisy_labels[self.val_mask].cpu().numpy()
            val_preds = output[self.val_mask].detach().cpu().numpy().argmax(1)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            
            if val_loss.item() < self.best_loss:
                self.best_loss = val_loss.item()
                self.wait = 0
                self.results['train'] = train_acc
                self.results['val'] = val_acc
                self.best_weights = deepcopy(self.model.state_dict())
            else:
                self.wait += 1

            if debug:
                print(f"Epoch {epoch:03d} | Train Loss: {loss_train:.4f}, Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                      f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
            
            if self.patience and self.wait >= self.patience:
                if debug:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

        self.model.eval()
        with torch.no_grad():
            output = self.model(self.data)
            test_labels = self.clean_labels[self.test_mask].cpu().numpy()
            test_preds = output[self.test_mask].cpu().numpy().argmax(1)
            test_acc = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, average='macro')
            
            test_loss = F.cross_entropy(output[self.test_mask], self.clean_labels[self.test_mask])

        self.results['test'] = test_acc

        if debug:
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
            total_time = time.time() - start_time
            print(f"\nUnionNET Training completed in {total_time:.2f}s")
            print(f"Final Results - Train: {self.results['train']:.4f}, "
                  f"Val: {self.results['val']:.4f}, Test: {self.results['test']:.4f}")

        return self.results
    