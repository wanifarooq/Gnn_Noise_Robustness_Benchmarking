import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

class GNN_Cleaner(nn.Module):
    def __init__(self, feat_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(1)

class GNNCleanerTrainer:

    def __init__(self, config, data, device, num_classes, gnn_model=None):
        self.config = config
        self.data = data
        self.device = device
        self.num_classes = num_classes
        
        self.epochs = config.get('epochs', 200)
        self.lr = config.get('lr', 0.01)
        self.weight_decay = config.get('weight_decay', 5e-4)
        self.hidden = config.get('hidden_channels', 64)
        
        self.alpha = config.get('alpha', 0.9)
        self.lp_iters = config.get('lp_iters', 30)
        self.sharpen_temp = config.get('sharpen_temp', 1.0)

        if gnn_model is not None:
            self.model = gnn_model.to(device)
        else:
            raise ValueError("GNN Cleaner wants a GNN model")
        
        feat_dim = data.x.size(1) + num_classes
        self.gnn_cleaner = GNN_Cleaner(feat_dim, self.hidden).to(device)
        
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.gnn_cleaner.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )
        
        self.results = {'train': -1, 'val': -1, 'test': -1}
        
        self._print_stats()
    
    def _print_stats(self):
        print("GNN Cleaner Dataset Statistics")
        print(f"Nodes: {self.data.x.shape[0]}")
        print(f"Features: {self.data.x.shape[1]}")
        print(f"Classes: {self.num_classes}")
        print(f"Edges: {self.data.edge_index.shape[1] // 2}")
        print(f"Train/Val/Test: {self.data.train_mask.sum()}/{self.data.val_mask.sum()}/{self.data.test_mask.sum()}")
        
        if hasattr(self.data, 'y_noisy'):
            n_changed = (self.data.y != self.data.y_noisy).sum().item()
            if n_changed > 0:
                actual_rate = n_changed / len(self.data.y)
                print(f"Label noise rate: {actual_rate:.3f} ({n_changed} labels changed)")
    
    def label_propagation(self, edge_index, y_soft, num_nodes, train_mask, alpha=0.9, iters=50, sharpen_temp=1.0):
        row, col = edge_index
        edges = np.vstack((row.cpu().numpy(), col.cpu().numpy()))
        A = sparse.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(num_nodes, num_nodes))
        A = A + sparse.eye(num_nodes)
        deg = np.array(A.sum(1)).flatten()
        deg[deg == 0] = 1e-12
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(deg))
        S = D_inv_sqrt.dot(A).dot(D_inv_sqrt)
        
        Y = y_soft.cpu().numpy().astype(np.float32)
        Z = Y.copy()
        train_mask_np = train_mask.cpu().numpy()
        
        for _ in range(iters):
            Z = alpha * (S.dot(Z)) + (1 - alpha) * Y
            Z[train_mask_np] = Y[train_mask_np]
        
        if sharpen_temp != 1.0:
            Z = np.power(Z, 1.0 / sharpen_temp)
            Z /= Z.sum(axis=1, keepdims=True)
        
        return torch.from_numpy(Z)
    
    def one_hot(self, labels, C):
        N = labels.size(0)
        oh = torch.zeros(N, C, device=labels.device)
        oh.scatter_(1, labels.unsqueeze(1), 1.0)
        return oh
    
    def train_step(self):
        self.model.train()
        self.gnn_cleaner.train()
        
        x, edge_index = self.data.x.to(self.device), self.data.edge_index.to(self.device)
        labels = self.data.y_noisy.to(self.device)
        num_nodes = x.size(0)
        
        try:
            logits = self.model(self.data)
        except:
            logits = self.model(x, edge_index)
        
        probs = F.softmax(logits, dim=1).detach()

        y_soft_init = self.one_hot(labels, self.num_classes).cpu()
        y_soft_init[~self.data.train_mask] = 0.0
        
        prop = self.label_propagation(
            edge_index.cpu(), y_soft_init, num_nodes,
            train_mask=self.data.train_mask,
            alpha=self.alpha, iters=self.lp_iters, 
            sharpen_temp=self.sharpen_temp
        ).to(self.device)

        with torch.no_grad():
            gnn_input = torch.cat([x, probs], dim=1)
        gnn = self.gnn_cleaner(gnn_input).unsqueeze(1)
        
        noisy_onehot = self.one_hot(labels, self.num_classes)
        corrected_soft = (1 - gnn) * noisy_onehot + gnn * prop
        
        train_mask = self.data.train_mask.to(self.device)
        loss = -(corrected_soft[train_mask] * F.log_softmax(logits[train_mask], dim=1)).sum(dim=1).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self):

        self.model.eval()
        
        x, edge_index = self.data.x.to(self.device), self.data.edge_index.to(self.device)
        
        try:
            logits = self.model(self.data)
        except:
            logits = self.model(x, edge_index)
        
        preds = logits.argmax(dim=1)
        
        labels = self.data.y.to(self.device)
        
        train_acc = (preds[self.data.train_mask] == labels[self.data.train_mask]).float().mean().item()
        val_acc = (preds[self.data.val_mask] == labels[self.data.val_mask]).float().mean().item()
        test_acc = (preds[self.data.test_mask] == labels[self.data.test_mask]).float().mean().item()
        
        return {'train': train_acc, 'val': val_acc, 'test': test_acc}
    
    def train(self, debug=True):
        start_time = time.time()
        
        best_val = 0.0
        best_train = 0.0
        best_state = None

        patience = 20
        counter = 0
        
        for epoch in range(1, self.epochs + 1):
            loss = self.train_step()
            accs = self.evaluate()

            if accs['val'] > best_val:
                best_val = accs['val']
                best_train = accs['train']
                best_state = {
                    "model": self.model.state_dict(),
                    "cleaner": self.gnn_cleaner.state_dict()
                }
                counter = 0
            else:
                counter += 1
            
            if counter >= patience:
                if debug:
                    print(f"Early stopping at epoch {epoch}")
                break

            if debug and (epoch % 20 == 0 or epoch == 1):
                print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Train {accs['train']:.4f} "
                    f"Val {accs['val']:.4f}")
        
        if best_state is not None:
            self.model.load_state_dict(best_state["model"])
            self.gnn_cleaner.load_state_dict(best_state["cleaner"])

        final_accs = self.evaluate()
        self.results = {'train': best_train, 'val': best_val, 'test': final_accs['test']}
        
        if debug:
            total_time = time.time() - start_time
            print(f"\nGNN Cleaner Training completed in {total_time:.2f}s")
            print(f"Best Results - Train: {self.results['train']:.4f}, "
                f"Val: {self.results['val']:.4f}, Test: {self.results['test']:.4f}")
        
        return self.results
    