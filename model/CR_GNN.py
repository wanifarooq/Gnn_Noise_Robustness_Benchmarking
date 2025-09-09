import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, mask_feature
from torch_geometric.data import Data
from copy import deepcopy
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ContrastiveProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.first_layer = torch.nn.Linear(input_dim, output_dim)
        self.second_layer = torch.nn.Linear(output_dim, output_dim)

    def forward(self, embeddings):
        embeddings = F.relu(self.first_layer(embeddings))
        return self.second_layer(embeddings)


class NodeClassificationHead(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = torch.nn.Linear(input_dim, num_classes)

    def forward(self, embeddings):
        return F.log_softmax(self.classifier(embeddings), dim=-1)


def contrastive_loss_original(z1, z2, tau):

    N = z1.size(0)
    
    pos1 = torch.exp(F.cosine_similarity(z1, z2, dim=1) / tau)
    neg1 = torch.sum(torch.exp(torch.mm(z1, z1.t()) / tau), dim=1) + pos1
    
    pos2 = torch.exp(F.cosine_similarity(z2, z1, dim=1) / tau) 
    neg2 = torch.sum(torch.exp(torch.mm(z2, z2.t()) / tau), dim=1) + pos2
    
    loss = (torch.sum(-torch.log(pos1 / (pos1 + neg1))) + 
            torch.sum(-torch.log(pos2 / (pos2 + neg2)))) / (2 * N)
    
    return loss


def dynamic_cross_entropy_loss_original(p1, p2, labels):

    labels = labels.long()
    pseudo1 = p1.argmax(dim=1)
    pseudo2 = p2.argmax(dim=1)
    consistent_mask = (pseudo1 == pseudo2)
    
    if consistent_mask.sum() > 0:
        loss = F.nll_loss(p1[consistent_mask], labels[consistent_mask])
    else:
        loss = torch.tensor(0.0, device=p1.device, requires_grad=True)
    
    return loss


def cross_space_consistency_loss(zm, pm):
    return F.mse_loss(zm, pm)


class CRGNNModel:
    def __init__(self, device='cuda', **config):
        self.device = torch.device(device)
        self.T = config.get('T', 0.5)
        self.tau = config.get('tau', 0.5)
        self.p = config.get('p', 0.5)
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 0.0)
        self.lr = config.get('lr', 0.001)
        self.weight_decay = config.get('weight_decay', 5e-4)
        self.epochs = config.get('epochs', 200)
        self.patience = config.get('patience', 10)
        self.hidden_channels = config.get('hidden_channels', 64)
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.best_weights = None

    def train_model(self, backbone_model, graph_data, model_config, model_factory_function):
        graph_data = graph_data.to(self.device)
        num_classes = graph_data.y.max().item() + 1
        
        backbone = backbone_model.to(self.device)
        
        with torch.no_grad():
            sample_out = backbone(graph_data)
        
        if sample_out.size(1) != self.hidden_channels:
            adapter = nn.Linear(sample_out.size(1), self.hidden_channels).to(self.device)
        else:
            adapter = nn.Identity().to(self.device)
        
        # Heads
        proj_head = ContrastiveProjectionHead(self.hidden_channels, self.hidden_channels).to(self.device)
        class_head = NodeClassificationHead(self.hidden_channels, num_classes).to(self.device)
        
        # Optimizer
        params = list(backbone.parameters()) + list(adapter.parameters()) + \
                list(proj_head.parameters()) + list(class_head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        
        # Masks
        if hasattr(graph_data, 'train_mask'):
            train_mask, val_mask, test_mask = graph_data.train_mask, graph_data.val_mask, graph_data.test_mask
        else:
            n = graph_data.x.size(0)
            idx = torch.randperm(n)
            train_mask = torch.zeros(n, dtype=torch.bool)
            val_mask = torch.zeros(n, dtype=torch.bool)
            test_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[idx[:int(0.6*n)]] = True
            val_mask[idx[int(0.6*n):int(0.8*n)]] = True
            test_mask[idx[int(0.8*n):]] = True
            train_mask = train_mask.to(self.device)
            val_mask = val_mask.to(self.device)
            test_mask = test_mask.to(self.device)
        
        clean_labels = getattr(graph_data, 'y_original', graph_data.y)
        noisy_labels = graph_data.y
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            backbone.train()
            adapter.train()
            proj_head.train()
            class_head.train()
            
            optimizer.zero_grad()
            
            loss, train_acc = self._train_step(backbone, adapter, proj_head, class_head,
                                             graph_data.x, graph_data.edge_index, 
                                             noisy_labels, train_mask)
            
            if loss is not None and torch.isfinite(loss):
                loss.backward()
                optimizer.step()
            
            val_loss, val_acc = self._evaluate(backbone, adapter, class_head,
                                             graph_data.x, graph_data.edge_index,
                                             noisy_labels, val_mask)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                self.best_weights = {
                    'backbone': deepcopy(backbone.state_dict()),
                    'adapter': deepcopy(adapter.state_dict()),
                    'proj_head': deepcopy(proj_head.state_dict()),
                    'class_head': deepcopy(class_head.state_dict())
                }
                training_time = time.time() - start_time
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    break
            
            print(f"Epoch {epoch+1:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if self.best_weights:
            backbone.load_state_dict(self.best_weights['backbone'])
            adapter.load_state_dict(self.best_weights['adapter'])
            proj_head.load_state_dict(self.best_weights['proj_head'])
            class_head.load_state_dict(self.best_weights['class_head'])
        
        test_loss, test_acc = self._evaluate(backbone, adapter, class_head,
                                           graph_data.x, graph_data.edge_index,
                                           clean_labels, test_mask)
        
        backbone.eval()
        adapter.eval()
        class_head.eval()
        with torch.no_grad():
            embeddings = backbone(Data(x=graph_data.x, edge_index=graph_data.edge_index))
            embeddings = adapter(embeddings)
            preds = class_head(embeddings)
            
            y_true = clean_labels[test_mask].cpu().numpy()
            y_pred = preds[test_mask].exp().argmax(dim=1).cpu().numpy()
            
            test_f1 = f1_score(y_true, y_pred, average='macro')
            test_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            test_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"Final Test - Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        return {
            'accuracy': float(test_acc),
            'f1': float(test_f1),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'oversmoothing': {'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                            'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0}
        }

    def _train_step(self, backbone, adapter, proj_head, class_head, x, edge_index, labels, mask):
        h = backbone(Data(x=x, edge_index=edge_index))
        h = adapter(h)
        
        # Augmentations
        edge_idx1, _ = dropout_adj(edge_index, p=0.3)
        edge_idx2, _ = dropout_adj(edge_index, p=0.3)
        x1, _ = mask_feature(x, p=0.3)
        x2, _ = mask_feature(x, p=0.3)
        
        h1 = backbone(Data(x=x1, edge_index=edge_idx1))
        h1 = adapter(h1)
        h2 = backbone(Data(x=x2, edge_index=edge_idx2))  
        h2 = adapter(h2)
        
        # Contrastive
        z1 = proj_head(h1)
        z2 = proj_head(h2)
        loss_con = contrastive_loss_original(z1, z2, self.tau)
        
        # Classification
        p1 = class_head(h1)
        p2 = class_head(h2)
        loss_sup = dynamic_cross_entropy_loss_original(p1[mask], p2[mask], labels[mask])
        
        # Total loss
        total_loss = self.alpha * loss_con + loss_sup
        
        # Consistency loss
        if self.beta > 0:
            zm = torch.exp(F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2) / self.T).mean(dim=1)
            pm = torch.exp(F.cosine_similarity(p1.unsqueeze(1), p2.unsqueeze(0), dim=2) / self.T).mean(dim=1)
            pm = torch.where(pm > self.p, pm, torch.zeros_like(pm))
            loss_ccon = cross_space_consistency_loss(zm, pm)
            total_loss += self.beta * loss_ccon
        
        with torch.no_grad():
            pred = class_head(h)[mask].exp().argmax(dim=1)
            acc = accuracy_score(labels[mask].cpu().numpy(), pred.cpu().numpy())
        
        return total_loss, acc

    def _evaluate(self, backbone, adapter, class_head, x, edge_index, labels, mask):
        backbone.eval()
        adapter.eval()
        class_head.eval()
        
        with torch.no_grad():
            h = backbone(Data(x=x, edge_index=edge_index))
            h = adapter(h)
            preds = class_head(h)
            
            loss = F.nll_loss(preds[mask], labels[mask])
            pred_labels = preds[mask].exp().argmax(dim=1)
            acc = accuracy_score(labels[mask].cpu().numpy(), pred_labels.cpu().numpy())
            
        return loss, acc

    def _compute_oversmoothing_metrics(self, embeddings, edge_index, node_mask, labels=None):
        if not self.oversmoothing_calc:
            return None
            
        try:
            masked_indices = torch.where(node_mask)[0]
            subset_embeddings = embeddings[node_mask]
            
            index_set = set(masked_indices.cpu().numpy())
            valid_edges = []
            for i, (src, tgt) in enumerate(edge_index.t()):
                if src.item() in index_set and tgt.item() in index_set:
                    valid_edges.append(i)
            
            if not valid_edges:
                return self._default_oversmoothing_metrics(subset_embeddings.shape[0])
            
            filtered_edge_index = edge_index[:, valid_edges]
            
            index_mapping = {orig_idx.item(): local_idx for local_idx, orig_idx in enumerate(masked_indices)}
            remapped_edges = torch.stack([
                torch.tensor([index_mapping[src.item()] for src in filtered_edge_index[0]], device=edge_index.device),
                torch.tensor([index_mapping[tgt.item()] for tgt in filtered_edge_index[1]], device=edge_index.device)
            ])
            
            graphs_in_class = [{
                'X': subset_embeddings,
                'edge_index': remapped_edges,
                'edge_weight': None
            }]
            
            metrics = self.oversmoothing_calc.compute_all_metrics(
                X=subset_embeddings,
                edge_index=remapped_edges,
                graphs_in_class=graphs_in_class
            )
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Could not compute oversmoothing metrics: {e}")
            return None

    def _default_oversmoothing_metrics(self, dim):
        return {
            'NumRank': float(dim),
            'Erank': float(dim),
            'EDir': 0.0,
            'EDir_traditional': 0.0,
            'EProj': 0.0,
            'MAD': 0.0
        }

    def _print_oversmoothing_metrics(self, train_metrics, val_metrics):

        print(f"Train DE: {train_metrics['EDir']:.4f}, Val DE: {val_metrics['EDir']:.4f} | "
              f"Train DE_trad: {train_metrics['EDir_traditional']:.4f}, Val DE_trad: {val_metrics['EDir_traditional']:.4f} | "
              f"Train EProj: {train_metrics['EProj']:.4f}, Val EProj: {val_metrics['EProj']:.4f} | "
              f"Train MAD: {train_metrics['MAD']:.4f}, Val MAD: {val_metrics['MAD']:.4f} | "
              f"Train NumRank: {train_metrics['NumRank']:.4f}, Val NumRank: {val_metrics['NumRank']:.4f} | "
              f"Train Erank: {train_metrics['Erank']:.4f}, Val Erank: {val_metrics['Erank']:.4f}")