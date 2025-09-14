import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, mask_feature
from torch_geometric.data import Data
from copy import deepcopy
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from model.evaluation import OversmoothingMetrics


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


def contrastive_loss_original_style(z1, z2, tau):

    N = z1.size(0)
    
    # Positive similarities
    pos1 = torch.exp(F.cosine_similarity(z1, z2, dim=1) / tau)
    pos2 = torch.exp(F.cosine_similarity(z2, z1, dim=1) / tau)
    
    # Negative similarities
    neg_matrix_1 = torch.exp(torch.mm(z1, z1.t()) / tau)  
    neg_matrix_2 = torch.exp(torch.mm(z2, z2.t()) / tau)

    neg1 = torch.sum(neg_matrix_1, dim=1) - torch.diag(neg_matrix_1)
    neg2 = torch.sum(neg_matrix_2, dim=1) - torch.diag(neg_matrix_2)
    
    loss1 = -torch.log(pos1 / (pos1 + neg1 + 1e-8))
    loss2 = -torch.log(pos2 / (pos2 + neg2 + 1e-8))
    
    return (torch.sum(loss1) + torch.sum(loss2)) / (2 * N)


def dynamic_cross_entropy_loss_corrected(p1, p2, labels):
    #Dynamic cross-entropy loss
    if len(labels) == 0:
        return torch.tensor(0.0, device=p1.device, requires_grad=True)
    
    labels = labels.long()
    pseudo_labels1 = p1.argmax(dim=1)
    pseudo_labels2 = p2.argmax(dim=1)
    consistent_mask = (pseudo_labels1 == pseudo_labels2)
    
    if consistent_mask.sum() > 0:

        loss = F.nll_loss(p1[consistent_mask], labels[consistent_mask])
    else:
        loss = torch.tensor(0.0, device=p1.device, requires_grad=True)
    
    return loss


def compute_cross_space_consistency_fixed(z1, z2, p1, p2, T, p_threshold):

    # Similarity matrix
    z1_expanded = z1.unsqueeze(1)
    z2_expanded = z2.unsqueeze(0)
    zm = torch.exp(F.cosine_similarity(z1_expanded, z2_expanded, dim=2) / T)
    zm = zm.mean(dim=1)
    
    # Similarity matrix
    p1_expanded = p1.unsqueeze(1)
    p2_expanded = p2.unsqueeze(0)
    pm = torch.exp(F.cosine_similarity(p1_expanded, p2_expanded, dim=2) / T)
    pm = pm.mean(dim=1)
    
    # Apply thresholding
    pm = torch.where(pm > p_threshold, pm, torch.zeros_like(pm))
    
    # MSE loss
    return F.mse_loss(zm, pm)


class CRGNNModel:
    def __init__(self, device='cuda', **config):
        self.device = torch.device(device)
        self.T = config.get('T', 0.5)
        self.tau = config.get('tau', 0.5)
        self.p = config.get('p', 0.5)

        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 1.0)
        self.lr = config.get('lr', 0.001)
        self.weight_decay = config.get('weight_decay', 5e-4)
        self.epochs = config.get('epochs', 200)
        self.patience = config.get('patience', 10)
        self.hidden_channels = config.get('hidden_channels', 64)
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.best_weights = None
        
        self.oversmoothing_evaluator = OversmoothingMetrics(device=self.device)

    def _compute_oversmoothing_metrics(self, embeddings, edge_index, node_mask):

        try:
            masked_node_indices = torch.where(node_mask)[0]
            masked_embeddings = embeddings[node_mask]
            mask_set = set(masked_node_indices.cpu().numpy())

            edge_mask = torch.tensor([
                src.item() in mask_set and tgt.item() in mask_set
                for src, tgt in edge_index.t()
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
            
            masked_edges = edge_index[:, edge_mask]
            node_mapping = {orig_idx.item(): local_idx 
                           for local_idx, orig_idx in enumerate(masked_node_indices)}
            
            remapped_edges = torch.stack([
                torch.tensor([node_mapping[src.item()] for src in masked_edges[0]], 
                           device=edge_index.device),
                torch.tensor([node_mapping[tgt.item()] for tgt in masked_edges[1]], 
                           device=edge_index.device)
            ])
            
            graphs_in_class = [{
                'X': masked_embeddings,
                'edge_index': remapped_edges,
                'edge_weight': None
            }]
            
            return self.oversmoothing_evaluator.compute_all_metrics(
                X=masked_embeddings,
                edge_index=remapped_edges,
                graphs_in_class=graphs_in_class
            )
        except Exception as e:
            print(f"Warning: Could not compute oversmoothing metrics for mask: {e}")
            return {
                'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0
            }

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
        
        proj_head = ContrastiveProjectionHead(self.hidden_channels, self.hidden_channels).to(self.device)
        class_head = NodeClassificationHead(self.hidden_channels, num_classes).to(self.device)
        
        params = list(backbone.parameters()) + list(adapter.parameters()) + \
                list(proj_head.parameters()) + list(class_head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        
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
            
            if epoch % 20 == 0:
                with torch.no_grad():
                    embeddings = backbone(Data(x=graph_data.x, edge_index=graph_data.edge_index))
                    embeddings = adapter(embeddings)
                    
                    train_oversmooth_metrics = self._compute_oversmoothing_metrics(
                        embeddings, graph_data.edge_index, train_mask
                    )
                    val_oversmooth_metrics = self._compute_oversmoothing_metrics(
                        embeddings, graph_data.edge_index, val_mask
                    )
                    
                    print(f"Epoch {epoch+1:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Total Loss: {loss:.4f}")
                    print(f"Train EDir: {train_oversmooth_metrics['EDir']:.4f}, Val EDir: {val_oversmooth_metrics['EDir']:.4f} | "
                          f"Train EDir_trad: {train_oversmooth_metrics['EDir_traditional']:.4f}, Val EDir_trad: {val_oversmooth_metrics['EDir_traditional']:.4f} | "
                          f"Train EProj: {train_oversmooth_metrics['EProj']:.4f}, Val EProj: {val_oversmooth_metrics['EProj']:.4f} | "
                          f"Train MAD: {train_oversmooth_metrics['MAD']:.4f}, Val MAD: {val_oversmooth_metrics['MAD']:.4f} | "
                          f"Train NumRank: {train_oversmooth_metrics['NumRank']:.4f}, Val NumRank: {val_oversmooth_metrics['NumRank']:.4f} | "
                          f"Train Erank: {train_oversmooth_metrics['Erank']:.4f}, Val Erank: {val_oversmooth_metrics['Erank']:.4f}")
            elif epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch+1:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Total Loss: {loss:.4f}")
            
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
            
            test_oversmooth_metrics = self._compute_oversmoothing_metrics(
                embeddings, graph_data.edge_index, test_mask
            )
        
        print(f"Final Test - Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        print("Final Test Oversmoothing Metrics:")
        print(f"Test EDir: {test_oversmooth_metrics['EDir']:.4f}, EDir_trad: {test_oversmooth_metrics['EDir_traditional']:.4f}, "
              f"EProj: {test_oversmooth_metrics['EProj']:.4f}, MAD: {test_oversmooth_metrics['MAD']:.4f}, "
              f"NumRank: {test_oversmooth_metrics['NumRank']:.4f}, Erank: {test_oversmooth_metrics['Erank']:.4f}")
        
        return {
            'accuracy': float(test_acc),
            'f1': float(test_f1),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'oversmoothing': test_oversmooth_metrics
        }

    def _train_step(self, backbone, adapter, proj_head, class_head, 
                   x, edge_index, labels, mask):

        edge_idx1, _ = dropout_adj(edge_index, p=0.3, training=True)
        edge_idx2, _ = dropout_adj(edge_index, p=0.3, training=True)  
        x1, _ = mask_feature(x, p=0.3)
        x2, _ = mask_feature(x, p=0.3)
        
        h1 = backbone(Data(x=x1, edge_index=edge_idx1))
        h1 = adapter(h1)
        h2 = backbone(Data(x=x2, edge_index=edge_idx2))
        h2 = adapter(h2)
        
        # Contrastive loss
        z1 = proj_head(h1)
        z2 = proj_head(h2)
        loss_con = contrastive_loss_original_style(z1, z2, self.tau)
        
        # Classification predictions
        p1 = class_head(h1)
        p2 = class_head(h2)
        
        # Dynamic cross-entropy loss
        if mask.sum() > 0:
            loss_sup = dynamic_cross_entropy_loss_corrected(p1[mask], p2[mask], labels[mask])
        else:
            loss_sup = torch.tensor(0.0, device=x.device, requires_grad=True)
        
        # Cross-space consistency
        loss_ccon = torch.tensor(0.0, device=x.device)
        if self.beta > 0:
            try:
                loss_ccon = compute_cross_space_consistency_fixed(z1, z2, p1, p2, self.T, self.p)
            except:
                loss_ccon = torch.tensor(0.0, device=x.device)
        
        # Total loss
        total_loss = self.alpha * loss_con + loss_sup + self.beta * loss_ccon
        
        with torch.no_grad():
            h_orig = backbone(Data(x=x, edge_index=edge_index))
            h_orig = adapter(h_orig)
            pred_orig = class_head(h_orig)[mask].exp().argmax(dim=1)
            if mask.sum() > 0:
                acc = accuracy_score(labels[mask].cpu().numpy(), pred_orig.cpu().numpy())
            else:
                acc = 0.0
        
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