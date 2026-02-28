import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge, mask_feature
from torch_geometric.data import Data
from copy import deepcopy
from collections import defaultdict
from model.evaluation import (OversmoothingMetrics, ClassificationMetrics,
                              compute_oversmoothing_for_mask, evaluate_model)
from model.base import BaseTrainer
from model.registry import register


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
        self.pr = config.get('pr', 0.3)
        self.oversmoothing_every = config.get('oversmoothing_every', 20)

        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        self.oversmoothing_evaluator = OversmoothingMetrics(device=self.device)
        self.cls_evaluator = ClassificationMetrics(average='macro')

    def train_model(self, backbone_model, graph_data, model_config, model_factory_function,
                    log_epoch_fn=None):
        per_epochs_oversmoothing = defaultdict(list)
        per_epochs_val_oversmoothing = defaultdict(list)
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

        # Assign early so get_checkpoint_state() can access during training
        self._backbone = backbone
        self._adapter = adapter
        self._proj_head = proj_head
        self._class_head = class_head
        self._graph_data = graph_data
        self._clean_labels = clean_labels

        for epoch in range(self.epochs):
            backbone.train()
            adapter.train()
            proj_head.train()
            class_head.train()
            
            optimizer.zero_grad(set_to_none=True)
            
            loss, train_acc, train_f1 = self._train_step(backbone, adapter, proj_head, class_head,
                                                        graph_data.x, graph_data.edge_index,
                                                        noisy_labels, train_mask)
            
            if loss is not None and torch.isfinite(loss):
                loss.backward()
                optimizer.step()
            
            val_loss, val_acc, val_f1, eval_pred = self._evaluate(backbone, adapter, class_head,
                                                                graph_data.x, graph_data.edge_index,
                                                                noisy_labels, val_mask)
            
            os_entry = None
            if epoch % self.oversmoothing_every == 0 or epoch == self.epochs - 1:
                with torch.no_grad():
                    embeddings = backbone.get_embeddings(Data(x=graph_data.x, edge_index=graph_data.edge_index))

                    train_oversmooth_metrics = compute_oversmoothing_for_mask(
                        self.oversmoothing_evaluator, embeddings, graph_data.edge_index, train_mask
                    )
                    val_oversmooth_metrics = compute_oversmoothing_for_mask(
                        self.oversmoothing_evaluator, embeddings, graph_data.edge_index, val_mask
                    )
                    for key, value in train_oversmooth_metrics.items():
                        per_epochs_oversmoothing[key].append(value)
                    for key, value in val_oversmooth_metrics.items():
                        per_epochs_val_oversmoothing[key].append(value)
                    os_entry = {'train': dict(train_oversmooth_metrics), 'val': dict(val_oversmooth_metrics)}
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
            is_best = val_loss < self.best_val_loss
            if log_epoch_fn is not None:
                log_epoch_fn(epoch, loss.item(), val_loss.item(), train_acc, val_acc,
                             train_f1=train_f1, val_f1=val_f1,
                             oversmoothing=os_entry, is_best=is_best,
                             train_predictions=eval_pred)
            if is_best:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    break

        self._backbone = backbone
        self._adapter = adapter
        self._class_head = class_head
        self._graph_data = graph_data
        self._clean_labels = clean_labels

        return {
            'train_oversmoothing': dict(per_epochs_oversmoothing),
            'val_oversmoothing': dict(per_epochs_val_oversmoothing),
            'stopped_at_epoch': epoch,
        }

    def _train_step(self, backbone, adapter, proj_head, class_head, 
                   x, edge_index, labels, mask):

        edge_idx1, _ = dropout_edge(edge_index, p=self.pr, training=True)
        edge_idx2, _ = dropout_edge(edge_index, p=self.pr, training=True)
        x1, _ = mask_feature(x, p=self.pr)
        x2, _ = mask_feature(x, p=self.pr)
        
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
            except Exception:
                loss_ccon = torch.tensor(0.0, device=x.device)
        
        # Total loss
        total_loss = self.alpha * loss_con + loss_sup + self.beta * loss_ccon
        
        with torch.no_grad():
            h_orig = backbone(Data(x=x, edge_index=edge_index))
            h_orig = adapter(h_orig)
            pred_orig = class_head(h_orig)[mask].exp().argmax(dim=1)
            if mask.sum() > 0:
                acc = self.cls_evaluator.compute_accuracy(pred_orig, labels[mask])
                f1 = self.cls_evaluator.compute_f1(pred_orig, labels[mask])
            else:
                acc = 0.0
                f1 = 0.0

        return total_loss, acc, f1

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
            acc = self.cls_evaluator.compute_accuracy(pred_labels, labels[mask])
            f1 = self.cls_evaluator.compute_f1(pred_labels, labels[mask])

        return loss, acc, f1, preds.exp().argmax(dim=1)


@register('cr_gnn')
class CRGNNMethodTrainer(BaseTrainer):
    def train(self):
        d = self.init_data
        cr_params = self.config.get('cr_gnn_params', {})

        combined_params = {
            'hidden_channels': self.config['model'].get('hidden_channels', 64),
            'lr': d['lr'],
            'weight_decay': d['weight_decay'],
            'epochs': d['epochs'],
            'patience': d['patience'],
            'oversmoothing_every': d['oversmoothing_every'],
        }
        combined_params.update(cr_params)

        self._cr = CRGNNModel(device=d['device'], **combined_params)
        return self._cr.train_model(
            d['backbone_model'], d['data_for_training'],
            d['backbone_model'], d['get_model'],
            log_epoch_fn=self.log_epoch,
        )

    def get_checkpoint_state(self) -> dict:
        cr = self._cr
        return {
            'backbone': deepcopy(cr._backbone.state_dict()),
            'adapter': deepcopy(cr._adapter.state_dict()),
            'proj_head': deepcopy(cr._proj_head.state_dict()),
            'class_head': deepcopy(cr._class_head.state_dict()),
        }

    def load_checkpoint_state(self, state):
        if not hasattr(self, '_cr'):
            self._setup_for_eval(state)
        cr = self._cr
        cr._backbone.load_state_dict(state['backbone'])
        cr._adapter.load_state_dict(state['adapter'])
        cr._proj_head.load_state_dict(state['proj_head'])
        cr._class_head.load_state_dict(state['class_head'])

    def _setup_for_eval(self, state):
        """Initialize CRGNNModel components for eval-only (no training)."""
        d = self.init_data
        cr_params = self.config.get('cr_gnn_params', {})
        combined_params = {
            'hidden_channels': self.config['model'].get('hidden_channels', 64),
            'lr': d['lr'],
            'weight_decay': d['weight_decay'],
            'epochs': d['epochs'],
            'patience': d['patience'],
            'oversmoothing_every': d['oversmoothing_every'],
        }
        combined_params.update(cr_params)
        self._cr = CRGNNModel(device=d['device'], **combined_params)

        device = torch.device(d['device'])
        graph_data = d['data_for_training'].to(device)
        backbone = d['backbone_model'].to(device)
        num_classes = graph_data.y.max().item() + 1
        hidden = self._cr.hidden_channels

        with torch.no_grad():
            sample_out = backbone(graph_data)
        if sample_out.size(1) != hidden:
            adapter = nn.Linear(sample_out.size(1), hidden).to(device)
        else:
            adapter = nn.Identity().to(device)
        class_head = NodeClassificationHead(hidden, num_classes).to(device)
        proj_head = ContrastiveProjectionHead(hidden, hidden).to(device)

        self._cr._backbone = backbone
        self._cr._adapter = adapter
        self._cr._proj_head = proj_head
        self._cr._class_head = class_head
        self._cr._graph_data = graph_data
        self._cr._clean_labels = getattr(graph_data, 'y_original', graph_data.y)

    def profile_flops(self):
        from util.profiling import profile_model_flops
        cr = self._cr
        backbone = cr._backbone
        adapter = cr._adapter
        class_head = cr._class_head
        data = cr._graph_data

        adapter.eval()
        class_head.eval()

        def fwd():
            h = backbone(Data(x=data.x, edge_index=data.edge_index))
            h = adapter(h)
            return class_head(h)

        return profile_model_flops(backbone, data, cr.device, forward_fn=fwd)

    def profile_training_step(self):
        from util.profiling import profile_training_step_flops
        cr = self._cr
        backbone = cr._backbone
        adapter = cr._adapter
        proj_head = cr._proj_head
        class_head = cr._class_head
        data = cr._graph_data
        noisy_labels = data.y
        train_mask = data.train_mask

        def step_fn():
            edge_idx1, _ = dropout_edge(data.edge_index, p=cr.pr, training=True)
            edge_idx2, _ = dropout_edge(data.edge_index, p=cr.pr, training=True)
            x1, _ = mask_feature(data.x, p=cr.pr)
            x2, _ = mask_feature(data.x, p=cr.pr)

            h1 = adapter(backbone(Data(x=x1, edge_index=edge_idx1)))
            h2 = adapter(backbone(Data(x=x2, edge_index=edge_idx2)))

            z1 = proj_head(h1)
            z2 = proj_head(h2)
            loss_con = contrastive_loss_original_style(z1, z2, cr.tau)

            p1 = class_head(h1)
            p2 = class_head(h2)
            loss_sup = F.nll_loss(p1[train_mask], noisy_labels[train_mask])

            return cr.alpha * loss_con + loss_sup

        models = [backbone, adapter, proj_head, class_head]
        return profile_training_step_flops(models, cr.device, step_fn)

    def evaluate(self):
        cr = self._cr
        backbone = cr._backbone
        adapter = cr._adapter
        class_head = cr._class_head
        graph_data = cr._graph_data
        clean_labels = cr._clean_labels

        backbone.eval()
        adapter.eval()
        class_head.eval()
        with torch.no_grad():
            def get_predictions():
                h = backbone(Data(x=graph_data.x, edge_index=graph_data.edge_index))
                h = adapter(h)
                return class_head(h).exp().argmax(dim=1)

            def get_embeddings():
                return backbone.get_embeddings(Data(x=graph_data.x, edge_index=graph_data.edge_index))

            return evaluate_model(
                get_predictions, get_embeddings, clean_labels,
                graph_data.train_mask, graph_data.val_mask, graph_data.test_mask,
                graph_data.edge_index, cr.device,
            )
