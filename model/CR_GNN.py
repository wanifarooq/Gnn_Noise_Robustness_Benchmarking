import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, mask_feature
from torch_geometric.data import Data
from copy import deepcopy
import time
from sklearn.metrics import accuracy_score, f1_score

from model.evaluation import OversmoothingMetrics

class ProjectionHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ClassificationHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)

def contrastive_loss(z1, z2, tau: float):
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    logits = (z1 @ z2.t()) / tau
    labels = torch.arange(z1.size(0), device=z1.device)

    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)

def dynamic_cross_entropy_loss(p1, p2, labels):
    loss1 = F.cross_entropy(p1, labels)
    loss2 = F.cross_entropy(p2, labels)
    return (loss1 + loss2) / 2

def cross_space_consistency_loss(zm, pm):
    return F.mse_loss(zm, pm)

class CRGNNTrainer:
    
    def __init__(self, device='cuda', **kwargs):
        self.device = torch.device(device)
        self.config = {
            'hidden_channels': kwargs.get('hidden_channels', 64),
            'lr': kwargs.get('lr', 0.001),
            'weight_decay': kwargs.get('weight_decay', 5e-4),
            'epochs': kwargs.get('epochs', 200),
            'patience': kwargs.get('patience', 10),
            'T': kwargs.get('T', 0.5),
            'tau': kwargs.get('tau', 0.5),
            'p': kwargs.get('p', 0.5),
            'alpha': kwargs.get('alpha', 1.0),
            'beta': kwargs.get('beta', 0.0),
            'debug': kwargs.get('debug', True),
            'eval_oversmoothing_freq': kwargs.get('eval_oversmoothing_freq', 10)
        }

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_weights = None

        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        
        self.oversmoothing_history = {
            'train': [],
            'val': [],
            'test': []
        }
    
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
    
    def get_prediction(self, encoder, projection_adapter, projection_head, classifier,
                    features, edge_index, labels=None, mask=None):
        h = encoder(Data(x=features, edge_index=edge_index))
        h = projection_adapter(h)
        output = h
        
        loss, acc, f1, de = None, None, None, None
        oversmoothing_metrics = None
        
        if labels is not None and mask is not None:
            edge_index1, _ = dropout_adj(edge_index, p=0.3)
            edge_index2, _ = dropout_adj(edge_index, p=0.3)
            x1, _ = mask_feature(features, p=0.3)
            x2, _ = mask_feature(features, p=0.3)

            h1 = encoder(Data(x=x1, edge_index=edge_index1))
            h1 = projection_adapter(h1)
            h2 = encoder(Data(x=x2, edge_index=edge_index2))
            h2 = projection_adapter(h2)

            z1 = projection_head(h1)
            z2 = projection_head(h2)
            loss_con = contrastive_loss(z1, z2, self.config['tau'])

            p1 = classifier(h1)
            p2 = classifier(h2)
            loss_sup = dynamic_cross_entropy_loss(p1[mask], p2[mask], labels[mask])

            if self.config['beta'] > 0:
                zm = torch.exp(F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2) / self.config['T']).mean(dim=1)
                pm = torch.exp(F.cosine_similarity(p1.unsqueeze(1), p2.unsqueeze(0), dim=2) / self.config['T']).mean(dim=1)
                pm = torch.where(pm > self.config['p'], pm, torch.zeros_like(pm))
                loss_ccon = cross_space_consistency_loss(zm, pm)
                loss = self.config['alpha'] * loss_con + loss_sup + self.config['beta'] * loss_ccon
            else:
                loss = self.config['alpha'] * loss_con + loss_sup

            with torch.no_grad():
                pred_output = classifier(output)
                pred_labels = pred_output[mask].argmax(dim=1)
                acc = accuracy_score(labels[mask].cpu().numpy(), pred_labels.cpu().numpy())
                f1 = f1_score(labels[mask].cpu().numpy(), pred_labels.cpu().numpy(), average='macro')

                de = None

                oversmoothing_metrics = self._compute_oversmoothing_for_mask(output, edge_index, mask, labels)

        return output, loss, acc, f1, de, oversmoothing_metrics

    def evaluate(self, encoder, projection_adapter, classifier, features, edge_index, labels, mask):
        encoder.eval()
        projection_adapter.eval()
        classifier.eval()
        
        with torch.no_grad():
            h = encoder(Data(x=features, edge_index=edge_index))
            h = projection_adapter(h)
            output = classifier(h)
            
            loss = F.cross_entropy(output[mask], labels[mask])
            pred_labels = output[mask].argmax(dim=1)
            acc = accuracy_score(labels[mask].cpu().numpy(), pred_labels.cpu().numpy())
            f1 = f1_score(labels[mask].cpu().numpy(), pred_labels.cpu().numpy(), average='macro')

            de = None

            oversmoothing_metrics = self._compute_oversmoothing_for_mask(h, edge_index, mask, labels)

        return loss, acc, f1, de, oversmoothing_metrics

    def fit(self, base_model, data, config, get_model_func):
        print(f"Training CR-GNN with {config['model_name'].upper()} backbone")
    
        if data.num_nodes > 10000:
            print(f"Warning: large dataset with {data.num_nodes} nodes.")
        
        data = data.to(self.device)
        n_classes = data.y.max().item() + 1

        encoder = base_model.to(self.device)

        with torch.no_grad():
            tmp_out = encoder(data)
        out_dim = tmp_out.size(1)
        hidden_dim = self.config['hidden_channels']
        
        if out_dim != hidden_dim:
            projection_adapter = nn.Linear(out_dim, hidden_dim).to(self.device)
            final_dim = hidden_dim
        else:
            projection_adapter = nn.Identity().to(self.device)
            final_dim = out_dim

        projection_head = ProjectionHead(final_dim, final_dim).to(self.device)
        classifier = ClassificationHead(final_dim, n_classes).to(self.device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + 
            list(projection_adapter.parameters()) +
            list(projection_head.parameters()) +
            list(classifier.parameters()),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        if hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask'):
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        else:
            num_nodes = data.x.size(0)
            train_size = int(0.6 * num_nodes)
            val_size = int(0.2 * num_nodes)
            
            indices = torch.randperm(num_nodes)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True
            
            train_mask = train_mask.to(self.device)
            val_mask = val_mask.to(self.device)
            test_mask = test_mask.to(self.device)
        
        clean_labels = getattr(data, 'y_original', data.y)
        noisy_labels = data.y
        
        start_time = time.time()
        total_time = 0
        
        for epoch in range(self.config['epochs']):
            encoder.train()
            projection_adapter.train()
            projection_head.train()
            classifier.train()

            optimizer.zero_grad()
            
            output, train_loss, train_acc, train_f1, train_de, train_oversmoothing = self.get_prediction(
                encoder, projection_adapter, projection_head, classifier,
                data.x, data.edge_index, noisy_labels, train_mask
            )

            if train_loss is not None and torch.isfinite(train_loss):
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) +
                    list(projection_adapter.parameters()) +
                    list(projection_head.parameters()) +
                    list(classifier.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            val_loss, val_acc, val_f1, val_de, val_oversmoothing = self.evaluate(
                encoder, projection_adapter, classifier,
                data.x, data.edge_index, noisy_labels, val_mask
            )

            if train_oversmoothing is not None:
                self.oversmoothing_history['train'].append(train_oversmoothing)
            if val_oversmoothing is not None:
                self.oversmoothing_history['val'].append(val_oversmoothing)

            if self.config['debug']:
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
                
                print(f"Epoch {epoch+1:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                      f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
                print(f"Train DE: {train_de:.4f}, Val DE: {val_de:.4f} | "
                      f"Train DE_trad: {train_de_traditional:.4f}, Val DE_trad: {val_de_traditional:.4f} | "
                      f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                      f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                      f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                      f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                total_time = time.time() - start_time
                self.best_weights = {
                    'encoder': deepcopy(encoder.state_dict()),
                    'projection_adapter': deepcopy(projection_adapter.state_dict()),
                    'projection_head': deepcopy(projection_head.state_dict()),
                    'classifier': deepcopy(classifier.state_dict())
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['patience']:
                    break
        
        if self.best_weights is not None:
            encoder.load_state_dict(self.best_weights['encoder'])
            projection_adapter.load_state_dict(self.best_weights['projection_adapter'])
            projection_head.load_state_dict(self.best_weights['projection_head'])
            classifier.load_state_dict(self.best_weights['classifier'])

        final_train_loss, final_train_acc, final_train_f1, final_train_de, final_train_oversmoothing = self.evaluate(
            encoder, projection_adapter, classifier,
            data.x, data.edge_index, clean_labels, train_mask
        )

        final_val_loss, final_val_acc, final_val_f1, final_val_de, final_val_oversmoothing = self.evaluate(
            encoder, projection_adapter, classifier,
            data.x, data.edge_index, clean_labels, val_mask
        )

        test_loss, test_acc, test_f1, test_de, test_oversmoothing = self.evaluate(
            encoder, projection_adapter, classifier,
            data.x, data.edge_index, clean_labels, test_mask
        )

        if test_oversmoothing is not None:
            self.oversmoothing_history['test'].append(test_oversmoothing)

        final_metrics = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1
        }

        if self.config['debug']:
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
            
            if test_oversmoothing is not None:
                print(f"Test: EDir: {test_oversmoothing['EDir']:.4f}, EDir_traditional: {test_oversmoothing['EDir_traditional']:.4f}, "
                      f"EProj: {test_oversmoothing['EProj']:.4f}, MAD: {test_oversmoothing['MAD']:.4f}, "
                      f"NumRank: {test_oversmoothing['NumRank']:.4f}, Erank: {test_oversmoothing['Erank']:.4f}")
        
        return test_acc
    
    def get_oversmoothing_history(self):
        return self.oversmoothing_history