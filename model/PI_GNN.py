import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import f1_score
from copy import deepcopy
import time

from model.evaluation import OversmoothingMetrics

def accuracy(output, labels):
    preds = output.argmax(dim=1)
    return (preds == labels).float().mean().item()

class InnerProductDecoder(nn.Module):
    def __init__(self, act=lambda x: x):
        super().__init__()
        self.act = act
    
    def forward(self, z):
        adj_pred = self.act(torch.mm(z, z.t()))
        return adj_pred

class Net(nn.Module):
    def __init__(self, gnn_model, supplementary_gnn=None):
        super().__init__()
        self.gnn = gnn_model
        self.supplementary_gnn = supplementary_gnn
    
    def forward(self, data):
        embeddings = self.gnn(data)
        out_supp = self.supplementary_gnn(embeddings) if self.supplementary_gnn is not None else None
        return F.log_softmax(embeddings, dim=1), out_supp

class InnerProductTrainer:
    def __init__(self, device, epochs=400, start_epoch=200, miself=False, 
                lr_main=0.01, lr_mi=0.01, weight_decay=5e-4, norm=None, 
                vanilla=False, patience=50, delta=1e-4):
        self.device = device
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.miself = miself
        self.lr_main = lr_main
        self.lr_mi = lr_mi
        self.weight_decay = weight_decay
        self.norm = norm
        self.vanilla = vanilla
        self.patience = patience
        self.delta = delta
        
        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        self.oversmoothing_history = {
            'train': [],
            'val': [],
            'test': []
        }

    def _compute_oversmoothing_for_mask(self, embeddings, edge_index, mask):
        try:
            if hasattr(mask, 'nonzero'):
                mask_indices = mask.nonzero(as_tuple=True)[0]
            else:
                mask_indices = torch.tensor(mask, device=self.device)
            
            mask_embeddings = embeddings[mask_indices]
            
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
            print(f"Warning: Could not compute oversmoothing metrics: {e}")
            return {
                'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0
            }

    def get_model_lr(self, config, model_name):
        if model_name.lower() in ['gat', 'gatv2']:
            return config.get('lr', 0.005)
        else:
            return config.get('lr', 0.01)

    def compute_metrics_and_de(self, model, data):
        model.eval()
        with torch.no_grad():
            out_train, _ = model(data)
            pred_train = out_train.argmax(dim=1)
            
            metrics = {}

            metrics['train_loss'] = F.nll_loss(out_train[data.train_mask], data.y[data.train_mask]).item()
            metrics['train_acc'] = pred_train[data.train_mask].eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
            metrics['train_f1'] = f1_score(data.y[data.train_mask].cpu(), pred_train[data.train_mask].cpu(), average='micro')
            
            metrics['val_loss'] = F.nll_loss(out_train[data.val_mask], data.y[data.val_mask]).item()
            metrics['val_acc'] = pred_train[data.val_mask].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
            metrics['val_f1'] = f1_score(data.y[data.val_mask].cpu(), pred_train[data.val_mask].cpu(), average='micro')

            hidden_embeddings = model.gnn(data)
            train_oversmoothing = self._compute_oversmoothing_for_mask(hidden_embeddings, data.edge_index, data.train_mask)
            val_oversmoothing = self._compute_oversmoothing_for_mask(hidden_embeddings, data.edge_index, data.val_mask)
            
            return metrics, train_oversmoothing, val_oversmoothing

    def compute_final_test_metrics(self, model, data):
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        model.eval()
        with torch.no_grad():
            out, _ = model(data)

            train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]).item()
            val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()
            test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask]).item()
            
            train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
            val_acc = accuracy(out[data.val_mask], data.y[data.val_mask])
            test_acc = accuracy(out[data.test_mask], data.y[data.test_mask])

            y_true_test = data.y[data.test_mask].cpu().numpy()
            y_pred_test = out[data.test_mask].argmax(dim=1).cpu().numpy()
            
            test_f1 = f1_score(y_true_test, y_pred_test, average='macro')
            test_precision = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
            test_recall = recall_score(y_true_test, y_pred_test, average='macro', zero_division=0)
            
            train_oversmoothing = self._compute_oversmoothing_for_mask(out, data.edge_index, data.train_mask)
            val_oversmoothing = self._compute_oversmoothing_for_mask(out, data.edge_index, data.val_mask)
            test_oversmoothing = self._compute_oversmoothing_for_mask(out, data.edge_index, data.test_mask)
            
            final_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall
            }
            
            return final_metrics, train_oversmoothing, val_oversmoothing, test_oversmoothing

    def fit(self, model, data, config=None, get_model_func=None):
        start_time = time.time()
        
        data = data.to(self.device)
        model = model.to(self.device)
        num_classes = data.y.max().item() + 1

        if get_model_func and config:
            gnn_mi = get_model_func(
                model_name=config['model_name'],
                in_channels=data.num_features,
                hidden_channels=config.get('hidden_channels', 64),
                out_channels=num_classes,
                n_layers=config.get('n_layers', 2),
                dropout=config.get('dropout', 0.5),
                mlp_layers=config.get('mlp_layers', 2),
                train_eps=config.get('train_eps', True),
                heads=config.get('heads', 8),
                concat=config.get('concat', True),
                self_loop=config.get('self_loop', True)
            )
            decoder_mi = InnerProductDecoder()
            model_mi = Net(gnn_model=gnn_mi, supplementary_gnn=decoder_mi).to(self.device)
            lr_main = self.get_model_lr(config, config['model_name'])
            lr_mi = self.get_model_lr(config, config['model_name'])
        else:
            model_mi = Net(
                gnn_model=type(model.gnn)(
                    in_channels=data.num_features,
                    hidden_channels=64,
                    out_channels=num_classes
                ),
                supplementary_gnn=InnerProductDecoder()
            ).to(self.device)
            lr_main = self.lr_main
            lr_mi = self.lr_mi

        optimizer = torch.optim.Adam(model.parameters(), lr=lr_main, weight_decay=self.weight_decay)
        optimizer_mi = torch.optim.Adam(model_mi.parameters(), lr=lr_mi, weight_decay=self.weight_decay)

        train_edges = data.edge_index.t().cpu().numpy()
        data_context = np.ones(train_edges.shape[0])
        adj_train = sp.csr_matrix(
            (data_context, (train_edges[:, 0], train_edges[:, 1])),
            shape=(data.num_nodes, data.num_nodes)
        )
        adj_train = (adj_train + adj_train.T) / 2
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray()).to(self.device)
        pos_weight = torch.tensor(
            [float(data.num_nodes ** 2 - len(data_context)) / len(data_context)], 
            device=self.device
        )
        norm = self.norm if self.norm is not None and self.norm != 10000 else \
              data.num_nodes ** 2 / float((data.num_nodes ** 2 - len(data_context)) * 2)

        corrupted_labels = data.y[data.train_mask]

        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            model.train()
            model_mi.train()
            optimizer.zero_grad()
            optimizer_mi.zero_grad()

            out, out_product = model(data)
            out_mi, out_product_mi = model_mi(data)

            loss_train = F.nll_loss(out[data.train_mask], corrupted_labels)
            loss_mi = norm * F.binary_cross_entropy_with_logits(out_product_mi, adj_label, pos_weight=pos_weight)
            loss_mi.backward()
            optimizer_mi.step()

            loss_context = 0
            if not self.vanilla:
                if epoch > self.start_epoch:
                    mask = torch.zeros_like(out_product_mi).view(-1).to(self.device)
                    pos_position = adj_label.view(-1).bool()
                    neg_position = (~adj_label.view(-1).bool())
                    if self.miself:
                        predictions = out_product
                        mask[pos_position] = torch.sigmoid(predictions).view(-1)[pos_position]
                        mask[neg_position] = 1 - torch.sigmoid(predictions).view(-1)[neg_position]
                    else:
                        mask[pos_position] = torch.sigmoid(out_product_mi).view(-1)[pos_position]
                        mask[neg_position] = 1 - torch.sigmoid(out_product_mi).view(-1)[neg_position]
                    mask = mask.view(adj_label.size(0), adj_label.size(1))
                    loss_context = norm * (F.binary_cross_entropy_with_logits(out_product, adj_label, pos_weight=pos_weight, reduction='none') * mask.detach()).mean()
                else:
                    loss_context = norm * F.binary_cross_entropy_with_logits(out_product, adj_label, pos_weight=pos_weight)

            total_loss = loss_train + loss_context
            total_loss.backward()
            optimizer.step()

            metrics, train_oversmoothing, val_oversmoothing = self.compute_metrics_and_de(model, data)

            if train_oversmoothing is not None:
                self.oversmoothing_history['train'].append(train_oversmoothing)
            if val_oversmoothing is not None:
                self.oversmoothing_history['val'].append(val_oversmoothing)

            if epoch%20 == 0:
                train_edir = train_oversmoothing['EDir'] if train_oversmoothing else 0.0
                train_edir_trad = train_oversmoothing['EDir_traditional'] if train_oversmoothing else 0.0
                train_eproj = train_oversmoothing['EProj'] if train_oversmoothing else 0.0
                train_mad = train_oversmoothing['MAD'] if train_oversmoothing else 0.0
                train_num_rank = train_oversmoothing['NumRank'] if train_oversmoothing else 0.0
                train_eff_rank = train_oversmoothing['Erank'] if train_oversmoothing else 0.0
                
                val_edir = val_oversmoothing['EDir'] if val_oversmoothing else 0.0
                val_edir_trad = val_oversmoothing['EDir_traditional'] if val_oversmoothing else 0.0
                val_eproj = val_oversmoothing['EProj'] if val_oversmoothing else 0.0
                val_mad = val_oversmoothing['MAD'] if val_oversmoothing else 0.0
                val_num_rank = val_oversmoothing['NumRank'] if val_oversmoothing else 0.0
                val_eff_rank = val_oversmoothing['Erank'] if val_oversmoothing else 0.0

                print(f"Epoch {epoch:03d} | Train Loss: {metrics['train_loss']:.4f}, Val Loss: {metrics['val_loss']:.4f} | "
                      f"Train Acc: {metrics['train_acc']:.4f}, Val Acc: {metrics['val_acc']:.4f} | "
                      f"Train F1: {metrics['train_f1']:.4f}, Val F1: {metrics['val_f1']:.4f}")
                print(f"Train DE: {train_edir:.4f}, Val DE: {val_edir:.4f} | "
                      f"Train DE_trad: {train_edir_trad:.4f}, Val DE_trad: {val_edir_trad:.4f} | "
                      f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                      f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                      f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                      f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")

            if metrics['val_loss'] < best_val_loss - self.delta:
                best_val_loss = metrics['val_loss']
                best_epoch = epoch
                patience_counter = 0
                best_model_state = deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}")
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        results = self.compute_final_test_metrics(model, data)
        final_metrics, final_train_oversmoothing, final_val_oversmoothing, final_test_oversmoothing = results

        if final_test_oversmoothing is not None:
            self.oversmoothing_history['test'].append(final_test_oversmoothing)

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

        train_dirichlet = final_train_oversmoothing.get('EDir', 0.0) if final_train_oversmoothing else 0.0
        val_dirichlet = final_val_oversmoothing.get('EDir', 0.0) if final_val_oversmoothing else 0.0
        test_dirichlet = final_test_oversmoothing.get('EDir', 0.0) if final_test_oversmoothing else 0.0

        print(f"Test Precision: {final_metrics['test_precision']:.4f} | Test Recall: {final_metrics['test_recall']:.4f}")
        return {
            'accuracy': final_metrics['test_acc'],
            'f1': final_metrics['test_f1'],
            'precision': final_metrics['test_precision'],
            'recall': final_metrics['test_recall'],
            'oversmoothing': final_test_oversmoothing if final_test_oversmoothing is not None else {
                'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0
            }
        }


    def get_oversmoothing_history(self):
        return self.oversmoothing_history