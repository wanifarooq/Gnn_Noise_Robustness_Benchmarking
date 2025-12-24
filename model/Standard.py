import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict

from model.evaluation import OversmoothingMetrics

def _compute_oversmoothing_for_mask(oversmoothing_evaluator, embeddings, edge_index, mask, labels=None):
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
        
        return oversmoothing_evaluator.compute_all_metrics(
            X=mask_embeddings,
            edge_index=remapped_edges,
            graphs_in_class=graphs_in_class
        )
        
    except Exception as e:
        print(f"Warning: Could not compute oversmoothing metrics for mask: {e}")
        return {
            'NumRank': 0.0,
            'Erank': 0.0,
            'EDir': 0.0,
            'EDir_traditional': 0.0,
            'EProj': 0.0,
            'MAD': 0.0
        }

def train_with_standard_loss(
    model, data, noisy_indices, device,
    total_epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    patience=20,
    debug=True
):

    per_epochs_oversmoothing = defaultdict(list)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    oversmoothing_evaluator = OversmoothingMetrics(device=device)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        loss_train = criterion(out[train_idx], data.y[train_idx])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_idx = data.val_mask.nonzero(as_tuple=True)[0]
            test_idx = data.test_mask.nonzero(as_tuple=True)[0]

            val_loss = criterion(out[val_idx], data.y[val_idx])

            pred = out.argmax(dim=1)
            train_acc = (pred[train_idx] == data.y[train_idx]).sum().item() / len(train_idx)
            val_acc = (pred[val_idx] == data.y[val_idx]).sum().item() / len(val_idx)
            train_f1 = f1_score(data.y[train_idx].cpu(), pred[train_idx].cpu(), average='macro')
            val_f1 = f1_score(data.y[val_idx].cpu(), pred[val_idx].cpu(), average='macro')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break

        if debug and epoch % 20 == 0:
            
            train_oversmoothing = _compute_oversmoothing_for_mask(
                oversmoothing_evaluator, out, data.edge_index, data.train_mask, data.y
            )
            val_oversmoothing = _compute_oversmoothing_for_mask(
                oversmoothing_evaluator, out, data.edge_index, data.val_mask, data.y
            )

            for key, value in train_oversmoothing.items():
                per_epochs_oversmoothing[key].append(value)
            train_de = train_oversmoothing['EDir']
            train_de_traditional = train_oversmoothing['EDir_traditional']
            train_eproj = train_oversmoothing['EProj']
            train_mad = train_oversmoothing['MAD']
            train_num_rank = train_oversmoothing['NumRank']
            train_eff_rank = train_oversmoothing['Erank']

            val_de = val_oversmoothing['EDir']
            val_de_traditional = val_oversmoothing['EDir_traditional']
            val_eproj = val_oversmoothing['EProj']
            val_mad = val_oversmoothing['MAD']
            val_num_rank = val_oversmoothing['NumRank']
            val_eff_rank = val_oversmoothing['Erank']

            print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
            print(f"Train DE: {train_de:.4f}, Val DE: {val_de:.4f} | "
                  f"Train DE_trad: {train_de_traditional:.4f}, Val DE_trad: {val_de_traditional:.4f} | "
                  f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
                  f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
                  f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
                  f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")
    model.eval()
    with torch.no_grad():
        test_idx = data.test_mask.nonzero(as_tuple=True)[0]
        test_loss = criterion(out[test_idx], data.y[test_idx])
        pred = out.argmax(dim=1)
        test_acc = (pred[test_idx] == data.y[test_idx]).sum().item() / len(test_idx)
        test_f1 = f1_score(data.y[test_idx].cpu(), pred[test_idx].cpu(), average='macro')
        test_precision = precision_score(data.y[test_idx].cpu(), pred[test_idx].cpu(), average='macro')
        test_recall = recall_score(data.y[test_idx].cpu(), pred[test_idx].cpu(), average='macro')

        final_train_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.train_mask, data.y
        )
        final_val_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.val_mask, data.y
        )
        test_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.test_mask, data.y
        )

    expected_keys = ['NumRank', 'Erank', 'EDir', 'EDir_traditional', 'EProj', 'MAD']

    def normalize_metrics(d):
        if d is None:
            return {k: 0.0 for k in expected_keys}
        return {k: d.get(k, 0.0) for k in expected_keys}

    final_train_oversmoothing = normalize_metrics(final_train_oversmoothing)
    final_val_oversmoothing = normalize_metrics(final_val_oversmoothing)
    final_test_oversmoothing = normalize_metrics(test_oversmoothing)

    if debug:
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | "
              f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        print("Final Oversmoothing Metrics:")
        print(f"Train: {final_train_oversmoothing}")
        print(f"Val: {final_val_oversmoothing}")
        print(f"Test: {final_test_oversmoothing}")

    results = {
        'accuracy': torch.tensor(test_acc),
        'f1': torch.tensor(test_f1),
        'precision': torch.tensor(test_precision),
        'recall': torch.tensor(test_recall),
        'oversmoothing': final_test_oversmoothing,
        'train_oversmoothing' : per_epochs_oversmoothing
    }

    return results
