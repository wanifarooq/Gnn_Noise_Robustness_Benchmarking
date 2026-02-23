import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

from model.evaluation import (OversmoothingMetrics, ClassificationMetrics,
                              compute_oversmoothing_for_mask, evaluate_model)

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
    cls_evaluator = ClassificationMetrics(average='macro')

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
            train_f1 = cls_evaluator.compute_f1(pred[train_idx], data.y[train_idx])
            val_f1 = cls_evaluator.compute_f1(pred[val_idx], data.y[val_idx])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break

        if debug and epoch % 20 == 0:
            train_oversmoothing = compute_oversmoothing_for_mask(
                oversmoothing_evaluator, out, data.edge_index, data.train_mask
            )
            val_oversmoothing = compute_oversmoothing_for_mask(
                oversmoothing_evaluator, out, data.edge_index, data.val_mask
            )

            for key, value in train_oversmoothing.items():
                per_epochs_oversmoothing[key].append(value)

            print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
            print(f"Train DE: {train_oversmoothing['EDir']:.4f}, Val DE: {val_oversmoothing['EDir']:.4f} | "
                  f"Train DE_trad: {train_oversmoothing['EDir_traditional']:.4f}, Val DE_trad: {val_oversmoothing['EDir_traditional']:.4f} | "
                  f"Train EProj: {train_oversmoothing['EProj']:.4f}, Val EProj: {val_oversmoothing['EProj']:.4f} | "
                  f"Train MAD: {train_oversmoothing['MAD']:.4f}, Val MAD: {val_oversmoothing['MAD']:.4f} | "
                  f"Train NumRank: {train_oversmoothing['NumRank']:.4f}, Val NumRank: {val_oversmoothing['NumRank']:.4f} | "
                  f"Train Erank: {train_oversmoothing['Erank']:.4f}, Val Erank: {val_oversmoothing['Erank']:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        get_predictions = lambda: model(data).argmax(dim=1)
        get_embeddings = lambda: model(data)
        results = evaluate_model(
            get_predictions, get_embeddings, data.y,
            data.train_mask, data.val_mask, data.test_mask,
            data.edge_index, device
        )

    results['train_oversmoothing'] = per_epochs_oversmoothing

    if debug:
        print(f"Test Acc: {results['accuracy']:.4f} | Test F1: {results['f1']:.4f} | "
              f"Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}")
        print(f"Test Oversmoothing: {results['oversmoothing']}")

    return results
