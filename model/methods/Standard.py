import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from model.evaluation import (OversmoothingMetrics, ClassificationMetrics,
                              compute_oversmoothing_for_mask)
from model.base import BaseTrainer
from model.registry import register

def train_with_standard_loss(
    model, data, noisy_indices, device,
    total_epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    patience=20,
    debug=True,
    oversmoothing_every=20,
    log_epoch_fn=None,
):

    per_epochs_oversmoothing = defaultdict(list)
    per_epochs_val_oversmoothing = defaultdict(list)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    oversmoothing_evaluator = OversmoothingMetrics(device=device)
    cls_evaluator = ClassificationMetrics(average='macro')

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(total_epochs):
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

            val_loss = criterion(out[val_idx], data.y[val_idx])

            pred = out.argmax(dim=1)
            train_acc = (pred[train_idx] == data.y[train_idx]).sum().item() / len(train_idx)
            val_acc = (pred[val_idx] == data.y[val_idx]).sum().item() / len(val_idx)
            train_f1 = cls_evaluator.compute_f1(pred[train_idx], data.y[train_idx])
            val_f1 = cls_evaluator.compute_f1(pred[val_idx], data.y[val_idx])

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        os_entry = None
        if debug and (epoch + 1) % oversmoothing_every == 0:
            with torch.no_grad():
                embeddings = model.get_embeddings(data)
            train_oversmoothing = compute_oversmoothing_for_mask(
                oversmoothing_evaluator, embeddings, data.edge_index, data.train_mask
            )
            val_oversmoothing = compute_oversmoothing_for_mask(
                oversmoothing_evaluator, embeddings, data.edge_index, data.val_mask
            )

            for key, value in train_oversmoothing.items():
                per_epochs_oversmoothing[key].append(value)
            for key, value in val_oversmoothing.items():
                per_epochs_val_oversmoothing[key].append(value)

            os_entry = {'train': dict(train_oversmoothing), 'val': dict(val_oversmoothing)}

            print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
            print(f"Train DE: {train_oversmoothing['EDir']:.4f}, Val DE: {val_oversmoothing['EDir']:.4f} | "
                  f"Train DE_trad: {train_oversmoothing['EDir_traditional']:.4f}, Val DE_trad: {val_oversmoothing['EDir_traditional']:.4f} | "
                  f"Train EProj: {train_oversmoothing['EProj']:.4f}, Val EProj: {val_oversmoothing['EProj']:.4f} | "
                  f"Train MAD: {train_oversmoothing['MAD']:.4f}, Val MAD: {val_oversmoothing['MAD']:.4f} | "
                  f"Train NumRank: {train_oversmoothing['NumRank']:.4f}, Val NumRank: {val_oversmoothing['NumRank']:.4f} | "
                  f"Train Erank: {train_oversmoothing['Erank']:.4f}, Val Erank: {val_oversmoothing['Erank']:.4f}")

        if log_epoch_fn is not None:
            log_epoch_fn(epoch, loss_train, val_loss, train_acc, val_acc,
                         train_f1=train_f1, val_f1=val_f1,
                         oversmoothing=os_entry, is_best=is_best)

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        'train_oversmoothing': dict(per_epochs_oversmoothing),
        'val_oversmoothing': dict(per_epochs_val_oversmoothing),
        'stopped_at_epoch': epoch,
    }


@register('standard')
class StandardMethodTrainer(BaseTrainer):
    def train(self):
        d = self.init_data
        return train_with_standard_loss(
            d['backbone_model'], d['data_for_training'],
            d['global_noisy_indices'], d['device'],
            total_epochs=d['epochs'], lr=d['lr'],
            weight_decay=d['weight_decay'], patience=d['patience'],
            oversmoothing_every=d['oversmoothing_every'],
            log_epoch_fn=self.log_epoch,
        )
