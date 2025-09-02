import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

from model.evaluation import OversmoothingMetrics

def dirichlet_energy(x, edge_index):
    row, col = edge_index
    diff = x[row] - x[col]
    return (diff ** 2).sum(dim=1).mean()

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

def train_with_standard_loss(model, data, noisy_indices, device, total_epochs=200, patience=20, debug=True):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
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

            train_oversmoothing = _compute_oversmoothing_for_mask(
                oversmoothing_evaluator, out, data.edge_index, data.train_mask, data.y
            )
            val_oversmoothing = _compute_oversmoothing_for_mask(
                oversmoothing_evaluator, out, data.edge_index, data.val_mask, data.y
            )

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

        if debug:
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

        final_train_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.train_mask, data.y
        )
        final_val_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.val_mask, data.y
        )
        test_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.test_mask, data.y
        )

    if debug:
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
        print("Final Oversmoothing Metrics:")
        
        print(f"Train: EDir: {final_train_oversmoothing['EDir']:.4f}, EDir_traditional: {final_train_oversmoothing['EDir_traditional']:.4f}, "
              f"EProj: {final_train_oversmoothing['EProj']:.4f}, MAD: {final_train_oversmoothing['MAD']:.4f}, "
              f"NumRank: {final_train_oversmoothing['NumRank']:.4f}, Erank: {final_train_oversmoothing['Erank']:.4f}")
        
        print(f"Val: EDir: {final_val_oversmoothing['EDir']:.4f}, EDir_traditional: {final_val_oversmoothing['EDir_traditional']:.4f}, "
              f"EProj: {final_val_oversmoothing['EProj']:.4f}, MAD: {final_val_oversmoothing['MAD']:.4f}, "
              f"NumRank: {final_val_oversmoothing['NumRank']:.4f}, Erank: {final_val_oversmoothing['Erank']:.4f}")
        
        print(f"Test: EDir: {test_oversmoothing['EDir']:.4f}, EDir_traditional: {test_oversmoothing['EDir_traditional']:.4f}, "
              f"EProj: {test_oversmoothing['EProj']:.4f}, MAD: {test_oversmoothing['MAD']:.4f}, "
              f"NumRank: {test_oversmoothing['NumRank']:.4f}, Erank: {test_oversmoothing['Erank']:.4f}")

    return test_acc


cross_entropy_val = nn.CrossEntropyLoss
mean = 1e-8
std = 1e-9

class ncodLoss(nn.Module):
    def __init__(self, sample_labels, device, num_examp=50000, num_classes=100, ratio_consistency=0, ratio_balance=0, encoder_features=64, total_epochs=100):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.USE_CUDA = torch.cuda.is_available()
        self.num_examp = num_examp
        self.total_epochs = total_epochs
        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance
        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.init_param(mean=mean, std=std)
        self.beginning = True
        self.prevSimilarity = torch.rand((num_examp, encoder_features), device=device)
        self.masterVector = torch.rand((num_classes, encoder_features), device=device)
        self.take = torch.zeros((num_examp, 1), device=device)
        self.weight = torch.zeros((num_examp, 1), device=device)
        self.sample_labels = sample_labels
        self.bins = [np.where(self.sample_labels == i)[0] for i in range(num_classes)]
        self.shuffledbins = copy.deepcopy(self.bins)
        for sublist in self.shuffledbins:
            random.shuffle(sublist)

    def init_param(self, mean=1e-8, std=1e-9):
        torch.nn.init.normal_(self.u, mean=mean, std=std)

    def forward(self, index, outputs, label, out, flag, epoch, train_acc_cater):
        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
            out1, out2 = torch.chunk(out, 2)
        else:
            output = outputs
            out1 = out

        eps = 1e-4
        u = self.u[index]
        weight = self.weight[index]

        if flag == 0:
            if self.beginning:
                percent = 100
                for i in range(len(self.bins)):
                    class_u = self.u.detach()[self.bins[i]]
                    bottomK = int((len(class_u) / 100) * percent)
                    important_indexs = torch.topk(class_u, bottomK, largest=False, dim=0)[1]
                    self.masterVector[i] = torch.mean(self.prevSimilarity[self.bins[i]][important_indexs.view(-1)], dim=0)

            masterVector_norm = self.masterVector.norm(p=2, dim=1, keepdim=True)
            masterVector_normalized = self.masterVector.div(masterVector_norm)
            self.masterVector_transpose = torch.transpose(masterVector_normalized, 0, 1)
            self.beginning = True

        self.prevSimilarity[index] = out1.detach()
        prediction = F.softmax(output, dim=1)
        out_norm = out1.detach().norm(p=2, dim=1, keepdim=True)
        out_normalized = out1.detach().div(out_norm)
        similarity = torch.mm(out_normalized, self.masterVector_transpose)
        similarity = similarity * label
        sim_mask = (similarity > 0.000).type(torch.float32)
        similarity = similarity * sim_mask
        u = u * label
        prediction = torch.clamp((prediction + ((1-weight)*u.detach())), min=eps, max=1.0)
        loss = torch.mean(-torch.sum(similarity * torch.log(prediction), dim=1))
        label_one_hot = self.soft_to_hard(output.detach())
        MSE_loss = F.mse_loss((label_one_hot + (weight*u)), label, reduction='sum') / len(label)
        loss += MSE_loss
        self.take[index] = torch.sum(label_one_hot * label, dim=1).view(-1, 1)
        kl_loss = F.kl_div(F.log_softmax(torch.sum(output * label, dim=1)), F.softmax(-self.u[index].detach().view(-1)))
        loss += kl_loss

        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)
            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            loss += self.ratio_balance * balance_kl

        if len(outputs) > len(index) and self.ratio_consistency > 0:
            consistency_loss = self.consistency_loss(output, output2)
            loss += self.ratio_consistency * torch.mean(consistency_loss)

        return loss

    def consistency_loss(self, output1, output2):
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
        return torch.sum(loss_kldiv, dim=1)

    def soft_to_hard(self, x):
        with torch.no_grad():
            return torch.zeros(len(x), self.num_classes).to(self.device).scatter_(1, x.argmax(dim=1).view(-1, 1), 1)
        
def train_with_ncod(model, data, noisy_indices, device, total_epochs=200, lambda_dir=0.1, num_classes=None, patience=20, debug=True):
    if num_classes is None:
        num_classes = int(data.y.max().item()) + 1

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    sample_labels = data.y.cpu().numpy()

    oversmoothing_evaluator = OversmoothingMetrics(device=device)

    ncod_loss_fn = ncodLoss(
        sample_labels=sample_labels,
        device=device,
        num_examp=data.num_nodes,
        num_classes=num_classes,
        ratio_consistency=1.5,
        ratio_balance=1.2,
        encoder_features=num_classes,
        total_epochs=total_epochs
    ).to(device)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        label_onehot_train = F.one_hot(data.y[train_idx], num_classes=num_classes).float()
        loss_ncod_train = ncod_loss_fn(
            index=train_idx,
            outputs=out[train_idx],
            label=label_onehot_train,
            out=out[train_idx],
            flag=0,
            epoch=epoch,
            train_acc_cater=None
        )
        loss_dir_train = dirichlet_energy(out, data.edge_index)
        loss_train = loss_ncod_train + lambda_dir * loss_dir_train
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_idx = data.val_mask.nonzero(as_tuple=True)[0]
            label_onehot_val = F.one_hot(data.y[val_idx], num_classes=num_classes).float()
            val_loss_ncod = ncod_loss_fn(
                index=val_idx,
                outputs=out[val_idx],
                label=label_onehot_val,
                out=out[val_idx],
                flag=0,
                epoch=epoch,
                train_acc_cater=None
            )
            val_loss_dir = dirichlet_energy(out, data.edge_index)
            val_loss = val_loss_ncod + lambda_dir * val_loss_dir

            pred = out.argmax(dim=1)
            train_acc = (pred[train_idx] == data.y[train_idx]).sum().item() / len(train_idx)
            val_acc = (pred[val_idx] == data.y[val_idx]).sum().item() / len(val_idx)
            train_f1 = f1_score(data.y[train_idx].cpu(), pred[train_idx].cpu(), average='macro')
            val_f1 = f1_score(data.y[val_idx].cpu(), pred[val_idx].cpu(), average='macro')

            train_oversmoothing = _compute_oversmoothing_for_mask(
                oversmoothing_evaluator, out, data.edge_index, data.train_mask, data.y
            )
            val_oversmoothing = _compute_oversmoothing_for_mask(
                oversmoothing_evaluator, out, data.edge_index, data.val_mask, data.y
            )

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

        if debug:
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
        label_onehot_test = F.one_hot(data.y[test_idx], num_classes=num_classes).float()
        loss_ncod_test = ncod_loss_fn(
            index=test_idx,
            outputs=out[test_idx],
            label=label_onehot_test,
            out=out[test_idx],
            flag=0,
            epoch=epoch,
            train_acc_cater=None
        )
        loss_dir_test = dirichlet_energy(out, data.edge_index)
        test_loss = loss_ncod_test + lambda_dir * loss_dir_test

        pred = out.argmax(dim=1)
        test_acc = (pred[test_idx] == data.y[test_idx]).sum().item() / len(test_idx)
        test_f1 = f1_score(data.y[test_idx].cpu(), pred[test_idx].cpu(), average='macro')

        final_train_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.train_mask, data.y
        )
        final_val_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.val_mask, data.y
        )
        test_oversmoothing = _compute_oversmoothing_for_mask(
            oversmoothing_evaluator, out, data.edge_index, data.test_mask, data.y
        )

    if debug:
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
        print("Final Oversmoothing Metrics:")
        
        print(f"Train: EDir: {final_train_oversmoothing['EDir']:.4f}, EDir_traditional: {final_train_oversmoothing['EDir_traditional']:.4f}, "
              f"EProj: {final_train_oversmoothing['EProj']:.4f}, MAD: {final_train_oversmoothing['MAD']:.4f}, "
              f"NumRank: {final_train_oversmoothing['NumRank']:.4f}, Erank: {final_train_oversmoothing['Erank']:.4f}")
        
        print(f"Val: EDir: {final_val_oversmoothing['EDir']:.4f}, EDir_traditional: {final_val_oversmoothing['EDir_traditional']:.4f}, "
              f"EProj: {final_val_oversmoothing['EProj']:.4f}, MAD: {final_val_oversmoothing['MAD']:.4f}, "
              f"NumRank: {final_val_oversmoothing['NumRank']:.4f}, Erank: {final_val_oversmoothing['Erank']:.4f}")
        
        print(f"Test: EDir: {test_oversmoothing['EDir']:.4f}, EDir_traditional: {test_oversmoothing['EDir_traditional']:.4f}, "
              f"EProj: {test_oversmoothing['EProj']:.4f}, MAD: {test_oversmoothing['MAD']:.4f}, "
              f"NumRank: {test_oversmoothing['NumRank']:.4f}, Erank: {test_oversmoothing['Erank']:.4f}")

    return test_acc