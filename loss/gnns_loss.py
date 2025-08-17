import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def dirichlet_energy(x, edge_index):
    row, col = edge_index
    diff = x[row] - x[col]
    return (diff ** 2).sum(dim=1).mean()


def torch_cdf_loss(tensor_a, tensor_b, p=1):
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)

    if p == 1:
        cdf_distance = torch.sum(torch.abs(cdf_tensor_a - cdf_tensor_b), dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow(cdf_tensor_a - cdf_tensor_b, 2), dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a - cdf_tensor_b), p), dim=-1), 1/p)

    return cdf_distance.mean()


def torch_wasserstein_loss(tensor_a, tensor_b):
    return torch_cdf_loss(tensor_a, tensor_b, p=1)


def torch_energy_loss(tensor_a, tensor_b):
    return (2**0.5) * torch_cdf_loss(tensor_a, tensor_b, p=2)


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


def train_with_standard_loss(model, data, noisy_indices, device, total_epochs=200):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(total_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[noisy_indices], data.y_noisy[noisy_indices])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{total_epochs}, Loss: {loss.item():.4f}")
    print(f"\nDone. Noise injected on {len(noisy_indices)} train samples.")


def train_with_dirichlet(model, data, noisy_indices, device, lambda_dir=0.1, epochs=200):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss_ce = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss_dir = dirichlet_energy(out, data.edge_index)
        loss = loss_ce + lambda_dir * loss_dir
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | CE Loss: {loss_ce:.4f} | Dir Energy: {loss_dir:.4f} | Test Acc: {acc:.4f}")
    print(f"\nDone. Noise injected on {len(noisy_indices)} train samples.")


def train_with_ncod(model, data, noisy_indices, device, total_epochs=200, lambda_dir=0.1, num_classes=None):
    if num_classes is None:
        num_classes = int(data.y.max().item()) + 1
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    sample_labels = data.y.cpu().numpy()
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
    for epoch in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        logits = out
        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        label_onehot = F.one_hot(data.y[train_idx], num_classes=num_classes).float()
        loss_ncod = ncod_loss_fn(
            index=train_idx,
            outputs=logits[train_idx],
            label=label_onehot,
            out=logits[train_idx],
            flag=0,
            epoch=epoch,
            train_acc_cater=None
        )
        loss_dir = dirichlet_energy(out, data.edge_index)
        loss = loss_ncod + lambda_dir * loss_dir
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | NCOD Loss: {loss_ncod:.4f} | Dir Energy: {loss_dir:.4f} | Test Acc: {acc:.4f}")
    print(f"\nDone. Noise injected on {len(noisy_indices)} train samples.")
