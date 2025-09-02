import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from copy import deepcopy
from torch_geometric.loader import NeighborLoader

def dirichlet_energy(x, edge_index):
    row, col = edge_index
    diff = x[row] - x[col]
    return (diff ** 2).sum(dim=1).mean()

class GCODLoss(nn.Module):
    def __init__(self, num_classes, device, num_samples, encoder_features=64):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.num_samples = num_samples
        self.encoder_features = encoder_features
        
        self.u = nn.Parameter(torch.zeros(num_samples, dtype=torch.float32))
        self.class_centroids = nn.Parameter(torch.randn(num_classes, encoder_features))
        
    def compute_soft_labels(self, embeddings, labels):
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        centroids_norm = F.normalize(self.class_centroids, p=2, dim=1)
        
        similarities = torch.mm(embeddings_norm, centroids_norm.t())
        soft_labels = F.softmax(similarities, dim=1)
        
        return soft_labels
    
    def forward(self, batch_indices, predictions, embeddings, labels, train_acc):
        batch_size = predictions.size(0)
        u_batch = self.u[batch_indices]
        
        if labels.dim() == 1:
            labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        else:
            labels_onehot = labels
        
        soft_labels = self.compute_soft_labels(embeddings, labels)

        u_diag = torch.diag(u_batch)
        modified_predictions = predictions + train_acc * torch.mm(u_diag, labels_onehot)
        modified_predictions = torch.clamp(modified_predictions, min=1e-7, max=1-1e-7)
        
        L1 = F.cross_entropy(modified_predictions, soft_labels)
        
        pred_onehot = F.one_hot(predictions.argmax(dim=1), num_classes=self.num_classes).float()
        regularized_pred = pred_onehot + torch.mm(u_diag, labels_onehot)
        L2 = F.mse_loss(regularized_pred, labels_onehot, reduction='mean')
        
        prediction_alignment = torch.diag(torch.mm(predictions, labels_onehot.t()))
        L_term = F.logsigmoid(prediction_alignment)
        
        u_term = torch.sigmoid(-torch.log(u_batch + 1e-8))
        
        L3 = (1 - train_acc) * F.kl_div(
            F.log_softmax(L_term, dim=0), 
            F.softmax(u_term, dim=0), 
            reduction='batchmean'
        )
        
        total_loss = L1 + L2 + L3
        
        return total_loss, L1, L2, L3


def train_with_gcod(model, data, noisy_indices=None, device='cuda', epochs=500, 
                   patience=100, lambda_dir=0.1, config=None, batch_size=32):

    model.to(device)
    data = data.to(device)
    
    num_classes = int(data.y.max().item()) + 1
    
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    gcod_loss_fn = GCODLoss(
        num_classes=num_classes,
        device=device,
        num_samples=data.num_nodes,
        encoder_features=num_classes
    ).to(device)
    
    u_optimizer = torch.optim.SGD([gcod_loss_fn.u], lr=1)
    
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    val_idx = data.val_mask.nonzero(as_tuple=True)[0]
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=batch_size,
        input_nodes=train_idx,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=batch_size,
        input_nodes=val_idx,
        shuffle=False
    )
    
    test_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=batch_size,
        input_nodes=test_idx,
        shuffle=False
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        gcod_loss_fn.train()
        
        total_train_loss = 0
        total_train_acc = 0
        total_train_f1 = 0
        num_train_batches = 0
        epoch_train_acc = 0
        
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for batch in train_loader:
                batch = batch.to(device)
                out = model(batch)
                
                batch_mask = batch.train_mask[:batch.batch_size]
                batch_target_nodes = batch_mask.nonzero(as_tuple=True)[0]
                
                if len(batch_target_nodes) > 0:
                    pred_train = out[batch_target_nodes].argmax(dim=1)
                    total_correct += (pred_train == batch.y[batch_target_nodes]).sum().item()
                    total_samples += len(batch_target_nodes)
            
            epoch_train_acc = total_correct / total_samples if total_samples > 0 else 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            out = model(batch)
            embeddings = out
            
            batch_mask = batch.train_mask[:batch.batch_size]
            batch_target_nodes = batch_mask.nonzero(as_tuple=True)[0]
            
            if len(batch_target_nodes) == 0:
                continue
            
            original_indices = batch.n_id[batch_target_nodes]
            
            gcod_total_loss, L1, L2, L3 = gcod_loss_fn(
                batch_indices=original_indices,
                predictions=out[batch_target_nodes],
                embeddings=embeddings[batch_target_nodes],
                labels=batch.y[batch_target_nodes],
                train_acc=epoch_train_acc
            )
            
            loss_dir = dirichlet_energy(embeddings, batch.edge_index)
            loss_train = gcod_total_loss + lambda_dir * loss_dir
            
            model_optimizer.zero_grad()
            (L1 + L3).backward(retain_graph=True)
            model_optimizer.step()
            
            u_optimizer.zero_grad()
            L2.backward()
            u_optimizer.step()
            
            with torch.no_grad():
                pred_train = out[batch_target_nodes].argmax(dim=1)
                batch_acc = (pred_train == batch.y[batch_target_nodes]).sum().item() / len(batch_target_nodes)
                batch_f1 = f1_score(batch.y[batch_target_nodes].cpu(), pred_train.cpu(), average='macro')
            
            total_train_loss += loss_train.item()
            total_train_acc += batch_acc
            total_train_f1 += batch_f1
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_acc = total_train_acc / num_train_batches
        avg_train_f1 = total_train_f1 / num_train_batches

        model.eval()
        gcod_loss_fn.eval()
        total_val_loss = 0
        total_val_acc = 0
        total_val_f1 = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out_val = model(batch)
                embeddings_val = out_val
                
                batch_mask = batch.val_mask[:batch.batch_size]
                batch_target_nodes = batch_mask.nonzero(as_tuple=True)[0]
                
                if len(batch_target_nodes) == 0:
                    continue
                
                original_indices = batch.n_id[batch_target_nodes]
                
                val_gcod_loss, val_L1, val_L2, val_L3 = gcod_loss_fn(
                    batch_indices=original_indices,
                    predictions=out_val[batch_target_nodes],
                    embeddings=embeddings_val[batch_target_nodes],
                    labels=batch.y[batch_target_nodes],
                    train_acc=epoch_train_acc
                )
                
                val_loss_dir = dirichlet_energy(embeddings_val, batch.edge_index)
                val_loss = val_gcod_loss + lambda_dir * val_loss_dir
                
                pred_val = out_val[batch_target_nodes].argmax(dim=1)
                batch_val_acc = (pred_val == batch.y[batch_target_nodes]).sum().item() / len(batch_target_nodes)
                batch_val_f1 = f1_score(batch.y[batch_target_nodes].cpu(), pred_val.cpu(), average='macro')
                
                total_val_loss += val_loss.item()
                total_val_acc += batch_val_acc
                total_val_f1 += batch_val_f1
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        avg_val_acc = total_val_acc / num_val_batches
        avg_val_f1 = total_val_f1 / num_val_batches
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break
        
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f} | "
              f"Train F1: {avg_train_f1:.4f}, Val F1: {avg_val_f1:.4f}")
    
    model.eval()
    gcod_loss_fn.eval()
    total_test_loss = 0
    total_test_acc = 0
    total_test_f1 = 0
    num_test_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out_test = model(batch.x, batch.edge_index)
            embeddings_test = out_test
            
            batch_mask = batch.test_mask[:batch.batch_size]
            batch_target_nodes = batch_mask.nonzero(as_tuple=True)[0]
            
            if len(batch_target_nodes) == 0:
                continue
            
            original_indices = batch.n_id[batch_target_nodes]
            
            test_gcod_loss, test_L1, test_L2, test_L3 = gcod_loss_fn(
                batch_indices=original_indices,
                predictions=out_test[batch_target_nodes],
                embeddings=embeddings_test[batch_target_nodes],
                labels=batch.y[batch_target_nodes],
                train_acc=epoch_train_acc
            )
            
            test_loss_dir = dirichlet_energy(embeddings_test, batch.edge_index)
            test_loss = test_gcod_loss + lambda_dir * test_loss_dir
            
            pred_test = out_test[batch_target_nodes].argmax(dim=1)
            batch_test_acc = (pred_test == batch.y[batch_target_nodes]).sum().item() / len(batch_target_nodes)
            batch_test_f1 = f1_score(batch.y[batch_target_nodes].cpu(), pred_test.cpu(), average='macro')
            
            total_test_loss += test_loss.item()
            total_test_acc += batch_test_acc
            total_test_f1 += batch_test_f1
            num_test_batches += 1
    
    avg_test_loss = total_test_loss / num_test_batches
    avg_test_acc = total_test_acc / num_test_batches
    avg_test_f1 = total_test_f1 / num_test_batches
    
    print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f} | Test F1: {avg_test_f1:.4f}")
    
    return avg_test_acc