import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from copy import deepcopy
from torch_geometric.loader import NeighborLoader

def dirichlet_energy(x, edge_index):
    row, col = edge_index
    diff = x[row] - x[col]
    return (diff ** 2).sum(dim=1).mean()

def enforce_positive_eigenvalues(weight_matrix):

    with torch.no_grad():

        eigenvalues, eigenvectors = torch.linalg.eigh(weight_matrix)
        
        positive_mask = eigenvalues > 0
        if positive_mask.sum() == 0:
            return torch.eye(weight_matrix.size(0), device=weight_matrix.device) * 0.01
        
        positive_eigenvalues = eigenvalues[positive_mask]
        positive_eigenvectors = eigenvectors[:, positive_mask]
        
        reconstructed = positive_eigenvectors @ torch.diag(positive_eigenvalues) @ positive_eigenvectors.T
        return reconstructed

def train_with_positive_eigenvalues(model, data, noisy_indices=None, device='cuda', epochs=200, 
                                  patience=20, lambda_dir=0.1, config=None, batch_size=32):

    model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
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
    
    def apply_eigenvalue_constraint():
        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and 'proj' in name.lower():
                    if module.weight.dim() == 2 and module.weight.size(0) == module.weight.size(1):
                        module.weight.data = enforce_positive_eigenvalues(module.weight.data)
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        total_train_loss = 0
        total_train_acc = 0
        total_train_f1 = 0
        num_train_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            embeddings = out

            batch_mask = batch.train_mask[:batch.batch_size]
            batch_target_nodes = batch_mask.nonzero(as_tuple=True)[0]
            
            if len(batch_target_nodes) == 0:
                continue
                
            loss_ce = F.cross_entropy(out[batch_target_nodes], batch.y[batch_target_nodes])
            loss_dir = dirichlet_energy(embeddings, batch.edge_index)
            
            loss_train = loss_ce + lambda_dir * loss_dir
            loss_train.backward()
            optimizer.step()
            
            apply_eigenvalue_constraint()
            
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
                
                val_loss_ce = F.cross_entropy(out_val[batch_target_nodes], batch.y[batch_target_nodes])
                val_loss_dir = dirichlet_energy(embeddings_val, batch.edge_index)
                val_loss = val_loss_ce + lambda_dir * val_loss_dir
                
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
    total_test_loss = 0
    total_test_acc = 0
    total_test_f1 = 0
    num_test_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out_test = model(batch)
            embeddings_test = out_test
            
            batch_mask = batch.test_mask[:batch.batch_size]
            batch_target_nodes = batch_mask.nonzero(as_tuple=True)[0]
            
            if len(batch_target_nodes) == 0:
                continue
            
            test_loss_ce = F.cross_entropy(out_test[batch_target_nodes], batch.y[batch_target_nodes])
            test_loss_dir = dirichlet_energy(embeddings_test, batch.edge_index)
            test_loss = test_loss_ce + lambda_dir * test_loss_dir
            
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
    avg_test_dir = 0
    
    print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f} | Test F1: {avg_test_f1:.4f} | "
          f"Dirichlet Energy: {avg_test_dir:.4f}")
    return avg_test_acc
