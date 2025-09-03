import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from copy import deepcopy
from torch_geometric.loader import NeighborLoader

from model.evaluation import OversmoothingMetrics

def dirichlet_energy(x, edge_index):
    row, col = edge_index
    diff = x[row] - x[col]
    return (diff ** 2).sum(dim=1).mean()

def _compute_oversmoothing_for_mask(oversmoothing_evaluator, embeddings, edge_index, mask):
    try:
        mask_indices = torch.where(mask)[0]
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
        
        return oversmoothing_evaluator.compute_all_metrics(
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

def train_with_positive_eigenvalues(model, data, noisy_indices=None, device='cuda', lr=0.01, weight_decay=5e-4, epochs=200, 
                                  patience=20, lambda_dir=0.1, config=None, batch_size=32):

    oversmoothing_evaluator = OversmoothingMetrics(device=device)
    oversmoothing_history = {
        'train': [],
        'val': [],
        'test': []
    }

    model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
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

    epoch_train_embeddings = []
    epoch_val_embeddings = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        total_train_loss = 0
        total_train_acc = 0
        total_train_f1 = 0
        num_train_batches = 0

        epoch_train_embeddings = []
        
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
            
            if len(epoch_train_embeddings) < 5:
                epoch_train_embeddings.append({
                    'embeddings': embeddings[:batch.batch_size].detach().cpu(),
                    'edge_index': batch.edge_index.cpu(),
                    'mask': batch_mask.cpu()
                })
        
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_acc = total_train_acc / num_train_batches
        avg_train_f1 = total_train_f1 / num_train_batches
        
        train_oversmoothing = None
        if epoch_train_embeddings:
            try:
                batch_data = epoch_train_embeddings[0]
                train_oversmoothing = _compute_oversmoothing_for_mask(
                    oversmoothing_evaluator,
                    batch_data['embeddings'].to(device),
                    batch_data['edge_index'].to(device),
                    batch_data['mask'].to(device)
                )
                if train_oversmoothing is not None:
                    oversmoothing_history['train'].append(train_oversmoothing)
            except:
                train_oversmoothing = None
        
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        total_val_f1 = 0
        num_val_batches = 0
        
        epoch_val_embeddings = []
        
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
                
                if len(epoch_val_embeddings) < 5:
                    epoch_val_embeddings.append({
                        'embeddings': embeddings_val[:batch.batch_size].detach().cpu(),
                        'edge_index': batch.edge_index.cpu(),
                        'mask': batch_mask.cpu()
                    })
        
        avg_val_loss = total_val_loss / num_val_batches
        avg_val_acc = total_val_acc / num_val_batches
        avg_val_f1 = total_val_f1 / num_val_batches

        val_oversmoothing = None
        if epoch_val_embeddings:
            try:
                batch_data = epoch_val_embeddings[0]
                val_oversmoothing = _compute_oversmoothing_for_mask(
                    oversmoothing_evaluator,
                    batch_data['embeddings'].to(device),
                    batch_data['edge_index'].to(device),
                    batch_data['mask'].to(device)
                )
                if val_oversmoothing is not None:
                    oversmoothing_history['val'].append(val_oversmoothing)
            except:
                val_oversmoothing = None
        
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
        
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f} | "
              f"Train F1: {avg_train_f1:.4f}, Val F1: {avg_val_f1:.4f}")
        print(f"Train DE: {train_edir:.4f}, Val DE: {val_edir:.4f} | "
              f"Train DE_trad: {train_edir_trad:.4f}, Val DE_trad: {val_edir_trad:.4f} | "
              f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
              f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
              f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
              f"Train Erank: {train_eff_rank:.4f}, Val Erank: {val_eff_rank:.4f}")
    
    model.eval()
    total_test_loss = 0
    total_test_acc = 0
    total_test_f1 = 0
    num_test_batches = 0
    
    epoch_test_embeddings = []
    
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
            
            if len(epoch_test_embeddings) < 5:
                epoch_test_embeddings.append({
                    'embeddings': embeddings_test[:batch.batch_size].detach().cpu(),
                    'edge_index': batch.edge_index.cpu(),
                    'mask': batch_mask.cpu()
                })
    
    avg_test_loss = total_test_loss / num_test_batches
    avg_test_acc = total_test_acc / num_test_batches
    avg_test_f1 = total_test_f1 / num_test_batches
    
    test_oversmoothing = None
    if epoch_test_embeddings:
        try:
            batch_data = epoch_test_embeddings[0]
            test_oversmoothing = _compute_oversmoothing_for_mask(
                oversmoothing_evaluator,
                batch_data['embeddings'].to(device),
                batch_data['edge_index'].to(device),
                batch_data['mask'].to(device)
            )
            if test_oversmoothing is not None:
                oversmoothing_history['test'].append(test_oversmoothing)
        except:
            test_oversmoothing = None
    
    print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f} | Test F1: {avg_test_f1:.4f}")
    
    print("Final Oversmoothing Metrics:")
    
    if oversmoothing_history['train']:
        final_train_oversmoothing = oversmoothing_history['train'][-1]
        print(f"Train: EDir: {final_train_oversmoothing['EDir']:.4f}, EDir_traditional: {final_train_oversmoothing['EDir_traditional']:.4f}, "
              f"EProj: {final_train_oversmoothing['EProj']:.4f}, MAD: {final_train_oversmoothing['MAD']:.4f}, "
              f"NumRank: {final_train_oversmoothing['NumRank']:.4f}, Erank: {final_train_oversmoothing['Erank']:.4f}")
    
    if oversmoothing_history['val']:
        final_val_oversmoothing = oversmoothing_history['val'][-1]
        print(f"Val: EDir: {final_val_oversmoothing['EDir']:.4f}, EDir_traditional: {final_val_oversmoothing['EDir_traditional']:.4f}, "
              f"EProj: {final_val_oversmoothing['EProj']:.4f}, MAD: {final_val_oversmoothing['MAD']:.4f}, "
              f"NumRank: {final_val_oversmoothing['NumRank']:.4f}, Erank: {final_val_oversmoothing['Erank']:.4f}")
    
    if test_oversmoothing is not None:
        print(f"Test: EDir: {test_oversmoothing['EDir']:.4f}, EDir_traditional: {test_oversmoothing['EDir_traditional']:.4f}, "
              f"EProj: {test_oversmoothing['EProj']:.4f}, MAD: {test_oversmoothing['MAD']:.4f}, "
              f"NumRank: {test_oversmoothing['NumRank']:.4f}, Erank: {test_oversmoothing['Erank']:.4f}")

    avg_test_dir = dirichlet_energy(epoch_test_embeddings[-1]['embeddings'].to(device), 
                                   epoch_test_embeddings[-1]['edge_index'].to(device)).item() if epoch_test_embeddings else 0.0
    
    print(f"Final Dirichlet Energy - Test: {avg_test_dir:.4f}")
    
    return avg_test_acc