import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from torch_geometric.data import Data
from copy import deepcopy
import time
from sklearn.metrics import accuracy_score

def mask_feature(features, p=0.5):
    if p <= 0:
        return features, None
    
    mask = torch.bernoulli(torch.full_like(features, 1-p))
    masked_features = features * mask
    return masked_features, mask

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

    with torch.no_grad():
        pseudo_labels = (F.softmax(p1, dim=1) + F.softmax(p2, dim=1)) / 2
        pseudo_labels = pseudo_labels.argmax(dim=1)
    
    loss1 = F.cross_entropy(p1, pseudo_labels)
    loss2 = F.cross_entropy(p2, pseudo_labels)
    
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
            'patience': kwargs.get('patience', 20),
            'T': kwargs.get('T', 0.5),
            'tau': kwargs.get('tau', 0.5),
            'p': kwargs.get('p', 0.5),
            'alpha': kwargs.get('alpha', 1.0),
            'beta': kwargs.get('beta', 0.0),
            'debug': kwargs.get('debug', True)
        }
    
    def fit(self, base_model, data, config, get_model_func):
        print(f"Training CRGNN with {config['model_name'].upper()} backbone...")
        
        data = data.to(self.device)
        n_classes = data.y.max().item() + 1

        encoder = base_model.to(self.device)

        with torch.no_grad():
            tmp_out = encoder(data)
        out_dim = tmp_out.size(1)

        hidden_dim = self.config['hidden_channels']

        if out_dim <= n_classes:
            projection_adapter = nn.Linear(out_dim, hidden_dim).to(self.device)
            encoder_out_dim = hidden_dim
        else:
            projection_adapter = nn.Identity().to(self.device)
            encoder_out_dim = out_dim

        projection_head = ProjectionHead(encoder_out_dim, hidden_dim).to(self.device)
        classifier = ClassificationHead(encoder_out_dim if isinstance(projection_adapter, nn.Identity) else hidden_dim, n_classes).to(self.device)


        optimizer = torch.optim.Adam(
            list(encoder.parameters()) +
            list(projection_head.parameters()) +
            list(classifier.parameters()),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        best_val_acc = 0
        patience_counter = 0
        best_weights = None
        
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
        
        for epoch in range(self.config['epochs']):

            encoder.train()
            projection_head.train()
            classifier.train()
            optimizer.zero_grad()
            
            edge_index1, _ = dropout_edge(data.edge_index, p=0.3)
            edge_index2, _ = dropout_edge(data.edge_index, p=0.3)
            x1, _ = mask_feature(data.x, p=0.3)
            x2, _ = mask_feature(data.x, p=0.3)
            
            data1 = Data(x=x1, edge_index=edge_index1)
            data2 = Data(x=x2, edge_index=edge_index2)

            h1 = encoder(data1)
            h2 = encoder(data2)


            h1 = projection_adapter(h1)
            h2 = projection_adapter(h2)
            
            z1 = projection_head(h1)
            z2 = projection_head(h2)
            
            p1 = classifier(h1)
            p2 = classifier(h2)
            
            loss_con = contrastive_loss(z1, z2, self.config['tau'])
            
            loss_pseudo = dynamic_cross_entropy_loss(
                p1[train_mask], p2[train_mask], noisy_labels[train_mask]
            )
            loss_sup = dynamic_cross_entropy_loss(p1[train_mask], p2[train_mask], noisy_labels[train_mask])

            
            if self.config['beta'] > 0:

                z1n = F.normalize(z1, p=2, dim=1)
                z2n = F.normalize(z2, p=2, dim=1)
                Sz = z1n @ z2n.t()
                Sz = (Sz / self.config['T']).clamp(min=-10, max=10)
                zm = torch.exp(Sz).mean(dim=1)

                p1_prob = F.softmax(p1, dim=1)
                p2_prob = F.softmax(p2, dim=1)

                Sp = p1_prob @ p2_prob.t()
                Sp = (Sp / self.config['T']).clamp(min=-10, max=10)
                pm = Sp.mean(dim=1)
                pm = torch.where(pm > self.config['p'], pm, torch.zeros_like(pm))
                pm = pm.detach()


                zm = torch.sigmoid(zm)
                pm = torch.sigmoid(pm)

                loss_ccon = F.mse_loss(zm, pm)
                total_loss = self.config['alpha'] * loss_con + loss_sup + self.config['beta'] * loss_ccon
            else:
                total_loss = self.config['alpha'] * loss_con + loss_sup

            if torch.isfinite(total_loss):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + 
                    list(projection_head.parameters()) +
                    list(classifier.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

            encoder.eval()
            classifier.eval()
            
            with torch.no_grad():
                data_clean = Data(x=data.x, edge_index=data.edge_index)
                h_val = encoder(data_clean)
                h_val = projection_adapter(h_val)
                h_val = projection_head(h_val)
                p_val = classifier(h_val)

                pred_labels = torch.argmax(p_val, dim=1)

                train_acc = accuracy_score(
                    noisy_labels[train_mask].cpu().numpy(),
                    pred_labels[train_mask].cpu().numpy()
                )
                val_pred_labels = torch.argmax(p_val, dim=1)
                val_acc = accuracy_score(
                    clean_labels[val_mask].cpu().numpy(),
                    val_pred_labels[val_mask].cpu().numpy()
                )

            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_weights = {
                    'encoder': deepcopy(encoder.state_dict()),
                    'projection_head': deepcopy(projection_head.state_dict()),
                    'classifier': deepcopy(classifier.state_dict())
                }
            else:
                patience_counter += 1
            
            if self.config['debug']:
                print(f"Epoch {epoch+1:05d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {total_loss.item():.4f}")
            
            if patience_counter >= self.config['patience']:
                if self.config['debug']:
                    print("Early stopping triggered.")
                break
        
        if best_weights is not None:
            encoder.load_state_dict(best_weights['encoder'])
            classifier.load_state_dict(best_weights['classifier'])
        
        encoder.eval()
        classifier.eval()
        
        with torch.no_grad():
            data_test = Data(x=data.x, edge_index=data.edge_index)
            h_test = encoder(data_test)
            h_test = projection_adapter(h_test)
            p_test = classifier(h_test)

            
            test_pred_labels = torch.argmax(p_test, dim=1)
            test_acc = accuracy_score(
                clean_labels[test_mask].cpu().numpy(),
                test_pred_labels[test_mask].cpu().numpy()
            )

        total_time = time.time() - start_time
        
        if self.config['debug']:
            print(f'CRGNN Training completed!')
            print(f'Time: {total_time:.4f}s')
            print(f'Best Val Acc: {best_val_acc:.4f}')
            print(f'Test Acc: {test_acc:.4f}')
        
        return test_acc