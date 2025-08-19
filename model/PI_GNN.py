import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import f1_score

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

    def get_model_lr(self, config, model_name):
        if model_name.lower() in ['gat', 'gat2']:
            return config.get('lr', 0.005)
        else:
            return config.get('lr', 0.01)

    def fit(self, model, data, config=None, get_model_func=None):
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

            model.eval()
            with torch.no_grad():
                out_train, _ = model(data)
                pred_train = out_train.argmax(dim=1)

                train_acc = pred_train[data.train_mask].eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
                train_f1 = f1_score(data.y[data.train_mask].cpu(), pred_train[data.train_mask].cpu(), average='micro')

                pred_val = pred_train[data.val_mask]
                val_acc = pred_val.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
                val_f1 = f1_score(data.y[data.val_mask].cpu(), pred_val.cpu(), average='micro')
                val_loss = F.nll_loss(out_train[data.val_mask], data.y[data.val_mask])

            print(f"Epoch {epoch:03d} | Train Loss: {loss_train:.4f}, Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | "
                f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            if val_loss < best_val_loss - self.delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}")
                model.load_state_dict(best_model_state)
                break

        model.eval()
        with torch.no_grad():
            out_test, _ = model(data)
            pred_test = out_test.argmax(dim=1)
            test_loss = F.nll_loss(out_test[data.test_mask], data.y[data.test_mask]).item()
            test_acc = pred_test[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
            test_f1 = f1_score(data.y[data.test_mask].cpu(), pred_test[data.test_mask].cpu(), average='micro')

        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
        return test_acc

