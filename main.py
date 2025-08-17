import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, WikiCS, Reddit
from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
from torch_geometric.utils import to_scipy_sparse_matrix
import time

from loss.gnns_loss import train_with_standard_loss, train_with_dirichlet, train_with_ncod
from model.gnns import GCN, GIN, GAT, GAT2
from model.NRGNN import NRGNN
from utilities.noises import label_process
from utilities.usefull import setup_seed_device

def load_dataset(name, root="data"):
    name_lower = name.lower()

    if name_lower == "corafull":
        dataset = CoraFull(root=f"{root}/CoraFull", transform=NormalizeFeatures())
    elif name_lower in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root=f"{root}/{name}", name=name.capitalize(), transform=NormalizeFeatures())
    elif name_lower in ["computers", "photo"]:
        dataset = Amazon(root=f"{root}/Amazon", name=name.capitalize(), transform=NormalizeFeatures())
    elif name_lower in ["coauthorcs", "coauthorphysics"]:
        dataset = Coauthor(root=f"{root}/Coauthor", name=name_lower.replace("coauthor", "").capitalize(), transform=NormalizeFeatures())
    elif name_lower == "wikics":
        dataset = WikiCS(root=f"{root}/WikiCS", transform=NormalizeFeatures())
    elif name_lower == "reddit":
        dataset = Reddit(root=f"{root}/Reddit")
    else:
        raise ValueError(f"Dataset {name} not supported.")

    data = dataset[0]

    if not hasattr(data, 'train_mask'):
        data = RandomNodeSplit(num_train_per_class=20, num_val=500, num_test=1000)(data)
    
    return data, dataset.num_classes


def get_model(model_name, in_channels, hidden_channels, out_channels, **kwargs):
    model_name = model_name.lower()
    
    if model_name == 'gcn':
        n_layers = kwargs.get('n_layers', 2)
        dropout = kwargs.get('dropout', 0.5)
        self_loop = kwargs.get('self_loop', True)
        return GCN(in_channels, hidden_channels, out_channels,
                   n_layers=n_layers, dropout=dropout, self_loop=self_loop)
    elif model_name == 'gin':
        n_layers = kwargs.get('n_layers', 3)
        mlp_layers = kwargs.get('mlp_layers', 2)
        dropout = kwargs.get('dropout', 0.5)
        train_eps = kwargs.get('train_eps', True)
        return GIN(in_channels, hidden_channels, out_channels,
                   n_layers=n_layers, mlp_layers=mlp_layers,
                   dropout=dropout, train_eps=train_eps)
    elif model_name == 'gat':
        n_layers = kwargs.get('n_layers', 2)
        heads = kwargs.get('heads', 8)
        dropout = kwargs.get('dropout', 0.5)
        return GAT(in_channels, hidden_channels, out_channels,
                   n_layers=n_layers, heads=heads, 
                   dropout=dropout)
    elif model_name == 'gat2':
        n_layers = kwargs.get('n_layers', 2)
        heads = kwargs.get('heads', 8)
        dropout = kwargs.get('dropout', 0.5)
        concat = kwargs.get('concat', True)
        return GAT2(in_channels, hidden_channels, out_channels,
            n_layers=n_layers, heads=heads,
            dropout=dropout)
    else:
        raise ValueError(f"Model {model_name} not recognized. "
                        f"Available models: gcn, gin, gat, gat2")

def train(model, data, noisy_indices, device, config):
    if config.get("supplementary_gnn", "").upper() == "NRGNN":
        print("Using NRGNN training")
        return "NRGNN"
    
    method = config.get("method", "standard").lower()

    if method == "standard":
        train_with_standard_loss(
            model, data, noisy_indices, device,
            total_epochs=config.get("total_epochs", 200)
        )
    elif method == "dirichlet":
        train_with_dirichlet(
            model, data, noisy_indices, device,
            lambda_dir=config.get("lambda_dir", 0.1),
            epochs=config.get("total_epochs", 200)
        )
    elif method == "ncod":
        train_with_ncod(
            model, data, noisy_indices, device,
            total_epochs=config.get("total_epochs", 200),
            lambda_dir=config.get("lambda_dir", 0.1),
            num_classes=config.get("num_classes", None)
        )
    else:
        raise ValueError(f"Training method '{method}' not recognized.")

def run_experiment(config):
    supp_gnn = config.get('supplementary_gnn', "").upper()
    method = config.get("method", "").lower()

    if supp_gnn == 'NRGNN':
        config['method'] = 'nrgnn'
        
    special_methods = ['nrgnn']
    
    if method in special_methods:
        if not supp_gnn:
            print(f"ERROR: can't use the method '{method.upper()}' without setting 'supplementary_gnn'.")
            return
        
        method_to_supp_gnn = {
            'nrgnn': 'NRGNN',
        }
        
        if method in method_to_supp_gnn and supp_gnn != method_to_supp_gnn[method]:
            print(f"ERROR: method {method.upper()} wants 'supplementary_gnn' set on '{method_to_supp_gnn[method]}'.")
            return

    if config.get('supplementary_gnn', "").upper() == 'NRGNN':
        config['method'] = 'nrgnn'

    for k, v in config.items():
        print(f"{k}: {v}")
    print("-"*50)

    setup_seed_device(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    data, num_classes = load_dataset(config['dataset_name'])
    data = data.to(device)
    
    noisy_labels, noisy_indices = label_process(
        data.y, data.x, num_classes,
        noise_type=config['noise_type'],
        noise_rate=config['noise_rate'],
        random_seed=config['train_noise_seed'],
        debug=True
    )

    data.y_noisy = noisy_labels

    if config['method'] == 'nrgnn':
        print("Using NRGNN")

        nrgnn_args = {
            'hidden_channels': config.get('hidden_channels', 64),
            'dropout': config.get('dropout', 0.5),
            'lr': config.get('lr', 0.01),
            'weight_decay': config.get('weight_decay', 5e-4),
            'epochs': config.get('total_epochs', 200),
            'edge_hidden': config.get('edge_hidden', 32),
            't_small': config.get('t_small', 0.1),
            'n_n': config.get('n_n', 1),
            'n_p': config.get('n_p', 5),
            'p_u': config.get('p_u', 0.8),
            'alpha': config.get('alpha', 0.5),
            'beta': config.get('beta', 0.5)
        }
    
        nrgnn_model = NRGNN(args=nrgnn_args, device=device, gnn_type=config.get('model_name', 'GCN').upper())
        adj_matrix = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    
        final_acc = nrgnn_model.fit(
            features=data.x,
            adj=adj_matrix,
            labels=data.y_noisy,
            idx_train=noisy_indices.cpu().numpy() if isinstance(noisy_indices, torch.Tensor) else noisy_indices
        )
    
        return

    model_params = {k: v for k, v in config.items() if k in [
        'n_layers', 'dropout', 'mlp_layers', 'train_eps'
    ]}
    
    model = get_model(
        model_name=config['model_name'],
        in_channels=data.num_features,
        hidden_channels=config.get('hidden_channels', 64),
        out_channels=num_classes,
        **model_params
    ).to(device)

    train(model, data, noisy_indices, device, config)

    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        acc = (pred[noisy_indices] == data.y_noisy[noisy_indices]).float().mean()
        print(f"Final Training Accuracy: {acc:.4f}")


config = {
    "seed": 42,
    "device": "cuda",
    "dataset_name": "cora",  # cora, citeseer, pubmed, corafull, computers, photo, coauthorcs, coauthorphysics, wikics, reddit
    "noise_type": "random", #clean, uniform, uniform_simple, random, pair, random_pair, instance
    "noise_rate": 0.3,
    "train_noise_seed": 42,
    "model_name": "gin", #gcn, gin, gat, gat2
    "n_layers": 2,
    "dropout": 0.3,
    "hidden_channels": 64,
    
    "method": "nrgnn",  # standard, dirichlet, ndoc, nrgnn
    "total_epochs": 200,
    
    # Supplementary GNN
    "supplementary_gnn": "nrgnn",  # None, nrgnn

    #GCN
    "self_loop": True,

    #GIN
    "mlp_layers": 2,
    "train_eps": True,
    
    #GAT
    "heads": 8,
    "concat": True,

    #NRGNN
    "edge_hidden": 64,
    "t_small": 0.1,
    "n_n": 1,
    "n_p": 5,
    "p_u": 0.8,
    "alpha": 0.5,
    "beta": 0.5
    
}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("="*60)
    run_experiment(config)
