import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, WikiCS, Reddit
from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
from torch_geometric.utils import to_scipy_sparse_matrix
import yaml

from utilities.usefull import setup_seed_device
from utilities.noises import label_process
from model.gnns import GCN, GIN, GAT, GAT2
from loss.gnns_loss import train_with_standard_loss, train_with_dirichlet, train_with_ncod
from model.NRGNN import NRGNN

def load_dataset(name, root=None):
    if root is None:
        root = "./data"
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
        return GCN(in_channels, hidden_channels, out_channels, **kwargs)
    elif model_name == 'gin':
        return GIN(in_channels, hidden_channels, out_channels, **kwargs)
    elif model_name == 'gat':
        return GAT(in_channels, hidden_channels, out_channels, **kwargs)
    elif model_name == 'gat2':
        return GAT2(in_channels, hidden_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not recognized.")

def train(model, data, noisy_indices, device, config):
    method = config['training']['method'].lower()
    supplementary_gnn = config['training'].get('supplementary_gnn', None)

    if supplementary_gnn and supplementary_gnn.lower() == "nrgnn":
        print("Using NRGNN training")
        return

    if method == "standard":
        train_with_standard_loss(model, data, noisy_indices, device, total_epochs=config['training']['total_epochs'])
    elif method == "dirichlet":
        train_with_dirichlet(
            model, data, noisy_indices, device,
            lambda_dir=config['training'].get('lambda_dir', 0.1),
            epochs=config['training']['total_epochs']
        )
    elif method == "ncod":
        train_with_ncod(
            model, data, noisy_indices, device,
            total_epochs=config['training']['total_epochs'],
            lambda_dir=config['training'].get('lambda_dir', 0.1),
            num_classes=config['dataset']['num_classes']
        )
    else:
        raise ValueError(f"Training method '{method}' not recognized.")

def run_experiment(config):
    setup_seed_device(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    data, num_classes = load_dataset(config['dataset']['name'], root=config['dataset'].get('root', './data'))
    data = data.to(device)

    noisy_labels, noisy_indices = label_process(
        data.y, data.x, num_classes,
        noise_type=config['noise']['type'],
        noise_rate=config['noise']['rate'],
        random_seed=config['noise'].get('seed', 42),
        debug=True
    )
    data.y_noisy = noisy_labels

    if config['training'].get('supplementary_gnn', "").lower() == 'nrgnn' or config['training']['method'].lower() == 'nrgnn':
        print("Using NRGNN")
        nrgnn_model = NRGNN(args=config.get('nrgnn_params', {}), device=device, gnn_type=config['model']['name'].upper())
        adj_matrix = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        nrgnn_model.fit(
            features=data.x,
            adj=adj_matrix,
            labels=data.y_noisy,
            idx_train=noisy_indices.cpu().numpy() if isinstance(noisy_indices, torch.Tensor) else noisy_indices
        )
        return

    model_params = {k: v for k, v in config['model'].items() if k != 'name'}
    model = get_model(config['model']['name'], data.num_features, config['model'].get('hidden_channels', 64), num_classes, **model_params).to(device)

    train(model, data, noisy_indices, device, config)

    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        acc = (pred[noisy_indices] == data.y_noisy[noisy_indices]).float().mean()
        print(f"Final Training Accuracy: {acc:.4f}")


if __name__ == "__main__":
    print("\n" + "-"*50)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Loaded configuration file\n" + "-"*50)
    run_experiment(config)
