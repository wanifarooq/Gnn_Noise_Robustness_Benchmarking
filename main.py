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
from model.PI_GNN import InnerProductTrainer, Net, InnerProductDecoder
from model.CR_GNN import CRGNNTrainer

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

    kwargs.pop('in_channels', None)
    kwargs.pop('hidden_channels', None) 
    kwargs.pop('out_channels', None)
    
    if model_name == 'gcn':
        gcn_params = {k: v for k, v in kwargs.items() if k in ['n_layers', 'dropout', 'self_loop']}
        return GCN(in_channels, hidden_channels, out_channels, **gcn_params)
    elif model_name == 'gin':
        gin_params = {k: v for k, v in kwargs.items() if k in ['n_layers', 'dropout', 'mlp_layers', 'train_eps']}
        return GIN(in_channels, hidden_channels, out_channels, **gin_params)
    elif model_name == 'gat':
        gat_params = {k: v for k, v in kwargs.items() if k in ['n_layers', 'dropout', 'heads']}
        return GAT(in_channels, hidden_channels, out_channels, **gat_params)
    elif model_name == 'gat2':
        gat2_params = {k: v for k, v in kwargs.items() if k in ['n_layers', 'dropout', 'heads']}
        return GAT2(in_channels, hidden_channels, out_channels, **gat2_params)
    else:
        raise ValueError(f"Model {model_name} not recognized.")

def train(model, data, noisy_indices, device, config):
    method = config['training']['method'].lower()
    supplementary_gnn = config['training'].get('supplementary_gnn', None)

    if supplementary_gnn and supplementary_gnn.lower() == "nrgnn":
        print("Using NRGNN training")
        return

    if supplementary_gnn and supplementary_gnn.lower() == "pi_gnn":
        print("Using PI-GNN training")
        return

    if supplementary_gnn and supplementary_gnn.lower() == "cr_gnn":
        print("Using CR-GNN training")
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
        raise ValueError(f"Training method '{method}' not recognized. Supported methods: standard, dirichlet, ncod")

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

    data.y_original = data.y.clone()
    data.y = noisy_labels
    data.y_noisy = noisy_labels

    # NRGNN Training
    if (config['training'].get('supplementary_gnn', "").lower() == 'nrgnn' or 
        config['training']['method'].lower() == 'nrgnn'):
        print("Using NRGNN")
        
        nrgnn_model = NRGNN(args=config.get('nrgnn_params', {}), device=device, gnn_type=config['model']['name'].upper())
        adj_matrix = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        nrgnn_model.fit(
            features=data.x,
            adj=adj_matrix,
            labels=data.y,
            idx_train=noisy_indices.cpu().numpy() if isinstance(noisy_indices, torch.Tensor) else noisy_indices
        )
        return

    # PI-GNN Training
    if (config['training'].get('supplementary_gnn', "").lower() == 'pi_gnn' or 
        config['training']['method'].lower() == 'pi_gnn'):
        print("Using PI-GNN")
        
        trainer_params = config.get('pi_gnn_params', {})
        trainer = InnerProductTrainer(
            device=device,
            epochs=int(trainer_params.get('epochs', 400)),
            start_epoch=int(trainer_params.get('start_epoch', 200)),
            miself=bool(trainer_params.get('miself', False)),
            lr_main=float(trainer_params.get('lr_main', 0.01)),
            lr_mi=float(trainer_params.get('lr_mi', 0.01)),
            weight_decay=float(trainer_params.get('weight_decay', 5e-4)),
            norm=trainer_params.get('norm', None),
            vanilla=bool(trainer_params.get('vanilla', False)),
        )
        
        model_params = {k: v for k, v in config['model'].items() if k not in ['name']}
        base_model = get_model(
            model_name=config['model']['name'], 
            in_channels=data.num_features,
            out_channels=num_classes, 
            **model_params
        )
        decoder = InnerProductDecoder()
        model = Net(gnn_model=base_model, supplementary_gnn=decoder)
        
        trainer_config = {
            'model_name': config['model']['name'],
            'hidden_channels': config['model'].get('hidden_channels', 64),
            'n_layers': config['model'].get('n_layers', 2),
            'dropout': config['model'].get('dropout', 0.5),
            'mlp_layers': config['model'].get('mlp_layers', 2),
            'train_eps': config['model'].get('train_eps', True),
            'heads': config['model'].get('heads', 8),
            'concat': config['model'].get('concat', True),
            'self_loop': config['model'].get('self_loop', True)
        }
        
        test_acc = trainer.fit(model, data, trainer_config, get_model)
        print(f"PI-GNN Training completed with test accuracy: {test_acc:.4f}")
        return

    # CR-GNN Training
    if (config['training'].get('supplementary_gnn', "").lower() == 'cr_gnn' or 
        config['training']['method'].lower() == 'cr_gnn'):
        print("Using CR-GNN")
        
        trainer_params = config.get('cr_gnn_params', {})
        trainer = CRGNNTrainer(
            device=device,
            hidden_channels=config['model'].get('hidden_channels', 64),
            lr=float(trainer_params.get('lr', 0.001)),
            weight_decay=float(trainer_params.get('weight_decay', 5e-4)),
            epochs=int(trainer_params.get('epochs', 200)),
            patience=int(trainer_params.get('patience', 20)),
            T=float(trainer_params.get('T', 0.5)),
            tau=float(trainer_params.get('tau', 0.5)),
            p=float(trainer_params.get('p', 0.5)),
            alpha=float(trainer_params.get('alpha', 1.0)),
            beta=float(trainer_params.get('beta', 0.0)),
            debug=bool(trainer_params.get('debug', True))
        )
        
        model_params = {k: v for k, v in config['model'].items() if k not in ['name']}

        model_params.pop('hidden_channels', None)
        base_model = get_model(
            model_name=config['model']['name'], 
            in_channels=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            out_channels=num_classes,
            **model_params
        )

        trainer_config = {
            'model_name': config['model']['name'],
            'hidden_channels': config['model'].get('hidden_channels', 64),
            'n_layers': config['model'].get('n_layers', 2),
            'dropout': config['model'].get('dropout', 0.5),
            'mlp_layers': config['model'].get('mlp_layers', 2),
            'train_eps': config['model'].get('train_eps', True),
            'heads': config['model'].get('heads', 8),
            'concat': config['model'].get('concat', True),
            'self_loop': config['model'].get('self_loop', True)
        }
        
        test_acc = trainer.fit(base_model, data, trainer_config, get_model)
        print(f"CR-GNN Training completed with test accuracy: {test_acc:.4f}")
        return

    model_params = {k: v for k, v in config['model'].items() if k not in ['name']}
    model = get_model(
        model_name=config['model']['name'], 
        in_channels=data.num_features,
        out_channels=num_classes, 
        **model_params
    ).to(device)

    result = train(model, data, noisy_indices, device, config)
    
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        acc = (pred[noisy_indices] == data.y[noisy_indices]).float().mean()
        print(f"Final Training Accuracy: {acc:.4f}")


if __name__ == "__main__":
    print("\n" + "-"*50)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Loaded configuration file\n" + "-"*50)
    run_experiment(config)