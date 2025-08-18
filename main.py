import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, WikiCS, Reddit
from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
import yaml

from utilities.usefull import setup_seed_device
from utilities.noises import label_process
from model.GNNs import GCN, GIN, GAT, GAT2
from loss.GNNs_loss import train_with_standard_loss, train_with_dirichlet, train_with_ncod
from model.NRGNN import NRGNN
from model.PI_GNN import InnerProductTrainer, Net, InnerProductDecoder
from model.CR_GNN import CRGNNTrainer
from model.LafAK import Attack, prepare_simpledata_attrs, resetBinaryClass_init
from model.RTGNN import RTGNN
from model.GraphCleaner import GraphCleanerDetector, get_noisy_ground_truth
from model.UnionNET import UnionNET

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
    if supplementary_gnn and supplementary_gnn.lower() == "lafak":
        print("Using LafAK training")
        return
    if supplementary_gnn and supplementary_gnn.lower() == "rtgnn":
        print("Using RTGNN training")
        return
    if supplementary_gnn and supplementary_gnn.lower() == "graphcleaner":
        print("Using GraphCleaner training")
        return
    if supplementary_gnn and supplementary_gnn.lower() == "unionnet":
        print("Using UnionNET training")
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
    
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    
    data.y_original = data.y.clone()
    
    train_labels = data.y[train_mask]
    train_features = data.x[train_mask] if config['noise']['type'] == 'instance' else None
    
    train_indices = train_mask.nonzero(as_tuple=True)[0]

    noisy_train_labels, relative_noisy_indices = label_process(
        train_labels,
        train_features,
        num_classes,
        noise_type=config['noise']['type'],
        noise_rate=config['noise']['rate'],
        random_seed=config['noise'].get('seed', 42),
        idx_train=train_indices,
        debug=True
    )
    
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    global_noisy_indices = train_indices[relative_noisy_indices]

    data.y_noisy = data.y_original.clone()
    data.y_noisy[train_mask] = noisy_train_labels
    
    data.y = data.y_original.clone()
    data.y[train_mask] = noisy_train_labels
    
    print(f"Applied noise to {len(relative_noisy_indices)} training samples out of {train_mask.sum().item()}")
    print(f"Noise rate: {len(relative_noisy_indices) / train_mask.sum().item():.4f}")
    
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
            idx_train=global_noisy_indices.cpu().numpy()
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

    # LafAK / GradientAttack Training
    if (config['training'].get('supplementary_gnn', "").lower() == 'lafak' or
        config['training']['method'].lower() == 'lafak'):
        print("Using LafAK / GradientAttack")
        
        data_for_lafak = data.clone()
        data_for_lafak.y = data.y_original
        data_dict = prepare_simpledata_attrs(data_for_lafak.cpu())
        
        target_classes = config.get('target_classes', None)
        actual_classes = np.unique(data.y_original.cpu().numpy())
        print(f"Available classes in dataset: {actual_classes}")
        
        if target_classes is None or len(target_classes) < 2:
            if len(actual_classes) >= 2:
                target_classes = actual_classes[:2].tolist()
            else:
                raise ValueError(f"Dataset has only {len(actual_classes)} classes, need at least 2 for binary attack")
        
        valid_targets = [tc for tc in target_classes if tc in actual_classes]
        if len(valid_targets) < 2:
            valid_targets = actual_classes[:2].tolist()
        target_classes = valid_targets[:2]
        
        print(f"Using target classes for attack: {target_classes}")
        
        K = int(data.y_original.max().item() + 1)
        target_classes = [min(tc, K-1) for tc in target_classes]
        
        resetBinaryClass_init(data_dict, a=target_classes[0], b=target_classes[1])
        
        gnn_model = get_model(
            model_name=config['model']['name'],
            in_channels=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            out_channels=2,
            n_layers=config['model'].get('n_layers', 2),
            dropout=config['model'].get('dropout', 0.5),
            self_loop=config['model'].get('self_loop', True)
        )
        
        attack = Attack(
            data_dict=data_dict,
            gpu_id=int(device.split(':')[-1]) if ':' in str(device) else 0,
            atkEpoch=config.get('attack_epochs', 500),
            gcnL2=config.get('gcn_l2', 5e-4)
        )
        
        print("Calculating clean accuracy...")
        resetBinaryClass_init(data_dict, a=target_classes[0], b=target_classes[1])
        acc_clean_runs = []
        for run in range(3):
            print(f"Clean run {run+1}/3...")
            acc_clean, *_ = attack.GNN_test(gnn_model)
            acc_clean_runs.append(acc_clean)
        
        acc_clean_avg = sum(acc_clean_runs) / len(acc_clean_runs)
        print(f"Clean accuracy (average): {acc_clean_avg:.4f}")
        
        print("Performing gradient attack...")
        results = attack.binaryAttack_multiclass_with_clean(
            c_max=config.get('c_max', 10),
            a=target_classes[0],
            b=target_classes[1],
            gnn_model=gnn_model
        )
        
        print(f"Final attack results")
        print(f"Binary Clean accuracy: {results['binary_clean_acc']:.4f}")
        print(f"Binary Attacked accuracy: {results['binary_attacked_acc']:.4f}")
        print(f"Binary Attack success: {results['attack_success']:.4f}")
        
        if results.get('gnn_clean_acc') and results.get('gnn_attacked_acc'):
            print(f"GNN Clean accuracy: {results['gnn_clean_acc']:.4f}")
            print(f"GNN Attacked accuracy: {results['gnn_attacked_acc']:.4f}")
            print(f"GNN Attack success: {results['gnn_attack_success']:.4f}")
        
        return {
            "clean_acc": results.get('gnn_clean_acc', results['binary_clean_acc']),
            "attacked_acc": results.get('gnn_attacked_acc', results['binary_attacked_acc']),
            "attack_success": results.get('gnn_attack_success', results['attack_success']),
            "target_classes": target_classes,
            "final_acc": results.get('gnn_attacked_acc', results['binary_attacked_acc'])
        }

    # RTGNN Training
    if (config['training'].get('supplementary_gnn', "").lower() == 'rtgnn' or
        config['training']['method'].lower() == 'rtgnn'):
        print("Using RTGNN")
        
        class RTGNNConfig:
            def __init__(self, config_dict):
                self.epochs = config_dict.get('total_epochs', 200)
                self.hidden = config_dict.get('hidden_channels', 64)
                self.edge_hidden = config_dict.get('edge_hidden', 64)
                self.dropout = config_dict.get('dropout', 0.5)
                self.lr = config_dict.get('lr', 0.01)
                self.weight_decay = config_dict.get('weight_decay', 5e-4)
                self.co_lambda = config_dict.get('co_lambda', 0.1)
                self.alpha = config_dict.get('alpha', 0.5)
                self.th = config_dict.get('th', 0.8)
                self.K = config_dict.get('K', 5)
                self.tau = config_dict.get('tau', 0.1)
                self.n_neg = config_dict.get('n_neg', 5)
        
        rtgnn_args = RTGNNConfig(config)
        
        features = data.x.cpu().numpy()
        labels = data.y_noisy.cpu().numpy()
        
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1
        adj.setdiag(0)
        
        features = sp.csr_matrix(features)
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        features = torch.FloatTensor(np.array(features.todense()))
        
        if hasattr(data, 'train_mask'):
            idx_train = data.train_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
            idx_val = data.val_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
            idx_test = data.test_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
        else:
            num_nodes = data.num_nodes
            idx_train = list(range(min(140, num_nodes // 5)))
            idx_val = list(range(len(idx_train), min(len(idx_train) + 500, num_nodes // 2)))
            idx_test = list(range(max(len(idx_train) + len(idx_val), num_nodes // 2), num_nodes))
        
        print(f"RTGNN - Dataset: {features.shape[0]} nodes, {features.shape[1]} features, "
              f"{len(np.unique(labels))} classes")
        print(f"RTGNN - Splits: {len(idx_train)} train, {len(idx_val)} val, {len(idx_test)} test")
        print(f"RTGNN - Noise rate: {config['noise']['rate']}")
        
        nfeat, nclass = features.shape[1], len(np.unique(labels))
        rtgnn_backbone = config.get('rtgnn_params', {}).get('gnn_type', config['model']['name'].lower())
        rtgnn_model = RTGNN(nfeat, nclass, rtgnn_args, device, gnn_type=rtgnn_backbone).to(device)
                
        rtgnn_model.fit(features, adj, labels, idx_train, idx_val, idx_test)
        
        clean_labels = data.y_original.cpu().numpy()
        test_acc = rtgnn_model.test(features, clean_labels, idx_test)
        
        print(f"RTGNN - Final test accuracy: {test_acc:.4f}")
        return {"test_acc": test_acc}
    
    # GraphCleaner Training
    if (config['training'].get('supplementary_gnn', "").lower() == 'graphcleaner' or
        config['training']['method'].lower() == 'graphcleaner'):
        print("Using GraphCleaner")
        
        data_for_detection = data.clone()
        
        test_labels = data.y_original[data.test_mask]
        test_features = data.x[data.test_mask] if config['noise']['type'] == 'instance' else None
        test_indices = data.test_mask.nonzero(as_tuple=True)[0]
        
        noisy_test_labels, relative_noisy_test_indices = label_process(
            test_labels,
            test_features, 
            num_classes,
            noise_type=config['noise']['type'],
            noise_rate=config['noise']['rate'],
            random_seed=config['noise'].get('seed', 42) + 1000,
            idx_train=test_indices,
            debug=False
        )
        
        data_for_detection.y_noisy = data.y_noisy.clone()
        data_for_detection.y_noisy[data.test_mask] = noisy_test_labels
        data_for_detection.y = data_for_detection.y_noisy.clone()
        
        global_noisy_test_indices = test_indices[relative_noisy_test_indices]
        all_noisy_indices = torch.cat([global_noisy_indices, global_noisy_test_indices])
        
        print(f"Added noise to {len(relative_noisy_test_indices)} test samples for detection evaluation")

        base_model = get_model(
            model_name=config['model']['name'],
            in_channels=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            out_channels=num_classes,
            n_layers=config['model'].get('n_layers', 2),
            dropout=config['model'].get('dropout', 0.5),
            self_loop=config['model'].get('self_loop', True)
        )
        detector = GraphCleanerDetector(config, device)

        predictions, probs, _ = detector.detect_noise(data_for_detection, base_model, num_classes)
        
        test_ground_truth = get_noisy_ground_truth(data_for_detection, all_noisy_indices)[data.test_mask.cpu()].cpu().numpy()
        detection_results = detector.evaluate_detection(predictions, test_ground_truth, probs)
        
        print(f"\nGraphCleaner Detection Summary:")
        print(f"Detected {np.sum(predictions)} out of {len(predictions)} test samples as noisy")
        print(f"Ground truth: {np.sum(test_ground_truth)} samples are actually noisy")
        print(f"Test noise rate: {np.sum(test_ground_truth) / len(test_ground_truth):.4f}")
        
        return detection_results
    
    # UnionNET Training
    if (config['training'].get('supplementary_gnn', "").lower() == 'unionnet' or
        config['training']['method'].lower() == 'unionnet'):
        print("Using UnionNET")
        
        gnn_model = get_model(
            model_name=config['model']['name'],
            in_channels=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            out_channels=num_classes,
            mlp_layers=config.get('mlp_layers', 2),
            train_eps=config.get('train_eps', True),
            heads=config.get('heads', 8),
            n_layers=config['model'].get('n_layers', 2),
            dropout=config['model'].get('dropout', 0.5),
            self_loop=config['model'].get('self_loop', True)
        ).to(device)
        
        unionnet_config = {
            'n_epochs': config.get('total_epochs', 200),
            'lr': config.get('lr', 0.01),
            'weight_decay': config.get('weight_decay', 5e-4),
            'patience': config.get('patience', 50),
            'k': config.get('k', 5),
            'alpha': config.get('alpha', 0.5),
            'feat_norm': config.get('feat_norm', True)
        }
        
        unionnet = UnionNET(gnn_model, data, num_classes, unionnet_config)
        result = unionnet.train(debug=True)
        
        print("UnionNET Results:")
        print(f"Train Acc: {result['train']:.4f}")
        print(f"Valid Acc: {result['val']:.4f}")
        print(f"Test Acc: {result['test']:.4f}")
        return result

    model_params = {k: v for k, v in config['model'].items() if k not in ['name']}
    model = get_model(
        model_name=config['model']['name'],
        in_channels=data.num_features,
        out_channels=num_classes,
        **model_params
    ).to(device)
    
    train(model, data, global_noisy_indices, device, config)
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        
        train_acc = (pred[train_mask] == data.y[train_mask]).float().mean()
        print(f"Final Training Accuracy (noisy labels): {train_acc:.4f}")
        
        test_acc = (pred[test_mask] == data.y_original[test_mask]).float().mean()
        print(f"Final Test Accuracy (clean labels): {test_acc:.4f}")
        
        return {"train_acc": train_acc.item(), "test_acc": test_acc.item()}

if __name__ == "__main__":
    print("\n" + "-"*50)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Loaded configuration file\n" + "-"*50)
    run_experiment(config)