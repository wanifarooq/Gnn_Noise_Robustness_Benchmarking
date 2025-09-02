import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
import yaml
import statistics

from utilities import setup_seed_device, load_dataset, get_model, train
from utilities import label_process
from model.NRGNN import NRGNN
from model.PI_GNN import InnerProductTrainer, Net, InnerProductDecoder
from model.CR_GNN import CRGNNTrainer
from model.LafAK import Attack, prepare_simpledata_attrs, resetBinaryClass_init, CommunityDefense
from model.RTGNN import RTGNN
from model.GraphCleaner import GraphCleanerDetector, get_noisy_ground_truth
from model.UnionNET import UnionNET
from model.GNN_Cleaner import GNNCleanerTrainer
from model.ERASE import ERASETrainer
from model.GNNGuard import GNNGuard

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data as PyGData

def run_experiment(config, run_id=1):
    seed = config['seed'] + run_id * 100
    setup_seed_device(seed)
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
        random_seed=config['noise'].get('seed', 42) + run_id * 10,
        idx_train=train_indices,
        debug=True
    )
    
    global_noisy_indices = train_indices[relative_noisy_indices]
    data.y_noisy = data.y.clone()
    data.y[train_mask] = noisy_train_labels
    
    print(f"Run {run_id}: Applied noise to {len(relative_noisy_indices)} training samples out of {train_mask.sum().item()}")
    
    supp_gnn = config['training'].get('supplementary_gnn', "")
    method = config['training']['method']
    
    # NRGNN Training
    if supp_gnn == 'nrgnn' or method == 'nrgnn':
        print(f"Run {run_id}: Using NRGNN")

        base_model = get_model(
            model_name=config['model']['name'],
            in_channels=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            out_channels=num_classes,
            n_layers=config['model'].get('n_layers', 2),
            dropout=config['model'].get('dropout', 0.5),
            self_loop=config['model'].get('self_loop', True)
        )

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.x.size(0))
        features = data.x.cpu().numpy()
        labels = data.y.cpu().numpy()
        idx_train = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        idx_val = val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        idx_test = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        
        nrgnn_config = config.get('nrgnn_params', {})
        
        class Args:
            def __init__(self, nrgnn_config):
                self.hidden = nrgnn_config.get('hidden', 64)
                self.edge_hidden = nrgnn_config.get('edge_hidden', 32)
                self.dropout = nrgnn_config.get('dropout', 0.5)
                self.lr = nrgnn_config.get('lr', 0.01)
                self.weight_decay = nrgnn_config.get('weight_decay', 5e-4)
                self.epochs = nrgnn_config.get('epochs', 400)
                self.n_p = nrgnn_config.get('n_p', 10)
                self.alpha = nrgnn_config.get('alpha', 1.0)
                self.beta = nrgnn_config.get('beta', 1.0)
                self.p_u = nrgnn_config.get('p_u', 0.7)
                self.t_small = nrgnn_config.get('t_small', 0.1)
                self.n_n = nrgnn_config.get('n_n', 1)
                self.debug = True
                self.patience = nrgnn_config.get('patience', 50)
        
        args = Args(nrgnn_config)
        
        model = NRGNN(args, device, base_model=base_model)
        
        model.fit(features, adj, labels, idx_train, idx_val)
        test_acc = model.test(idx_test)
        
        return test_acc

    # PI-GNN Training
    if supp_gnn in ['pi_gnn'] or method == 'pi_gnn':
        print(f"Run {run_id}: Using PI-GNN")
        trainer_params = config.get('pi_gnn_params', {})
        trainer = InnerProductTrainer(
            device=device,
            epochs=int(trainer_params.get('epochs', 400)),
            start_epoch=int(trainer_params.get('start_epoch', 200)),
            miself=bool(trainer_params.get('miself', False)),
            lr_main=float(trainer_params.get('lr_main', 0.01)),
            lr_mi=float(trainer_params.get('lr_mi', 0.01)),
            weight_decay=float(trainer_params.get('weight_decay', 5e-4)),
            patience=float(trainer_params.get('patience', 20)),
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
        return test_acc

    # CR-GNN Training
    if supp_gnn in ['cr_gnn'] or method == 'cr_gnn':
        print(f"Run {run_id}: Using CR-GNN")
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
            debug=True
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
        return test_acc

    # LafAK / GradientAttack Training
    if supp_gnn in ['lafak'] or method == 'lafak':
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
            gcnL2=config.get('gcn_l2', 5e-4),
            smooth_coefficient=config.get('smooth_coefficient', 1.0),
            c_max=10,
        )
        
        print("Performing gradient attack with comprehensive evaluation")
        results = attack.binaryAttack_multiclass_with_clean(
            a=target_classes[0], 
            b=target_classes[1], 
            gnn_model=gnn_model
        )
        
        print("Final attack results")
        print(f"Binary Clean accuracy: {results['binary_clean_acc']:.4f}")
        print(f"Binary Attacked accuracy: {results['binary_attacked_acc']:.4f}")
        print(f"Binary Attack success: {results['attack_success']:.4f}")

        if results.get('gnn_clean_acc') and results.get('gnn_attacked_acc'):
            print(f"GNN Clean accuracy: {results['gnn_clean_acc']:.4f}")
            print(f"GNN Attacked accuracy: {results['gnn_attacked_acc']:.4f}")
            print(f"GNN Attack success: {results['gnn_attack_success']:.4f}")


            x = torch.tensor(data_dict["_X_obs"], dtype=torch.float)
            num_nodes = x.size(0)

            y_original = torch.tensor(data_dict["_z_obs_original"], dtype=torch.long)

            y_attacked = y_original.clone()
            if 'flipped_nodes' in results:
                flipped_nodes = results['flipped_nodes']
                print(f"Applying flips to {len(flipped_nodes)} nodes in multiclass labels")
                
                binary_classes = data_dict.get('binary_classes', (target_classes[0], target_classes[1]))
                a, b = binary_classes
                
                for node_idx in flipped_nodes:
                    current_class = y_original[node_idx].item()
                    if current_class == a:
                        y_attacked[node_idx] = b
                    elif current_class == b:
                        y_attacked[node_idx] = a

            y = y_attacked

            edge_index, edge_weight = from_scipy_sparse_matrix(data_dict["_A_obs"])

            pyg_data = PyGData(x=x, edge_index=edge_index, y=y)
            if edge_weight is not None:
                pyg_data.edge_weight = edge_weight

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[data_dict['split_train']] = True
            pyg_data.train_mask = train_mask
            
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask[data_dict['split_val']] = True
            pyg_data.val_mask = val_mask
            
            test_mask = torch.zeros(num_nodes, dtype=torch.bool) 
            test_mask[data_dict['split_test']] = True
            pyg_data.test_mask = test_mask

            print(f"[Debug] Nodi totali: {num_nodes}")
            print(f"[Debug] Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
            print(f"[Debug] Classi nel dataset: {torch.unique(y)}")

            defense_cfg = config.get('lafak_defense_params', {})
            
            num_classes_full = int(y.max().item()) + 1
            print(f"[Debug] Numero classi per il modello di difesa: {num_classes_full}")
            print(f"[Debug] Min/Max labels: {y.min().item()}/{y.max().item()}")

            gnn_model_defense = get_model(
                model_name=config['model']['name'],
                in_channels=pyg_data.num_node_features,
                hidden_channels=config['model'].get('hidden_channels', 64),
                out_channels=num_classes_full,
                n_layers=config['model'].get('n_layers', 2),
                dropout=config['model'].get('dropout', 0.5),
                self_loop=config['model'].get('self_loop', True)
            )
            
            print(f"[Debug] Modello creato con {gnn_model_defense.out_channels if hasattr(gnn_model_defense, 'out_channels') else 'unknown'} output channels")

            defense = CommunityDefense(
                pyg_data,
                community_method=defense_cfg.get("community_method", "louvain"),
                lambda_comm=defense_cfg.get("lambda_comm", 2.0),
                pos_weight=1.0,
                neg_weight=2.0,
                margin=1.5,
                num_neg_samples=3,
                verbose=True
            )

            test_acc_def = defense.train_with_defense(
                gnn_model_defense,
                epochs=defense_cfg.get("epochs", 250),
                early_stopping=20,
                lr=0.005,
                weight_decay=1e-3
            )

            print(f"Accuracy with defense (all nodes): {test_acc_def:.4f}")

            return {
                "binary_attacked_acc": results['binary_attacked_acc'],
                "gnn_attacked_acc": results.get('gnn_attacked_acc'),
                "defense_acc": test_acc_def
            }

    # RTGNN Training
    if supp_gnn in ['rtgnn'] or method == 'rtgnn':
        print(f"Run {run_id}: Using RTGNN")
        
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
        
        nfeat, nclass = features.shape[1], len(np.unique(labels))
        rtgnn_backbone = config.get('rtgnn_params', {}).get('gnn_type', config['model']['name'].lower())
        rtgnn_model = RTGNN(nfeat, nclass, rtgnn_args, device, gnn_type=rtgnn_backbone).to(device)
                
        rtgnn_model.fit(features, adj, labels, idx_train, idx_val, idx_test)
        
        clean_labels = data.y_original.cpu().numpy()
        test_acc = rtgnn_model.test(features, clean_labels, idx_test)
        
        return test_acc
    
    # GraphCleaner Training
    if supp_gnn in ['graphcleaner'] or method == 'graphcleaner':
        print(f"Run {run_id}: Using GraphCleaner")
        
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
            random_seed=config['noise'].get('seed', 42) + 1000 + run_id * 10,
            idx_train=test_indices,
            debug=True
        )
        
        data_for_detection.y_noisy = data.y_noisy.clone()
        data_for_detection.y_noisy[data.test_mask] = noisy_test_labels
        data_for_detection.y = data_for_detection.y_noisy.clone()
        
        global_noisy_test_indices = test_indices[relative_noisy_test_indices]
        all_noisy_indices = torch.cat([global_noisy_indices, global_noisy_test_indices])

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
        
        return detection_results.get('accuracy', 0.0)
    
    # UnionNET Training
    if supp_gnn in ['unionnet'] or method == 'unionnet':
        print(f"Run {run_id}: Using UnionNET")
        
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
            'patience': config.get('patience', 10),
            'k': config.get('k', 5),
            'alpha': config.get('alpha', 0.5),
            'beta': config.get('alpha', 1),
            'feat_norm': config.get('feat_norm', True)
        }
        
        unionnet = UnionNET(gnn_model, data, num_classes, unionnet_config)
        result = unionnet.train(debug=True)
        
        return result['test']
    
    # GNN Cleaner Training
    if supp_gnn in ['gnn_cleaner'] or method == 'gnn_cleaner':
        print(f"Run {run_id}: Using GNN Cleaner")

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
            self_loop=config['model'].get('self_loop', True),
        )
        
        gnn_cleaner_config = {
            'epochs': config.get('total_epochs', 200),
            'lr': config.get('lr', 0.01),
            'weight_decay': config.get('weight_decay', 5e-4),
            'patience': config.get('patience', 10),
            'hidden_channels': config.get('hidden_channels', 64),
            'alpha': config.get('alpha', 0.9),
            'lp_iters': config.get('lp_iters', 30),
            'sharpen_temp': config.get('sharpen_temp', 1.0),
            'epochs': config["training"].get('total_epochs', 200)
        }

        trainer = GNNCleanerTrainer(gnn_cleaner_config, data, device, num_classes, gnn_model)
        result = trainer.train(debug=True)
        
        return result['test']
    
    # ERASE Training
    if supp_gnn in ['erase'] or method == 'erase':
        print(f"Run {run_id}: Using ERASE")
        
        erase_config = {
            'seed': seed,
            'erase_gnn_type': config['model']['name'],
            'hidden_channels': config['model'].get('hidden_channels', 128),
            'n_embedding': config['training'].get('erase_params', {}).get('n_embedding', 512),
            'n_layers': config['model'].get('n_layers', 2),
            'n_heads': config['training'].get('erase_params', {}).get('n_heads', config.get('heads', 8)),
            'dropout': config['model'].get('dropout', 0.5),
            'self_loop': config['model'].get('self_loop', True),
            'mlp_layers': config.get('mlp_layers', 2),
            'train_eps': config.get('train_eps', True),
            'lr': config.get('lr', 0.001),
            'weight_decay': config.get('weight_decay', 0.0005),
            'total_epochs': config.get('total_epochs', 200),
            'patience': config.get('patience', 5),
            'gam1': config['training'].get('erase_params', {}).get('gam1', 1.0),
            'gam2': config['training'].get('erase_params', {}).get('gam2', 2.0),
            'eps': config['training'].get('erase_params', {}).get('eps', 0.05),
            'corafull': config['training'].get('erase_params', {}).get('corafull', False),
            'alpha': config['training'].get('erase_params', {}).get('alpha', 0.6),
            'beta': config['training'].get('erase_params', {}).get('beta', 0.6),
            'T': config['training'].get('erase_params', {}).get('T', 3),
            'noise_rate': config.get('noise', {}).get('rate', 0.3),
        }
        
        trainer = ERASETrainer(erase_config, device, num_classes, get_model)
        result = trainer.train(data, debug=True)
        
        return result['test']
    
    # GNNGuard Training
    if supp_gnn in ['gnnguard'] or method == 'gnnguard':
        print(f"Run {run_id}: Using GNNGuard")
        
        gnnguard_args = {
            'dropout': config.get('dropout', 0.5),
            'lr': config.get('lr', 0.01),
            'weight_decay': config.get('weight_decay', 5e-4),
            'P0': config.get('P0', 0.5),
            'K': config.get('K', 2),
            'D2': config.get('D2', 16),
            'attention': config.get('attention', True),
            'device': device,
        }
        
        gnnguard_model = GNNGuard(
            nfeat=data.num_features,
            nhid=config.get('hidden_channels', 64),
            nclass=num_classes,
            **gnnguard_args
        ).to(device)

        adj_matrix = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

        if hasattr(data, 'train_mask'):
            idx_train = data.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            idx_val = data.val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            idx_test = data.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        else:
            num_nodes = data.num_nodes
            idx_train = np.arange(min(140, num_nodes // 5))
            idx_val = np.arange(len(idx_train), min(len(idx_train) + 500, num_nodes // 2))
            idx_test = np.arange(max(len(idx_train) + len(idx_val), num_nodes // 2), num_nodes)
        
        gnnguard_model.fit(
            features=data.x,
            adj=adj_matrix,
            labels=data.y_noisy,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            epochs=config["training"].get('total_epochs', 200),
            verbose=True,
            patience=config.get('patience', 5)
        )
        
        test_acc, test_f1 = gnnguard_model.test(idx_test)
        return test_acc

    model_params = {k: v for k, v in config['model'].items() if k not in ['name']}
    print(f"Run {run_id}: Using {config['model']['name']} with method {config['training']['method']}")
    model = get_model(
        model_name=config['model']['name'],
        in_channels=data.num_features,
        out_channels=num_classes,
        **model_params
    ).to(device)
    
    result = train(model, data, global_noisy_indices, device, config)
    return result

if __name__ == "__main__":
    print("\n" + "-"*50)
    print("Multi-run experiment: 10 runs")
    print("-"*50)
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Loaded configuration file")

    method_name = config['training'].get('supplementary_gnn', config['training']['method'])
    dataset_name = config['dataset']['name']
    noise_rate = config['noise']['rate']
    
    print(f"Dataset: {dataset_name}")
    print(f"Method: {method_name}")
    print(f"Noise Rate: {noise_rate}")
    print(f"Running 10 experiments")
    print("-"*50)
    
    test_accuracies = []
    
    for run in range(1, 11):
        try:
            print(f"\nRun {run}/10:")
            test_acc = run_experiment(config, run_id=run)
            
            if isinstance(test_acc, dict):
                if 'test_acc' in test_acc:
                    acc_value = test_acc['test_acc']
                elif 'test' in test_acc:
                    acc_value = test_acc['test']
                else:
                    acc_value = list(test_acc.values())[0]
            else:
                acc_value = test_acc
            
            test_accuracies.append(acc_value)
            print(f"Run {run} completed - Test Accuracy: {acc_value:.4f}")
            
        except Exception as e:
            print(f"Run {run} failed with error: {str(e)}")
            print("Continuing with remaining runs")
            continue
    
    if test_accuracies:
        mean_acc = np.mean(test_accuracies)
        std_acc = np.std(test_accuracies)
        min_acc = np.min(test_accuracies)
        max_acc = np.max(test_accuracies)
        
        print("\n" + "-"*50)
        print("Final results")
        print("-"*50)
        print(f"Method: {method_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Noise Rate: {noise_rate}")
        print(f"Completed Runs: {len(test_accuracies)}/10")
        print("-"*50)
        print(f"Mean Test Accuracy: {mean_acc:.4f}")
        print(f"Std Test Accuracy: {std_acc:.4f}")
        print(f"Min Test Accuracy: {min_acc:.4f}")
        print(f"Max Test Accuracy: {max_acc:.4f}")
        print("-"*50)
        print("Individual Run Results:")
        for i, acc in enumerate(test_accuracies, 1):
            print(f"  Run {i:2d}: {acc:.4f}")
        
        print(f"\nFormatted Result: {mean_acc:.4f} ± {std_acc:.4f}")
        print("-"*50)
        
    else:
        print("\n" + "-"*50)
        print("ERROR: No successful runs completed!")
        print("-"*50)