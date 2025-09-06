import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
import yaml

from utilities import setup_seed_device, load_dataset, get_model
from utilities import label_process
from model.Baseline_loss import train_with_standard_loss, train_with_ncod
from model.Positive_Eigenvalues import train_with_positive_eigenvalues
from model.GCOD_loss import train_with_gcod
from model.NRGNN import NRGNN
from model.PI_GNN import PiGnnModel, PiGnnTrainer, GraphLinkDecoder
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

def prepare_data_for_method(data, train_mask, val_mask, test_mask, noisy_train_labels, method_name):

    data_for_method = data.clone()

    data_for_method.y = data.y_original.clone()
    data_for_method.y[train_mask] = noisy_train_labels
    
    return data_for_method

def verify_label_distribution(data, train_mask, val_mask, test_mask, run_id, method_name):

    print(f"[DEBUG Run {run_id}] {method_name} - Label distribution:")
    
    if hasattr(data, 'y_original'):

        train_corrupted = (data.y[train_mask] != data.y_original[train_mask]).sum()
        print(f"  Training labels corrupted: {train_corrupted}/{train_mask.sum()} nodes")
        
        val_clean = (data.y[val_mask] == data.y_original[val_mask]).all()
        test_clean = (data.y[test_mask] == data.y_original[test_mask]).all()
        print(f"  Val labels clean: {val_clean}")
        print(f"  Test labels clean: {test_clean}")

def run_experiment(config, run_id=1):
    seed = config['seed'] + run_id * 100
    setup_seed_device(seed)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print()
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

    data.y_original = data.y.clone()
    data.y_noisy = data.y.clone()
    data.y_noisy[train_mask] = noisy_train_labels

    print(f"Run {run_id}: Applied noise to {len(relative_noisy_indices)} training samples out of {train_mask.sum().item()}")
    
    method = config['training']['method']
    
    data_for_training = prepare_data_for_method(data, train_mask, val_mask, test_mask, noisy_train_labels, method)
    
    verify_label_distribution(data_for_training, train_mask, val_mask, test_mask, run_id, method)

    if method == 'standard':
        trainer_params = config.get('training', {})
        lr = float(trainer_params.get('lr', 0.01))
        weight_decay = float(trainer_params.get('weight_decay', 5e-4))
        epochs = int(trainer_params.get('epochs'))
        patience = int(trainer_params.get('patience', 100))

        model_params = {k: v for k, v in config['model'].items() if k not in ['name', 'hidden_channels', 'n_layers', 'dropout', 'self_loop']}
        base_model = get_model(
            model_name=config['model']['name'],
            in_channels=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            out_channels=num_classes,
            n_layers=config['model'].get('n_layers', 2),
            dropout=config['model'].get('dropout', 0.5),
            self_loop=config['model'].get('self_loop', True),
            **model_params
        )

        result = train_with_standard_loss(
            base_model, data_for_training, global_noisy_indices, device,
            total_epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience
        )
        
        return {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'precision': result['precision'],
            'recall': result['recall'],
            'oversmoothing': result['oversmoothing']
        }

    elif method == 'ncod':
        trainer_params = config.get('training', {})
        lr = float(trainer_params.get('lr', 0.01))
        weight_decay = float(trainer_params.get('weight_decay', 5e-4))
        epochs = int(trainer_params.get('epochs'))
        patience = int(trainer_params.get('patience', 100))

        trainer_params_method = config.get('ncod_params', {})
        lambda_dir = float(trainer_params_method.get('lambda_dir', 0.1))

        model_params = {k: v for k, v in config['model'].items() if k not in ['name', 'hidden_channels', 'n_layers', 'dropout', 'self_loop']}
        base_model = get_model(
            model_name=config['model']['name'],
            in_channels=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            out_channels=num_classes,
            n_layers=config['model'].get('n_layers', 2),
            dropout=config['model'].get('dropout', 0.5),
            self_loop=config['model'].get('self_loop', True),
            **model_params
        )

        result = train_with_ncod(
            base_model, data_for_training, global_noisy_indices, device,
            total_epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            lambda_dir=lambda_dir,
            patience=patience
        )
        
        return {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'precision': result['precision'],
            'recall': result['recall'],
            'oversmoothing': result['oversmoothing']
        }

    elif method == 'positive_eigenvalues':
        trainer_params = config.get('training', {})
        lr = float(trainer_params.get('lr', 0.01))
        weight_decay = float(trainer_params.get('weight_decay', 5e-4))
        epochs = int(trainer_params.get('epochs'))
        patience = int(trainer_params.get('patience', 100))

        trainer_params_method = config.get('positive_eigenvalues_params', {})
        batch_size = int(trainer_params_method.get('batch_size', 32))

        model_params = {k: v for k, v in config['model'].items() if k not in ['name', 'hidden_channels', 'n_layers', 'dropout', 'self_loop']}
        base_model = get_model(
            model_name=config['model']['name'],
            in_channels=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            out_channels=num_classes,
            n_layers=config['model'].get('n_layers', 2),
            dropout=config['model'].get('dropout', 0.5),
            self_loop=config['model'].get('self_loop', True),
            **model_params
        )

        result = train_with_positive_eigenvalues(
            base_model, data_for_training, global_noisy_indices, device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            patience=patience
        )
        
        return {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'precision': result['precision'],
            'recall': result['recall'],
            'oversmoothing': result['oversmoothing']
        }

    elif method == 'gcod':
        trainer_params = config.get('training', {})
        lr = float(trainer_params.get('lr', 0.01))
        weight_decay = float(trainer_params.get('weight_decay', 5e-4))
        epochs = int(trainer_params.get('epochs'))
        patience = int(trainer_params.get('patience', 100))

        trainer_params_method = config.get('gcod_params', {})
        batch_size = int(trainer_params_method.get('batch_size', 32))
        u_lr = float(trainer_params_method.get('u_lr', 1.0))

        model_params = {k: v for k, v in config['model'].items() if k not in ['name', 'hidden_channels', 'n_layers', 'dropout', 'self_loop']}
        base_model = get_model(
            model_name=config['model']['name'],
            in_channels=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            out_channels=num_classes,
            n_layers=config['model'].get('n_layers', 2),
            dropout=config['model'].get('dropout', 0.5),
            self_loop=config['model'].get('self_loop', True),
            **model_params
        )

        result = train_with_gcod(
            base_model, data_for_training, global_noisy_indices, device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            u_lr=u_lr,
            patience=patience
        )
        
        return {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'precision': result['precision'],
            'recall': result['recall'],
            'oversmoothing': result['oversmoothing']
        }
    
    # NRGNN Training
    elif method == 'nrgnn':
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
        
        adj = to_scipy_sparse_matrix(data_for_training.edge_index, num_nodes=data_for_training.x.size(0))
        features = data_for_training.x.to(device)
        labels = data_for_training.y.to(device)
        idx_train = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        idx_val = val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        idx_test = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()

        nrgnn_config = {
            'lr': config['training'].get('lr', 0.01),
            'weight_decay': config['training'].get('weight_decay', 5e-4),
            'epochs': config['training'].get('epochs', 1000),
            'patience': config['training'].get('patience', 100),
            'nrgnn_params': config['training'].get('nrgnn_params', {})
        }
        
        nrgnn_model = NRGNN(nrgnn_config, device, base_model=base_model)
        nrgnn_model.fit(features, adj, labels, idx_train, idx_val)
        
        test_results = nrgnn_model.test(idx_test)
        
        return {
            'accuracy': test_results['test_acc'],
            'f1': test_results['test_f1'],
            'precision': test_results['test_precision'],
            'recall': test_results['test_recall'],
            'oversmoothing': test_results['test_oversmoothing']
        }

    # PI-GNN Training
    elif method == 'pi_gnn':
        print(f"Run {run_id}: Using PI-GNN")

        pi_gnn_config = config.get('pi_gnn_params', {})
        training_config = config.get('training', {})
        
        trainer = PiGnnTrainer(
            device=device,
            epochs=int(training_config.get('epochs', 400)),
            mutual_info_start_epoch=int(pi_gnn_config.get('start_epoch', 200)),
            use_self_mi=bool(pi_gnn_config.get('miself', False)),
            main_learning_rate=float(pi_gnn_config.get('lr_main', 0.01)),
            mi_learning_rate=float(pi_gnn_config.get('lr_mi', 0.01)),
            weight_decay=float(training_config.get('weight_decay', 5e-4)),
            early_stopping_patience=int(training_config.get('patience', 50)),
            normalization_factor=pi_gnn_config.get('norm', None),
            use_vanilla_training=bool(pi_gnn_config.get('vanilla', False)),
        )
        
        model_parameters = {k: v for k, v in config['model'].items() if k not in ['name']}
        
        base_gnn_model = get_model(
            model_name=config['model']['name'],
            in_channels=data.num_features,
            out_channels=num_classes,
            **model_parameters
        )
        
        link_decoder = GraphLinkDecoder()
        
        pi_gnn_model = PiGnnModel(backbone_gnn=base_gnn_model, supplementary_decoder=link_decoder)
        
        trainer_model_config = {
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
        
        test_results = trainer.train_model(pi_gnn_model, data_for_training, trainer_model_config, get_model)
        
        return {
            'accuracy': torch.tensor(test_results['accuracy']),
            'f1': torch.tensor(test_results['f1']),
            'precision': torch.tensor(test_results['precision']),
            'recall': torch.tensor(test_results['recall']),
            'oversmoothing': test_results['oversmoothing']
        }

    # CR-GNN Training
    elif method == 'cr_gnn':
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
        
        test_results = trainer.fit(base_model, data_for_training, trainer_config, get_model)
        
        return {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'oversmoothing': test_results['oversmoothing']
        }

    # LafAK / GradientAttack Training
    elif method == 'lafak':
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

            test_results = defense.train_with_defense(
                gnn_model_defense,
                epochs=defense_cfg.get("epochs", 250),
                early_stopping=20,
                lr=0.005,
                weight_decay=1e-3
            )

            print(f"Defense completed!")
            print(f"Test Accuracy: {test_results['accuracy']:.4f}")

            return {
                'accuracy': test_results['accuracy'],
                'f1': test_results['f1'],
                'precision': test_results['precision'],
                'recall': test_results['recall'],
                'oversmoothing': test_results['oversmoothing']
            }

    # RTGNN Training
    elif method == 'rtgnn':
        print(f"Run {run_id}: Using RTGNN")
        
        class RTGNNTrainingConfig:
            def __init__(self, config_dict):
                rtgnn_params = config_dict.get('rtgnn_params', {})
                training_params = config_dict.get('training', {})
                model_params = config_dict.get('model', {})
                
                self.epochs = training_params.get('epochs', 200)
                self.lr = training_params.get('lr', 0.001)  
                self.weight_decay = float(training_params.get('weight_decay', 5e-4))
                self.patience = training_params.get('patience', 8)
                self.dropout = model_params.get('dropout', 0.5)

                self.hidden = model_params.get('hidden', 128)
                self.edge_hidden = rtgnn_params.get('edge_hidden', 64)
                self.n_layers = model_params.get('n_layers', 2)
                self.self_loop = model_params.get('self_loop', True)
                self.mlp_layers = model_params.get('mlp_layers', 2)
                self.train_eps = model_params.get('train_eps', True)
                self.heads = model_params.get('heads', 8)
                
                self.co_lambda = rtgnn_params.get('co_lambda', 0.1)
                self.alpha = rtgnn_params.get('alpha', 0.3)
                self.th = rtgnn_params.get('th', 0.8)
                self.K = rtgnn_params.get('K', 50)
                self.tau = rtgnn_params.get('tau', 0.05)
                self.n_neg = rtgnn_params.get('n_neg', 100)
        
        rtgnn_training_config = RTGNNTrainingConfig(config)
        
        node_features = data_for_training.x.cpu().numpy()
        node_labels = data_for_training.y.cpu().numpy()
        
        print(f"[DEBUG] Checking label corruption in RTGNN training:")
        if hasattr(data_for_training, 'y_original'):
            corrupted_count = (data_for_training.y[train_mask] != data_for_training.y_original[train_mask]).sum().item()
            print(f"[DEBUG] Training labels corrupted: {corrupted_count}/{train_mask.sum().item()} nodes")
            val_clean = (data_for_training.y[val_mask] == data_for_training.y_original[val_mask]).all().item()
            test_clean = (data_for_training.y[test_mask] == data_for_training.y_original[test_mask]).all().item()
            print(f"[DEBUG] Validation labels clean: {val_clean}, Test labels clean: {test_clean}")
        
        adjacency_matrix = to_scipy_sparse_matrix(data_for_training.edge_index, num_nodes=data_for_training.num_nodes)
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        adjacency_matrix = adjacency_matrix.tolil()
        adjacency_matrix[adjacency_matrix > 1] = 1
        adjacency_matrix.setdiag(0)
        
        sparse_features = sp.csr_matrix(node_features)
        row_sums = np.array(sparse_features.sum(1))
        reciprocal_row_sums = np.power(row_sums, -1).flatten()
        reciprocal_row_sums[np.isinf(reciprocal_row_sums)] = 0.
        row_normalization_matrix = sp.diags(reciprocal_row_sums)
        normalized_features = row_normalization_matrix.dot(sparse_features)
        normalized_features = torch.FloatTensor(np.array(normalized_features.todense()))

        if hasattr(data_for_training, 'train_mask'):
            train_node_indices = data_for_training.train_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
            val_node_indices = data_for_training.val_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
            test_node_indices = data_for_training.test_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
        else:
            num_nodes = data_for_training.num_nodes
            train_node_indices = list(range(min(140, num_nodes // 5)))
            val_node_indices = list(range(len(train_node_indices), min(len(train_node_indices) + 500, num_nodes // 2)))
            test_node_indices = list(range(max(len(train_node_indices) + len(val_node_indices), num_nodes // 2), num_nodes))
        
        num_input_features, num_classes = normalized_features.shape[1], len(np.unique(node_labels))
        rtgnn_backbone_type = config.get('rtgnn_params', {}).get('gnn_type', config['model']['name'].lower())
        
        rtgnn_trainer = RTGNN(
            input_features=num_input_features, 
            num_classes=num_classes, 
            training_config=rtgnn_training_config, 
            device=device, 
            gnn_backbone=rtgnn_backbone_type
        ).to(device)
                
        rtgnn_trainer.train_model(
            node_features=normalized_features, 
            adjacency_matrix=adjacency_matrix, 
            node_labels=node_labels,
            train_indices=train_node_indices, 
            val_indices=val_node_indices, 
            test_indices=test_node_indices
        )
        
        clean_labels = data.y_original.cpu().numpy()
        test_results = rtgnn_trainer.evaluate_final_performance(
            node_features=normalized_features, 
            node_labels=clean_labels,
            test_indices=test_node_indices
        )

        print(f"RTGNN Training completed!")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test F1: {test_results['f1']:.4f}")
        print(f"Test Precision: {test_results['precision']:.4f}")
        print(f"Test Recall: {test_results['recall']:.4f}")

        return {
            'accuracy': torch.tensor(test_results['accuracy']),
            'f1': torch.tensor(test_results['f1']),
            'precision': torch.tensor(test_results['precision']),
            'recall': torch.tensor(test_results['recall']),
            'oversmoothing': test_results['oversmoothing']
        }
    
    # GraphCleaner Training 
    elif method == 'graphcleaner':
        print(f"Run {run_id}: Using GraphCleaner")
        data_for_detection = data_for_training.clone()
        
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
        
        data_for_detection.y[data.test_mask] = noisy_test_labels
        
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
        predictions, probs, classifier, trained_model = detector.detect_noise(data_for_detection, base_model, num_classes)
        
        test_ground_truth = get_noisy_ground_truth(data_for_detection, all_noisy_indices)[data.test_mask.cpu()].cpu().numpy()
        
        detection_results = detector.evaluate_detection(
            predictions, 
            test_ground_truth, 
            probs, 
            model=trained_model, 
            data=data_for_detection
        )
        
        return detection_results
    
    # UnionNET Training
    elif method == 'unionnet':
        print(f"Run {run_id}: Using UnionNET")
        gnn_model = get_model(
            model_name=config['model']['name'],
            in_channels=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            out_channels=num_classes,
            mlp_layers=config['model'].get('mlp_layers', 2),
            train_eps=config['model'].get('train_eps', True),
            heads=config['model'].get('heads', 8),
            n_layers=config['model'].get('n_layers', 2),
            dropout=config['model'].get('dropout', 0.5),
            self_loop=config['model'].get('self_loop', True)
        ).to(device)
        
        unionnet_config = {
            'n_epochs': config['training'].get('epochs', 200),
            'lr': config['training'].get('lr', 0.01),
            'weight_decay': config['training'].get('weight_decay', 5e-4),
            'patience': config['training'].get('patience', 10),
            'k': config['unionnet_params'].get('k', 5),
            'alpha': config['unionnet_params'].get('alpha', 0.5),
            'beta': config['unionnet_params'].get('beta', 1.0),
            'feat_norm': config['unionnet_params'].get('feat_norm', True)
        }

        unionnet_trainer = UnionNET(gnn_model, data_for_training, num_classes, unionnet_config)
        test_results = unionnet_trainer.train_model(enable_debug=True)
        
        return {
            'accuracy': torch.tensor(test_results['accuracy']),
            'f1': torch.tensor(test_results['f1']),
            'precision': torch.tensor(test_results['precision']),
            'recall': torch.tensor(test_results['recall']),
            'oversmoothing': test_results['oversmoothing']
        }

    # GNN Cleaner Training
    elif method == 'gnn_cleaner':
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
            'epochs': config["training"].get('total_epochs', 200),
            'lr': config.get('lr', 0.01),
            'net_lr': config.get('net_lr', 0.001),
            'weight_decay': config.get('weight_decay', 5e-4),
            'lp_iters': config.get('lp_iters', 50),
            'epsilon': config.get('epsilon', 1e-8),
            'patience': config.get('patience', 10),
        }
        
        trainer = GNNCleanerTrainer(gnn_cleaner_config, data_for_training, device, num_classes, gnn_model)
        result = trainer.train(debug=True)
        
        return result
    
    # ERASE Training
    elif method == 'erase':
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
        result = trainer.train(data_for_training, debug=True)
        
        return result
    
    # GNNGuard Training
    elif method == 'gnnguard':
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
        
        adj_matrix = to_scipy_sparse_matrix(data_for_training.edge_index, num_nodes=data_for_training.num_nodes)
        if hasattr(data_for_training, 'train_mask'):
            idx_train = data_for_training.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            idx_val = data_for_training.val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            idx_test = data_for_training.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        else:
            num_nodes = data_for_training.num_nodes
            idx_train = np.arange(min(140, num_nodes // 5))
            idx_val = np.arange(len(idx_train), min(len(idx_train) + 500, num_nodes // 2))
            idx_test = np.arange(max(len(idx_train) + len(idx_val), num_nodes // 2), num_nodes)
        
        gnnguard_model.fit(
            features=data_for_training.x,
            adj=adj_matrix,
            labels=data_for_training.y,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            epochs=config["training"].get('total_epochs', 200),
            verbose=True,
            patience=config.get('patience', 5)
        )
        
        test_results = gnnguard_model.test(idx_test)
        
        return {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'oversmoothing': test_results['oversmoothing']
        }
    
    else:
        raise ValueError(
            f"Run {run_id}: Training method '{method}' not implemented. "
            "Please choose one of the implemented methods: "
            "[standard, ncod, positive_eigenvalues, gcod, nrgnn, pi_gnn, cr_gnn, lafak, rtgnn, graphcleaner, unionnet, gnn_cleaner, erase, gnnguard]"
        )

if __name__ == "__main__":
    print("\n" + "-"*50)
    print("Multi-run experiment: 5 runs")
    print("-"*50)
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Loaded configuration file")

    method_name = config['training']['method']
    dataset_name = config['dataset']['name']
    noise_rate = config['noise']['rate']
    
    print(f"Dataset: {dataset_name}")
    print(f"Method: {method_name}")
    print(f"Noise Rate: {noise_rate}")
    device_str = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.device(device_str))}")

    print(f"Running 5 experiments")
    print("-"*50)
    
    test_accuracies = []
    test_f1s = []
    test_precisions = []
    test_recalls = []

    oversmoothing_metrics = {
        'NumRank': [],
        'Erank': [],
        'EDir': [],
        'EDir_traditional': [],
        'EProj': [],
        'MAD': []
    }

    for run in range(1, 6):

            print(f"\nRun {run}/5:")
            test_metrics = run_experiment(config, run_id=run)
            test_acc = test_metrics['accuracy']
            test_f1 = test_metrics['f1']
            test_precision = test_metrics['precision']
            test_recall = test_metrics['recall']
            test_overs = test_metrics['oversmoothing']

            test_accuracies.append(test_acc.item())
            test_f1s.append(test_f1.item())
            test_precisions.append(test_precision.item())
            test_recalls.append(test_recall.item())

            for key in oversmoothing_metrics:
                oversmoothing_metrics[key].append(test_overs[key])

            print(f"Run {run} completed - Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}, "
                  f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
            

        
    if test_accuracies:
        mean_std_dict = {
            'Accuracy': (np.mean(test_accuracies), np.std(test_accuracies)),
            'F1': (np.mean(test_f1s), np.std(test_f1s)),
            'Precision': (np.mean(test_precisions), np.std(test_precisions)),
            'Recall': (np.mean(test_recalls), np.std(test_recalls))
        }

        print("\n" + "-"*50)
        print("Final results")
        print("-"*50)
        for metric, (mean_val, std_val) in mean_std_dict.items():
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("\nOversmoothing Metrics:")
        for key in oversmoothing_metrics:
            mean_val = np.mean(oversmoothing_metrics[key])
            std_val = np.std(oversmoothing_metrics[key])
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")

        print("-"*50)
    else:
        print("\nERROR: No successful runs completed!")
        print("-"*50)