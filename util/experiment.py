"""Experiment orchestration — setup pipeline and 13-method dispatch."""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix

from util.seed import setup_seed_device
from util.data import load_dataset, ensure_splits, prepare_data_for_method, verify_label_distribution
from util.noise import noise_operation
from util.profiling import get_model, profile_model_flops, _reduce_oversmoothing

from model.Standard import train_with_standard_loss
from model.Positive_Eigenvalues import PositiveEigenvaluesTrainer
from model.GCOD_loss import GCODTrainer
from model.NRGNN import NRGNN
from model.PI_GNN import PiGnnModel, PiGnnTrainer, GraphLinkDecoder
from model.CR_GNN import CRGNNModel
from model.CommunityDefense import GraphCommunityDefenseTrainer
from model.RTGNN import RTGNN, RTGNNTrainingConfig
from model.GraphCleaner import GraphCleanerNoiseDetector
from model.UnionNET import UnionNET
from model.GNN_Cleaner import GNNCleanerTrainer
from model.ERASE import ERASETrainer
from model.GNNGuard import GNNGuardTrainer


def initialize_experiment(config, run_id=1):
    """Full setup pipeline: seed -> data -> noise -> backbone -> profile.

    Returns a dict with all objects and scalars needed by ``run_experiment``.
    """

    # Seed e device
    seed = config.get('seed', 42) + run_id * 100
    setup_seed_device(seed)
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    # Dataset
    data, num_classes = load_dataset(config['dataset'].get('name', 'cora'), root=config['dataset'].get('root', './data'))
    if not isinstance(data, Data):
        data = data[0]
    data = data.to(device)
    if not hasattr(data, "train_mask"):
        train_mask, val_mask, test_mask = ensure_splits(data, config["seed"])
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    elif data.train_mask.dim() > 1 and data.train_mask.shape[1] > 1:
        train_mask = data.train_mask[:,0]
        val_mask = data.val_mask[:,0]
        test_mask = data.test_mask[:,0]
    else:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    data.y_original = data.y.clone()

    train_labels = data.y[train_mask]
    train_features = data.x[train_mask] if config['noise'].get('type', 'clean') == 'instance' else None
    train_indices = train_mask.nonzero(as_tuple=True)[0]

    noisy_train_labels, relative_noisy_indices = noise_operation(
        train_labels,
        train_features,
        num_classes,
        noise_type=config['noise'].get('type', 'clean'),
        noise_rate=config['noise'].get('rate', 0),
        noise_seed=config['noise'].get('seed', 42) + run_id * 10,
        idx_train=train_indices,
        debug=True
    )

    global_noisy_indices = train_indices[relative_noisy_indices]

    data.y_noisy = data.y.clone()
    data.y_noisy[train_mask] = noisy_train_labels

    print(f"Run {run_id}: Applied noise to {len(relative_noisy_indices)} training samples out of {train_mask.sum().item()}")

    method = config['training']['method']
    data_for_training = prepare_data_for_method(data, train_mask, val_mask, test_mask, noisy_train_labels, method)

    verify_label_distribution(data_for_training, train_mask, val_mask, test_mask, run_id, method)

    # backbone model
    backbone_model = get_model(
        model_name=config['model'].get('name', 'standard'),
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

    flops_info = profile_model_flops(backbone_model, data_for_training, device)


    # training parameters
    trainer_params = config.get('training', {})
    lr = float(trainer_params.get('lr', 0.01))
    weight_decay = float(trainer_params.get('weight_decay', 5e-4))
    epochs = int(trainer_params.get('epochs', 20))
    patience = int(trainer_params.get('patience', 100))

    return {
        'device': device,
        'data': data,
        'num_classes': num_classes,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'train_indices': train_indices,
        'relative_noisy_indices': relative_noisy_indices,
        'noisy_train_labels': noisy_train_labels,
        'global_noisy_indices': global_noisy_indices,
        'data_for_training': data_for_training,
        'backbone_model': backbone_model,
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'patience': patience,
        'method': method,
        'seed': seed,
        'flops_info' : flops_info,
    }

def run_experiment(config, run_id=1):
    init_data = initialize_experiment(config, run_id)

    device = init_data['device']
    data = init_data['data']
    num_classes = init_data['num_classes']
    train_mask = init_data['train_mask']
    val_mask = init_data['val_mask']
    test_mask = init_data['test_mask']
    global_noisy_indices = init_data['global_noisy_indices']
    data_for_training = init_data['data_for_training']
    backbone_model = init_data['backbone_model']
    lr = init_data['lr']
    weight_decay = init_data['weight_decay']
    epochs = init_data['epochs']
    patience = init_data['patience']
    method = init_data['method']
    seed = init_data['seed']
    flops_info = init_data['flops_info']

    # Standard Training
    if method == 'standard':

        result = train_with_standard_loss(
            backbone_model, data_for_training, global_noisy_indices, device,
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
            'oversmoothing': result['oversmoothing'],
            'train_oversmoothing': _reduce_oversmoothing(result['train_oversmoothing']),
            'flops_info' : flops_info,
        }

    # Positive eigenvalues Training
    elif method == 'positive_eigenvalues':

        trainer_params_method = config.get('positive_eigenvalues_params', {})
        batch_size = int(trainer_params_method.get('batch_size', 32))

        trainer = PositiveEigenvaluesTrainer(
            model=backbone_model,
            data=data_for_training,
            device=device,
            learning_rate=lr,
            weight_decay=weight_decay
        )

        result = trainer.train_with_positive_eigenvalue_constraint(
            max_epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            noisy_indices=global_noisy_indices
        )

        return {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'precision': result['precision'],
            'recall': result['recall'],
            'oversmoothing': result['oversmoothing'],
            'train_oversmoothing': _reduce_oversmoothing(result['train_oversmoothing']),
            'flops_info' : flops_info,
        }

    # GCOD Training
    elif method == 'gcod':

        gcod_params = config.get('gcod_params', {})
        batch_size = int(gcod_params.get('batch_size', 32))
        uncertainty_lr = float(gcod_params.get('uncertainty_lr', 1.0))

        trainer = GCODTrainer(
            model=backbone_model,
            data=data_for_training,
            noisy_indices=global_noisy_indices,
            device=device,
            learning_rate=lr,
            weight_decay=weight_decay,
            uncertainty_lr=uncertainty_lr,
            total_epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            debug=True
        )

        result, train_oversmoothing = trainer.train_full_model()

        return {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'precision': result['precision'],
            'recall': result['recall'],
            'oversmoothing': result['oversmoothing'],
            'train_oversmoothing': train_oversmoothing,
            'flops_info' : flops_info,
        }

    # NRGNN Training
    elif method == 'nrgnn':
        print(f"Run {run_id}: Using NRGNN")

        nrgnn_config = {
            'lr': lr,
            'weight_decay': weight_decay,
            'epochs': epochs,
            'patience': patience,
            'nrgnn_params': config.get('nrgnn_params', {})
        }

        nrgnn_model = NRGNN(nrgnn_config, device, base_model=backbone_model)
        train_oversmoothing = nrgnn_model.fit(data_for_training.x.to(device), to_scipy_sparse_matrix(data_for_training.edge_index, num_nodes=data_for_training.x.size(0)), data_for_training.y.to(device), train_mask.nonzero(as_tuple=True)[0].cpu().numpy(), val_mask.nonzero(as_tuple=True)[0].cpu().numpy())

        test_results = nrgnn_model.test(test_mask.nonzero(as_tuple=True)[0].cpu().numpy())

        return {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': _reduce_oversmoothing(train_oversmoothing),
            'flops_info' : flops_info,
        }

    # PI-GNN Training
    elif method == 'pi_gnn':
        print(f"Run {run_id}: Using PI-GNN")

        pi_gnn_config = config.get('pi_gnn_params', {})

        trainer = PiGnnTrainer(
            device=device,
            epochs=int(epochs),
            main_learning_rate=float(lr),
            mi_learning_rate=float(lr),
            weight_decay=float(weight_decay),
            early_stopping_patience=int(patience),
            mutual_info_start_epoch=int(pi_gnn_config.get('start_epoch', 200)),
            use_self_mi=bool(pi_gnn_config.get('miself', False)),
            normalization_factor=pi_gnn_config.get('norm', None),
            use_vanilla_training=bool(pi_gnn_config.get('vanilla', False)),
        )

        link_decoder = GraphLinkDecoder()

        pi_gnn_model = PiGnnModel(backbone_gnn=backbone_model, supplementary_decoder=link_decoder)

        test_results = trainer.train_model(pi_gnn_model, data_for_training, config, get_model)

        return {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': _reduce_oversmoothing(test_results['train_oversmoothing']),
            'flops_info' : flops_info,
        }

    # CR-GNN Training
    elif method == 'cr_gnn':
        print(f"Run {run_id}: Using CR-GNN")

        cr_gnn_specific_params = config.get('cr_gnn_params', {})

        combined_params = {
            'hidden_channels': config['model'].get('hidden_channels', 64),
            'lr': lr,
            'weight_decay': float(weight_decay),
            'epochs': epochs,
            'patience': patience,
        }

        combined_params.update(cr_gnn_specific_params)

        cr_gnn_model = CRGNNModel(
            device=device,
            **combined_params
        )

        test_results = cr_gnn_model.train_model(backbone_model, data_for_training, backbone_model, get_model)

        return {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': _reduce_oversmoothing(test_results['train_oversmoothing']),
            'flops_info' : flops_info,
        }

    # Community Defense Training
    elif method == 'community_defense':
        print(f"Run {run_id}: Using Community Defense")

        comm_params = config.get('community_defense_params', {})

        defense_trainer = GraphCommunityDefenseTrainer(
            graph_data=data_for_training,
            community_detection_method=comm_params.get("community_method", "louvain"),
            num_communities=comm_params.get("num_communities", None),
            community_loss_weight=float(comm_params.get("lambda_comm", 2.0)),
            positive_pair_weight=float(comm_params.get("pos_weight", 1.0)),
            negative_pair_weight=float(comm_params.get("neg_weight", 2.0)),
            similarity_margin=float(comm_params.get("margin", 1.5)),
            negative_samples_per_node=int(comm_params.get("num_neg_samples", 3)),
            device=device,
            verbose=True
        )

        test_results = defense_trainer.train_with_community_defense(
            gnn_model=backbone_model,
            training_epochs=int(epochs),
            early_stopping_patience=int(patience),
            learning_rate=float(lr),
            weight_decay_rate=float(weight_decay),
            enable_debug=True
        )

        print("Defense completed!")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")


        return {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': _reduce_oversmoothing(test_results['train_oversmoothing']),
            'flops_info' : flops_info
        }

    # RTGNN Training
    elif method == 'rtgnn':
        print(f"Run {run_id}: Using RTGNN")

        rtgnn_training_config = RTGNNTrainingConfig(config)

        rtgnn_trainer = RTGNN(
            training_config=rtgnn_training_config,
            device=device,
            gnn_backbone=config['model']['name'].lower(),
            data_for_training=data_for_training
        ).to(device)

        results = rtgnn_trainer.train_model()

        clean_labels = data.y_original.cpu().numpy()
        test_results = rtgnn_trainer.evaluate_final_performance(clean_labels=clean_labels)

        print("RTGNN Training completed!")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test F1: {test_results['f1']:.4f}")
        print(f"Test Precision: {test_results['precision']:.4f}")
        print(f"Test Recall: {test_results['recall']:.4f}")

        return {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'oversmoothing': test_results['oversmoothing'],
            'flops_info' : flops_info,
            'train_oversmoothing' : _reduce_oversmoothing(results)
        }

    # GraphCleaner Training
    elif method == 'graphcleaner':
        print(f"Run {run_id}: Using GraphCleaner for robust training")

        graphcleaner_detector = GraphCleanerNoiseDetector(
            configuration_params=config,
            computation_device=device,
            random_seed=seed
        )

        clean_train_mask, cleaned_data = graphcleaner_detector.clean_training_data(
            graph_data=data_for_training,
            neural_network_model=backbone_model,
            num_classes=num_classes
        )

        final_training_data = data.clone()
        final_training_data.train_mask = clean_train_mask
        final_training_data.y = data_for_training.y.clone()
        noisy_indices_after_cleaning = (~clean_train_mask & data_for_training.train_mask).nonzero(as_tuple=True)[0]

        result = train_with_standard_loss(
            backbone_model,
            final_training_data,
            noisy_indices_after_cleaning,
            device=device,
            total_epochs=int(epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            patience=int(patience)
        )


        return {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'precision': result['precision'],
            'recall': result['recall'],
            'oversmoothing': result['oversmoothing'],
            'train_oversmoothing': _reduce_oversmoothing(result['train_oversmoothing']),
            'flops_info' : flops_info,
        }

    # UnionNET Training
    elif method == 'unionnet':
        print(f"Run {run_id}: Using UnionNET")

        unionnet_config = {
            'n_epochs': epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'patience': patience,
            'k': config['unionnet_params'].get('k', 5),
            'alpha': config['unionnet_params'].get('alpha', 0.5),
            'beta': config['unionnet_params'].get('beta', 1.0),
            'feat_norm': config['unionnet_params'].get('feat_norm', True)
        }

        unionnet_trainer = UnionNET(backbone_model, data_for_training, num_classes, unionnet_config)
        test_results = unionnet_trainer.train_model(enable_debug=True)

        return {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': _reduce_oversmoothing(test_results['train_oversmoothing']),
            'flops_info' : flops_info,
        }

    # GNN Cleaner Training
    elif method == 'gnn_cleaner':
        print(f"Run {run_id}: Using GNN Cleaner method")

        train_corrupted = (data_for_training.y[train_mask] != data_for_training.y_original[train_mask]).sum()
        print(f"Training corruption: {train_corrupted}/{train_mask.sum()} nodes")

        gnn_cleaner_configuration = {
            'max_epochs': epochs,
            'model_learning_rate': lr,
            'net_learning_rate': lr,
            'weight_decay': weight_decay,
            'early_stopping_patience': patience,
            'label_propagation_iterations': config['gnn_cleaner_params'].get('label_propagation_iterations', 50),
            'similarity_epsilon': config['gnn_cleaner_params'].get('similarity_epsilon', 1e-8),
        }

        gnn_cleaner_trainer = GNNCleanerTrainer(
            gnn_cleaner_configuration,
            data_for_training,
            device,
            num_classes,
            backbone_model
        )

        test_results = gnn_cleaner_trainer.execute_full_training(enable_debug_output=True)

        return {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': _reduce_oversmoothing(test_results['train_oversmoothing']),
            'flops_info' : flops_info,
        }

    # ERASE Training
    elif method == 'erase':
        print(f"Run {run_id}: Using ERASE method")

        erase_specific_params = config['erase_params']

        erase_training_config = {
            'erase_gnn_type': config['model']['name'],
            'in_channels': data.num_features,
            'hidden_channels': config['model'].get('hidden_channels', 128),
            'n_embedding': erase_specific_params.get('n_embedding', 512),
            'n_layers': config['model'].get('n_layers', 2),
            'n_heads': erase_specific_params.get('n_heads', config['model'].get('heads', 8)),
            'dropout': config['model'].get('dropout', 0.5),
            'self_loop': config['model'].get('self_loop', True),
            'mlp_layers': config['model'].get('mlp_layers', 2),
            'train_eps': config['model'].get('train_eps', True),

            'lr': lr,
            'weight_decay': float(weight_decay),
            'total_epochs': epochs,
            'patience': patience,

            'gam1': erase_specific_params.get('gam1', 1.0),
            'gam2': erase_specific_params.get('gam2', 2.0),
            'eps': erase_specific_params.get('eps', 0.05),
            'alpha': erase_specific_params.get('alpha', 0.6),
            'beta': erase_specific_params.get('beta', 0.6),
            'T': erase_specific_params.get('T', 3),
            'use_layer_norm': erase_specific_params.get('use_layer_norm', False),
            'use_residual': erase_specific_params.get('use_residual', False),
            'use_residual_linear': erase_specific_params.get('use_residual_linear', False),
            'seed': seed,
        }

        erase_trainer = ERASETrainer(
            training_config=erase_training_config,
            computation_device=device,
            num_node_classes=num_classes,
            model_creation_function=get_model
        )

        test_results = erase_trainer.train_erase_model(data_for_training, enable_debug_output=True)

        return {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': _reduce_oversmoothing(test_results['train_oversmoothing']),
            'flops_info' : flops_info,
        }

    # GNNGuard Training
    elif method == 'gnnguard':
        print(f"Run {run_id}: Using GNNGuard")

        gnnguard_config = config.get('gnnguard_params', {})

        gnnguard_trainer = GNNGuardTrainer(
            input_features=data.num_features,
            hidden_channels=config['model'].get('hidden_channels', 64),
            num_classes=num_classes,
            dropout=config['model'].get('dropout', 0.5),
            lr=lr,
            weight_decay=float(weight_decay),
            attention=gnnguard_config.get('attention', True),
            device=device,
            similarity_threshold=gnnguard_config.get('P0', 0.5),
            num_layers=gnnguard_config.get('K', 2),
            attention_dim=gnnguard_config.get('D2', 16),
            data_for_training=data_for_training,
            backbone=backbone_model,
        )

        gnnguard_trainer.prepare_data()

        train_oversmoothing = gnnguard_trainer.train_model(
            node_features=data_for_training.x,
            node_labels=data_for_training.y,
            max_epochs=epochs,
            verbose=True,
            patience=patience
        )

        test_results = gnnguard_trainer.evaluate_model()

        return {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': _reduce_oversmoothing(train_oversmoothing),
            'flops_info' : flops_info,
        }

    else:
        raise ValueError(
            f"Run {run_id}: Training method '{method}' not implemented. "
            "Please choose one of the implemented methods: "
            "[standard, positive_eigenvalues, gcod, nrgnn, pi_gnn, cr_gnn, community_defense, rtgnn, graphcleaner, unionnet, gnn_cleaner, erase, gnnguard]"
        )
