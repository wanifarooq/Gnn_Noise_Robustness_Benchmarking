"""Experiment orchestration — setup pipeline and registry-based dispatch."""

import os
import time

import torch
from torch_geometric.data import Data

from util.seed import setup_seed_device
from util.data import load_dataset, ensure_splits, prepare_data_for_method, verify_label_distribution
from util.noise import noise_operation
from util.profiling import get_model, profile_model_flops

from model.registry import discover_trainers, get_trainer


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
    compute_info = {
        'flops_inference': flops_info['total_flops'],
        'flops_training_total': 0,
        'time_training_total': 0.0,
        'time_inference': 0.0,
    }

    # training parameters
    trainer_params = config.get('training', {})
    lr = float(trainer_params.get('lr', 0.01))
    weight_decay = float(trainer_params.get('weight_decay', 5e-4))
    epochs = int(trainer_params.get('epochs', 20))
    patience = int(trainer_params.get('patience', 100))
    oversmoothing_every = int(trainer_params.get('oversmoothing_every', 20))

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
        'oversmoothing_every': oversmoothing_every,
        'compute_info': compute_info,
        'get_model': get_model,
    }

def run_experiment(config, run_id=1, *, checkpoint_path=None, eval_only=False):
    """Run a single experiment: setup -> train -> evaluate via registry.

    Parameters
    ----------
    checkpoint_path : str or None
        When provided in normal mode, the trained model state is saved here
        after ``trainer.run()``.  In *eval_only* mode the checkpoint is loaded
        and only evaluation is performed (no training).
    eval_only : bool
        If *True*, skip training and evaluate from a saved checkpoint.
        Requires *checkpoint_path* to point at an existing file.
    """
    discover_trainers()
    init_data = initialize_experiment(config, run_id)
    trainer = get_trainer(init_data['method'], init_data, config)

    if eval_only:
        if checkpoint_path is None:
            raise ValueError("eval_only=True requires a checkpoint_path")
        if not trainer.supports_eval_only:
            raise NotImplementedError(
                f"Method '{init_data['method']}' does not support eval_only "
                f"(its evaluate() depends on state created during train()). "
                f"Override get_checkpoint_state/load_checkpoint_state to enable it."
            )
        # weights_only=False: checkpoint contains plain dicts (oversmoothing
        # metrics) in addition to tensor state_dicts.
        state = torch.load(checkpoint_path, map_location=init_data['device'],
                           weights_only=False)
        trainer.load_checkpoint_state(state)
        t0 = time.perf_counter()
        eval_result = trainer.evaluate()
        time_inference = time.perf_counter() - t0
        init_data['compute_info']['time_inference'] = round(time_inference, 4)
        return trainer._make_result(
            eval_result,
            state.get('train_oversmoothing', {}),
            state.get('val_oversmoothing'),
            reduce=False,
        )

    result = trainer.run()

    if checkpoint_path is not None:
        ckpt_dir = os.path.dirname(checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        state = trainer.get_checkpoint_state()
        state['train_oversmoothing'] = result.get('train_oversmoothing', {})
        state['val_oversmoothing'] = result.get('val_oversmoothing', {})
        torch.save(state, checkpoint_path)

    return result
