import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
from scipy import stats
from torch_geometric.utils import to_scipy_sparse_matrix
import argparse
import pandas as pd
from torch_geometric.data import Data
from torch.profiler import profile, ProfilerActivity
import requests
import zipfile
from io import BytesIO

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
from model.gnns import GCN, GIN, GAT, GATv2, GPS

def make_random_splits(num_nodes: int,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       seed: int = 42,
                       device=None):
    """
    Create boolean train/val/test masks for node-level tasks.
    Deterministic w.r.t. seed.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    g = torch.Generator()
    g.manual_seed(int(seed))

    perm = torch.randperm(num_nodes, generator=g)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)
    n_test = num_nodes - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    dev = device if device is not None else torch.device("cpu")

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=dev)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool, device=dev)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool, device=dev)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def ensure_splits(data,
                  seed: int,
                  split_id: int = 0,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1):
    """
    Ensure data has 1D train/val/test masks.
    If they do not exist, create them.
    If they exist and are 2D, pick split_id.
    """
    num_nodes = data.num_nodes
    device = data.x.device if hasattr(data, "x") and data.x is not None else torch.device("cpu")

    train_mask, val_mask, test_mask = make_random_splits(
        num_nodes=num_nodes,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        device=device
    )
   
    return train_mask, val_mask, test_mask

# Noises
def simple_uniform_noise(labels, n_classes, noise_rate, seed):
    if noise_rate == 0:
        return labels.clone()
    rng = np.random.RandomState(seed)
    
    n_samples = len(labels)
    noisy_labels = labels.clone()
    for i in range(n_samples):
        if rng.rand() < noise_rate:
            available_classes = list(range(n_classes))
            available_classes.remove(labels[i].item())
            noisy_labels[i] = rng.choice(available_classes)
    return noisy_labels

def uniform_noise(n_classes, noise_rate):
    P = np.full((n_classes, n_classes), noise_rate / (n_classes - 1), dtype=np.float64)
    np.fill_diagonal(P, 1 - noise_rate)
    P[np.arange(n_classes), np.arange(n_classes)] += 1 - P.sum(axis=1)
    return P

def pair_noise(n_classes, noise_rate):
    P = (1 - noise_rate) * np.eye(n_classes, dtype=np.float64)
    for i in range(n_classes):
        P[i, (i - 1) % n_classes] = noise_rate
    return P

def random_noise_cp(n_classes, noise_rate, seed):
    rng = np.random.RandomState(seed)
    
    P = (1 - noise_rate) * np.eye(n_classes, dtype=np.float64)
    for i in range(n_classes):
        probs = rng.rand(n_classes)
        probs[i] = 0
        probs /= probs.sum()
        P[i, :] += noise_rate * probs
    return P

def random_pair_noise(n_classes, noise_rate, seed):
    rng = np.random.default_rng(seed)
    P = np.eye(n_classes, dtype=np.float64) * (1 - noise_rate)
    for i in range(n_classes):
        candidates = list(range(i)) + list(range(i + 1, n_classes))
        chosen = rng.choice(candidates)
        P[i, chosen] = noise_rate
    return P

def deterministic(labels, idx_train, noise_rate, seed):
    rng = np.random.RandomState(seed)
    
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    idx_train_np = idx_train.cpu().numpy() if torch.is_tensor(idx_train) else idx_train

    corrupted_labels = labels_np.copy()
    num_classes = len(np.unique(labels_np))
    
    num_noise = int(len(idx_train_np) * noise_rate)
    noise_indices = rng.choice(idx_train_np, num_noise, replace=False)
    
    noise_idx, clean_idx = [], []
    for idx in idx_train_np:
        if idx in noise_indices:
            original = labels_np[idx]
            possible = [i for i in range(num_classes) if i != original]
            corrupted_labels[idx] = rng.choice(possible)
            noise_idx.append(idx)
        else:
            clean_idx.append(idx)
    
    return corrupted_labels, noise_idx, clean_idx

def flip_noise(n_classes, noise_rate):
    P = np.eye(n_classes, dtype=np.float64) * (1 - noise_rate)

    P[0, 1] = noise_rate

    for i in range(1, n_classes - 1):
        P[i, i + 1] = noise_rate

    P[n_classes - 1, 0] = noise_rate
    
    return P

def uniform_mix_noise(n_classes, noise_rate):

    P = np.eye(n_classes, dtype=np.float64) * (1 - noise_rate)
    P += noise_rate / n_classes

    return P

def instance_independent_noise(labels, cp, seed):
    rs = np.random.RandomState(seed)
    noisy_labels = np.array([np.where(rs.multinomial(1, cp[label]))[0][0] for label in labels])
    return noisy_labels

def instance_dependent_noise(noise_rate, feature, labels, num_classes, norm_std, seed):
    num_nodes, feature_size = feature.shape
    
    rng = np.random.RandomState(seed)
    flip_dist = stats.truncnorm((0 - noise_rate) / norm_std, (1 - noise_rate) / norm_std, loc=noise_rate, scale=norm_std)
    flip_rate = flip_dist.rvs(num_nodes, random_state=rng)
    
    labels_t = torch.tensor(labels, dtype=torch.long, device=feature.device)
    
    torch.manual_seed(seed)
    W = torch.randn(num_classes, feature_size, num_classes, device=feature.device)
    
    P = []
    for i in range(num_nodes):
        x = feature[i].unsqueeze(0)
        y = labels_t[i]
        A = x @ W[y]
        A[0, y] = float('-inf')
        A = flip_rate[i] * F.softmax(A, dim=1)
        A[0, y] += 1 - flip_rate[i]
        P.append(A.squeeze(0))
    P = torch.stack(P).cpu().numpy()
    
    new_label = np.array([rng.choice(num_classes, p=P[i]) for i in range(num_nodes)])
    return new_label

def noise_operation(labels, features, n_classes, noise_type='clean', noise_rate=0, noise_seed=1, idx_train=None, debug=True):
    assert 0 <= noise_rate <= 1

    allowed_noise_types = [
        'clean', 'uniform_simple', 'uniform', 'random', 'pair', 
        'random_pair', 'flip', 'uniform_mix', 'deterministic', 'instance'
    ]

    if noise_type not in allowed_noise_types:
        raise ValueError(
            f"Invalid noise_type '{noise_type}'. "
            f"Please choose one of: {allowed_noise_types}"
        )

    cp = None
    noisy_labels = None

    if noise_rate == 0:
        cp = np.eye(n_classes)
        noisy_labels = labels.clone()
    else:
        if noise_type == 'clean':
            cp = np.eye(n_classes)
            noisy_labels = labels.clone()
        elif noise_type == 'uniform_simple':
            noisy_labels = simple_uniform_noise(labels, n_classes, noise_rate, seed=noise_seed)
        elif noise_type == 'uniform':
            cp = uniform_noise(n_classes, noise_rate)
        elif noise_type == 'random':
            cp = random_noise_cp(n_classes, noise_rate, seed=noise_seed)
        elif noise_type == 'pair':
            cp = pair_noise(n_classes, noise_rate)
        elif noise_type == 'random_pair':
            cp = random_pair_noise(n_classes, noise_rate, seed=noise_seed)
        elif noise_type == 'flip':
            cp = flip_noise(n_classes, noise_rate)
        elif noise_type == 'uniform_mix':
            cp = uniform_mix_noise(n_classes, noise_rate)
        elif noise_type == 'deterministic':
            if idx_train is None:
                raise ValueError("idx_train must be provided for deterministic noise")
            noisy_labels_np, _, _ = deterministic(labels, idx_train, noise_rate=noise_rate, seed=noise_seed)
            noisy_labels = torch.tensor(noisy_labels_np, dtype=torch.long, device=labels.device)
        elif noise_type == 'instance':
            if features is None:
                raise ValueError("features must be provided for instance-dependent noise")
            noisy_labels_np = instance_dependent_noise(
                noise_rate=noise_rate,
                feature=features,
                labels=labels.cpu().numpy(),
                num_classes=n_classes,
                norm_std=0.1,
                seed=noise_seed
            )
            noisy_labels = torch.tensor(noisy_labels_np, dtype=torch.long, device=labels.device)
        else:
            print(f"[Warning] Noise type '{noise_type}' does not exist. Labels will remain unchanged.")
            noisy_labels = labels.clone()

    if noisy_labels is None:
        if cp is not None:
            noisy_labels_np = instance_independent_noise(labels.cpu().numpy(), cp, seed=noise_seed)
            noisy_labels = torch.tensor(noisy_labels_np, dtype=torch.long, device=labels.device)
        else:
            noisy_labels = labels.clone()

    if debug and noise_rate > 0:
        noise_count = (noisy_labels != labels).sum().item()
        actual_noise_rate = noise_count / len(labels)
        print(f"[Noise info] Type: {noise_type}, Target rate: {noise_rate:.2f}, Actual rate: {actual_noise_rate:.2f}")
        print(f"[Noise info] Total corrupted labels: {noise_count} out of {len(labels)}")

    noisy_indices = (noisy_labels != labels).nonzero(as_tuple=True)[0].cpu().numpy()
    return noisy_labels, noisy_indices

# Useful
def setup_seed_device(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

def load_dataset(name, root="./data"):
    name_lower = name.lower()
    
    if name_lower in ["cora", "citeseer", "pubmed"]:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=f"{root}/{name}", name=name.capitalize(), transform=NormalizeFeatures(), split='public')
        data = dataset[0]
        return data, dataset.num_classes
    
    elif name_lower in ["amazon-ratings", "tolokers", "roman-empire", "minesweeper", "questions"]:
        from torch_geometric.datasets import HeterophilousGraphDataset
        dataset = HeterophilousGraphDataset(root=f"{root}/{name}", name=name, transform=NormalizeFeatures())
        def _fix_split_masks(data):
            for key in ["train_mask", "val_mask", "test_mask"]:
                if hasattr(data, key):
                    m = getattr(data, key)
                    if m is None:
                        continue
                    if m.dim() == 2:
                        m = m.any(dim=1)
                    setattr(data, key, m.bool())
            return data

        data = dataset[0]
        data = _fix_split_masks(data)
        return data, dataset.num_classes
    
    elif name_lower == "dblp":
        from torch_geometric.datasets import CitationFull
        dataset = CitationFull(root=f"{root}/{name}", name="dblp", transform=NormalizeFeatures())
        data = dataset[0]
        return data, dataset.num_classes
    
    elif name_lower in ["amazon-computers", "amazon-photo"]:
        from torch_geometric.datasets import Amazon
        amazon_name = name.split('-')[1].capitalize()
        dataset = Amazon(root=f"{root}/{name}", name=amazon_name, transform=NormalizeFeatures())
        data = dataset[0]
        return data, dataset.num_classes
    
    elif name_lower in ["blogcatalog", "flickr"]:
        from torch_geometric.datasets import AttributedGraphDataset
        dataset = AttributedGraphDataset(root=f"{root}/{name}", name=name.capitalize(), transform=NormalizeFeatures())
        data = dataset[0]
        return data, dataset.num_classes
    
    elif name_lower in ["hm-categories", "pokec-regions", "web-topics", "tolokers-2", "city-reviews", "artnet-exp", "web-fraud"]:
        
        dataset_dir = os.path.join(root, name)
        if not os.path.exists(dataset_dir):
            url = f"https://zenodo.org/records/16895532/files/{name_lower}.zip"
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall("data")
            
            #raise FileNotFoundError(f"GraphLAND dataset {name} not found in {dataset_dir}. ")
        
        edgelist = pd.read_csv(os.path.join(dataset_dir, "edgelist.csv"))
        edge_index = torch.tensor(edgelist.values.T, dtype=torch.long)
        features = pd.read_csv(os.path.join(dataset_dir, "features.csv"))
        features_filled = features.fillna(0)

        x = torch.tensor(features_filled.values, dtype=torch.float)
        targets_series = pd.read_csv(os.path.join(dataset_dir, "targets.csv")).iloc[:, -1]
        targets_values = targets_series.fillna(-1).to_numpy()

        valid_label_mask = targets_values != -1
        y = torch.tensor(targets_values, dtype=torch.long)
        
        valid_labels = targets_values[valid_label_mask]
        num_classes = len(pd.Series(valid_labels).unique())
        
        split_masks = pd.read_csv(os.path.join(dataset_dir, "split_masks_RL.csv"))
        train_mask = torch.tensor(split_masks["train"].values, dtype=torch.bool)
        val_mask = torch.tensor(split_masks["val"].values, dtype=torch.bool)
        test_mask = torch.tensor(split_masks["test"].values, dtype=torch.bool)
        valid_mask_torch = torch.from_numpy(valid_label_mask)
        train_mask = train_mask & valid_mask_torch
        val_mask = val_mask & valid_mask_torch
        test_mask = test_mask & valid_mask_torch
        
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        return data, num_classes
    
    elif name_lower in ["pattern", "cluster"]:
        from torch_geometric.datasets import GNNBenchmarkDataset
        dataset = GNNBenchmarkDataset(root=f"{root}/{name}", name=name.upper(), split='train', transform=NormalizeFeatures())
        return dataset, dataset.num_classes
    
    elif name_lower in ["pascalvoc-sp", "coco-sp"]:
        from torch_geometric.datasets import LRGBDataset
        name_lower = name.lower()
        from torch_geometric.datasets import LRGBDataset
        dataset_train = LRGBDataset(root=root, name=name, split='train')
        dataset_val   = LRGBDataset(root=root, name=name, split='val')
        dataset_test  = LRGBDataset(root=root, name=name, split='test')

        # Total nodes across ALL graphs (train+val+test)
        total_nodes = 0
        for ds in (dataset_train, dataset_val, dataset_test):
            for g in ds:
                total_nodes += g.num_nodes

        # Global masks (length == total_nodes)
        train_mask = torch.zeros(total_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(total_nodes, dtype=torch.bool)

        x_list, edge_index_list, y_list = [], [], []
        offset = 0

        # Helper to append graphs and set the correct split mask ranges
        def add_split(split_dataset, split_name):
            nonlocal offset
            for g in split_dataset:
                n = g.num_nodes

                # Append node features and labels
                x_list.append(g.x)
                y_list.append(g.y)

                # Shift edges by current offset and append
                edge_index_list.append(g.edge_index + offset)

                # Mark the corresponding node range in the proper mask
                if split_name == "train":
                    train_mask[offset:offset+n] = True
                elif split_name == "val":
                    val_mask[offset:offset+n] = True
                elif split_name == "test":
                    test_mask[offset:offset+n] = True
                else:
                    raise ValueError(f"Unknown split_name: {split_name}")

                offset += n

        add_split(dataset_train, "train")
        add_split(dataset_val, "val")
        add_split(dataset_test, "test")

        # Concatenate everything into a single big graph
        x = torch.cat(x_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        y = torch.cat(y_list, dim=0)

        # Safety checks
        assert x.size(0) == total_nodes
        assert y.size(0) == total_nodes
        assert train_mask.size(0) == total_nodes
        assert val_mask.size(0) == total_nodes
        assert test_mask.size(0) == total_nodes
        assert (train_mask & val_mask).sum().item() == 0
        assert (train_mask & test_mask).sum().item() == 0
        assert (val_mask & test_mask).sum().item() == 0

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        num_classes = int(y.max().item() + 1)
        return data, num_classes

    
    else:
        raise ValueError(f"Dataset {name} not supported. Supported datasets: cora, citeseer, pubmed, amazon-ratings, tolokers, roman-empire, minesweeper, questions, dblp, amazon-computers, amazon-photo, blogcatalog, flickr, hm-categories, pokec-regions, web-topics, tolokers-2, city-reviews, artnet-exp, web-fraud, pattern, cluster, pascalvoc-sp, coco-sp")

def prepare_data_for_method(data, train_mask, val_mask, test_mask, noisy_train_labels, method_name):

    data_for_method = data.clone()

    data_for_method.y = data.y_original.clone()
    data_for_method.y[train_mask] = noisy_train_labels
    
    return data_for_method

def verify_label_distribution(data, train_mask, val_mask, test_mask, run_id, method_name):

    print(f"[DEBUG Run {run_id}] {method_name} - Label distribution:")
    
    if hasattr(data, 'y_original'):

        train_corrupted = (data.y[train_mask] != data.y_original[train_mask]).sum()
        print(f"Training labels corrupted: {train_corrupted}/{train_mask.sum()} nodes")
        
        val_clean = (data.y[val_mask] == data.y_original[val_mask]).all()
        test_clean = (data.y[test_mask] == data.y_original[test_mask]).all()
        print(f"Val labels clean: {val_clean}")
        print(f"Test labels clean: {test_clean}")

def get_model(model_name, in_channels, hidden_channels, out_channels, **kwargs):
    model_name = model_name.lower()
    kwargs.pop('in_channels', None)
    kwargs.pop('hidden_channels', None)
    kwargs.pop('out_channels', None)
    
    model_registry = {
        'gcn':    (GCN, ['n_layers', 'dropout', 'self_loop']),
        'gin':    (GIN, ['n_layers', 'dropout', 'mlp_layers', 'train_eps']),
        'gat':    (GAT, ['n_layers', 'dropout', 'heads']),
        'gatv2':  (GATv2, ['n_layers', 'dropout', 'heads']),
        'gps':    (GPS, ['n_layers', 'dropout', 'heads', 'use_pe', 'pe_dim']),
    }
    
    if model_name not in model_registry:
        raise ValueError(f"Model {model_name} not recognized.")
    
    model_cls, valid_params = model_registry[model_name]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return model_cls(in_channels, hidden_channels, out_channels, **filtered_kwargs)

import inspect
import torch
from torch.profiler import profile, ProfilerActivity

def _forward_call(model, data):
    """
    Call model forward with the correct signature.

    Tries in this order:
    1) model(data)
    2) model(data.x)
    3) model(data.x, data.edge_index)
    """
    # Try to inspect the forward signature (best effort)
    try:
        sig = inspect.signature(model.forward)
        # Number of parameters excluding self
        n_args = len(sig.parameters) - 1
    except Exception:
        n_args = None

    # If forward seems to take 1 arg (besides self), prefer model(data)
    if n_args == 1:
        try:
            return model(data)
        except TypeError:
            pass

    # Fallback attempts (works across different model styles)
    try:
        return model(data)
    except TypeError:
        pass

    try:
        return model(data.x)
    except TypeError:
        pass

    # Most PyG-style GNNs: forward(x, edge_index)
    return model(data.x, data.edge_index)

def _reduce_oversmoothing(oversmoothing_dict):
    return {
        k: float(np.mean(v)) if isinstance(v, (list, tuple, np.ndarray)) else float(v)
        for k, v in oversmoothing_dict.items()
    }


def profile_model_flops(model, data, device, n_warmup=1, n_iters=1):
    """
    Profile FLOPs with torch.profiler (forward pass).
    """
    model.eval()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Warmup outside profiler
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = _forward_call(model, data)
        if device.type == "cuda":
            torch.cuda.synchronize()

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(n_iters):
                _ = _forward_call(model, data)
            if device.type == "cuda":
                torch.cuda.synchronize()

    total_flops = 0
    for e in prof.key_averages():
        fl = getattr(e, "flops", None)
        if fl:
            total_flops += fl

    table = prof.key_averages().table(
        sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
        row_limit=15
    )

    return {"total_flops": int(total_flops), "profiler_table": table}




def initialize_experiment(config, run_id=1):

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
            'train_oversmoothing': result['train_oversmoothing'],
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
            'accuracy': torch.tensor(result['accuracy']),
            'f1': torch.tensor(result['f1']),
            'precision': torch.tensor(result['precision']),
            'recall': torch.tensor(result['recall']),
            'oversmoothing': result['oversmoothing'],
            'train_oversmoothing': result['train_oversmoothing'],
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

        result = trainer.train_full_model()
        
        return {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'precision': result['precision'],
            'recall': result['recall'],
            'oversmoothing': result['oversmoothing'],
            'train_oversmoothing': result['train_oversmoothing'],
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
            'accuracy': test_results['test_acc'],
            'f1': test_results['test_f1'],
            'precision': test_results['test_precision'],
            'recall': test_results['test_recall'],
            'oversmoothing': test_results['test_oversmoothing'],
            'train_oversmoothing': train_oversmoothing,
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
            'accuracy': torch.tensor(test_results['accuracy']),
            'f1': torch.tensor(test_results['f1']),
            'precision': torch.tensor(test_results['precision']),
            'recall': torch.tensor(test_results['recall']),
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': test_results['train_oversmoothing'],
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
            'accuracy': torch.tensor(test_results['accuracy']),
            'f1': torch.tensor(test_results['f1']),
            'precision': torch.tensor(test_results['precision']),
            'recall': torch.tensor(test_results['recall']),
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


        print("test_results['train_oversmoothing']", test_results['train_oversmoothing'])
        return {
            'accuracy': torch.tensor(test_results['accuracy']),
            'f1': torch.tensor(test_results['f1']),
            'precision': torch.tensor(test_results['precision']),
            'recall': torch.tensor(test_results['recall']),
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
            'oversmoothing': test_results['oversmoothing'],
            'flops_info' : flops_info,
            'train_oversmoothing' : results
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
            'train_oversmoothing': result['train_oversmoothing'],
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
            'accuracy': torch.tensor(test_results['accuracy']),
            'f1': torch.tensor(test_results['f1']),
            'precision': torch.tensor(test_results['precision']),
            'recall': torch.tensor(test_results['recall']),
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': test_results['train_oversmoothing'],
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
            'accuracy': torch.tensor(test_results['accuracy']),
            'f1': torch.tensor(test_results['f1']),
            'precision': torch.tensor(test_results['precision']),
            'recall': torch.tensor(test_results['recall']),
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': test_results['train_oversmoothing'],
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
            'accuracy': torch.tensor(test_results['accuracy']),
            'f1': torch.tensor(test_results['f1']),
            'precision': torch.tensor(test_results['precision']),
            'recall': torch.tensor(test_results['recall']),
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': test_results['train_oversmoothing'],
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
            'accuracy': torch.tensor(test_results['accuracy']),
            'f1': torch.tensor(test_results['f1']),
            'precision': torch.tensor(test_results['precision']),
            'recall': torch.tensor(test_results['recall']),
            'oversmoothing': test_results['oversmoothing'],
            'train_oversmoothing': train_oversmoothing,
            'flops_info' : flops_info,
        }
    
    else:
        raise ValueError(
            f"Run {run_id}: Training method '{method}' not implemented. "
            "Please choose one of the implemented methods: "
            "[standard, positive_eigenvalues, gcod, nrgnn, pi_gnn, cr_gnn, community_defense, rtgnn, graphcleaner, unionnet, gnn_cleaner, erase, gnnguard]"
        )

# Multithreading
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run parallel benchmarks with selected methods')
    parser.add_argument('--methods', '-m', 
                       nargs='+', 
                       default=['standard', 'cr_gnn', 'nrgnn'],
                       help='Methods to benchmark (space separated)')
    parser.add_argument('--run-id', '-r',
                       type=int,
                       default=1,
                       help='Fixed run ID for consistent seeds (default: 1)')
    return parser.parse_args()

def print_table(headers, rows, col_widths):

    def separator():
        return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def row_line(row):
        return "|" + "|".join(f" {str(val).ljust(w)} " for val, w in zip(row, col_widths)) + "|"

    print(separator())
    print(row_line(headers))
    print(separator())
    for row in rows:
        print(row_line(row))
        print(separator())