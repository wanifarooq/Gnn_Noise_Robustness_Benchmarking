"""Dataset I/O, split creation, and label preparation."""

import os
import torch
import pandas as pd
import requests
import zipfile
from io import BytesIO
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data


SUPPORTED_DATASETS = (
    "cora", "citeseer", "pubmed",
    "amazon-ratings", "tolokers", "roman-empire", "minesweeper", "questions",
    "dblp",
    "amazon-computers", "amazon-photo",
    "blogcatalog", "flickr",
    "hm-categories", "pokec-regions", "web-topics", "tolokers-2",
    "city-reviews", "artnet-exp", "web-fraud",
    "pattern", "cluster",
    "pascalvoc-sp", "coco-sp",
)


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
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1):
    """
    Ensure data has 1D train/val/test masks.
    If they do not exist, create them.
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


def load_dataset(name, root="./data"):
    """Load a graph dataset by name.

    Supports Planetoid, HeterophilousGraph, CitationFull, Amazon,
    AttributedGraph, GraphLAND (Zenodo), GNNBenchmark, and LRGB families.
    """
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
        raise ValueError(
            f"Dataset {name} not supported. "
            f"Supported datasets: {', '.join(SUPPORTED_DATASETS)}"
        )


def prepare_data_for_method(data, train_mask, val_mask, test_mask, noisy_train_labels, method_name):
    """Prepare a data clone where val/test keep original labels and train gets noisy labels."""
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
