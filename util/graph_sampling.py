"""Graph data loader factory — supports multiple sampling strategies.

When ``training.batch_size`` is set in the config, this module creates PyG
data loaders for mini-batch training, validation, and evaluation.

Supported samplers (``training.sampler``):
    neighbor     — NeighborLoader (message-passing-aware, default)
    cluster      — ClusterData + ClusterLoader (METIS graph partitioning)
    graphsaint   — GraphSAINTRandomWalkSampler (random-walk subgraph sampling)
    random_node  — RandomNodeLoader (simple random node partitioning)
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch_geometric.data import Data


@dataclass
class GraphLoaders:
    """Container for train / val / test / inference data loaders."""

    train_loader: object = None
    val_loader: object = None
    test_loader: object = None
    inference_loader: object = None  # batched inference over ALL nodes
    sampler_type: str = 'none'
    batch_size: int = 0

    @property
    def is_batched(self) -> bool:
        return self.train_loader is not None


# ── Public API ─────────────────────────────────────────────────────────────


def create_loaders(
    data: Data,
    config: dict,
    device: torch.device,
    *,
    input_nodes_train=None,
    input_nodes_val=None,
    input_nodes_test=None,
) -> GraphLoaders:
    """Build data loaders from config.

    Args:
        data: PyG Data object.
        config: Full experiment configuration.
        device: Target device.
        input_nodes_{train,val,test}: Override node indices for each split.
            If None, derived from ``data.{train,val,test}_mask``.

    Returns:
        ``GraphLoaders`` with ``is_batched=True``, or a no-op stub when
        ``training.batch_size`` is absent.
    """
    training_cfg = config.get('training', {})
    batch_size = training_cfg.get('batch_size')

    if batch_size is None:
        return GraphLoaders()  # full-batch stub

    batch_size = int(batch_size)
    sampler_type = training_cfg.get('sampler', 'neighbor').lower()
    sampler_params = training_cfg.get('sampler_params', {})

    # Derive split node indices from masks if not provided
    if input_nodes_train is None and hasattr(data, 'train_mask'):
        input_nodes_train = data.train_mask.nonzero(as_tuple=True)[0]
    if input_nodes_val is None and hasattr(data, 'val_mask'):
        input_nodes_val = data.val_mask.nonzero(as_tuple=True)[0]
    if input_nodes_test is None and hasattr(data, 'test_mask'):
        input_nodes_test = data.test_mask.nonzero(as_tuple=True)[0]

    builder = _SAMPLER_BUILDERS.get(sampler_type)
    if builder is None:
        supported = ', '.join(sorted(_SAMPLER_BUILDERS.keys()))
        raise ValueError(
            f"Unknown sampler '{sampler_type}'. Supported: {supported}"
        )

    return builder(
        data, batch_size, sampler_params,
        input_nodes_train, input_nodes_val, input_nodes_test,
        sampler_type,
    )


def get_seed_indices(batch, sampler_type: str) -> int:
    """Return the number of *seed* (target) nodes in a batch.

    For NeighborLoader batches the first ``batch.batch_size`` nodes are seeds;
    for cluster / graphsaint / random_node *all* nodes in the batch are seeds.
    """
    if sampler_type == 'neighbor' and hasattr(batch, 'batch_size'):
        return batch.batch_size
    return batch.num_nodes


def get_global_ids(batch, sampler_type: str):
    """Return global node IDs for the seed nodes in *batch*.

    Returns ``None`` when no mapping is available (caller should skip
    scatter-back).
    """
    n_seed = get_seed_indices(batch, sampler_type)
    if hasattr(batch, 'n_id'):
        return batch.n_id[:n_seed]
    return None


# ── Builder functions (private) ────────────────────────────────────────────


def _make_inference_loader(data, batch_size, sampler_params):
    """NeighborLoader over ALL nodes — used for batched predictions."""
    from torch_geometric.loader import NeighborLoader

    num_neighbors = sampler_params.get('num_neighbors', [15, 10])
    all_nodes = torch.arange(data.num_nodes)
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=all_nodes,
        shuffle=False,
    )


# ──────────── NeighborLoader ──────────────────────────────────────────────

def _build_neighbor(data, batch_size, params, train_n, val_n, test_n, stype):
    from torch_geometric.loader import NeighborLoader

    num_neighbors = params.get('num_neighbors', [15, 10])

    def _loader(nodes, shuffle):
        if nodes is None or len(nodes) == 0:
            return None
        return NeighborLoader(
            data, num_neighbors=num_neighbors, batch_size=batch_size,
            input_nodes=nodes, shuffle=shuffle,
        )

    all_nodes = torch.arange(data.num_nodes)
    return GraphLoaders(
        train_loader=_loader(train_n, shuffle=True),
        val_loader=_loader(val_n, shuffle=False),
        test_loader=_loader(test_n, shuffle=False),
        inference_loader=_loader(all_nodes, shuffle=False),
        sampler_type=stype,
        batch_size=batch_size,
    )


# ──────────── ClusterLoader ───────────────────────────────────────────────

def _build_cluster(data, batch_size, params, train_n, val_n, test_n, stype):
    from torch_geometric.loader import ClusterData, ClusterLoader

    num_parts = params.get('num_parts', 100)
    cluster_data = ClusterData(data, num_parts=num_parts)

    train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)
    eval_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False)

    return GraphLoaders(
        train_loader=train_loader,
        val_loader=eval_loader,
        test_loader=eval_loader,
        inference_loader=eval_loader,
        sampler_type=stype,
        batch_size=batch_size,
    )


# ──────────── GraphSAINT ──────────────────────────────────────────────────

def _build_graphsaint(data, batch_size, params, train_n, val_n, test_n, stype):
    from torch_geometric.loader import GraphSAINTRandomWalkSampler

    walk_length = params.get('walk_length', 4)
    num_steps = params.get('num_steps', 30)

    train_loader = GraphSAINTRandomWalkSampler(
        data, batch_size=batch_size, walk_length=walk_length,
        num_steps=num_steps, sample_coverage=100,
    )

    # Eval / inference uses NeighborLoader for full-coverage scatter-back
    inference_loader = _make_inference_loader(data, batch_size, params)

    return GraphLoaders(
        train_loader=train_loader,
        val_loader=inference_loader,
        test_loader=inference_loader,
        inference_loader=inference_loader,
        sampler_type='neighbor',  # eval path uses neighbor semantics
        batch_size=batch_size,
    )


# ──────────── RandomNodeLoader ────────────────────────────────────────────

def _build_random_node(data, batch_size, params, train_n, val_n, test_n, stype):
    from torch_geometric.loader import RandomNodeLoader

    num_parts = max(1, data.num_nodes // batch_size)
    train_loader = RandomNodeLoader(data, num_parts=num_parts, shuffle=True)
    eval_loader = RandomNodeLoader(data, num_parts=num_parts, shuffle=False)

    return GraphLoaders(
        train_loader=train_loader,
        val_loader=eval_loader,
        test_loader=eval_loader,
        inference_loader=eval_loader,
        sampler_type=stype,
        batch_size=batch_size,
    )


# ── Registry ──────────────────────────────────────────────────────────────

_SAMPLER_BUILDERS = {
    'neighbor': _build_neighbor,
    'cluster': _build_cluster,
    'graphsaint': _build_graphsaint,
    'random_node': _build_random_node,
}
