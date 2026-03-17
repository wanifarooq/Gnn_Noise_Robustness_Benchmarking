"""Inductive graph partitioning — split a graph into disjoint subgraphs.

For inductive learning the train / val / test nodes live in completely
separate subgraphs.  Edges that cross partition boundaries are removed so
that information cannot leak between splits during message passing.

Usage::

    train_sub, val_sub, test_sub = partition_graph_inductive(data)

Each returned ``Data`` object contains:

* ``x``, ``y``, ``edge_index`` — features, labels, and intra-partition edges
  (node indices remapped to 0 … N_partition − 1).
* ``original_node_ids`` — ``LongTensor`` mapping local → global indices
  (e.g. ``global_id = sub.original_node_ids[local_id]``).
* ``train_mask`` / ``val_mask`` / ``test_mask`` — trivially all-True for the
  owning split, all-False for the others.
"""

import torch
from torch_geometric.data import Data


def partition_graph_inductive(data: Data):
    """Create three disjoint subgraphs for inductive learning.

    Args:
        data: PyG Data object with ``train_mask``, ``val_mask``,
            ``test_mask``, ``x``, ``y``, ``edge_index``.

    Returns:
        ``(train_data, val_data, test_data)`` — each a standalone ``Data``
        object with remapped node indices.
    """
    return (
        _extract_subgraph(data, data.train_mask, 'train'),
        _extract_subgraph(data, data.val_mask, 'val'),
        _extract_subgraph(data, data.test_mask, 'test'),
    )


def _extract_subgraph(data: Data, mask, split_name: str) -> Data:
    """Extract an induced subgraph for nodes selected by *mask*.

    Only edges where *both* endpoints belong to *mask* are retained.
    Node indices are remapped to a contiguous 0 … N−1 range.
    """
    node_ids = mask.nonzero(as_tuple=True)[0]  # global ids of selected nodes
    num_nodes_sub = node_ids.size(0)

    # Build global→local mapping (−1 for nodes outside the partition)
    global_to_local = torch.full(
        (data.num_nodes,), -1, dtype=torch.long, device=data.x.device,
    )
    global_to_local[node_ids] = torch.arange(
        num_nodes_sub, device=data.x.device,
    )

    # Filter edges: keep only those with both endpoints in the partition
    src, dst = data.edge_index[0], data.edge_index[1]
    edge_mask = (global_to_local[src] >= 0) & (global_to_local[dst] >= 0)
    new_edge_index = global_to_local[data.edge_index[:, edge_mask]]

    # Build subgraph Data
    sub = Data(
        x=data.x[node_ids],
        y=data.y[node_ids],
        edge_index=new_edge_index,
    )

    # Copy original labels if present (for noise-split metrics)
    if hasattr(data, 'y_original'):
        sub.y_original = data.y_original[node_ids]
    if hasattr(data, 'y_noisy'):
        sub.y_noisy = data.y_noisy[node_ids]

    # Edge weights (if any method previously set them)
    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        sub.edge_weight = data.edge_weight[edge_mask]

    # Trivial masks: all nodes belong to this split
    sub.train_mask = torch.zeros(num_nodes_sub, dtype=torch.bool, device=data.x.device)
    sub.val_mask = torch.zeros(num_nodes_sub, dtype=torch.bool, device=data.x.device)
    sub.test_mask = torch.zeros(num_nodes_sub, dtype=torch.bool, device=data.x.device)

    if split_name == 'train':
        sub.train_mask[:] = True
    elif split_name == 'val':
        sub.val_mask[:] = True
    elif split_name == 'test':
        sub.test_mask[:] = True

    # Store mapping back to global indices
    sub.original_node_ids = node_ids

    return sub
