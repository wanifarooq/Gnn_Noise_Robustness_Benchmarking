"""CommunityDefense method helper — community-aware regularization training.

Runs community detection (Louvain or spectral) on the graph and adds an
adaptive community regularization loss that encourages node embeddings to
align with detected community structure.  The regularization weight ramps
up linearly over the first 50 epochs.

Mirrors the logic in model/methods/CommunityDefense.py.
"""

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base_helper import MethodHelper
from methods.registry import register_helper

try:
    import networkx as nx
except ImportError:
    nx = None


# ---------------------------------------------------------------------------
# Community detection utilities
# ---------------------------------------------------------------------------

def _build_adjacency_matrix(edge_index, num_nodes):
    """Build a symmetric CSR adjacency matrix from a PyG edge_index."""
    edges = edge_index.cpu().numpy()
    row, col = edges[0], edges[1]
    adj = sp.coo_matrix(
        (np.ones_like(row, dtype=np.float32), (row, col)),
        shape=(num_nodes, num_nodes),
    )
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data, dtype=np.float32)
    adj.eliminate_zeros()
    return adj.tocsr()


def _import_louvain_module():
    """Try several import paths for the Louvain community detection library."""
    try:
        import community.community_louvain as mod
        return mod
    except ImportError:
        pass
    try:
        import community as mod
        if hasattr(mod, 'best_partition'):
            return mod
    except ImportError:
        pass
    try:
        import community_louvain as mod
        return mod
    except ImportError:
        pass
    return None


def _detect_louvain(adj_csr, num_nodes):
    """Louvain community detection with NetworkX fallback."""
    if nx is None:
        raise ImportError("networkx is required for community detection")

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    row, col = adj_csr.nonzero()
    graph.add_edges_from(zip(row.tolist(), col.tolist()))

    louvain = _import_louvain_module()
    if louvain is not None:
        mapping = louvain.best_partition(graph, random_state=0)
        labels = np.array([mapping[i] for i in range(num_nodes)], dtype=np.int64)
        print(f"CommunityDefense: Louvain detected {len(set(labels.tolist()))} communities")
        return labels

    # Fallback: NetworkX built-in community detection
    import networkx.algorithms.community as nx_comm
    if hasattr(nx_comm, 'louvain_communities'):
        communities = nx_comm.louvain_communities(graph, seed=0)
    else:
        communities = nx_comm.greedy_modularity_communities(graph)

    node_to_comm = {}
    for cid, nodes in enumerate(communities):
        for n in nodes:
            node_to_comm[n] = cid
    labels = np.array([node_to_comm.get(i, 0) for i in range(num_nodes)], dtype=np.int64)
    print(f"CommunityDefense: NetworkX fallback detected {len(set(labels.tolist()))} communities")
    return labels


def _detect_spectral(adj_csr, num_nodes, k, y_tensor=None):
    """Spectral clustering community detection."""
    from sklearn.cluster import KMeans
    from scipy.sparse.linalg import eigsh

    if k is None:
        if y_tensor is not None and y_tensor.dim() == 1:
            k = int(y_tensor.max().item()) + 1
        else:
            k = 8

    degree = np.array(adj_csr.sum(1)).flatten()
    d_inv_sqrt = np.power(np.maximum(degree, 1e-12), -0.5)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L_norm = sp.eye(num_nodes) - D_inv_sqrt @ adj_csr @ D_inv_sqrt

    _, eigvecs = eigsh(L_norm.asfptype(), k=k, which='SM')
    eigvecs = eigvecs / (np.linalg.norm(eigvecs, axis=1, keepdims=True) + 1e-12)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(eigvecs).astype(np.int64)
    print(f"CommunityDefense: Spectral clustering with k={k} communities")
    return labels


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

@register_helper('community_defense')
class CommunityDefenseHelper(MethodHelper):
    """Community-aware regularization training."""

    def supports_batched_training(self):
        return True

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        lr = float(training_cfg.get('lr', 0.01))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))
        comm_params = config.get('community_defense_params', {})

        backbone_model.to(device)

        # --- community detection ---
        adj = _build_adjacency_matrix(data.edge_index, data.num_nodes)
        method = comm_params.get('community_method', 'louvain').lower()
        num_communities_hint = comm_params.get('num_communities', None)

        if method == 'spectral':
            community_labels = _detect_spectral(
                adj, data.num_nodes, num_communities_hint,
                y_tensor=data.y if hasattr(data, 'y') else None,
            )
        else:
            community_labels = _detect_louvain(adj, data.num_nodes)

        num_communities = len(np.unique(community_labels))
        community_labels_tensor = torch.as_tensor(
            community_labels, dtype=torch.long, device=device,
        )

        # The community classifier takes the embedding dim as input.
        # At setup time we don't know the exact logits dim yet, so we use
        # num_features as a placeholder; it will be rebuilt on first forward
        # if the dim differs.
        community_classifier = nn.Linear(
            data.x.shape[1], num_communities,
        ).to(device)

        # Single Adam optimizer over backbone + community classifier
        optimizer = torch.optim.Adam(
            list(backbone_model.parameters()) + list(community_classifier.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        community_loss_weight = float(comm_params.get('lambda_comm', 2.0))

        return {
            'models': [backbone_model],
            'optimizers': [optimizer],
            'backbone': backbone_model,
            'optimizer': optimizer,
            'community_classifier': community_classifier,
            'community_labels': community_labels_tensor,
            'num_communities': num_communities,
            'community_loss_weight': community_loss_weight,
            'device': device,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, state, data, epoch):
        model = state['backbone']
        optimizer = state['optimizer']
        community_classifier = state['community_classifier']
        community_labels = state['community_labels']
        community_loss_weight = state['community_loss_weight']
        device = state['device']

        model.train()
        community_classifier.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(data)

        # Supervised CE loss
        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        ce_loss = F.cross_entropy(logits[train_idx], data.y[train_idx])

        # Use dropout(logits) as surrogate embeddings (no model exposes last_hidden)
        embeddings = F.dropout(logits, p=0.3, training=True)

        # Rebuild community classifier if embedding dim changed
        if community_classifier.in_features != embeddings.shape[1]:
            community_classifier = nn.Linear(
                embeddings.shape[1], state['num_communities'],
            ).to(device)
            state['community_classifier'] = community_classifier
            # Rebuild optimizer with new classifier params
            state['optimizer'] = torch.optim.Adam(
                list(model.parameters()) + list(community_classifier.parameters()),
                lr=optimizer.param_groups[0]['lr'],
                weight_decay=optimizer.param_groups[0]['weight_decay'],
            )
            optimizer = state['optimizer']
            optimizer.zero_grad(set_to_none=True)
            # Recompute (graph is still alive)
            logits = model(data)
            ce_loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
            embeddings = F.dropout(logits, p=0.3, training=True)

        # Community regularization loss (adaptive weight ramps over 50 epochs)
        adaptive_weight = community_loss_weight * min(1.0, (epoch + 1) / 50.0)
        comm_logits = community_classifier(embeddings)
        comm_loss = F.cross_entropy(comm_logits, community_labels)

        total_loss = ce_loss + adaptive_weight * comm_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        return {'train_loss': total_loss.item()}

    # ------------------------------------------------------------------
    # Checkpointing — community classifier is training-only
    # ------------------------------------------------------------------

    def get_checkpoint_state(self, state):
        from copy import deepcopy
        return {'model_0': deepcopy(state['backbone'].state_dict())}

    def load_checkpoint_state(self, state, checkpoint):
        if 'model_0' in checkpoint:
            state['backbone'].load_state_dict(checkpoint['model_0'])
