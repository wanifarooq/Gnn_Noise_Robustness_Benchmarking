import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import svd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


class ClassificationMetrics:
    """
    Standardized classification metrics using sklearn.
    All methods across the project should use this class for consistent evaluation.

    Usage:
        cls_evaluator = ClassificationMetrics()
        metrics = cls_evaluator.compute_all_metrics(predictions, labels)
        cls_evaluator.print_metrics(metrics)
    """

    def __init__(self, average='macro', zero_division=0):
        self.average = average
        self.zero_division = zero_division

    def compute_all_metrics(self, predictions, labels, average=None):
        """
        Compute accuracy, precision, recall, and F1 score.

        Args:
            predictions: np.ndarray or torch.Tensor of predicted labels
            labels: np.ndarray or torch.Tensor of true labels
            average: str, override default averaging ('macro', 'micro', 'weighted', None)

        Returns:
            dict with keys: 'accuracy', 'precision', 'recall', 'f1'
        """
        avg = average or self.average

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        return {
            'accuracy': float(accuracy_score(labels, predictions)),
            'precision': float(precision_score(labels, predictions, average=avg, zero_division=self.zero_division)),
            'recall': float(recall_score(labels, predictions, average=avg, zero_division=self.zero_division)),
            'f1': float(f1_score(labels, predictions, average=avg, zero_division=self.zero_division)),
        }

    def compute_accuracy(self, predictions, labels):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        return float(accuracy_score(labels, predictions))

    def compute_f1(self, predictions, labels, average=None):
        avg = average or self.average
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        return float(f1_score(labels, predictions, average=avg, zero_division=self.zero_division))

    def get_confusion_matrix(self, predictions, labels):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        return confusion_matrix(labels, predictions)

    def get_classification_report(self, predictions, labels, target_names=None):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        return classification_report(labels, predictions, target_names=target_names, zero_division=self.zero_division)

    def print_metrics(self, metrics):
        print("Classification Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.6f}")
        print(f"  Precision: {metrics['precision']:.6f}")
        print(f"  Recall:    {metrics['recall']:.6f}")
        print(f"  F1 Score:  {metrics['f1']:.6f}")
        print("-" * 50)


class OversmoothingMetrics:

    def __init__(self, device='cpu'):
        self.device = device

    def compute_all_metrics(self, X, edge_index, edge_weight=None, batch_size=None):
        metrics = {}
        X_np = X.detach().cpu().numpy()

        try:
            metrics['EDir'] = self._compute_edir_average(X, edge_index, edge_weight)
        except Exception as e:
            print(f"Warning: Could not compute EDir: {e}")
            metrics['EDir'] = 0.0

        try:
            metrics['NumRank'] = self._compute_numerical_rank(X_np)
        except Exception as e:
            print(f"Warning: Could not compute NumRank: {e}")
            metrics['NumRank'] = float(min(X_np.shape))

        try:
            metrics['Erank'] = self._compute_effective_rank(X_np)
        except Exception as e:
            print(f"Warning: Could not compute Erank: {e}")
            metrics['Erank'] = float(min(X_np.shape))

        try:
            metrics['EDir_traditional'] = self._compute_dirichlet_energy_traditional(X, edge_index, edge_weight)
        except Exception as e:
            print(f"Warning: Could not compute EDir_traditional: {e}")
            metrics['EDir_traditional'] = 0.0

        try:
            metrics['EProj'] = self._compute_projection_energy(X, edge_index, edge_weight)
        except Exception as e:
            print(f"Warning: Could not compute EProj: {e}")
            metrics['EProj'] = 0.0

        try:
            metrics['MAD'] = self._compute_mad(X, edge_index)
        except Exception as e:
            print(f"Warning: Could not compute MAD: {e}")
            metrics['MAD'] = 0.0

        return metrics

    # def _compute_edir_average(self, graphs_in_class):
    #     if not graphs_in_class or len(graphs_in_class) == 0:
    #         return 0.0
    #     total_energy = 0.0
    #     num_graphs = len(graphs_in_class)
    #     for graph_data in graphs_in_class:
    #         X = graph_data['X']
    #         edge_index = graph_data['edge_index']
    #         edge_weight = graph_data.get('edge_weight', None)
    #         if edge_index.size(1) == 0:
    #             continue
    #         graph_energy = 0.0
    #         num_edges = edge_index.size(1)
    #         for i in range(num_edges):
    #             u, v = edge_index[0, i], edge_index[1, i]
    #             grad = X[u] - X[v]
    #             edge_energy = torch.norm(grad, p=2)**2 # ||X[u]-X[v]||²
    #             if edge_weight is not None:
    #                 edge_energy *= edge_weight[i]
    #             graph_energy += edge_energy.item()
    #         total_energy += graph_energy
    #     return total_energy / (2 * num_graphs)
    
    def _compute_edir_average(self, X, edge_index, edge_weight=None):
        """ Refactored to compute energy in a vectorized manner for efficiency.

        The logic is preserved. Here's why:
        1. Per-edge energy: torch.norm(grad, p=2)**2 = sum(grad²) = (diff**2).sum(dim=1) — mathematically identical.
        2. Edge weight handling: Both multiply the per-edge energy by the weight — identical.
        3. Division by 2 * num_graphs: The original divided by 2 * num_graphs. The new divides by 2.0. This is correct because graphs_in_class was always constructed as a
        single-element list (line 367-371 in the original), so num_graphs was always 1. The averaging across multiple graphs was dead code — compute_oversmoothing_for_mask was the only
        call site, and it always wrapped a single graph dict in a list.
        """
        if edge_index.size(1) == 0:
            return 0.0
        diff = X[edge_index[0]] - X[edge_index[1]]
        energies = (diff ** 2).sum(dim=1)  # ||X[u]-X[v]||² per edge
        if edge_weight is not None:
            energies = energies * edge_weight
        return energies.sum().item() / 2.0

    def _compute_numerical_rank(self, X):
        frobenius_norm_sq = np.sum(X**2)
        try:
            s = svd(X, full_matrices=False, compute_uv=False)
            spectral_norm_sq = s[0]**2 if len(s) > 0 else 1e-8
        except Exception:
            spectral_norm_sq = np.linalg.norm(X, ord=2)**2

        if spectral_norm_sq < 1e-12:
            return 1.0

        return frobenius_norm_sq / spectral_norm_sq

    def _compute_effective_rank(self, X):
        try:
            s = svd(X, full_matrices=False, compute_uv=False)
            s = s[s > 1e-12]

            if len(s) == 0:
                return 1.0

            p = s / np.sum(s)

            p = p[p > 1e-12]
            entropy = -np.sum(p * np.log(p))

            return np.exp(entropy)
        except Exception:
            return float(min(X.shape))

    def _compute_message_passing_matrix_eigenvector(self, edge_index, num_nodes, edge_weight=None):
        # Original: used to_dense_adj(edge_index, ...) to create a full
        # N×N dense numpy array, then passed it to scipy.sparse.linalg.eigsh.
        # This costs O(N²) memory — prohibitive for large graphs.
        #
        # Optimized: build a scipy COO sparse matrix directly from the
        # edge_index tensor.  eigsh() is designed for sparse input, so
        # this costs only O(E) memory (number of edges).
        try:
            row = edge_index[0].cpu().numpy()
            col = edge_index[1].cpu().numpy()
            vals = edge_weight.cpu().numpy() if edge_weight is not None else np.ones(len(row))
            adj_sparse = coo_matrix((vals, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
            adj_sym = (adj_sparse + adj_sparse.T) / 2

            try:
                eigenvalues, eigenvectors = eigsh(adj_sym, k=1, which='LA', maxiter=1000)
                dominant_eigenvector = np.abs(eigenvectors[:, 0])

                dominant_eigenvector = np.maximum(dominant_eigenvector, 1e-4)

                return dominant_eigenvector
            except Exception:
                degrees = np.array(adj_sym.sum(axis=1)).flatten()

                return np.maximum(degrees, 1e-4)

        except Exception:
            return np.ones(num_nodes)

    # def _compute_dirichlet_energy_traditional(self, X, edge_index, edge_weight=None):
    #     num_nodes = X.size(0)
    #     u = self._compute_message_passing_matrix_eigenvector(edge_index, num_nodes, edge_weight)
    #     # Normalization
    #     u = u / np.max(u)
    #     u = np.maximum(u, 1e-4)
    #     total_energy = 0.0
    #     num_edges = edge_index.size(1)
    #     for i in range(num_edges):
    #         node_i, node_j = edge_index[0, i].item(), edge_index[1, i].item()
    #         X_i_norm = X[node_i] / u[node_i]
    #         X_j_norm = X[node_j] / u[node_j]
    #         diff = X_i_norm - X_j_norm
    #         energy = torch.norm(diff, p=2)**2 # ||diff||²
    #         if edge_weight is not None:
    #             energy *= edge_weight[i]
    #         total_energy += energy.item()
    #     return total_energy
    
    def _compute_dirichlet_energy_traditional(self, X, edge_index, edge_weight=None):
        """ Refactored to vectorised form for efficiency.  The logic is preserved. Step by step:

        1. Eigenvector computation + normalization — identical in both (u / np.max(u), np.maximum(u, 1e-4)).
        2. Per-node feature normalization — Original: X[node_i] / u[node_i] (torch tensor / numpy scalar, per edge in a loop). New: X / u_t where u_t is (N, 1) — broadcasts so each row
        X[i] is divided by u[i]. Mathematically identical, but the new version computes it once for all nodes upfront rather than redundantly per-edge.
        3. Per-edge energy — torch.norm(diff, p=2)**2 = sum(diff²) = (diff**2).sum(dim=1). Identical.
        4. Edge weight handling — Both multiply per-edge energy by the corresponding weight. Identical.
        5. Final return — Both return the raw sum (no division). Identical.
        """
        num_nodes = X.size(0)

        u = self._compute_message_passing_matrix_eigenvector(edge_index, num_nodes, edge_weight)

        # Normalization
        u = u / np.max(u)
        u = np.maximum(u, 1e-4)

        u_t = torch.tensor(u, device=X.device, dtype=X.dtype).unsqueeze(1)  # (N, 1)
        X_norm = X / u_t # broadcast: each row X[i] / u[i]
        diff = X_norm[edge_index[0]] - X_norm[edge_index[1]]
        energies = (diff ** 2).sum(dim=1) # ||diff||² per edge
        if edge_weight is not None:
            energies = energies * edge_weight
        return energies.sum().item()

    def _compute_projection_energy(self, X, edge_index, edge_weight=None):
        num_nodes = X.size(0)

        u = self._compute_message_passing_matrix_eigenvector(edge_index, num_nodes, edge_weight)
        u = torch.tensor(u, device=X.device, dtype=X.dtype).unsqueeze(1)

        # Normalization
        u = u / torch.norm(u)

        # Original: P = torch.mm(u, u.t()) then PX = torch.mm(P, X)
        # This creates an N×N dense matrix P, costing O(N²) memory
        # (e.g. 10 GB for N=50K nodes).
        #
        # Optimized: exploit P = u @ u^T being rank-1, so
        # P @ X = u @ (u^T @ X).  Memory drops from O(N²) to O(N·d).
        utX = torch.mm(u.t(), X)   # (1, d)
        PX = torch.mm(u, utX)      # (N, d) — same result, no N×N matrix

        diff = X - PX
        energy = torch.norm(diff, p='fro')**2

        return energy.item()

    # def _compute_mad(self, X, edge_index):
    #     num_edges = edge_index.size(1)
    #     if num_edges == 0:
    #         return 0.0
    #     total_distance = 0.0
    #     for i in range(num_edges):
    #         node_i, node_j = edge_index[0, i], edge_index[1, i]
    #         X_i, X_j = X[node_i], X[node_j]
    #         norm_i = torch.norm(X_i, p=2)
    #         norm_j = torch.norm(X_j, p=2)
    #         if norm_i > 1e-8 and norm_j > 1e-8:
    #             cosine_sim = torch.dot(X_i, X_j) / (norm_i * norm_j)
    #             cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
    #             distance = 1.0 - cosine_sim
    #         else:
    #             distance = 1.0
    #         if isinstance(distance, float):
    #             total_distance += distance
    #         else:
    #             total_distance += distance.item()
    #     return total_distance / num_edges
    
    def _compute_mad(self, X, edge_index):
        """ Mean Angular Distance (MAD) between connected node embeddings. For each edge, compute 1 - cosine similarity, then average across all edges.

        Refactored from a loop-based implementation to a fully vectorized one for efficiency.

        The logic is preserved. Here's the trace through each case:
        1. Empty edges — Both return 0.0. Identical.
        2. Invalid edges (either norm <= 1e-8) — Original: distance = 1.0. New: distances is initialized to 1.0 and only valid positions are overwritten, so invalid edges keep 1.0.
        Identical.
        3. Valid edges (both norms > 1e-8) — Original: torch.dot(X_i, X_j) / (norm_i * norm_j). New: F.cosine_similarity(X_src, X_tgt, dim=1) which computes sum(X_src * X_tgt, dim=1) /
        max(||X_src|| * ||X_tgt||, eps). For valid edges the norm product is > 1e-16, well above the default eps=1e-8, so the max is a no-op. Same formula. Both clamp to [-1, 1], both
        compute 1.0 - cos_sim. Identical.
        4. Aggregation — Original: total_distance / num_edges. New: distances.mean() = sum / count. Identical.

        One subtlety worth noting: F.cosine_similarity is computed for all edges including invalid ones (it won't error because of its internal eps), but those values are never used —
        only cos_sim[valid] is written into distances. So the invalid-edge garbage is harmless.
        """
        if edge_index.size(1) == 0:
            return 0.0
        X_src = X[edge_index[0]]
        X_tgt = X[edge_index[1]]
        norm_src = torch.norm(X_src, p=2, dim=1)
        norm_tgt = torch.norm(X_tgt, p=2, dim=1)
        valid = (norm_src > 1e-8) & (norm_tgt > 1e-8)
        cos_sim = torch.clamp(
            F.cosine_similarity(X_src, X_tgt, dim=1), -1.0, 1.0)
        distances = torch.ones(edge_index.size(1), device=X.device)
        distances[valid] = 1.0 - cos_sim[valid]
        return distances.mean().item()

    def evaluate_model_oversmoothing(self, model, data, device='cpu'):
        model.eval()

        with torch.no_grad():
            data = data.to(device)

            if hasattr(model, 'get_embeddings'):
                embeddings = model.get_embeddings(data.x, data.edge_index, data.edge_weight)
            else:
                embeddings = model(data.x, data.edge_index, data.edge_weight)

            metrics = self.compute_all_metrics(
                X=embeddings,
                edge_index=data.edge_index,
                edge_weight=getattr(data, 'edge_weight', None)
            )

        return metrics

    def print_metrics(self, metrics):
        print("Oversmoothing metrics:")
        print(f"NumRank: {metrics['NumRank']:.6f}")
        print(f"Erank: {metrics['Erank']:.6f}")
        print(f"Dirichlet energy: {metrics['EDir']:.6f}")
        print(f"Traditional Dirichlet energy: {metrics['EDir_traditional']:.6f}")
        print(f"EProj: {metrics['EProj']:.6f}")
        print(f"MAD: {metrics['MAD']:.6f}")
        print("-"*50)


# --- Shared evaluation utilities ---

OVERSMOOTHING_KEYS = ['NumRank', 'Erank', 'EDir', 'EDir_traditional', 'EProj', 'MAD']
DEFAULT_OVERSMOOTHING = {k: 0.0 for k in OVERSMOOTHING_KEYS}
ZERO_CLS = {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}


def compute_train_noise_split_cls(predictions, noisy_labels, clean_labels, train_mask):
    """Classification metrics on clean vs mislabelled training subsets.

    Mislabelled nodes are detected dynamically: noisy_labels != clean_labels on train_mask.

    Returns dict with keys:
        train_only_clean_cls, train_only_mislabelled_factual_cls,
        train_only_mislabelled_corrected_cls
    """
    cls_evaluator = ClassificationMetrics(average='macro')

    train_indices = train_mask.nonzero(as_tuple=True)[0]
    is_mislabelled = noisy_labels[train_mask] != clean_labels[train_mask]
    clean_idx = train_indices[~is_mislabelled]
    mislabelled_idx = train_indices[is_mislabelled]

    result = {}
    result['train_only_clean_cls'] = (
        cls_evaluator.compute_all_metrics(predictions[clean_idx], clean_labels[clean_idx])
        if len(clean_idx) > 0 else dict(ZERO_CLS)
    )
    result['train_only_mislabelled_factual_cls'] = (
        cls_evaluator.compute_all_metrics(predictions[mislabelled_idx], noisy_labels[mislabelled_idx])
        if len(mislabelled_idx) > 0 else dict(ZERO_CLS)
    )
    result['train_only_mislabelled_corrected_cls'] = (
        cls_evaluator.compute_all_metrics(predictions[mislabelled_idx], clean_labels[mislabelled_idx])
        if len(mislabelled_idx) > 0 else dict(ZERO_CLS)
    )
    return result


def compute_oversmoothing_for_mask(oversmoothing_evaluator, embeddings, edge_index, mask):
    """Compute oversmoothing metrics for a node subset defined by a boolean mask.
    """
    try:
        mask_indices = torch.where(mask)[0]
        mask_embeddings = embeddings[mask]

        src_in = torch.isin(edge_index[0], mask_indices)
        tgt_in = torch.isin(edge_index[1], mask_indices)
        edge_mask = src_in & tgt_in

        if not edge_mask.any():
            return {
                'NumRank': float(min(mask_embeddings.shape)),
                'Erank': float(min(mask_embeddings.shape)),
                'EDir': 0.0,
                'EDir_traditional': 0.0,
                'EProj': 0.0,
                'MAD': 0.0
            }

        masked_edges = edge_index[:, edge_mask]
        remapped_edges = torch.stack([
            torch.searchsorted(mask_indices, masked_edges[0]),
            torch.searchsorted(mask_indices, masked_edges[1]),
        ])
        # graphs_in_class = [{
        #     'X': mask_embeddings,
        #     'edge_index': remapped_edges,
        #     'edge_weight': None
        # }]
        return oversmoothing_evaluator.compute_all_metrics(
            X=mask_embeddings,
            edge_index=remapped_edges,
        )

    except Exception as e:
        print(f"Warning: Could not compute oversmoothing metrics for mask: {e}")
        return dict(DEFAULT_OVERSMOOTHING)


@torch.no_grad()
def evaluate_model(get_predictions, get_embeddings, labels, train_mask, val_mask, test_mask,
                   edge_index, device):
    """Shared final test evaluation.

    Args:
        get_predictions: callable returning Tensor[num_nodes] of integer class labels
        get_embeddings: callable returning Tensor[num_nodes, dim] for oversmoothing
        labels: Tensor[num_nodes] of true labels
        train_mask, val_mask, test_mask: boolean masks
        edge_index: edge index tensor for oversmoothing computation
        device: computation device

    Returns:
        dict with test_cls, train_cls, val_cls (each a dict with accuracy,
        f1, precision, recall), test_oversmoothing, train_oversmoothing_final,
        val_oversmoothing_final
    """
    oversmoothing_evaluator = OversmoothingMetrics(device=device)
    cls_evaluator = ClassificationMetrics(average='macro')

    predictions = get_predictions()
    test_cls_metrics = cls_evaluator.compute_all_metrics(predictions[test_mask], labels[test_mask])
    train_cls_metrics = cls_evaluator.compute_all_metrics(predictions[train_mask], labels[train_mask])
    val_cls_metrics = cls_evaluator.compute_all_metrics(predictions[val_mask], labels[val_mask])

    embeddings = get_embeddings()
    test_oversmoothing = compute_oversmoothing_for_mask(
        oversmoothing_evaluator, embeddings, edge_index, test_mask)
    train_oversmoothing = compute_oversmoothing_for_mask(
        oversmoothing_evaluator, embeddings, edge_index, train_mask)
    val_oversmoothing = compute_oversmoothing_for_mask(
        oversmoothing_evaluator, embeddings, edge_index, val_mask)

    def normalize_metrics(d):
        if d is None:
            return dict(DEFAULT_OVERSMOOTHING)
        return {k: d.get(k, 0.0) for k in OVERSMOOTHING_KEYS}

    result = {
        'test_cls': test_cls_metrics,
        'train_cls': train_cls_metrics,
        'val_cls': val_cls_metrics,
        'test_oversmoothing': normalize_metrics(test_oversmoothing),
        'train_oversmoothing_final': normalize_metrics(train_oversmoothing),
        'val_oversmoothing_final': normalize_metrics(val_oversmoothing),
    }
    result['_predictions'] = predictions  # consumed by _make_result, not serialised
    return result


def compute_training_metrics(predictions, labels, train_mask, val_mask,
                             embeddings=None, edge_index=None, device=None,
                             compute_oversmoothing=False):
    """In-loop monitoring utility.

    Args:
        predictions: Tensor[num_nodes] of integer class labels
        labels: Tensor[num_nodes] of true labels
        train_mask, val_mask: boolean masks
        embeddings: optional Tensor[num_nodes, dim] for oversmoothing
        edge_index: optional edge index for oversmoothing
        device: computation device
        compute_oversmoothing: whether to compute oversmoothing metrics

    Returns:
        dict with train_acc, val_acc, train_f1, val_f1,
        and optionally train_oversmoothing, val_oversmoothing
    """
    cls_evaluator = ClassificationMetrics(average='macro')

    result = {
        'train_acc': cls_evaluator.compute_accuracy(predictions[train_mask], labels[train_mask]),
        'val_acc': cls_evaluator.compute_accuracy(predictions[val_mask], labels[val_mask]),
        'train_f1': cls_evaluator.compute_f1(predictions[train_mask], labels[train_mask]),
        'val_f1': cls_evaluator.compute_f1(predictions[val_mask], labels[val_mask]),
    }

    if compute_oversmoothing and embeddings is not None and edge_index is not None:
        oversmoothing_evaluator = OversmoothingMetrics(
            device=device or embeddings.device)
        result['train_oversmoothing'] = compute_oversmoothing_for_mask(
            oversmoothing_evaluator, embeddings, edge_index, train_mask)
        result['val_oversmoothing'] = compute_oversmoothing_for_mask(
            oversmoothing_evaluator, embeddings, edge_index, val_mask)

    return result
