import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.utils import degree


def remove_duplicate_edges(edge_index):
    processed_edges = set()
    new_edge_index = []

    for e in range(edge_index.size(1)):
        source, target = sorted((edge_index[0, e].item(), edge_index[1, e].item()))
        if (source, target) in processed_edges:
            continue
        processed_edges.add((source, target))
        new_edge_index.append([source, target])
    print(f"Removed {edge_index.size(1) - len(new_edge_index)} edges")
    return torch.tensor(new_edge_index, dtype=torch.long).t()

def batched_sym_matrix_pow(matrices: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrices: A batch of matrices.
        p: Power.
        positive_definite: If positive definite
    Returns:
        Power of each matrix in the batch.
    """
    # vals, vecs = torch.linalg.eigh(matrices)
    # SVD is much faster than  vals, vecs = torch.linalg.eigh(matrices) for large batches.
    vecs, vals, _ = torch.linalg.svd(matrices)
    good = vals > vals.max(-1, True).values * vals.size(-1) * torch.finfo(vals.dtype).eps
    vals = vals.pow(p).where(good, torch.zeros((), device=matrices.device, dtype=matrices.dtype))
    matrix_power = (vecs * vals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)
    return matrix_power

def compute_learnable_laplacian_indices(size, edge_index, learned_d, total_d):
    assert torch.all(edge_index[0] < edge_index[1])

    row, col = edge_index
    device = edge_index.device
    row_template = torch.arange(0, learned_d, device=device).view(1, -1, 1).tile(1, 1, learned_d)
    col_template = torch.transpose(row_template, dim0=1, dim1=2)

    non_diag_row_indices = (row_template + total_d*row.reshape(-1, 1, 1)).reshape(1, -1)
    non_diag_col_indices = (col_template + total_d*col.reshape(-1, 1, 1)).reshape(1, -1)
    non_diag_indices = torch.cat((non_diag_row_indices, non_diag_col_indices), dim=0)

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (row_template + total_d*diag.reshape(-1, 1, 1)).reshape(1, -1)
    diag_col_indices = (col_template + total_d*diag.reshape(-1, 1, 1)).reshape(1, -1)
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices

def compute_fixed_diag_laplacian_indices(size, edge_index, learned_d, total_d):
    assert torch.all(edge_index[0] < edge_index[1])
    row, col = edge_index
    device = edge_index.device
    row_template = torch.arange(learned_d, total_d, device=device).view(1, -1)
    col_template = row_template.clone()

    non_diag_row_indices = (row_template + total_d*row.unsqueeze(1)).reshape(1, -1)
    non_diag_col_indices = (col_template + total_d*col.unsqueeze(1)).reshape(1, -1)
    non_diag_indices = torch.cat((non_diag_row_indices, non_diag_col_indices), dim=0)

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (row_template + total_d*diag.unsqueeze(1)).reshape(1, -1)
    diag_col_indices = (col_template + total_d*diag.unsqueeze(1)).reshape(1, -1)
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices


def compute_left_right_map_index(edge_index, full_matrix=False):
    """Computes indices for lower triangular matrix or full matrix"""
    edge_to_idx = dict()
    for e in range(edge_index.size(1)):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        edge_to_idx[(source, target)] = e

    left_index, right_index = [], []
    row, col = [], []
    for e in range(edge_index.size(1)):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        if source < target or full_matrix:
            left_index.append(e)
            right_index.append(edge_to_idx[(target, source)])

            row.append(source)
            col.append(target)

    left_index = torch.tensor(left_index, dtype=torch.long, device=edge_index.device)
    right_index = torch.tensor(right_index, dtype=torch.long, device=edge_index.device)
    left_right_index = torch.vstack([left_index, right_index])

    row = torch.tensor(row, dtype=torch.long, device=edge_index.device)
    col = torch.tensor(col, dtype=torch.long, device=edge_index.device)
    new_edge_index = torch.vstack([row, col])

    if full_matrix:
        assert len(left_index) == edge_index.size(1)
    else:
        assert len(left_index) == edge_index.size(1) // 2

    return left_right_index, new_edge_index


def mergesp(index1, value1, index2, value2):
    """Merges two sparse matrices with disjoint indices into one."""
    assert index1.dim() == 2 and index2.dim() == 2
    assert value1.dim() == 1 and value2.dim() == 1
    assert index1.size(1) == value1.numel()
    assert index2.size(1) == value2.numel()
    assert index1.size(0) == 2 and index2.size(0) == 2

    index = torch.cat([index1, index2], dim=1)
    val = torch.cat([value1, value2])
    return index, val


class LaplacianBuilder(nn.Module):
    def __init__(self, size, edge_index, d, normalised=False, deg_normalised=False, add_hp=False, add_lp=False,
                 augmented=True):
        super(LaplacianBuilder, self).__init__()
        assert not (normalised and deg_normalised)

        self.d = d
        self.final_d = d
        if add_hp:
            self.final_d += 1
        if add_lp:
            self.final_d += 1
        self.size = size
        self.edges = edge_index.size(1) // 2
        self.edge_index = edge_index
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.device = edge_index.device
        self.add_hp = add_hp
        self.add_lp = add_lp
        self.augmented = augmented

        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.full_left_right_idx, _ = compute_left_right_map_index(edge_index, full_matrix=True)
        self.left_right_idx, self.vertex_tril_idx = compute_left_right_map_index(edge_index)
        if self.add_lp or self.add_hp:
            self.fixed_diag_indices, self.fixed_tril_indices = compute_fixed_diag_laplacian_indices(
                size, self.vertex_tril_idx, self.d, self.final_d)
        self.deg = degree(self.edge_index[0], num_nodes=self.size)

    def get_fixed_maps(self, size, dtype):
        assert self.add_lp or self.add_hp

        fixed_diag, fixed_non_diag = [], []
        if self.add_lp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(torch.ones(size=(size, 1), device=self.device, dtype=dtype))
        if self.add_hp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(-torch.ones(size=(size, 1), device=self.device, dtype=dtype))

        fixed_diag = torch.cat(fixed_diag, dim=1)
        fixed_non_diag = torch.cat(fixed_non_diag, dim=1)

        assert self.fixed_tril_indices.size(1) == fixed_non_diag.numel()
        assert self.fixed_diag_indices.size(1) == fixed_diag.numel()

        return fixed_diag, fixed_non_diag

    def scalar_normalise(self, diag, tril, row, col):
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)

        if self.augmented:
            diag_sqrt_inv = (diag + 1).pow(-0.5)
        else:
            diag_sqrt_inv = diag.pow(-0.5)
            diag_sqrt_inv.masked_fill_(diag_sqrt_inv == float('inf'), 0)
        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if tril.dim() > 2 else diag_sqrt_inv.view(-1, d)
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        non_diag_maps = left_norm * tril * right_norm

        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if diag.dim() > 2 else diag_sqrt_inv.view(-1, d)
        diag_maps = diag_sqrt_inv**2 * diag

        return diag_maps, non_diag_maps

    def append_fixed_maps(self, size, diag_indices, diag_maps, tril_indices, tril_maps):
        if not self.add_lp and not self.add_hp:
            return (diag_indices, diag_maps), (tril_indices, tril_maps)

        fixed_diag, fixed_non_diag = self.get_fixed_maps(size, tril_maps.dtype)
        tril_row, tril_col = self.vertex_tril_idx

        # Normalise the fixed parts.
        if self.normalised:
            fixed_diag, fixed_non_diag = self.scalar_normalise(fixed_diag, fixed_non_diag, tril_row, tril_col)
        fixed_diag, fixed_non_diag = fixed_diag.view(-1), fixed_non_diag.view(-1)
        # Combine the learnable and fixed parts.
        tril_indices, tril_maps = mergesp(self.fixed_tril_indices, fixed_non_diag, tril_indices, tril_maps)
        diag_indices, diag_maps = mergesp(self.fixed_diag_indices, fixed_diag, diag_indices, diag_maps)

        return (diag_indices, diag_maps), (tril_indices, tril_maps)

    def create_with_new_edge_index(self, edge_index):
        assert edge_index.max() <= self.size
        new_builder = self.__class__(
            self.size, edge_index, self.d,
            normalised=self.normalised, deg_normalised=self.deg_normalised, add_hp=self.add_hp, add_lp=self.add_lp,
            augmented=self.augmented)
        new_builder.train(self.training)
        return new_builder


class GeneralLaplacianBuilder(LaplacianBuilder):
    """Learns a multi-dimensional Sheaf Laplacian from data."""

    def __init__(self, size, edge_index, d, normalised=False, deg_normalised=False,
                 add_hp=False, add_lp=False, augmented=True):
        super(GeneralLaplacianBuilder, self).__init__(size, edge_index, d,
                                                      normalised=normalised, deg_normalised=deg_normalised,
                                                      add_hp=add_hp, add_lp=add_lp, augmented=augmented)

        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.diag_indices, self.tril_indices = compute_learnable_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d)

    def normalise(self, diag_maps, non_diag_maps, tril_row, tril_col):
        if self.normalised:
            # Normalise the entries if the normalised Laplacian is used.
            if self.training:
                # During training, we perturb the matrices to ensure they have different singular values.
                # Without this, the gradients of batched_sym_matrix_pow, which uses SVD are non-finite.
                eps = torch.FloatTensor(self.d).uniform_(-0.001, 0.001).to(device=self.device)
            else:
                eps = torch.zeros(self.d, device=self.device)

            to_be_inv_diag_maps = diag_maps + torch.diag(1. + eps).unsqueeze(0) if self.augmented else diag_maps
            d_sqrt_inv = batched_sym_matrix_pow(to_be_inv_diag_maps, -0.5)
            assert torch.all(torch.isfinite(d_sqrt_inv))
            left_norm = d_sqrt_inv[tril_row]
            right_norm = d_sqrt_inv[tril_col]
            non_diag_maps = (left_norm @ non_diag_maps @ right_norm).clamp(min=-1, max=1)
            diag_maps = (d_sqrt_inv @ diag_maps @ d_sqrt_inv).clamp(min=-1, max=1)
            assert torch.all(torch.isfinite(non_diag_maps))
            assert torch.all(torch.isfinite(diag_maps))
        elif self.deg_normalised:
            # These are general d x d maps so we need to divide by 1 / sqrt(deg * d), their maximum possible norm.
            deg_sqrt_inv = (self.deg * self.d + 1).pow(-1/2) if self.augmented else (self.deg * self.d + 1).pow(-1/2)
            deg_sqrt_inv = deg_sqrt_inv.view(-1, 1, 1)
            left_norm = deg_sqrt_inv[tril_row]
            right_norm = deg_sqrt_inv[tril_col]
            non_diag_maps = left_norm * non_diag_maps * right_norm
            diag_maps = deg_sqrt_inv * diag_maps * deg_sqrt_inv
        return diag_maps, non_diag_maps

    def forward(self, maps):
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = self.edge_index

        # Compute transport maps.
        assert torch.all(torch.isfinite(maps))
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -torch.bmm(torch.transpose(left_maps, dim0=-1, dim1=-2), right_maps)
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = torch.bmm(torch.transpose(maps, dim0=-1, dim1=-2), maps)
        diag_maps = scatter_add(diag_maps, row, dim=0, dim_size=self.size)

        # Normalise the transport maps.
        diag_maps, tril_maps = self.normalise(diag_maps, tril_maps, tril_row, tril_col)
        diag_maps, tril_maps = diag_maps.view(-1), tril_maps.view(-1)

        # Append fixed diagonal values in the non-learnable dimensions.
        (diag_indices, diag_maps), (tril_indices, tril_maps) = self.append_fixed_maps(
            len(left_maps), diag_indices, diag_maps, tril_indices, tril_maps)

        # Add the upper triangular part.
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = mergesp(tril_indices, tril_maps, triu_indices, tril_maps)

        # Merge diagonal and non-diagonal
        edge_index, weights = mergesp(non_diag_indices, non_diag_values, diag_indices, diag_maps)

        return (edge_index, weights), saved_tril_maps