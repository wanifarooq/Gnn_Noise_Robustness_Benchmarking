"""Noise generation — transition matrices and label corruption strategies."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


def simple_uniform_noise(labels, n_classes, noise_rate, seed):
    """Per-node coin-flip noise with uniform random class replacement.

    For each node, independently flips its label with probability ``noise_rate``
    to a uniformly chosen *different* class.

    Args:
        labels: 1-D tensor of original integer labels.
        n_classes: Total number of classes.
        noise_rate: Probability of flipping each label (0 = no noise).
        seed: RNG seed for reproducibility.

    Returns:
        Tensor of (possibly corrupted) labels, same shape as *labels*.
    """
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
    """Build a C x C symmetric transition matrix with uniform off-diagonal noise.

    P[i,j] = noise_rate / (C - 1)  for i != j
    P[i,i] = 1 - noise_rate

    Rows are renormalized to sum to 1.

    Args:
        n_classes: Number of classes C.
        noise_rate: Total probability mass shifted away from the diagonal.

    Returns:
        numpy array of shape (C, C).
    """
    P = np.full((n_classes, n_classes), noise_rate / (n_classes - 1), dtype=np.float64)
    np.fill_diagonal(P, 1 - noise_rate)
    P[np.arange(n_classes), np.arange(n_classes)] += 1 - P.sum(axis=1)
    return P


def pair_noise(n_classes, noise_rate):
    """Circular pair-flip transition matrix.

    Each class i flips to its predecessor (i - 1) mod C with probability
    ``noise_rate`` and stays unchanged with probability 1 - ``noise_rate``.

    Args:
        n_classes: Number of classes C.
        noise_rate: Flip probability.

    Returns:
        numpy array of shape (C, C).
    """
    P = (1 - noise_rate) * np.eye(n_classes, dtype=np.float64)
    for i in range(n_classes):
        P[i, (i - 1) % n_classes] = noise_rate
    return P


def random_noise_cp(n_classes, noise_rate, seed):
    """Random off-diagonal transition matrix (seeded).

    For each class i the off-diagonal probability mass ``noise_rate`` is
    distributed randomly among the other C - 1 classes.

    Args:
        n_classes: Number of classes C.
        noise_rate: Total off-diagonal mass per row.
        seed: RNG seed for reproducibility.

    Returns:
        numpy array of shape (C, C).
    """
    rng = np.random.RandomState(seed)

    P = (1 - noise_rate) * np.eye(n_classes, dtype=np.float64)
    for i in range(n_classes):
        probs = rng.rand(n_classes)
        probs[i] = 0
        probs /= probs.sum()
        P[i, :] += noise_rate * probs
    return P


def random_pair_noise(n_classes, noise_rate, seed):
    """Each class flips to one randomly chosen other class (seeded).

    Unlike ``pair_noise`` which uses a fixed circular pattern, here each class
    independently picks a single target class at random.

    Args:
        n_classes: Number of classes C.
        noise_rate: Flip probability.
        seed: RNG seed for reproducibility.

    Returns:
        numpy array of shape (C, C).
    """
    rng = np.random.default_rng(seed)
    P = np.eye(n_classes, dtype=np.float64) * (1 - noise_rate)
    for i in range(n_classes):
        candidates = list(range(i)) + list(range(i + 1, n_classes))
        chosen = rng.choice(candidates)
        P[i, chosen] = noise_rate
    return P


def deterministic(labels, idx_train, noise_rate, seed):
    """Corrupt exactly ``rate * |train|`` training nodes.

    Selects a fixed-size subset of training indices and flips each to a
    uniformly random different class.  Returns the corrupted label array plus
    the split into noise / clean index lists.

    Args:
        labels: Full label array (tensor or numpy).
        idx_train: Training node indices (tensor or numpy).
        noise_rate: Fraction of training nodes to corrupt.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (corrupted_labels_np, noise_idx_list, clean_idx_list).
    """
    rng = np.random.RandomState(seed)

    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    idx_train_np = idx_train.cpu().numpy() if torch.is_tensor(idx_train) else idx_train

    corrupted_labels = labels_np.copy()
    num_classes = len(np.unique(labels_np))

    num_noise = int(len(idx_train_np) * noise_rate)
    noise_indices = rng.choice(idx_train_np, num_noise, replace=False)

    noise_indices_set = set(noise_indices)
    noise_idx, clean_idx = [], []
    for idx in idx_train_np:
        if idx in noise_indices_set:
            original = labels_np[idx]
            possible = [i for i in range(num_classes) if i != original]
            corrupted_labels[idx] = rng.choice(possible)
            noise_idx.append(idx)
        else:
            clean_idx.append(idx)

    return corrupted_labels, noise_idx, clean_idx


def flip_noise(n_classes, noise_rate):
    """Chain-flip transition matrix: class i -> i + 1, last -> 0.

    Creates a directed cycle 0 -> 1 -> ... -> C-1 -> 0 where each link
    carries probability ``noise_rate``.

    Args:
        n_classes: Number of classes C.
        noise_rate: Flip probability along the chain.

    Returns:
        numpy array of shape (C, C).
    """
    P = np.eye(n_classes, dtype=np.float64) * (1 - noise_rate)

    P[0, 1] = noise_rate

    for i in range(1, n_classes - 1):
        P[i, i + 1] = noise_rate

    P[n_classes - 1, 0] = noise_rate

    return P


def uniform_mix_noise(n_classes, noise_rate):
    """Uniform-mix transition matrix: P = (1 - rate) * I + rate / C.

    Every class has equal probability ``rate / C`` of transitioning to *any*
    class (including itself), plus ``(1 - rate)`` on the diagonal.

    Args:
        n_classes: Number of classes C.
        noise_rate: Mixing weight.

    Returns:
        numpy array of shape (C, C).
    """
    P = np.eye(n_classes, dtype=np.float64) * (1 - noise_rate)
    P += noise_rate / n_classes

    return P


def instance_independent_noise(labels, cp, seed):
    """Apply a pre-computed transition matrix via multinomial sampling.

    For each node, draws a new label from the row of ``cp`` corresponding to
    its current label.

    Args:
        labels: 1-D numpy array of integer labels.
        cp: Transition matrix of shape (C, C), rows sum to 1.
        seed: RNG seed for reproducibility.

    Returns:
        1-D numpy array of (possibly corrupted) labels.
    """
    rs = np.random.RandomState(seed)
    noisy_labels = np.array([np.where(rs.multinomial(1, cp[label]))[0][0] for label in labels])
    return noisy_labels


def instance_dependent_noise(noise_rate, feature, labels, num_classes, norm_std, seed):
    """Instance-dependent noise: per-node flip rate from truncated normal.

    Each node's flip probability is drawn from TruncNorm(noise_rate, norm_std)
    clipped to [0, 1].  Class transition probabilities depend on node features
    via a learned weight tensor W, so nodes with similar features tend to
    receive similar corruption patterns.

    Args:
        noise_rate: Mean flip rate (center of truncated normal).
        feature: Node feature tensor of shape (N, F).
        labels: 1-D numpy array of integer labels.
        num_classes: Number of classes C.
        norm_std: Std-dev of the truncated normal for per-node flip rates.
        seed: RNG seed for reproducibility.

    Returns:
        1-D numpy array of corrupted labels.
    """
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
    """Dispatcher: validate noise_type, delegate to the specific generator, log stats.

    Validates ``noise_type`` against the known set, dispatches to the
    appropriate generator function, applies the transition matrix (if
    applicable) via ``instance_independent_noise``, and optionally prints
    corruption statistics.

    Args:
        labels: 1-D tensor of original integer labels.
        features: Node feature tensor (required only for 'instance' noise).
        n_classes: Number of classes.
        noise_type: One of 'clean', 'uniform_simple', 'uniform', 'random',
            'pair', 'random_pair', 'flip', 'uniform_mix', 'deterministic',
            'instance'.
        noise_rate: Corruption rate in [0, 1].
        noise_seed: RNG seed.
        idx_train: Training indices (required for 'deterministic' noise).
        debug: If True, print noise statistics when noise_rate > 0.

    Returns:
        Tuple of (noisy_labels tensor, noisy_indices numpy array).
    """
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
