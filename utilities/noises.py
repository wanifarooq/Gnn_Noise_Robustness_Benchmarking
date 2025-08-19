import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from numpy.testing import assert_array_almost_equal

def simple_uniform_noise(labels, n_classes, noise_rate, random_seed):
    if noise_rate == 0:
        return labels.clone()
    n_samples = len(labels)
    noisy_labels = labels.clone()
    for i in range(n_samples):
        if np.random.rand() < noise_rate:
            available_classes = list(range(n_classes))
            available_classes.remove(labels[i].item())
            noisy_labels[i] = np.random.choice(available_classes)
    return noisy_labels

def uniform_noise_cp(n_classes, noise_rate):
    P = np.full((n_classes, n_classes), noise_rate / (n_classes - 1), dtype=np.float64)
    np.fill_diagonal(P, 1 - noise_rate)
    P[np.arange(n_classes), np.arange(n_classes)] += 1 - P.sum(axis=1)
    assert_array_almost_equal(P.sum(axis=1), 1, decimal=6)
    return P

def pair_noise_cp(n_classes, noise_rate):
    P = (1 - noise_rate) * np.eye(n_classes, dtype=np.float64)
    for i in range(n_classes):
        P[i, (i - 1) % n_classes] = noise_rate
    assert_array_almost_equal(P.sum(axis=1), 1, decimal=6)
    return P

def random_noise_cp(n_classes, noise_rate):
    P = (1 - noise_rate) * np.eye(n_classes, dtype=np.float64)
    for i in range(n_classes):
        probs = np.random.rand(n_classes)
        probs[i] = 0
        probs /= probs.sum()
        P[i, :] += noise_rate * probs
    assert_array_almost_equal(P.sum(axis=1), 1, decimal=6)
    return P

def random_pair_noise_cp(n_classes, noise_rate, seed=1):
    rng = np.random.default_rng(seed)
    P = np.eye(n_classes, dtype=np.float64) * (1 - noise_rate)
    for i in range(n_classes):
        candidates = list(range(i)) + list(range(i + 1, n_classes))
        chosen = rng.choice(candidates)
        P[i, chosen] = noise_rate
    assert_array_almost_equal(P.sum(axis=1), 1, decimal=6)
    return P

def deterministic(labels, idx_train, noise_rate=0.2):

    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    idx_train_np = idx_train.cpu().numpy() if torch.is_tensor(idx_train) else idx_train

    corrupted_labels = labels_np.copy()
    num_classes = len(np.unique(labels_np))
    
    num_noise = int(len(idx_train_np) * noise_rate)
    noise_indices = np.random.choice(idx_train_np, num_noise, replace=False)
    
    noise_idx, clean_idx = [], []
    for idx in idx_train_np:
        if idx in noise_indices:
            original = labels_np[idx]
            possible = [i for i in range(num_classes) if i != original]
            corrupted_labels[idx] = np.random.choice(possible)
            noise_idx.append(idx)
        else:
            clean_idx.append(idx)
    
    return corrupted_labels, noise_idx, clean_idx


def flip_noise_cp(n_classes, noise_rate, seed=1):
    P = np.eye(n_classes, dtype=np.float64) * (1 - noise_rate)

    P[0, 1] = noise_rate

    for i in range(1, n_classes - 1):
        P[i, i + 1] = noise_rate

    P[n_classes - 1, 0] = noise_rate
    
    assert_array_almost_equal(P.sum(axis=1), 1, decimal=6)
    return P

def uniform_mix_revised_noise_cp(n_classes, noise_rate):

    P = np.eye(n_classes, dtype=np.float64) * (1 - noise_rate)
    P += noise_rate / n_classes
    assert_array_almost_equal(P.sum(axis=1), 1, decimal=6)
    return P

def add_instance_independent_label_noise(labels, cp, random_seed):
    assert_array_almost_equal(cp.sum(axis=1), np.ones(cp.shape[0]), decimal=6)
    rs = np.random.RandomState(random_seed)
    noisy_labels = np.array([np.where(rs.multinomial(1, cp[label]))[0][0] for label in labels])
    return noisy_labels

def add_instance_dependent_label_noise(noise_rate, feature, labels, num_classes, norm_std, seed):
    num_nodes, feature_size = feature.shape
    flip_dist = stats.truncnorm((0 - noise_rate) / norm_std, (1 - noise_rate) / norm_std, loc=noise_rate, scale=norm_std)
    flip_rate = flip_dist.rvs(num_nodes)
    labels_t = torch.tensor(labels, dtype=torch.long, device=feature.device)
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
    new_label = np.array([np.random.choice(num_classes, p=P[i]) for i in range(num_nodes)])
    return new_label

def label_process(labels, features, n_classes, noise_type='uniform', noise_rate=0, random_seed=5, idx_train=None, debug=True):
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
            noisy_labels = simple_uniform_noise(labels, n_classes, noise_rate, random_seed)
        elif noise_type == 'uniform':
            cp = uniform_noise_cp(n_classes, noise_rate)
        elif noise_type == 'random':
            cp = random_noise_cp(n_classes, noise_rate)
        elif noise_type == 'pair':
            cp = pair_noise_cp(n_classes, noise_rate)
        elif noise_type == 'random_pair':
            cp = random_pair_noise_cp(n_classes, noise_rate, seed=random_seed)
        elif noise_type == 'flip':
            cp = flip_noise_cp(n_classes, noise_rate, seed=random_seed)
        elif noise_type == 'uniform_mix':
            cp = uniform_mix_revised_noise_cp(n_classes, noise_rate)
        elif noise_type == 'deterministic':
            if idx_train is None:
                raise ValueError("idx_train must be provided for deterministic noise")
            noisy_labels_np, _, _ = deterministic(labels, idx_train, noise_rate=noise_rate)
            noisy_labels = torch.tensor(noisy_labels_np, dtype=torch.long, device=labels.device)
        elif noise_type == 'instance':
            if features is None:
                raise ValueError("features must be provided for instance-dependent noise")
            noisy_labels_np = add_instance_dependent_label_noise(
                noise_rate=noise_rate,
                feature=features,
                labels=labels.cpu().numpy(),
                num_classes=n_classes,
                norm_std=0.1,
                seed=random_seed
            )
            noisy_labels = torch.tensor(noisy_labels_np, dtype=torch.long, device=labels.device)
        else:
            print(f"[Warning] Noise type '{noise_type}' does not exist. Labels will remain unchanged.")
            noisy_labels = labels.clone()

    if noisy_labels is None:
        if cp is not None:
            noisy_labels_np = add_instance_independent_label_noise(labels.cpu().numpy(), cp, random_seed)
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
