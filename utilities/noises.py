import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from numpy.testing import assert_array_almost_equal
from utilities.usefull import setup_seed_device

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

def random_pair_noise_cp(n_classes, noise_rate, seed=1):
    rng = np.random.default_rng(seed)
    P = np.eye(n_classes, dtype=np.float64) * (1 - noise_rate)
    for i in range(n_classes):
        candidates = list(range(i)) + list(range(i + 1, n_classes))
        chosen = rng.choice(candidates)
        P[i, chosen] = noise_rate
    assert_array_almost_equal(P.sum(axis=1), 1, decimal=6)
    return P

def label_process(labels, features, n_classes, noise_type='uniform', noise_rate=0, random_seed=5, debug=True):
    assert 0 <= noise_rate <= 1
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
        elif noise_type == 'instance':
            pass
        else:
            cp = np.eye(n_classes)
    if noisy_labels is None:
        if noise_rate > 0 and cp is not None:
            noisy_labels_np = add_instance_independent_label_noise(labels.cpu().numpy(), cp, random_seed)
            noisy_labels = torch.tensor(noisy_labels_np, dtype=torch.long, device=labels.device)
        elif noise_rate > 0 and cp is None:
            noisy_labels_np = add_instance_dependent_label_noise(noise_rate, features, labels.cpu().numpy(), n_classes, 0.1, random_seed)
            noisy_labels = torch.tensor(noisy_labels_np, dtype=torch.long, device=labels.device)
        else:
            noisy_labels = labels.clone()
    noisy_indices = (noisy_labels != labels).nonzero(as_tuple=True)[0].cpu().numpy()
    return noisy_labels, noisy_indices
