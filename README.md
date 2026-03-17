# Benchmarking Robustness Strategies for Graph Neural Networks under Noise

A benchmarking framework for systematically evaluating GNN robustness strategies under label noise. It reproduces 13 robustness methods under standardized conditions across 24 datasets with 10 noise types, enabling fair and reproducible comparisons. The framework measures classification performance, oversmoothing behaviour, and computational cost in a unified pipeline.

<div style="display: flex; justify-content: center; gap: 10px; margin-bottom: 20px;">
  <img src="images/diagram.png" alt="Symmetric" width="100%">
</div>

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [How to Run](#how-to-run)
  - [Automatic Benchmarking (main.py)](#1-automatic-benchmarking--mainpy)
  - [Parallel Single Run (main_multithreading.py)](#2-parallel-single-run--main_multithreadingpy)
  - [Programmatic API](#3-programmatic-api)
- [Configuration](#configuration)
  - [General Settings](#general-settings)
  - [Dataset](#dataset)
  - [Noise](#noise)
  - [Model Backbone](#model-backbone)
  - [Training](#training)
  - [Parameter Sweeps](#parameter-sweeps)
- [Robustness Methods](#robustness-methods)
  - [Method Overview](#method-overview)
  - [Method-Specific Parameters](#method-specific-parameters)
- [Batch Processing](#batch-processing)
- [Transductive and Inductive Learning](#transductive-and-inductive-learning)
- [Evaluation Metrics](#evaluation-metrics)
  - [Classification Metrics](#classification-metrics)
  - [Oversmoothing Metrics](#oversmoothing-metrics)
  - [Noise-Split Analysis](#noise-split-analysis)
  - [Compute Metrics](#compute-metrics)
- [Output and Results](#output-and-results)
- [Adding New Methods](#adding-new-methods)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
- [Author](#author)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
# For GPU (CUDA 11.8), run setup.sh instead

# 2. Edit config.yaml to set dataset, noise, method, etc.

# 3. Run the benchmark
python main.py -c config.yaml

# Results are saved to results/<experiment_dir>/experiment.json
```

---

## Installation

**CPU:**
```bash
pip install -r requirements.txt
```

**GPU (CUDA 11.8):**
```bash
./setup.sh
```

This installs PyTorch 2.2.1, PyTorch Geometric, and all dependencies.

**Dependencies:** PyTorch, PyTorch Geometric (torch-scatter, torch-sparse, torch-cluster, pyg-lib), scikit-learn, networkx, pandas, matplotlib, codecarbon, pytest.

---

## How to Run

### 1. Automatic Benchmarking &mdash; `main.py`

The primary entry point. Runs multi-run sweeps with automatic incremental execution, checkpointing, and aggregated statistics.

```bash
python main.py -c config.yaml
```

| Flag | Default | Description |
|------|---------|-------------|
| `-c, --config` | `config.yaml` | Path to YAML configuration file |
| `--num-runs` | from config or 5 | Number of runs per configuration |
| `--force` | off | Re-run all experiments, ignore completed runs |
| `--eval-only` | off | Skip training, evaluate from saved checkpoints |
| `--no-checkpoint` | off | Disable saving model checkpoints |

**Incremental execution:** Only missing runs are executed. Completed runs are detected from `run_N/training_log.json`. Increasing `--num-runs` from 5 to 10 executes only runs 6-10.

```bash
# Start with 2 runs
python main.py -c config.yaml --num-runs 2

# Later, bump to 5 -- only runs 3-5 execute
python main.py -c config.yaml --num-runs 5

# Force full re-run
python main.py -c config.yaml --num-runs 5 --force

# Re-evaluate from checkpoints without training
python main.py -c config.yaml --eval-only
```

### 2. Parallel Single Run &mdash; `main_multithreading.py`

Runs multiple methods in parallel with the same seed for quick comparison. Reads all non-method settings from `config.yaml` directly.

```bash
python main_multithreading.py -m standard nrgnn cr_gnn -r 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --methods` | standard cr_gnn nrgnn | Methods to run (max 4) |
| `-r, --run-id` | 1 | Fixed run ID (determines seed) |
| `--no-checkpoint` | off | Disable checkpointing |
| `--eval-only` | off | Evaluate from saved checkpoints |
| `--checkpoint-dir` | `checkpoints` | Checkpoint directory |

**Workers:** 2 parallel workers on GPU, 4 on CPU (auto-detected).

### 3. Programmatic API

```python
import yaml
from util.experiment import run_experiment

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Single experiment run
result = run_experiment(config, run_id=1)

print(f"Accuracy: {result['test_cls']['accuracy']:.4f}")
print(f"F1 Score: {result['test_cls']['f1']:.4f}")

# With checkpointing
result = run_experiment(config, run_id=1, checkpoint_path="best.pt", run_dir="run_1")

# Eval-only from checkpoint
result = run_experiment(config, run_id=1, checkpoint_path="best.pt", eval_only=True)
```

---

## Configuration

All experiment settings are specified in a YAML file. A minimal example:

```yaml
seed: 42
device: cpu          # cpu or cuda
num_runs: 5

dataset:
  name: cora
  root: data

noise:
  type: uniform
  rate: 0.2
  seed: 42

model:
  name: gcn
  hidden_channels: 16
  n_layers: 2
  dropout: 0.5

training:
  method: standard
  lr: 0.001
  weight_decay: 5e-4
  epochs: 200
  patience: 20
  # mode: inductive        # optional: transductive (default) or inductive
  # batch_size: 512        # optional: enables mini-batch processing
  # sampler: neighbor      # optional: neighbor, cluster, graphsaint, random_node
  # sampler_params:        # optional: sampler-specific parameters
  #   num_neighbors: [15, 10]
```

### General Settings

| Parameter | Description |
|-----------|-------------|
| `seed` | Random seed for reproducibility. Controls model init, data splits, and noise generation. Each run uses `seed + run_id` for independent runs. |
| `device` | `cpu` or `cuda`. Falls back to CPU if CUDA is unavailable. |
| `num_runs` | Number of independent runs per configuration (default: 5). |

### Dataset

| Parameter | Description |
|-----------|-------------|
| `name` | Dataset name (see list below) |
| `root` | Directory for dataset storage/download (default: `data`) |

**24 supported datasets:**

| Family | Datasets |
|--------|----------|
| Planetoid | `cora`, `citeseer`, `pubmed` |
| Heterophilous | `amazon-ratings`, `tolokers`, `roman-empire`, `minesweeper`, `questions` |
| Citation | `dblp` |
| Amazon | `amazon-computers`, `amazon-photo` |
| Social | `blogcatalog`, `flickr` |
| GraphLAND (Zenodo) | `hm-categories`, `pokec-regions`, `web-topics`, `tolokers-2`, `city-reviews`, `artnet-exp`, `web-fraud` |
| GNN Benchmark | `pattern`, `cluster` |
| LRGB | `pascalvoc-sp`, `coco-sp` |

Datasets are automatically downloaded on first use. GraphLAND datasets are fetched from Zenodo. LRGB datasets are merged into a single graph with global node masks.

**Train/val/test splits:** If the dataset provides predefined splits, those are used. Otherwise, random splits are generated (80/10/10 by default) with a deterministic seed.

### Noise

Label noise is injected into **training and validation labels**. Test labels remain clean (ground truth for final evaluation). This reflects real-world conditions where noisy annotations affect all labeled data, including the validation set used for early stopping.

Train and validation noise is applied independently (different random seed) so that the specific corrupted nodes differ, but the noise type and rate are identical.

| Parameter | Description |
|-----------|-------------|
| `type` | Noise type (see table below) |
| `rate` | Fraction of labels corrupted (0.0 to 1.0), applied to both train and val |
| `seed` | Seed for noise generation (combined with `run_id` for per-run variation) |

**10 noise types:**

| Type | Mechanism |
|------|-----------|
| `clean` | No corruption. Identity transition matrix. |
| `uniform` | Transition matrix: P[i,i] = 1 - rate, P[i,j] = rate / (C-1) for i != j. Each label resampled from its row. |
| `uniform_simple` | Per-node coin flip with probability `rate` to a uniformly random class. |
| `pair` | Circular chain: class i flips to class (i-1) mod C with probability `rate`. |
| `random` | Random transition matrix per class, seeded by noise seed. |
| `random_pair` | Each class flips to one randomly chosen other class with probability `rate`. |
| `flip` | Sequential circular: 0 &rarr; 1 &rarr; 2 &rarr; ... &rarr; C-1 &rarr; 0 with probability `rate`. |
| `uniform_mix` | P = (1 - rate) * I + rate / C. Uniform confusion matrix. |
| `deterministic` | Exactly floor(rate * num_train) nodes corrupted (fixed count, not stochastic). |
| `instance` | Instance-dependent: per-node flip rate from truncated normal, transitions depend on node features via learned projection. |

### Model Backbone

| Parameter | Description | Used by |
|-----------|-------------|---------|
| `name` | Architecture: `gcn`, `gin`, `gat`, `gatv2`, `gps` | All |
| `hidden_channels` | Hidden representation size | All |
| `n_layers` | Number of GNN layers | All |
| `dropout` | Dropout probability | All |
| `self_loop` | Add self-loops to nodes | GCN |
| `mlp_layers` | MLP layers inside GIN convolutions | GIN |
| `train_eps` | Learnable epsilon in GIN | GIN |
| `heads` | Number of attention heads | GAT, GATv2, GPS |
| `use_pe` | Use positional encoding | GPS |
| `pe_dim` | Positional encoding dimension | GPS |

**Edge weight support:** GCN, GAT, and GATv2 support edge weights (used by methods like NRGNN, RTGNN, GNNGuard that modify graph structure). GIN silently ignores edge weights. GPS does not use edge weights.

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | — | Robustness method to use (see [Robustness Methods](#robustness-methods)) |
| `lr` | 0.001 | Learning rate |
| `weight_decay` | 5e-4 | L2 regularization |
| `epochs` | 200 | Maximum training epochs |
| `patience` | 20 | Early stopping patience (epochs without val loss improvement) |
| `oversmoothing_every` | 20 | Compute oversmoothing metrics every N epochs. Set to 1 for per-epoch tracking. |
| `checkpoint_every_epoch` | true | Save a `.pt` checkpoint every epoch. Set to false to only keep the best-epoch checkpoint. |
| `mode` | transductive | Learning mode: `transductive` (shared graph) or `inductive` (disjoint subgraphs). See [Transductive and Inductive Learning](#transductive-and-inductive-learning). |
| `batch_size` | — | Mini-batch size. When set, enables batched training/evaluation. See [Batch Processing](#batch-processing). |
| `sampler` | neighbor | Graph sampler type: `neighbor`, `cluster`, `graphsaint`, or `random_node`. Only used when `batch_size` is set. |
| `sampler_params` | — | Sampler-specific parameters (dict). See [Batch Processing](#batch-processing). |

**Early stopping:** Training halts when validation loss does not improve for `patience` consecutive epochs. The model is restored to the best-epoch weights before evaluation.

### Parameter Sweeps

Use the `£[...]` syntax in any YAML field to create a parameter sweep. `main.py` expands all sweeps into a Cartesian product and runs each configuration independently.

```yaml
dataset:
  name: £[cora, citeseer, pubmed]

noise:
  type: £[uniform, pair, instance]

training:
  method: £[standard, nrgnn, cr_gnn]

# Produces 3 x 3 x 3 = 27 configurations, each run num_runs times
```

Sweep values are parsed with YAML types: numbers become int/float, `true`/`false` become booleans, `null` becomes None.

---

## Robustness Methods

### Method Overview

| # | Method | Strategy | Batched Training | Key Technique |
|---|--------|----------|:----------------:|---------------|
| 1 | `standard` | Baseline | Yes | Standard cross-entropy, no robustness mechanism |
| 2 | `positive_eigenvalues` | Weight constraint | Yes | SVD constraint on final layer: enforces positive singular values after each step |
| 3 | `gcod` | Outlier discounting | Yes | Dual optimizers (model + uncertainty), soft-label CE + KL divergence |
| 4 | `nrgnn` | Edge augmentation | — | 3 models (main + predictor + edge estimator), potential/confident edge augmentation |
| 5 | `pi_gnn` | Mutual information | — | Dual models, context-aware MI regularization + link reconstruction |
| 6 | `cr_gnn` | Contrastive learning | — | Double graph augmentation (edge dropout + feature masking), contrastive + consistency loss |
| 7 | `community_defense` | Community regularization | Yes | Louvain/spectral community detection, community-preserving auxiliary loss |
| 8 | `rtgnn` | Dual-branch co-teaching | — | Two GNN branches, structure estimation, adaptive co-teaching + pseudo-labeling |
| 9 | `graphcleaner` | Noise detection | Yes | Two-phase: detect noisy nodes via binary classifier, then train on cleaned subset |
| 10 | `unionnet` | KNN aggregation | Yes | Support-set label aggregation via k-nearest neighbors, reweighted loss |
| 11 | `gnn_cleaner` | Label propagation | — | Dual optimizers, expanding clean-sample identification via label propagation |
| 12 | `erase` | Self-supervised | — | MCR2 loss for embedding learning, LogisticRegression probe for inference |
| 13 | `gnnguard` | Edge attention | — | Cosine-similarity edge reweighting with learnable attention |

**Batched Training column:** "Yes" means the method supports mini-batch training when `training.batch_size` is set. "—" means the method requires full-graph training (its algorithm is inherently full-graph). All 13 methods support batched **evaluation** and **inference** regardless of this column.

### Method-Specific Parameters

Each method accepts optional parameters under a `<method>_params` key in the config.

#### Standard
No additional parameters. Uses default cross-entropy training.

#### Positive Eigenvalues
```yaml
positive_eigenvalues_params:
  batch_size: 32       # Mini-batch size for NeighborLoader
```

#### GCOD
```yaml
gcod_params:
  batch_size: 32       # Mini-batch size for NeighborLoader
  uncertainty_lr: 1.0  # Learning rate for uncertainty parameters
```

#### NRGNN
```yaml
nrgnn_params:
  edge_hidden: 16   # Hidden dim in edge predictor
  n_p: 10           # Max potential edges per node from most similar nodes
  p_u: 0.7          # Confidence threshold for unlabeled node selection
  alpha: 0.05       # Edge reconstruction loss weight
  beta: 1.0         # Consistency loss weight (main model vs predictor on confident nodes)
  t_small: 0.1      # Connection threshold for edge predictor
  n_n: 50           # Negative samples for edge reconstruction
```

#### PI-GNN
```yaml
pi_gnn_params:
  start_epoch: 200   # Epoch to begin MI regularization
  miself: false       # Use self mutual information in contextual loss
  norm: null          # Normalization factor in loss computation
  vanilla: false      # Disable context-aware regularization
```

#### CR-GNN
```yaml
cr_gnn_params:
  T: 2          # Temperature for similarity matrices
  tau: 0.6      # Temperature for contrastive loss
  p: 0.9        # Filtering threshold for low similarity values
  alpha: 0.2    # Contrastive loss weight
  beta: 0.9     # Cross-space consistency weight
  pr: 0.3       # Edge dropout / feature masking probability
```

#### Community Defense
```yaml
community_defense_params:
  community_method: louvain  # 'louvain' or 'spectral'
  num_communities: null      # Auto-detect if null
  lambda_comm: 1.0           # Community loss weight
  pos_weight: 2.0            # Weight for same-community node pairs
  neg_weight: 2.0            # Weight for cross-community node pairs
  margin: 1.5                # Min embedding distance for negative pairs
  num_neg_samples: 3         # Negative samples per node
```

#### RTGNN
```yaml
rtgnn_params:
  edge_hidden: 16    # Hidden dim in edge predictor
  co_lambda: 0.1     # Intra-view regularization weight
  alpha: 0.3         # Reconstruction loss weight
  th: 0.8            # Pseudo-label confidence threshold
  K: 50              # KNN candidates for edge augmentation
  tau: 0.05          # Min similarity for edge filtering
  n_neg: 100         # Negative samples per node for reconstruction
```

#### GraphCleaner
```yaml
graphcleaner_params:
  k: 5                      # Neighbourhood hops in mislabel detector
  sample_rate: 0.5           # Fraction of nodes for synthetic mislabel generation
  max_iter_classifier: 5000  # Max iterations for binary classifier training
  held_split: valid          # Split for noise transition matrix estimation
```

#### UnionNET
```yaml
unionnet_params:
  k: 10          # KNN neighbours for support set
  alpha: 0.5     # Reweighted loss weight
  beta: 1        # KL divergence regularization weight
  feat_norm: true  # Normalize node features
```

#### GNN Cleaner
```yaml
gnn_cleaner_params:
  label_propagation_iterations: 50  # LP iterations for label correction
  similarity_epsilon: 1e-8          # Numerical stability constant
```

#### ERASE
```yaml
erase_params:
  n_embedding: 512      # Embedding dimension
  n_heads: 8            # Attention heads in first GAT layer
  use_layer_norm: false  # Apply layer normalization
  use_residual: false    # Residual connections
  use_residual_linear: false  # Learnable residual projection
  gam1: 1.0             # MCR2 compression coefficient
  gam2: 2.0             # MCR2 discrimination coefficient
  eps: 0.05             # Covariance stabilization constant
  alpha: 0.6            # Label propagation neighbour influence
  beta: 0.6             # Pseudo-label / denoised-label blending weight
  T: 3                  # Label propagation depth (steps)
```

#### GNNGuard
```yaml
gnnguard_params:
  P0: 0.5          # Edge-pruning similarity threshold
  K: 2             # Number of GNN layers
  D2: 16           # Hidden embedding dimension
  attention: true  # Enable attention-based edge reweighting
```

---

## Batch Processing

When graphs are too large to fit in GPU memory (or for any dataset where full-batch forward passes are impractical), set `training.batch_size` to enable mini-batch processing. This activates batched **training** (for methods that support it), **evaluation**, and **inference** across the entire pipeline.

### Enabling Batched Mode

```yaml
training:
  method: standard
  batch_size: 512           # activate mini-batch mode
  sampler: neighbor         # graph sampler (default)
  sampler_params:
    num_neighbors: [15, 10] # 2-hop sampling
```

When `batch_size` is set:
- **Methods that support batched training** (6 of 13 — see [Method Overview](#method-overview)) will train on mini-batches.
- **Methods that do NOT support batched training** (7 of 13) will train full-batch but still use batched evaluation and inference. A warning is printed: `[WARNING] <Method> does not support batched training. Using full-batch training with batched evaluation.`
- **All 13 methods** use batched inference for predictions and embeddings when `batch_size` is set, regardless of whether they support batched training.

### Supported Samplers

Four graph sampling strategies are available via the `training.sampler` parameter:

| Sampler | Description | Seed Nodes | Best For |
|---------|-------------|------------|----------|
| `neighbor` | **NeighborLoader** — samples a fixed number of neighbours per hop for each target node. First `batch_size` nodes in each batch are seeds (targets); remaining nodes are their expanded neighbourhood. | First `batch_size` | General-purpose; preserves local neighbourhood structure |
| `cluster` | **ClusterLoader** — METIS graph partitioning into `num_parts` clusters. Each batch is one cluster; all nodes in the cluster are targets. | All nodes | Large graphs with community structure |
| `graphsaint` | **GraphSAINTRandomWalkSampler** — random-walk-based subgraph sampling for training. Evaluation/inference uses NeighborLoader for full-coverage scatter-back. | All nodes (train), first `batch_size` (eval) | Large graphs; unbiased gradient estimates |
| `random_node` | **RandomNodeLoader** — simple random node partitioning. Graph is divided into `num_parts` approximately equal parts. | All nodes | Fast; no pre-processing overhead |

### Sampler-Specific Parameters

```yaml
# NeighborLoader (default)
training:
  sampler: neighbor
  sampler_params:
    num_neighbors: [15, 10]   # neighbours per hop (default: [15, 10])

# ClusterLoader
training:
  sampler: cluster
  sampler_params:
    num_parts: 100            # number of METIS partitions (default: 100)

# GraphSAINT
training:
  sampler: graphsaint
  sampler_params:
    walk_length: 4            # random walk length (default: 4)
    num_steps: 30             # batches per epoch (default: 30)

# RandomNodeLoader
training:
  sampler: random_node
  # num_parts is automatically derived from batch_size and graph size
```

### How It Works

1. **Training:** Each epoch iterates over the train loader. For each batch, the model does a forward pass, computes loss on seed/target nodes, and updates weights.
2. **Validation:** Val loss is computed by iterating the val loader and averaging per-batch CE loss on validation nodes.
3. **Predictions:** The inference loader iterates over **all** nodes. Predictions are scattered back to a full-graph tensor via the batch's `n_id` (global node ID) mapping.
4. **Embeddings:** Same scatter-back mechanism as predictions, used for oversmoothing metrics.

### Example: Large Graph Configuration

```yaml
dataset:
  name: pokec-regions

model:
  name: gcn
  hidden_channels: 64
  n_layers: 2

training:
  method: standard
  batch_size: 1024
  sampler: neighbor
  sampler_params:
    num_neighbors: [25, 15]
  epochs: 100
  patience: 20
```

---

## Transductive and Inductive Learning

The framework supports both **transductive** and **inductive** learning modes, controlled by the `training.mode` flag.

### Transductive Mode (Default)

```yaml
training:
  mode: transductive   # default — can be omitted
```

All nodes and edges are visible during training; only labels are masked by split:
- The full graph (all nodes, all edges) is loaded into memory.
- `train_mask`, `val_mask`, `test_mask` control which labels are used for training, validation, and testing.
- The GNN performs message passing over the **entire graph** (including test nodes as neighbours), but loss is computed only on `train_mask` nodes.
- Both training and validation labels are noisy (same noise type/rate). Test labels are never corrupted.

This is the standard setup for transductive node classification benchmarks (Cora, CiteSeer, Pubmed, etc.).

### Inductive Mode

```yaml
training:
  mode: inductive
```

The graph is partitioned into **three disjoint subgraphs** for train, validation, and test:
- Edges that cross partition boundaries are removed so that information cannot leak between splits during message passing.
- Each subgraph has its own node indices (remapped to 0 … N_partition - 1) and contains only nodes from its split.
- Training is performed entirely on the train subgraph.
- Validation loss and metrics are computed on the val subgraph via a separate forward pass.
- Final evaluation runs on the test subgraph independently.
- Global predictions are reconstructed by mapping local predictions back to original node IDs.

This simulates a realistic deployment scenario where the model must generalize to previously unseen graph regions.

### Combining Modes with Batching

Both modes can be combined with batched processing:

```yaml
# Transductive + batched
training:
  mode: transductive
  batch_size: 512
  sampler: neighbor

# Inductive + batched
training:
  mode: inductive
  batch_size: 512
  sampler: neighbor
```

In inductive + batched mode, separate data loaders are created for each subgraph (train and val), and batched evaluation/inference operates within each subgraph independently.

---

## Evaluation Metrics

### Classification Metrics

Computed on node-level predictions using scikit-learn with **macro averaging**:

| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of correctly classified nodes |
| Precision | Per-class precision, macro-averaged |
| Recall | Per-class recall, macro-averaged |
| F1 | Per-class F1 score, macro-averaged |

Reported separately for **train**, **val**, and **test** splits.

### Oversmoothing Metrics

Computed on hidden-layer embeddings (not output logits) at intervals controlled by `oversmoothing_every`. These measure how distinguishable node representations remain as information propagates through GNN layers.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **NumRank** | &#8741;X&#8741;<sub>F</sub>&sup2; / &#8741;X&#8741;<sub>2</sub>&sup2; | Numerical rank via Frobenius/spectral norms. Higher = more diverse representations. |
| **Erank** | exp(-&sum; p<sub>i</sub> log p<sub>i</sub>) | Effective rank via entropy of singular value distribution. Higher = more diverse. |
| **EDir** | &sum;<sub>edges</sub> &#8741;X[u] - X[v]&#8741;&sup2; / 2 | Dirichlet energy. Higher = more variation between neighbours. |
| **EDir_traditional** | Normalized Dirichlet energy | Dirichlet energy normalized by dominant eigenvector of the graph Laplacian. |
| **EProj** | &#8741;X - uu<sup>T</sup>X&#8741;<sub>F</sub>&sup2; | Projection energy onto rank-1 approximation. Higher = more energy outside dominant direction. |
| **MAD** | mean(1 - cos(X[u], X[v])) | Mean angular distance between connected nodes. Higher = less oversmoothing. |

Reported for **train**, **val**, and **test** splits at each checkpoint epoch.

### Noise-Split Analysis

For both training and validation nodes, the framework automatically identifies which labels were corrupted by noise and reports separate metrics:

| Split | Evaluated Against | Description |
|-------|-------------------|-------------|
| `train_only_clean` | Original clean labels | Performance on training nodes that were NOT corrupted |
| `train_only_mislabelled_factual` | Noisy labels | Performance on corrupted training nodes, measured against the noisy labels they were trained on |
| `train_only_mislabelled_corrected` | Original clean labels | Performance on corrupted training nodes, measured against their true labels (did the model learn the correct label despite noise?) |
| `val_only_clean` | Original clean labels | Performance on validation nodes that were NOT corrupted |
| `val_only_mislabelled_factual` | Noisy labels | Performance on corrupted validation nodes, measured against their noisy labels |
| `val_only_mislabelled_corrected` | Original clean labels | Performance on corrupted validation nodes, measured against their true labels |

This reveals whether a method memorises noisy labels or learns to correct them. The val noise-split metrics are particularly informative: since val labels are noisy too, methods that can recover true labels on the validation set demonstrate genuine robustness.

### Compute Metrics

| Metric | Description |
|--------|-------------|
| `flops_inference` | FLOPs for a single inference forward pass (via `torch.profiler`) |
| `flops_training_total` | Estimated total training FLOPs (per-step FLOPs &times; epochs) |
| `time_training_total` | Wall-clock training time (seconds) |
| `time_inference` | Wall-clock inference time (seconds) |

Carbon emissions are also tracked via [CodeCarbon](https://codecarbon.io/) when available, saved to `emissions.csv` in the experiment directory.

---

## Output and Results

### Directory Layout

```
results/
  <hash>_<dataset>_<method>_<seed>_noise-<type>-<rate>/
    experiment.json           # Aggregated results across all runs
    emissions.csv             # Carbon emissions (if codecarbon available)
    best_run_1.pt             # Best-epoch checkpoint for run 1
    best_run_2.pt             # Best-epoch checkpoint for run 2
    run_1/
      training_log.json       # Full per-epoch log for run 1
      epoch_000_valloss_1.2345.pt  # Per-epoch checkpoints
      epoch_001_valloss_1.1234.pt
      training_plots.png      # Training curves
      oversmoothing_plots.png # Oversmoothing evolution
    run_2/
      ...
```

### Per-Run Training Log (`training_log.json`)

Contains the complete training history:

```json
{
  "run_id": 1,
  "config": { "..." },
  "training_params": {
    "method": "standard",
    "lr": 0.001,
    "epochs": 200,
    "patience": 20,
    "oversmoothing_every": 20
  },
  "duration_seconds": 12.34,
  "stopped_at_epoch": 45,
  "best_epoch": 32,
  "best_val_loss": 0.5678,
  "best_checkpoint": "epoch_032_valloss_0.5678.pt",
  "epoch_log": [
    {
      "epoch": 0,
      "train_loss": 1.50,
      "val_loss": 1.40,
      "train_acc": 0.45,
      "val_acc": 0.50,
      "train_f1": 0.40,
      "val_f1": 0.45,
      "oversmoothing": { "NumRank": 45.3, "EDir": 0.123, "MAD": 0.567, "..." }
    }
  ],
  "final_result": {
    "test_cls": { "accuracy": 0.82, "f1": 0.80, "precision": 0.81, "recall": 0.79 },
    "train_cls": { "..." },
    "val_cls": { "..." },
    "train_only_clean_cls": { "..." },
    "train_only_mislabelled_factual_cls": { "..." },
    "train_only_mislabelled_corrected_cls": { "..." },
    "test_oversmoothing": { "NumRank": 45.0, "Erank": 42.0, "EDir": 0.12, "..." },
    "compute_info": {
      "flops_inference": 1234567,
      "flops_training_total": 12345670,
      "time_training_total": 12.34,
      "time_inference": 0.56
    }
  }
}
```

### Aggregated Results (`experiment.json`)

Mean/std computed across all runs:

```json
{
  "config": { "..." },
  "classification": {
    "test":  { "accuracy": [0.82, 0.81, 0.83], "f1": [...], "precision": [...], "recall": [...] },
    "train": { "..." },
    "val":   { "..." },
    "train_only_clean": { "..." },
    "train_only_mislabelled_factual": { "..." },
    "train_only_mislabelled_corrected": { "..." },
    "val_only_clean": { "..." },
    "val_only_mislabelled_factual": { "..." },
    "val_only_mislabelled_corrected": { "..." }
  },
  "oversmoothing": {
    "test":  { "NumRank": [45.0, 46.0, ...], "Erank": [...], "EDir": [...], "..." },
    "train": { "..." },
    "val":   { "..." }
  },
  "compute": {
    "flops_inference": [...],
    "flops_training_total": [...],
    "time_training_total": [...],
    "time_inference": [...]
  }
}
```

Values are lists (one entry per run), allowing downstream computation of mean, std, confidence intervals, etc.

---

## Adding New Methods

Methods use a two-layer registration pattern: a **trainer** (thin wrapper) and a **helper** (training logic).

### Step 1: Create the helper (`methods/my_method_helper.py`)

```python
import torch
import torch.nn.functional as F
import torch.optim as optim

from methods.base_helper import MethodHelper
from methods.registry import register_helper


@register_helper('my_method')
class MyMethodHelper(MethodHelper):

    def setup(self, backbone_model, data, config, device, init_data):
        optimizer = optim.Adam(
            backbone_model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training'].get('weight_decay', 5e-4),
        )
        return {
            'models': [backbone_model],
            'optimizers': [optimizer],
            'model': backbone_model,
            'optimizer': optimizer,
        }

    def train_step(self, state, data, epoch):
        model = state['model']
        optimizer = state['optimizer']

        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        return {'train_loss': loss.item()}

    def compute_val_loss(self, state, data):
        model = state['model']
        model.eval()
        with torch.no_grad():
            out = model(data)
            return F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item()

    def get_predictions(self, state, data):
        model = state['model']
        model.eval()
        with torch.no_grad():
            return model(data).argmax(dim=1)

    def get_embeddings(self, state, data):
        model = state['model']
        model.eval()
        with torch.no_grad():
            return model.get_embeddings(data)

    # Optional: enable mini-batch training
    def supports_batched_training(self):
        return True  # default is False
```

If `supports_batched_training()` returns `True`, the default mini-batch training implementation in the base class iterates the train loader with CE loss. Override `train_step_batched()` for custom batched logic. Even methods that return `False` get batched evaluation/inference automatically.

### Step 2: Create the trainer (`model/methods/MyMethod.py`)

```python
from model.base import BaseTrainer
from model.registry import register


@register('my_method')
class MyMethodTrainer(BaseTrainer):
    def train(self):
        from methods.registry import get_helper
        from training.training_loop import TrainingLoop

        d = self.init_data
        helper = get_helper('my_method')
        loop = TrainingLoop(helper, log_epoch_fn=self.log_epoch)
        result = loop.run(
            d['backbone_model'], d['data_for_training'],
            self.config, d['device'], d,
        )
        self._loop = loop
        return result
```

### Step 3: Use it

Set `training.method: my_method` in `config.yaml` and run. No other wiring needed -- both registries auto-discover files in their directories.

---

## Testing

```bash
# Run all tests (99 tests across 13 methods)
./test.sh

# Run a single method
./test.sh -k standard

# Run multiple methods
./test.sh -k "nrgnn or rtgnn"

# Stop on first failure
./test.sh -x

# Verbose tracebacks
./test.sh --tb=long
```

Tests use 5 epochs on Cora with reduced hyperparameters for speed. The test suite includes:

- **Smoke tests** (`test_smoke.py`): End-to-end training + evaluation for all 13 methods
- **Evaluation tests** (`test_evaluation.py`): Unit tests for classification and oversmoothing metrics
- **Checkpoint tests** (`test_checkpoint_consistency.py`): Verify checkpoint save/load round-trip produces identical predictions

---

## Project Structure

```
.
+-- config.yaml                     # Experiment configuration (sweeps, hyperparams)
+-- main.py                         # Production sweep runner (multi-run, incremental)
+-- main_multithreading.py          # Parallel single-run comparison
+-- sweep_utils.py                  # YAML sweep expansion, config hashing, run detection
|
+-- methods/                        # Method helpers (training logic for each method)
|   +-- base_helper.py              # MethodHelper ABC (setup, train_step, val, predict, embed)
|   +-- registry.py                 # @register_helper decorator + auto-discovery
|   +-- standard_helper.py
|   +-- positive_eigenvalues_helper.py
|   +-- gcod_helper.py
|   +-- nrgnn_helper.py
|   +-- pi_gnn_helper.py
|   +-- cr_gnn_helper.py
|   +-- community_defense_helper.py
|   +-- rtgnn_helper.py
|   +-- graphcleaner_helper.py
|   +-- unionnet_helper.py
|   +-- gnn_cleaner_helper.py
|   +-- erase_helper.py
|   +-- gnnguard_helper.py
|
+-- training/                       # Shared training infrastructure
|   +-- training_loop.py            # Unified training loop (epochs, early stopping, metrics)
|   +-- early_stopping.py           # Early stopping tracker
|
+-- evaluation/                     # Metrics (re-exports from model.evaluation)
|   +-- metrics.py
|
+-- model/
|   +-- base.py                     # BaseTrainer ABC (checkpoint, logging, profiling, plotting)
|   +-- registry.py                 # @register decorator + trainer auto-discovery
|   +-- gnns.py                     # GCN, GIN, GAT, GATv2, GPS backbones
|   +-- evaluation.py               # ClassificationMetrics, OversmoothingMetrics
|   +-- methods/                    # Trainer wrappers + algorithm-specific model components
|       +-- Standard.py
|       +-- GCOD_loss.py            # + GraphCentroidOutlierDiscounting loss
|       +-- NRGNN.py                # + NRGNN algorithm class
|       +-- RTGNN.py                # + DualBranchGNNModel, GraphStructureEstimator, etc.
|       +-- CR_GNN.py               # + ContrastiveProjectionHead, loss functions
|       +-- PI_GNN.py               # + GraphLinkDecoder, PiGnnModel
|       +-- ERASE.py                # + MCR2 loss, EnhancedGNNWrapper, AdjacencyProcessor
|       +-- GNNGuard.py             # + GNNGuardModel
|       +-- GraphCleaner.py
|       +-- UnionNET.py
|       +-- GNN_Cleaner.py
|       +-- Positive_Eigenvalues.py
|       +-- CommunityDefense.py
|
+-- util/
|   +-- experiment.py               # initialize_experiment(), run_experiment()
|   +-- data.py                     # Dataset loading, splits, preprocessing
|   +-- noise.py                    # Noise injection (10 types)
|   +-- graph_sampling.py           # Graph sampler factory (NeighborLoader, Cluster, GraphSAINT, RandomNode)
|   +-- inductive.py                # Inductive graph partitioning (disjoint subgraphs)
|   +-- profiling.py                # FLOPs profiling (inference + training)
|   +-- plot.py                     # Training curves, oversmoothing, noise-split plots
|   +-- seed.py                     # Deterministic seeding (torch, numpy, python, cuDNN)
|   +-- cli.py                      # CLI helpers and formatting
|
+-- tests/
|   +-- test_smoke.py               # Smoke tests for all 13 methods
|   +-- test_evaluation.py          # Unit tests for metrics
|   +-- test_checkpoint_consistency.py  # Checkpoint round-trip tests
|
+-- results/                        # Generated experiment outputs
+-- images/                         # Figures for README
+-- requirements.txt
+-- setup.sh                        # GPU installation script
+-- test.sh                         # Test runner
+-- LICENSE
```

---

## Experiments

### Cora

Results on the Cora dataset under uniform and instance noise:

<p align="center">
  <img src="images/Symmetric.png" alt="Uniform Noise" width="49%">
  <img src="images/Asymmetric.png" alt="Instance Noise" width="49%">
</p>

<i>Results on the Cora dataset with (left) Uniform noise and (right) Instance noise.</i>

### CiteSeer and Pubmed

<p align="center">
  <img src="images/CiteSeer_and_Pubmed.png" alt="Uniform noise with 0.6 noise ratio on Cora dataset" width="100%">
</p>

<i>Performance of models on CiteSeer and Pubmed datasets with Uniform noise (ratio 0.6).</i>

### Ablation Studies

#### Type of Noise

<p align="center">
  <img src="images/Different_noise_type.png" alt="Uniform noise with 0.6 noise ratio on Cora dataset" width="100%">
</p>

<i>Performance of GraphCleaner with different noise types (Cora dataset, noise ratio 0.6).</i>

#### Backbone Architectures

<p align="center">
  <img src="images/Different_Backbone.png" alt="Uniform noise with 0.6 noise ratio on Cora dataset" width="70%">
</p>

<i>Results on the Cora dataset with Uniform noise (noise ratio 0.6).</i>

#### Oversmoothing Metrics

<p align="center">
  <img src="images/Loss.png" alt="Loss" width="32%">
  <img src="images/Accuracy.png" alt="Accuracy" width="31.6%">
  <img src="images/F1.png" alt="F1-Score" width="31.2%">
</p>

<i>Evolution of Loss (left), Accuracy (center), and F1-Score (right) during training and validation.</i>

<p align="center">
  <img src="images/E_dir.png" alt="E_dir" width="32%">
  <img src="images/Mad.png" alt="MAD" width="31.5%">
  <img src="images/NumRank.png" alt="Numrank" width="31.2%">
</p>

<i>Evolution of E<sup>dir</sup> (left), MAD (center), and NumRank (right) during training and validation.</i>

---

## Author

Farooq Ahmad Wani, Antonio Purificato, Andrea Giussepe Di Francesco, Fabrizio Silvestri, Michael Corelli, Maria Sofia Bucarelli, Oleksandr Pryymak
