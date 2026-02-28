# Benchmarking Robustness Strategies for Graph Neural Networks under Noise

This study contributes to the analysis of the robustness of Graph Neural Networks by developing a benchmarking framework that systematically evaluates methods designed to address graph noise. The framework reproduces selected robustness strategies under standardized conditions across commonly used datasets, enabling fair and reproducible comparisons. Although numerous robustness approaches have been proposed for GNNs, their evaluations are often inconsistent. This work analyzes the performance of existing robustness strategies under uniform experimental setups and across different types of graph noise, with the goal of identifying the methods that generalize most effectively. Several strategies have been introduced in the literature, including noise-resistant training, structure-learning techniques, and specialized frameworks. However, no standard benchmarking framework currently exists to directly compare these methods under consistent protocols.

This work therefore establishes a unified benchmarking effort to identify which strategies perform best under controlled and comparable conditions. To this end, extensive experiments and advanced ablation studies are conducted to evaluate not only standard performance metrics but also oversmoothing measures and alternative experimental setups. Furthermore, the study investigates how different backbone architectures influence model performance. By providing a systematic and in-depth analysis under consistent conditions, this study sheds light on the differences among robustness frameworks and highlights which approaches are most effective in various scenarios.

<div style="display: flex; justify-content: center; gap: 10px; margin-bottom: 20px;">
  <img src="images/diagram.png" alt="Symmetric" width="100%">
</div>

# Experiments

This section presents some of the experiments conducted during the work.

## Cora
The following figures show the results on the Cora dataset under a uniform setup, with different noise levels. Two types of noise are considered: **Uniform** and **Instance**.

<p align="center">
  <img src="images/Symmetric.png" alt="Uniform Noise" width="49%">
  <img src="images/Asymmetric.png" alt="Instance Noise" width="49%">
</p>

<i>Results on the Cora dataset with (left) Uniform noise and (right) Instance noise.</i>

## Citeseer and Pubmed
The following table reports the performance across different datasets:

<p align="center">
  <img src="images/CiteSeer_and_Pubmed.png" alt="Uniform noise with 0.6 noise ratio on Cora dataset" width="100%">
</p>

<i>Performance of models on CiteSeer and Pubmed datasets with Uniform noise (ratio 0.6).</i>

## Ablation study
An ablation study is carried out considering different noise types, backbone architectures and oversmoothing metrics.

### Type of noise
Results obtained using different noise types to evaluate their impact on performance:

<p align="center">
  <img src="images/Different_noise_type.png" alt="Uniform noise with 0.6 noise ratio on Cora dataset" width="100%">
</p>

<i>Performance of GraphCleaner with different noise types (Cora dataset, noise ratio 0.6).</i>

### Different Backbone Architectures
This analysis investigates the role of the backbone architecture and its influence on performance:

<p align="center">
  <img src="images/Different_Backbone.png" alt="Uniform noise with 0.6 noise ratio on Cora dataset" width="70%">
</p>

<i>Results on the Cora dataset with Uniform noise (noise ratio 0.6).</i>

### Oversmoothing metrics
The following plots illustrate the behavior of classic metrics as well as the oversmoothing metrics **MAD**, **$E^{dir}$**, and **Numrank** during training and validation. These are computed on hidden-layer embeddings (not output logits) and are generated automatically for all methods.
These oversmoothing metrics are included in the ablation study to provide insights into the model's behavior beyond standard evaluation measures.

<p align="center">
  <img src="images/Loss.png" alt="Loss" width="32%">
  <img src="images/Accuracy.png" alt="Accuracy" width="31.6%">
  <img src="images/F1.png" alt="F1-Score" width="31.2%">
</p>

<i>Evolution of Loss (left), Accuracy (center), and F1-Score (right) metrics during training and validation.</i>

<p align="center">
  <img src="images/E_dir.png" alt="E_dir" width="32%">
  <img src="images/Mad.png" alt="MAD" width="31.5%">
  <img src="images/NumRank.png" alt="Numrank" width="31.2%">
</p>

<i>Evolution of E<sup>dir</sup> (left), MAD (center), and NumRank (right) metrics during training and validation.</i>

# Configuration

## General
- **seed**: Random seed used to initialize model weights and ensure reproducibility of experiments.
- **device**: Specifies the hardware for training (`cpu` or `cuda`).

## Dataset
- **name**: The name of the dataset to be used (`cora`, `citeseer`, `pubmed`, `amazon-ratings`, `tolokers`, `roman-empire`, `minesweeper`, `questions`, `dblp`, `amazon-computers`, `amazon-photo`, `blogcatalog`, `flickr`, `hm-categories`, `pokec-regions`, `web-topics`, `tolokers-2`, `city-reviews`, `artnet-exp`, `web-fraud`, `pattern`, `cluster`, `pascalvoc-sp`, `coco-sp`).
- **root**: Path where the dataset is stored or will be downloaded.

## Noise
- **type**: Type of label noise injected into the dataset (`clean`, `uniform`, `uniform_simple`, `random`, `pair`, `random_pair`, `flip`, `uniform_mix`, `instance`, `deterministic`).
- **rate**: Proportion of nodes affected by noise.
- **seed**: Random seed used specifically for noise generation.

## Model
- **name**: Backbone architecture (`gcn`, `gin`, `gat`, `gatv2`, `gps`).
- **hidden_channels**: Size of hidden representations in each layer.
- **n_layers**: Number of layers in the model.
- **dropout**: Dropout probability to prevent overfitting.
- **self_loop**: Whether to add self-loops to nodes in the graph.
- **mlp_layers**: Number of layers in an MLP applied on node features.
- **train_eps**: Whether the epsilon parameter in GIN is trainable.
- **heads**: Number of attention heads for GAT, GATv2, or GPS models.
- **use_pe**: Whether to use positional encoding in GPS.
- **pe_dim**: Dimension of the positional encoding features.

## Training
- **method**: Training method (`standard`, `positive_eigenvalues`, `gcod`, `nrgnn`, `pi_gnn`, `cr_gnn`, `community_defense`, `rtgnn`, `graphcleaner`, `unionnet`, `gnn_cleaner`, `erase`, `gnnguard`).
- **lr**: Learning rate for the optimizer.
- **weight_decay**: L2 regularization weight to avoid overfitting.
- **epochs**: Maximum number of training epochs.
- **patience**: Number of epochs without improvement before early stopping.
- **oversmoothing_every**: Compute oversmoothing metrics every N epochs (default: 20). Set to 1 for per-epoch tracking.
- **checkpoint_every_epoch**: Whether to save a `.pt` checkpoint every epoch (default: true). Set to false to only keep the best-epoch checkpoint.

## Framework-Specific Parameters

### Standard Training
- **standard_params**: Empty; uses default training.

### Positive Eigenvalues
- **batch_size**: Number of model weight matrices updated before applying the positive singular value (eigenvalue) constraint.

### GCOD
- **batch_size**: Number of nodes processed at a time when computing the GCOD loss.
- **uncertainty_lr**: Learning rate controlling how fast uncertainty parameters are updated during GCOD loss computation.

### NRGNN
- **edge_hidden**: Dimension of hidden representations in the edge predictor.
- **n_p**: Maximum number of potential edges per node added from the most similar nodes.
- **p_u**: Confidence threshold to select unlabeled nodes for model updates.
- **alpha**: Weight of edge reconstruction loss in the total loss.
- **beta**: Weight of consistency loss aligning the main model with the predictor on confident nodes.
- **t_small**: Threshold above which nodes are considered connected by the edge predictor.
- **n_n**: Number of negative samples used to balance edge reconstruction.

### PI-GNN
- **start_epoch**: Epoch at which joint training with the main model begins.
- **miself**: Whether to use self mutual information in contextual loss computation.
- **norm**: Normalization factor applied in loss computation.
- **vanilla**: Whether to apply contextual regularization based on PI model predictions.

### CR-GNN
- **T**: Temperature for constructing similarity matrices in embedding and prediction spaces.
- **tau**: Temperature for contrastive loss; affects separation of positive and negative pairs.
- **p**: Filtering threshold for low similarity values in prediction space.
- **alpha**: Weight of contrastive loss.
- **beta**: Weight for cross-space consistency between embedding and prediction spaces.
- **pr**: Probability of dropout/masking

### Community Defense
- **community_method**: Algorithm used for community detection (`louvain`, `spectral`).
- **num_communities**: Number of communities; automatically detected if null.
- **lambda_comm**: Weight for the community-preserving auxiliary task.
- **pos_weight**: Weight for positive node pairs within the same community.
- **neg_weight**: Weight for negative node pairs from different communities.
- **margin**: Minimum embedding distance between negative pairs.
- **num_neg_samples**: Number of negative samples per node.

### RTGNN
- **edge_hidden**: Dimension of hidden representations in the edge predictor.
- **co_lambda**: Weight for co-teaching regularization; balances intra-view regularization.
- **alpha**: Weight assigned to the reconstruction loss; preserves graph structure.
- **th**: Confidence threshold for creating pseudo labels.
- **K**: Number of nearest-neighbor candidates to consider for additional edges.
- **tau**: Minimum similarity threshold for filtering unreliable edges.
- **n_neg**: Number of negative samples generated per node for reconstruction loss.

### GraphCleaner
- **k**: Number of neighborhood propagation hops in the mislabel detector.
- **sample_rate**: Fraction of nodes used for synthetic mislabel generation.
- **max_iter_classifier**: Maximum number of iterations to train the binary classifier.
- **held_split**: Split used to estimate the noise transition matrix.

### UnionNET
- **k**: Number of most similar neighbors used to construct the support set for label propagation.
- **alpha**: Weight balancing the reweighted loss and the correction loss.
- **beta**: Weight for the KL-divergence regularization term.
- **feat_norm**: Whether to normalize node features so that each node's feature vector sums to one.

### GNN Cleaner
- **label_propagation_iterations**: Number of label propagation iterations to correct noisy labels.
- **similarity_epsilon**: Small constant added to similarity computation to prevent division by zero.

### ERASE
- **n_embedding**: Dimensionality of latent representations generated by the encoder.
- **n_heads**: Number of attention heads in the first GAT layer.
- **use_layer_norm**: Whether to apply layer normalization.
- **use_residual**: Whether to add residual connections between input and output.
- **use_residual_linear**: Whether to linearly project residual connections to match dimensions.
- **gam1**: Coefficient in the coding rate controlling compression and discrimination.
- **gam2**: Coefficient in the coding rate controlling compression and discrimination.
- **eps**: Small constant added to stabilize covariance computation.
- **alpha**: Coefficient for label propagation; influence of neighbors' labels.
- **beta**: Weight combining pseudo-labels with denoised labels.
- **T**: Propagation depth; number of steps labels are propagated.

### GNNGuard
- **P0**: Edge-pruning threshold.
- **K**: Number of GNN layers.
- **D2**: Embedding dimension in the hidden layer.
- **attention**: Whether to use attention mechanism on edges to weight neighbors differently.


## How to run the code

This repository allows you to run experiments in three different modes:
1. **Single experiment run** -- execute a single experiment via `run_experiment()`.
2. **Automatic benchmarking** -- multi-run sweeps with mean/std, incremental execution, and checkpointing.
3. **Multithreading single run** -- test multiple methods in parallel using the same seed.

Before running any experiment:

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Set the desired parameters and hyperparameters in the **`config.yaml`** file

### Single experiment run

```python
import yaml
from util.experiment import run_experiment

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Run a single experiment (any method -- set in config.yaml under training.method)
result = run_experiment(config, run_id=1)

# Print classification results
print("Single run results:")
print(f"Accuracy: {result['test_cls']['accuracy']:.4f}")
print(f"F1 Score: {result['test_cls']['f1']:.4f}")
print(f"Precision: {result['test_cls']['precision']:.4f}")
print(f"Recall: {result['test_cls']['recall']:.4f}")

# Print oversmoothing metrics (computed on hidden embeddings)
os_results = result['test_oversmoothing']
print("Oversmoothing metrics:")
for key in ('NumRank', 'Erank', 'EDir', 'EDir_traditional', 'EProj', 'MAD'):
    print(f"  {key}: {os_results[key]:.4f}")

# Print compute cost
ci = result['compute_info']
print(f"Training: {ci['time_training_total']:.2f}s, {ci['flops_training_total']:.0f} FLOPs")
print(f"Inference: {ci['time_inference']:.2f}s, {ci['flops_inference']:.0f} FLOPs")
```

To save a checkpoint and later re-evaluate without retraining:

```python
# Train and save checkpoint
result = run_experiment(config, run_id=1, checkpoint_path="best.pt", run_dir="run_1")

# Later: evaluate from checkpoint (no training)
result = run_experiment(config, run_id=1, checkpoint_path="best.pt", eval_only=True)
```

### Automatic benchmarking

Run the main script:
```bash
python main.py -c config.yaml
```

The number of runs per experiment is controlled by `num_runs` in `config.yaml` (default: 5) or overridden via CLI:
```bash
python main.py -c config.yaml --num-runs 3
```

**Incremental runs** -- increasing `num_runs` only executes the new runs. Existing results are loaded from each `run_N/training_log.json`, so completed work is never repeated.

**Skip behavior** -- when all requested runs are already complete and `experiment.json` exists, the experiment is skipped entirely.

**Force re-run** -- pass `--force` on the CLI (or set `force: true` in `config.yaml`) to discard previous results and re-execute every run from scratch.

**Eval-only mode** -- `--eval-only` always re-evaluates every run from saved checkpoints. It does not write `training_log.json`, so eval-only results are not detected by incremental runs.

**Example workflow:**
```bash
# Start with 2 runs
python main.py -c config.yaml --num-runs 2

# Later, bump to 5 -- only runs 3-5 execute
python main.py -c config.yaml --num-runs 5

# Running again with 5 skips immediately (all complete)
python main.py -c config.yaml --num-runs 5
```

### Multithreading options for single run (Thread pooling)

Run the multithreading main using command-line arguments:
```bash
# Use default methods (standard, cr_gnn, nrgnn)
python main_multithreading.py

# Specify custom methods (short version: -m)
python main_multithreading.py --methods standard gcod

# Use different run ID for different seed (short version: -r)
python main_multithreading.py -m standard gcod --run-id 5

# Get help (short version: -h)
python main_multithreading.py --help
```
**Limitations:** Maximum 4 methods can be tested simultaneously. Parallel workers: GPU max 2 and CPU max 4 (workers are automatically adjusted based on available resources).

## How to add frameworks

Methods use a self-registration pattern. To add a new robustness method:

1. Create a file in `model/methods/` (e.g., `model/methods/MyMethod.py`)
2. Subclass `BaseTrainer` and implement the `train()` method (the per-epoch training loop)
3. Decorate the class with `@register('my_method')`

```python
from model.base import BaseTrainer
from model.registry import register

@register('my_method')
class MyMethodTrainer(BaseTrainer):
    def train(self):
        d = self.init_data
        model, data = d['backbone_model'], d['data_for_training']
        optimizer = torch.optim.Adam(model.parameters(), lr=d['lr'])

        for epoch in range(d['epochs']):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(data)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Compute validation loss for early stopping
            model.eval()
            with torch.no_grad():
                val_out = model(data)
                val_loss = F.cross_entropy(val_out[data.val_mask], data.y[data.val_mask])

            # log_epoch handles checkpointing, early stopping, noise-split metrics
            self.log_epoch(epoch, float(loss), float(val_loss),
                           train_acc=..., val_acc=...,
                           is_best=(float(val_loss) < self.best_val_loss),
                           train_predictions=out.argmax(dim=1))

        return {'train_oversmoothing': {}, 'val_oversmoothing': {},
                'stopped_at_epoch': epoch}
```

No other wiring is needed -- the registry auto-discovers all files in `model/methods/`. Set `training.method: my_method` in `config.yaml` and run.

## Structure
```
.
+-- config.yaml                  # experiment configuration (sweeps, hyperparams)
+-- main.py                      # production sweep runner (multi-run, incremental)
+-- main_multithreading.py       # parallel single-run comparison
+-- sweep_utils.py               # YAML sweep expansion, config hashing, run detection
|
+-- model/
|   +-- base.py                  # BaseTrainer ABC (train/eval/checkpoint/logging/plotting)
|   +-- registry.py              # @register decorator + auto-discovery
|   +-- gnns.py                  # GCN, GIN, GAT, GATv2, GPS backbones
|   +-- evaluation.py            # ClassificationMetrics, OversmoothingMetrics
|   +-- methods/                 # 13 robustness method trainers
|       +-- Standard.py
|       +-- GCOD_loss.py
|       +-- NRGNN.py
|       +-- RTGNN.py
|       +-- CR_GNN.py
|       +-- PI_GNN.py
|       +-- ERASE.py
|       +-- GNNGuard.py
|       +-- GraphCleaner.py
|       +-- UnionNET.py
|       +-- GNN_Cleaner.py
|       +-- Positive_Eigenvalues.py
|       +-- CommunityDefense.py
|
+-- util/
|   +-- experiment.py            # initialize_experiment(), run_experiment()
|   +-- data.py                  # dataset loading, splits, preprocessing
|   +-- noise.py                 # noise injection (10 types)
|   +-- profiling.py             # FLOPs profiling (inference + training)
|   +-- plot.py                  # training curves, oversmoothing, noise-split plots
|   +-- seed.py                  # deterministic seeding
|   +-- cli.py                   # CLI helpers and formatting
|
+-- tests/
|   +-- test_smoke.py            # smoke tests for all 13 methods
|   +-- test_evaluation.py       # unit tests for metrics
|   +-- test_checkpoint_consistency.py  # checkpoint round-trip tests
|
+-- results/                     # generated experiment outputs (per-run logs, checkpoints, plots)
+-- images/                      # figures for README
+-- requirements.txt
+-- LICENSE
```

## Author
Michael Corelli
