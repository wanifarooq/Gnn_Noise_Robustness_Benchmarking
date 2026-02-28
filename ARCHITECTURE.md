# Architecture Overview

## Entry Points

| File | Purpose |
|------|---------|
| `main.py` | Production sweep runner. Runs configurable repeats per config (default 5), computes mean/std, saves JSON. Supports incremental runs, eval-only, and checkpoint management. |
| `main_multithreading.py` | Quick parallel single-run comparison of methods. Up to 4 CPU / 2 GPU threads. |

Both call `run_experiment()` from `util.experiment`.

### CLI Flags (`main.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--config`, `-c` | `config.yaml` | Path to YAML config file |
| `--num-runs` | from config or 5 | Number of runs per sweep config |
| `--eval-only` | off | Skip training, evaluate from saved checkpoints |
| `--no-checkpoint` | off | Disable saving model checkpoints |
| `--force` | off | Re-run all experiments, ignoring completed runs |

---

## Project Structure

```
model/
    __init__.py              # package marker
    base.py                  # BaseTrainer ABC — train/eval/checkpoint/logging/plotting
    registry.py              # @register decorator + trainer discovery
    gnns.py                  # GCN, GIN, GAT, GATv2, GPS backbones
    evaluation.py            # ClassificationMetrics, OversmoothingMetrics, noise-split metrics
    methods/
        __init__.py          # package marker
        Standard.py          # + 12 more method files (see Training Methods)

util/
    __init__.py
    experiment.py            # initialize_experiment(), run_experiment()
    data.py                  # load_dataset(), ensure_splits(), prepare_data_for_method()
    noise.py                 # noise_operation() + transition matrices
    seed.py                  # setup_seed_device()
    profiling.py             # get_model(), profile_model_flops(), profile_training_step_flops()
    cli.py                   # print_table(), fmt_mean_std(), parse_arguments()
    plot.py                  # save_training_plots(), save_oversmoothing_plots()

sweep_utils.py               # expand_yaml_sweeps(), get_config_hash(), get_result_filename(),
                             # detect_completed_runs(), json_serializer()
main.py                      # production sweep runner (incremental, eval-only, force)
main_multithreading.py       # parallel single-run runner

tests/
    test_smoke.py            # smoke tests — minimal training loop for all 13 methods
    test_evaluation.py       # unit tests — classification metrics, oversmoothing metrics
    test_checkpoint_consistency.py  # checkpoint save/load round-trip for all methods
```

---

## Pipeline Flow

```
config.yaml
        │
        ▼
  expand_yaml_sweeps()           # sweep_utils.py — Cartesian product over £[...] params
        │
        ▼
  ┌─ detect_completed_runs()     # sweep_utils.py — scan run_N/training_log.json
  │   (skip completed, resume partial, force re-run)
  │
  ▼
  run_experiment(config, run_id) # util/experiment.py
        │
        ├── discover_trainers()  # model/registry.py — auto-import model/methods/*.py
        │
        ├── initialize_experiment()
        │     ├── setup_seed_device()        # util/seed.py
        │     ├── load_dataset()             # util/data.py
        │     ├── ensure_splits()            # util/data.py
        │     ├── noise_operation()          # util/noise.py
        │     ├── prepare_data_for_method()  # util/data.py
        │     └── get_model()               # util/profiling.py
        │
        ├── get_trainer(method)  # model/registry.py — look up BaseTrainer subclass
        │
        ▼
  trainer.run()                  # model/base.py — unified template
        │
        ├── train()                          # subclass: epoch loop + log_epoch()
        ├── restore best checkpoint          # from in-memory snapshot
        ├── profile_flops()                  # inference FLOPs (overrideable)
        ├── profile_training_step()          # training FLOPs (overrideable)
        ├── evaluate()                       # classification + oversmoothing on all splits
        ├── _make_result()                   # assemble standardised result dict
        ├── save_training_log()              # write run_N/training_log.json
        └── save_plots()                     # training curves + oversmoothing curves
        │
        ▼
  Save checkpoint (.pt)          # util/experiment.py — best model state to disk
        │
        ▼
  Return result dict
```

### Eval-only path

When `--eval-only` is passed, `run_experiment()` loads the checkpoint from disk, runs `evaluate()` and `profile_flops()` only (no training), and returns the result. No `training_log.json` is written, so eval-only results are invisible to incremental run detection.

---

## Trainer Infrastructure

### BaseTrainer (`model/base.py`)

Abstract base class for all 13 method trainers. Provides training orchestration, checkpointing, per-epoch logging, FLOPs profiling, plotting, and result assembly.

```python
class BaseTrainer(ABC):
    def __init__(self, init_data: dict, config: dict): ...

    # ── Core contract (subclass implements train) ──
    @abstractmethod
    def train(self) -> dict: ...       # epoch loop; must return {train_oversmoothing,
                                        #   val_oversmoothing, stopped_at_epoch}

    def evaluate(self) -> dict: ...     # backbone_model.forward() for predictions,
                                        # backbone_model.get_embeddings() for oversmoothing

    def run(self) -> dict: ...          # template: train → restore best → profile → evaluate
                                        #   → _make_result → save_training_log → save_plots

    # ── Per-epoch logging ──
    def log_epoch(epoch, train_loss, val_loss, ...): ...  # records metrics + optional checkpoint
    def save_training_log(run_id, config, ...): ...       # writes training_log.json

    # ── Checkpointing (override for multi-model methods) ──
    def get_checkpoint_state(self) -> dict: ...
    def load_checkpoint_state(self, state: dict): ...

    # ── FLOPs profiling (override for multi-model methods) ──
    def profile_flops(self) -> dict: ...            # inference forward pass
    def profile_training_step(self) -> dict: ...    # forward + backward

    # ── Plotting ──
    def save_plots(self): ...           # calls util/plot.py

    # ── Result assembly ──
    def _make_result(self, result_dict, train_oversmoothing, val_oversmoothing,
                     *, reduce=True) -> dict: ...
```

Key instance state:
- `epoch_log: list` — per-epoch metric entries (fed to training_log.json and plots)
- `best_epoch: int` — epoch index of lowest validation loss
- `best_val_loss: float` — the corresponding validation loss
- `_best_checkpoint_state: dict` — in-memory snapshot of model state at best epoch

Key behaviors:
- **`run()`** orchestrates the full pipeline: calls `train()`, restores the best-epoch checkpoint, profiles FLOPs, evaluates on all splits, assembles the result dict, writes `training_log.json`, and saves plots.
- **`log_epoch()`** records one epoch's metrics and optionally saves a per-epoch `.pt` checkpoint to `run_dir`. When `is_best=True`, it deep-copies the model state as the in-memory best checkpoint. Also computes noise-split train metrics (clean vs. mislabelled) when `train_predictions` is provided.
- **`evaluate()`** uses `backbone_model(data).argmax(1)` for classification predictions and `backbone_model.get_embeddings(data)` for oversmoothing metrics. Override in methods that use wrapper models (GNNGuard, ERASE, NRGNN, RTGNN).
- **`_make_result()`** assembles the standardised result dict with keys: `test_cls`, `train_cls`, `val_cls`, `train_only_clean_cls`, `train_only_mislabelled_factual_cls`, `train_only_mislabelled_corrected_cls`, `test_oversmoothing`, `train_oversmoothing`, `val_oversmoothing`, `compute_info`.
- **Checkpoint hooks**: `get_checkpoint_state()` / `load_checkpoint_state()` default to saving/restoring the backbone model. Complex methods (CR_GNN, GNNGuard, NRGNN, RTGNN, ERASE) override these to include auxiliary models (edge predictors, wrapper networks, dual branches).
- **FLOPs hooks**: `profile_flops()` / `profile_training_step()` default to profiling the backbone only. Complex methods override to profile the full inference/training pipeline including auxiliary models.

### Registry (`model/registry.py`)

```python
TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = {}

@register('method_name')   # class decorator — adds to registry
discover_trainers()         # auto-imports model/methods/*.py via pkgutil
get_trainer(name, init_data, config) -> BaseTrainer  # lookup + instantiate
```

Adding a new method: drop a file in `model/methods/` with `@register('name')` on a `BaseTrainer` subclass. No other wiring needed.

---

## Config System

YAML configs use a custom sweep syntax: `£[val1, val2, ...]`

`expand_yaml_sweeps()` produces the Cartesian product of all sweep fields. Example:
```yaml
dataset:
  name: £[cora, pubmed]
training:
  method: £[standard, gcod]
```
Produces 4 configs: (cora,standard), (cora,gcod), (pubmed,standard), (pubmed,gcod).

### Result directory naming

Each sweep config gets a human-readable directory under `results/`:
```
results/{hash8}_{dataset}_{method}_{seed}_noise-{type}-{rate}/
```
The 8-char prefix is a SHA1 hash of all result-affecting config fields (dataset, noise, model, training, method-specific params) via `get_config_hash()`.

### Incremental run detection

`detect_completed_runs()` scans `run_N/training_log.json` files within an experiment directory. A run is considered complete if its `training_log.json` exists, is valid JSON, and contains a `final_result` with a `test_cls` key. Corrupt or incomplete files are silently treated as missing.

| Scenario | Behavior |
|----------|----------|
| No runs exist | Run all, save `experiment.json` |
| 2 of 5 complete | `[RESUME]` — run only missing 3, merge with loaded results |
| All complete, `experiment.json` present | `[SKIP]` — print and continue |
| All complete, `experiment.json` missing | `[REBUILD]` — reconstruct from training logs |
| `--force` | `[FORCE]` — re-run everything from scratch |
| `--eval-only` | `[EVAL-ONLY]` — always re-evaluate (no training_log.json written) |

---

## Backbone Models (`model/gnns.py`)

5 GNN architectures + MLP. All accept a `data` object with `data.x`, `data.edge_index`, and optional `data.edge_weight`.

| Model | Key Params | Notes |
|-------|-----------|-------|
| MLP | num_layers, dropout | Feature-only baseline (no message passing) |
| GCN | n_layers, dropout, self_loop | Standard GCN convolution |
| GIN | n_layers, dropout, mlp_layers, train_eps | GIN with configurable MLP per conv. **Does not support edge_weights** — silently drops them with a warning. |
| GAT | n_layers, dropout, heads | Multi-head attention (v1, concatenates heads) |
| GATv2 | n_layers, dropout, heads | Dynamic attention (head-normalised, per-head dim = hidden/heads). Validates dimension compatibility at init. |
| GPS | n_layers, dropout, heads, use_pe, pe_dim | Graph transformer with optional positional encoding. **Does not support edge_weights** — silently substitutes zero edge attributes. |

All have `initialize()` for weight reset.

### Two-method API: `forward()` and `get_embeddings()`

Every backbone exposes two methods via a shared `_forward_body()`:

- **`forward(data)`** — returns `out_channels`-dim logits (= num_classes) for classification.
- **`get_embeddings(data)`** — returns `hidden_channels`-dim representations for oversmoothing metrics.

The split is implemented as:
- `_forward_body()` runs all layers **except** the final projection. Returns `hidden_channels` dim.
- `forward()` calls `_forward_body()` then applies the final projection layer.
- `get_embeddings()` returns `_forward_body()` directly.

| Model | `_forward_body` runs | `forward` adds | Embedding dim |
|-------|---------------------|----------------|---------------|
| MLP | `layers[:-1]` | `layers[-1]` | hidden_channels |
| GCN (default) | `convs[:-1]` | `convs[-1]` | hidden_channels |
| GCN (output_layer) | `convs[:-1]` + `convs[-1]` (hidden→hidden) | output_norm/act/dropout + output_linear | hidden_channels |
| GIN | `convs[:-1]` | `convs[-1]` | hidden_channels |
| GAT | `convs[:-1]` | `convs[-1]` (1 head) | hidden_channels * heads |
| GATv2 | `convs[:-1]` | `convs[-1]` (1 head) | hidden_channels |
| GPS | `lin_in` + all convs | `lin_out` | hidden_channels |

GAT's embedding dim is `hidden_channels * heads` because intermediate layers concatenate multi-head outputs. GATv2 normalises per-head dim to `hidden_channels // heads`, so concat still yields `hidden_channels`.

### Wrapper models

Four method-specific wrappers also expose `get_embeddings()`:

| Wrapper | File | Strategy |
|---------|------|----------|
| GNNGuardModel | GNNGuard.py | Backbone path: delegates to `backbone.get_embeddings(data)`. Own-GCN path: runs `gcn_layers[:-1]`. |
| EnhancedGNNWrapper | ERASE.py | Delegates to `base_gnn_model.get_embeddings(data)` (skips wrapper enhancements). |
| NRGNN GNN wrapper | NRGNN.py | Delegates to `base_gnn_model.get_embeddings(data)`. |
| DualBranchGNNModel | RTGNN.py | Averages `get_embeddings()` from both branches. |

### Edge weight handling

`_get_edge_attr(data)` (module-level helper) extracts scalar edge weights and unsqueezes `(E,)` → `(E,1)` for GAT/GATv2's `edge_dim=1`. GCN uses `edge_weight` directly via `getattr(data, 'edge_weight', None)`. GIN warns and drops edge weights; GPS silently substitutes zero edge attributes (see table above).

---

## Training Methods (13 total)

Each method lives in `model/methods/<Name>.py` and follows a common pattern:

1. `@register('name')` class extending `BaseTrainer`
2. Implements `train()` — the per-epoch training loop with method-specific logic
3. Calls `self.log_epoch()` each epoch for unified logging and checkpointing
4. The base class `run()` handles everything else: best-model restoration, FLOPs profiling, final evaluation, result assembly, training log, and plots

Complex methods (Group B: CR_GNN, GNNGuard, NRGNN, RTGNN, ERASE) additionally override:
- `get_checkpoint_state()` / `load_checkpoint_state()` — to include auxiliary models
- `profile_flops()` / `profile_training_step()` — to profile multi-model pipelines
- `evaluate()` — to use wrapper models for inference

### Method Summary

| # | Method | File | Description |
|---|--------|------|-------------|
| 1 | `standard` | `model/methods/Standard.py` | Cross-entropy, no defense |
| 2 | `positive_eigenvalues` | `model/methods/Positive_Eigenvalues.py` | Eigenvalue constraint on Hessian |
| 3 | `gcod` | `model/methods/GCOD_loss.py` | Centroid-based soft labels + uncertainty |
| 4 | `nrgnn` | `model/methods/NRGNN.py` | Dual-model: predictor + GNN with edge reweighting |
| 5 | `pi_gnn` | `model/methods/PI_GNN.py` | Mutual information regularization |
| 6 | `cr_gnn` | `model/methods/CR_GNN.py` | Contrastive regularization with augmentation |
| 7 | `community_defense` | `model/methods/CommunityDefense.py` | Community-structure loss regularization |
| 8 | `rtgnn` | `model/methods/RTGNN.py` | Robust topology-aware with edge re-estimation |
| 9 | `graphcleaner` | `model/methods/GraphCleaner.py` | Noise detection + filtered standard training |
| 10 | `unionnet` | `model/methods/UnionNET.py` | Label smoothing with neighbor consensus |
| 11 | `gnn_cleaner` | `model/methods/GNN_Cleaner.py` | Dual network: GNN + sample weighting network |
| 12 | `erase` | `model/methods/ERASE.py` | Self-supervised + MCR2 loss for robust features |
| 13 | `gnnguard` | `model/methods/GNNGuard.py` | Attention-based edge pruning defense |

---

## Evaluation

### Classification Metrics (`model/evaluation.py` — `ClassificationMetrics`)

Uses sklearn with `macro` averaging:
- Accuracy (`accuracy_score`)
- Precision (`precision_score`, macro)
- Recall (`recall_score`, macro)
- F1 (`f1_score`, macro)

Collected on three splits (test, train, val) at final evaluation via `evaluate_model()`.

### Noise-Split Train Metrics

In addition to standard per-split metrics, `_make_result()` computes classification metrics on three training subsets via `compute_train_noise_split_cls()`:

| Key | Subset | Evaluated against |
|-----|--------|-------------------|
| `train_only_clean_cls` | Training nodes with uncorrupted labels | Original (clean) labels |
| `train_only_mislabelled_factual_cls` | Training nodes whose labels were corrupted | Corrupted (noisy) labels |
| `train_only_mislabelled_corrected_cls` | Training nodes whose labels were corrupted | Original (clean) labels |

These are also tracked per-epoch via `log_epoch(train_predictions=...)`.

### Oversmoothing Metrics (`model/evaluation.py` — `OversmoothingMetrics`)

6 metrics computed on node embeddings + graph edges using sparse operations (O(E) memory, not O(N²)):

| Metric | What it measures |
|--------|-----------------|
| NumRank | Numerical rank: frobenius_norm^2 / spectral_norm^2 |
| Erank | Effective rank via singular value entropy |
| EDir | Dirichlet energy (class-grouped, edge-by-edge, sparse) |
| EDir_traditional | Dirichlet energy normalized by dominant eigenvector |
| EProj | Projection energy onto dominant eigenvector |
| MAD | Mean Average Distance (cosine-based, edge-by-edge, sparse) |

### When Oversmoothing is Collected

During training: every `oversmoothing_every` epochs (default 20), train and val splits are computed and stored in the epoch log.

At final evaluation: train, val, and test splits are computed. All three are included in the result dict as `test_oversmoothing`, `train_oversmoothing`, and `val_oversmoothing`.

### Embedding source for oversmoothing

All 13 trainers use `get_embeddings()` (hidden_channels-dim) rather than `forward()` (num_classes-dim logits) for oversmoothing computation. This provides semantically meaningful representations: logits collapse to near-identical vectors for same-class neighbors in a well-trained model, making metrics like MAD trivially near-zero. Hidden representations preserve structural diversity.

---

## Compute Profiling

FLOPs and wall-clock time are measured per run via `BaseTrainer.run()`:

| Metric | How measured |
|--------|-------------|
| `flops_inference` | One forward pass under `torch.profiler` after training |
| `flops_training_total` | One training step (forward + backward) profiled, then multiplied by `actual_epochs` |
| `time_training_total` | Wall-clock time for the entire `train()` call |
| `time_inference` | Wall-clock time for the `evaluate()` call |

Methods with multi-model pipelines (NRGNN, RTGNN, CR_GNN, GNNGuard, ERASE) override `profile_flops()` and `profile_training_step()` to capture the full computation including auxiliary models.

Known limitations: methods whose graph structure changes per epoch (NRGNN adds edges, RTGNN filters edges) use a fixed-size graph for profiling, so FLOPs are approximate. Non-neural operations (scipy, sklearn) are invisible to `torch.profiler`.

Carbon emissions tracked via `codecarbon` (optional, requires sudo on macOS).

---

## Per-Epoch Logging and Training Logs

Every trainer records per-epoch data via `log_epoch()`. After training completes, `save_training_log()` writes a `training_log.json` to the run directory:

```
results/{experiment_dir}/run_1/
    training_log.json          # full training history + final result
    epoch_000_valloss_1.2345.pt  # per-epoch checkpoint (if enabled)
    epoch_001_valloss_1.1234.pt
    ...
    training_curves.png        # loss, accuracy, F1 over epochs
    oversmoothing_curves.png   # MAD, E_dir, NumRank over epochs
    noise_split_curves.png     # clean vs noisy subset metrics
```

`training_log.json` structure:
```json
{
    "run_id": 1,
    "config": { ... },
    "training_params": { "method": "...", "lr": 0.01, ... },
    "duration_seconds": 12.34,
    "stopped_at_epoch": 45,
    "best_epoch": 32,
    "best_val_loss": 0.5678,
    "best_checkpoint": "epoch_032_valloss_0.5678.pt",
    "epoch_log": [
        {
            "epoch": 0,
            "train_loss": 1.5, "val_loss": 1.4,
            "train_acc": 0.45, "val_acc": 0.50,
            "train_f1": 0.40, "val_f1": 0.45,
            "train_acc_clean_only": 0.55, "train_f1_clean_only": 0.40,
            "train_acc_only_mislabelled_factual": 0.30, "train_f1_only_mislabelled_factual": 0.25,
            "train_acc_only_mislabelled_corrected": 0.35, "train_f1_only_mislabelled_corrected": 0.30,
            "oversmoothing": { "NumRank": ..., "MAD": ... }
        },
        ...
    ],
    "final_result": {
        "test_cls": { "accuracy": 0.82, "f1": 0.80, ... },
        "train_cls": { ... }, "val_cls": { ... },
        "train_only_clean_cls": { ... },
        "train_only_mislabelled_factual_cls": { ... },
        "train_only_mislabelled_corrected_cls": { ... },
        "test_oversmoothing": { ... },
        "train_oversmoothing": { ... },
        "val_oversmoothing": { ... },
        "compute_info": { "flops_inference": ..., "time_training_total": ... }
    }
}
```

The `final_result` key is what `detect_completed_runs()` reads to support incremental run detection.

---

## Results Storage

Each sweep config produces an experiment directory with per-run subdirectories and a summary:

```
results/{hash8}_{dataset}_{method}_{seed}_noise-{type}-{rate}/
    run_1/training_log.json
    run_1/*.pt                   # per-epoch checkpoints
    run_1/*.png                  # training/oversmoothing plots
    ...
    run_5/training_log.json
    best_run_1.pt                # best-epoch checkpoint (for eval-only)
    ...
    experiment.json              # aggregated mean/std across all runs
    emissions.csv                # codecarbon output (optional)
```

`experiment.json` structure:
```json
{
    "config": { ... },
    "classification": {
        "test":  { "accuracy": [r1, r2, ...], "f1": [...], ... },
        "train": { ... },
        "val":   { ... },
        "train_only_clean": { ... },
        "train_only_mislabelled_factual": { ... },
        "train_only_mislabelled_corrected": { ... }
    },
    "oversmoothing": {
        "test":  { "NumRank": [r1, r2, ...], "MAD": [...], ... },
        "train": { ... },
        "val":   { ... }
    },
    "compute": {
        "flops_inference": [r1, r2, ...],
        "flops_training_total": [...],
        "time_training_total": [...],
        "time_inference": [...]
    }
}
```

---

## Noise Injection (`util/noise.py`)

Applied **only to training labels**. Val/test labels remain clean (original).

| Noise Type | Mechanism |
|-----------|-----------|
| `clean` | No corruption |
| `uniform_simple` | Each label flipped with probability `rate` to random other class |
| `uniform` | Transition matrix: off-diagonal = rate/(C-1) |
| `pair` | Each class flips to previous class with probability `rate` |
| `random` | Random transition matrix |
| `random_pair` | Random single-target flip per class |
| `flip` | Sequential circular flip |
| `uniform_mix` | Uniform transition with self-loops |
| `deterministic` | Exact `rate * n_train` samples corrupted |
| `instance` | Instance-dependent via learned projection |

Noise seed varies per run: `noise_seed = config_seed + run_id * 10`.

The original labels are preserved as `data.y_original` and the noisy labels as `data.y_noisy` — both stored on the data object for noise-split metric computation.

---

## Plotting (`util/plot.py`)

All methods automatically generate training curve plots via `BaseTrainer.save_plots()` when `run_dir` is set:

- **Training curves** — loss, accuracy, and F1 for train/val splits over epochs
- **Oversmoothing curves** — MAD, E_dir, NumRank over epochs (at `oversmoothing_every` intervals)
- **Noise-split curves** — accuracy and F1 on clean, noisy-factual, and noisy-corrected subsets over epochs

Plots are saved as PNG files in each `run_N/` directory. Uses `Agg` backend (non-interactive) to avoid blocking popups during batch runs.

---

## Seeding Strategy

Global seed per run: `seed = config_seed + run_id * 100`

Sets: Python hashseed, `random`, numpy, torch manual seed, CUDA seeds, cuDNN deterministic mode, `CUBLAS_WORKSPACE_CONFIG`.

Noise seed per run: `config_noise_seed + run_id * 10`.

Data splits are fixed by `config_seed` (not per-run), so all runs share the same train/val/test partition.
