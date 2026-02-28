# Changelog

All notable changes to the GNN Robustness Benchmarking framework (starting from the review effort).

---

## 2026-02-22 17:54 — `da03bef` macos cpu-only setup file

**Files:** `setup_macos.sh` (+5)

Added a shell script for setting up the project on macOS without a GPU. Enables local development and testing on CPU-only machines.

---

## 2026-02-22 17:54 — `2be83fc` fix import - not existing folder

**Files:** `model/RTGNN.py` (+1 −1)

Fixed a broken import in RTGNN that referenced a non-existent directory path. Prevented the model from loading at all.

---

## 2026-02-22 17:56 — `a0a2183` minimal cpu config for testing

**Files:** `config.yaml` (+113)

Added a minimal configuration file tuned for CPU testing — small hidden dimensions, few epochs, and lightweight dataset defaults. Allows quick smoke-testing without GPU resources.

---

## 2026-02-22 17:57 — `6485550` fix threading Segfault by adding profiler thread lock

**Files:** `utilities.py` (+35 −31)

The multithreading entry point (`main_multithreading.py`) was crashing with a segfault due to unsynchronized access to the FLOPs profiler. Added a thread lock around profiling calls to make concurrent runs safe.

---

## 2026-02-22 18:07 — `37823cf` fix - fails on trial runs when metrics empty (less than 20 epochs)

**Files:** `main.py` (+5 −9)

Short training runs (fewer than 20 epochs) produced empty metric arrays, causing the aggregation in `main.py` to crash. Added guards so that runs with insufficient data are handled gracefully.

---

## 2026-02-22 18:16 — `51cff99` macos fix - allow running without EmissionsTracker

**Files:** `main.py` (+11 −5)

`codecarbon.EmissionsTracker` requires `sudo` on macOS to read hardware power metrics. Wrapped the tracker in a try/except so the benchmarking loop continues without carbon tracking when permissions are unavailable.

---

## 2026-02-22 18:24 — `8bc4049` macos cpu-only setup file ext

**Files:** `setup_macos.sh` (+4)

Extended the macOS setup script with additional dependency installation steps.

---

## 2026-02-22 19:13 — `f7182fe` CRITICAL - fix stale evaluation; final test computed on last training epoch, not the best model

**Files:** `model/Standard.py` (+4 −1)

**Impact: Results correctness.** The Standard trainer was evaluating the test set using the model state from the *last* training epoch rather than the *best* (early-stopped) checkpoint. This means all previously reported Standard method results were potentially using an overfitted or suboptimal model. Fixed by reloading the best checkpoint before final evaluation.

---

## 2026-02-22 19:16 — `10ddc6e` CRITICAL - fix oversmoothing metric for ERASE to follow the other models

**Files:** `model/ERASE.py` (+1 −1)

**Impact: Metric consistency.** ERASE computed oversmoothing metrics differently from all other methods — it was passing the wrong tensor. Aligned it with the shared evaluation convention so oversmoothing comparisons across methods are valid.

---

## 2026-02-22 19:19 — `f9fc607` nit - GNNGuard make patience counter consistent non-inverted as in other models

**Files:** `model/GNNGuard.py` (+5 −5)

GNNGuard's early-stopping patience counter was counting in the opposite direction compared to all other trainers (counting down vs. counting up). Inverted the logic to match the convention used everywhere else, making the codebase more readable and consistent.

---

## 2026-02-22 19:31 — `797b0c3` fix - config hash did not capture all experiment variables

**Files:** `sweep_utils.py` (+13 −9)

**Impact: Result deduplication.** The SHA1 hash used to name experiment result directories was missing several config fields (model params, method-specific params). Two experiments with different hyperparameters could collide to the same hash, silently overwriting each other's results. Fixed by including all result-affecting fields in the hash input.

---

## 2026-02-22 19:50 — `b4e6885` CRITICAL - PI_GNN f1 logged micro in training, while macro in reporting

**Files:** `model/PI_GNN.py` (+2 −2)

**Impact: Reported metrics were wrong.** PI_GNN computed F1 using `micro` averaging during training but reported it as if it were `macro`. Since all other methods use `macro` F1, PI_GNN's training-time F1 was incomparable. Changed to `macro` to match the benchmark convention.

---

## 2026-02-22 19:51 — `c302daf` CRITICAL - ERASE f1 computed weighted, while all others report macro

**Files:** `model/ERASE.py` (−2)

**Impact: Reported metrics were wrong.** ERASE was computing F1 with `weighted` averaging instead of `macro`. Removed the override so it falls through to the shared evaluation code that uses `macro`, making ERASE comparable to other methods.

---

## 2026-02-23 10:55 — `e4a3dbf` refactor 1 - move most of metrics out of models to evaluation.py

**Files:** 14 files (+578 −1,544)

**Major refactor.** Each of the 12 method trainers had its own copy of accuracy, F1, precision, recall, and oversmoothing computation — with subtle inconsistencies between them (as discovered above). Extracted all shared metric logic into a single `model/evaluation.py` module. Every trainer now calls the same functions, eliminating the risk of per-method metric drift. Net reduction of ~1,000 lines.

---

## 2026-02-23 11:01 — `b7179c6` olderfiles - delete, kept in git history

**Files:** 14 files (−7,482)

Deleted the `model/olderfiles/` directory containing pre-refactor copies of all trainers. These were kept as a safety net during the metric extraction refactor and are preserved in git history.

---

## 2026-02-23 11:19 — `e9b6288` testing - create small local test runs

**Files:** 6 files (+252 −2)

Introduced the project's first automated test suite: `tests/test_smoke.py` with smoke tests that run a minimal training loop (3 epochs, tiny hidden dim) for each method on a small dataset. Also added `pytest` to `requirements.txt` and a `test.sh` convenience script.

---

## 2026-02-23 11:23 — `60e5bf9` gitignore - PyG JIT codegen cache, remove added

**Files:** 4 files (+4 −726)

Added PyTorch Geometric's JIT-compiled kernel cache directory to `.gitignore` and removed the three auto-generated `.py` files that had been accidentally committed.

---

## 2026-02-23 11:38 — `ec520d6` results_old - delete results leaked into github

**Files:** 265 files (−36,452)

Removed a large set of old experiment result JSON files and emission CSVs that had been accidentally committed. These hash-named files bloated the repo. Results are now generated locally and excluded from version control.

---

## 2026-02-23 11:49 — `d585b3e` Del - unused model param

**Files:** `model/GNNGuard.py` (−1)

Removed an unused constructor parameter from GNNGuard.

---

## 2026-02-23 11:53 — `f78de86` Fix - remove dead condition

**Files:** `model/GraphCleaner.py` (+1 −2)

Removed a dead conditional branch in GraphCleaner that could never be reached.

---

## 2026-02-23 12:17 — `cbbcb34` Del - dead code

**Files:** 6 files (−54)

Removed dead code across multiple models: unused methods in ERASE, GCOD, GNN_Cleaner, GraphCleaner, PI_GNN, and utilities. Identified through static analysis and confirmed unreachable.

---

## 2026-02-23 12:22 — `e9b6128` Del - CommunityDefense `_generate_community_pairs()` is dead code

**Files:** `model/CommunityDefense.py` (−45)

Removed an unused 45-line method that generated community pairs but was never called from any code path.

---

## 2026-02-23 12:23 — `7973380` Del - GraphCleaner `_calculate_dirichlet_energy()` is dead code

**Files:** `model/GraphCleaner.py` (−23)

Removed an unused Dirichlet energy calculation method from GraphCleaner.

---

## 2026-02-23 12:37 — `1e4f3dd` Del&Fix - in utilities.py dead code, perf improve, clean

**Files:** `utilities.py` (+19 −25)

Cleaned up the main utilities module: removed dead code paths, simplified control flow, and applied minor performance improvements.

---

## 2026-02-23 12:54 — `a8f4e5c` Refactor utils - split into logical files, document, lint

**Files:** 13 files (+1,462 −1,233)

**Major refactor.** Split the monolithic `utilities.py` (1,228 lines) into focused modules under `util/`:
- `util/data.py` — dataset loading and preprocessing
- `util/noise.py` — noise injection functions
- `util/experiment.py` — experiment orchestration (`run_experiment()`)
- `util/profiling.py` — FLOPs and timing measurement
- `util/cli.py` — CLI formatting helpers
- `util/seed.py` — seed management

Added docstrings and fixed lint warnings throughout. All imports in `main.py`, `main_multithreading.py`, and tests updated.

---

## 2026-02-23 13:22 — `f12b4f7` Refactor - create model registry, each model knows how to run itself

**Files:** 18 files (+553 −520)

**Architecture change.** Introduced `model/registry.py` and `model/base.py`. Each method trainer now registers itself with a decorator and defines a `from_config()` classmethod. The experiment runner no longer needs a giant if/elif chain — it looks up the method by name in the registry. This makes adding new methods a one-file operation.

---

## 2026-02-23 13:33 — `17f7e9b` Ref - move all method experiments to `model/methods/`

**Files:** 15 files (+12 −16)

Moved all 13 method trainers from `model/` into `model/methods/` subdirectory. Keeps the `model/` package clean with only shared modules (`base.py`, `evaluation.py`, `gnns.py`, `registry.py`) at the top level.

---

## 2026-02-23 14:32 — `e6d550b` Ref&clean - separate methods, clean all unused variables and fix lint and type errors

**Files:** 15 files (+175 −236)

Systematic cleanup pass across all method trainers and shared modules. Removed unused variables, fixed type annotations, resolved lint warnings, and ensured consistent code style across all 13 methods.

---

## 2026-02-23 22:18 — `ce5c9a9` Eval - vectorize oversmoothing metrics, store val oversmoothing, configurable frequency

**Files:** 19 files (+658 −219)

**Performance + feature.** Rewrote oversmoothing metric computation (MAD, E_dir, NumRank, etc.) using vectorized operations instead of Python loops — significant speedup on large graphs. Added validation-split oversmoothing metrics (previously only test). Made metric computation frequency configurable to reduce overhead during training. Added `tests/test_evaluation.py` with comprehensive unit tests for all oversmoothing metrics.

---

## 2026-02-24 17:41 — `74f6599` CRITICAL - fix GPS model variant, residual dimension must match

**Files:** `model/gnns.py` (+21 −10)

**Impact: GPS model was broken.** The GPS (General Powerful Scalable) backbone had a dimension mismatch in its residual connections — the input projection and skip-connection dimensions didn't align, causing runtime errors or silent shape mismatches. Fixed by ensuring the linear projection maps to the correct hidden dimension.

---

## 2026-02-24 17:44 — `2342d45` Nit - show error when dims mismatch in GATv2

**Files:** `model/gnns.py` (+6)

Added an explicit dimension check in GATv2 initialization. Previously a mismatch between `hidden_channels` and `heads * out_channels` would produce a cryptic downstream error; now it fails immediately with a clear message.

---

## 2026-02-24 18:27 — `0d2c267` GAT and GATv2 - add edge_weights correctly, fix in v2; add support into NRGNN and RTGNN

**Files:** 3 files (+28 −17)

Fixed how edge weights are passed through attention-based architectures. GATv2 was ignoring edge weights entirely. NRGNN and RTGNN, which construct their own edge sets, now correctly propagate edge weights through their GAT/GATv2 backbones.

---

## 2026-02-24 19:11 — `ca62529` GPS does not support edge_weights even as edge_attr

**Files:** `model/gnns.py` (+9 −4)

GPS's `GPSConv` layer does not accept edge weights. Added a guard that silently drops edge weights when the GPS backbone is used, with a warning on first occurrence. Prevents runtime crashes when GPS is combined with methods that produce edge weights (NRGNN, RTGNN).

---

## 2026-02-24 19:12 — `0595785` Nit - rename all edge_attr to edge_weights

**Files:** 2 files (+4 −7)

Renamed `edge_attr` to `edge_weights` throughout the codebase. The framework only uses scalar edge weights, not multi-dimensional edge attributes. The rename makes the semantics explicit and avoids confusion with PyG's `edge_attr` convention.

---

## 2026-02-24 20:35 — `5556056` nit GCN - had no activation before output linear layer

**Files:** `model/gnns.py` (+2)

Added a missing ReLU activation before GCN's final linear output layer. Not used in the current benchmark configurations (which use GCN without the output linear), but corrects the architecture for completeness.

---

## 2026-02-24 21:37 — `a2d5c5d` nits - remove gnns.py unused norms and init params

**Files:** `model/gnns.py` (+3 −4)

Removed unused normalization layer references and initialization parameters from the backbone module.

---

## 2026-02-24 22:12 — `fee51f2` BaseTrainer: add train/evaluate/checkpoint infrastructure

**Files:** `model/base.py` (+57 −4)

**Architecture change.** Extended `BaseTrainer` with a standardized `train_full_model()` flow that handles: training loop orchestration, early stopping, best-model checkpointing (in-memory), final evaluation on all splits, and result packaging. This is the foundation for migrating all 13 method trainers to a shared control flow.

---

## 2026-02-24 22:16 — `6f12169` Migrate Standard + GraphCleaner to `train()`

**Files:** 2 files (+9 −29)

Migrated Standard and GraphCleaner trainers to use `BaseTrainer.train_full_model()`. Each now only implements `train_epoch()` — the training loop, checkpointing, and evaluation are handled by the base class. Net reduction of 20 lines per method.

---

## 2026-02-24 22:25 — `8bdf456` Migrate 7 Group A methods to `train()`

**Files:** 7 files (+90 −200)

Migrated CommunityDefense, ERASE, GCOD, GNN_Cleaner, PI_GNN, Positive_Eigenvalues, and UnionNET to the shared `train_full_model()` flow. These "Group A" methods have a standard single-model training loop and were straightforward to migrate.

---

## 2026-02-24 22:30 — `5c255b8` Migrate 4 Group B methods to `train()`, make `train()` abstract

**Files:** 6 files (+77 −52)

Migrated CR_GNN, GNNGuard, NRGNN, and RTGNN — the "Group B" methods that train multiple models (e.g., edge predictor + classifier) or have non-standard training flows. Made `train_epoch()` abstract in `BaseTrainer` to enforce the contract. All 13 methods now use the shared infrastructure.

---

## 2026-02-24 22:34 — `a73e6c1` Add checkpoint I/O to run_experiment and tests

**Files:** 4 files (+125 −4)

Added model checkpoint save/load to the experiment runner. After training, the best model state is saved to disk as a `.pt` file. Added `--eval-only` mode that loads a checkpoint and runs evaluation without retraining. Includes tests for checkpoint round-trip correctness.

---

## 2026-02-24 22:45 — `b1ac1d7` Code review fixes: eval_only guard, snapshot checkpoints, minor cleanup

**Files:** 9 files (+92 −13)

Post-migration code review fixes. Added a guard preventing eval-only mode when no checkpoint exists. Changed checkpointing to snapshot (deep copy) model state at best epoch rather than saving at the end. Fixed several minor issues in Group B methods where auxiliary models weren't included in checkpoints. Added tests validating these fixes.

---

## 2026-02-24 22:52 — `1755211` Add checkpoint support to main_multithreading.py

**Files:** 2 files (+47 −14)

Extended the multithreading entry point with checkpoint save/load and eval-only CLI flags, matching the capabilities of `main.py`.

---

## 2026-02-24 23:04 — `55ca725` Checkpoint by default

**Files:** 3 files (+4 −4)

Changed the default behavior from `--no-checkpoint` (opt-in saving) to checkpoints enabled by default with `--no-checkpoint` to disable. Checkpoints are now always saved unless explicitly opted out.

---

## 2026-02-24 23:44 — `6718333` Add eval-only command line flags

**Files:** `main.py` (+22 −8)

Added `--eval-only` and `--no-checkpoint` as proper CLI arguments to `main.py` with `argparse`. Previously these were only configurable via `config.yaml`.

---

## 2026-02-25 00:39 — `ebb7163` Fix - store compute cost metrics for every run, not just first; add config CLI param

**Files:** `main.py` (+18 −14)

**Impact: Data completeness.** Compute metrics (FLOPs, training time, inference time) were only captured for the first run in a multi-run benchmark and reused for all subsequent runs. Fixed to capture independently per run, since training time can vary. Also added `--config` / `-c` CLI parameter for specifying the config file path.

---

## 2026-02-25 00:47 — `ceab129` Nit - do not show plots as new popup windows

**Files:** `model/methods/GCOD_loss.py` (+2 −2)

Switched matplotlib backend to `Agg` (non-interactive) for GCOD's built-in plotting. Prevents blocking popup windows during batch runs.

---

## 2026-02-25 01:04 — `895f41d` FIX - flops measure inference GCN, now should measure whole inference which differs between methods

**Files:** 5 files (+59 −3)

**Impact: Metric accuracy.** FLOPs measurement was only profiling a single GCN forward pass, ignoring method-specific inference steps (e.g., ERASE's label propagation, CR_GNN's dual-view fusion, RTGNN's edge predictor). Added `inference_forward()` overrides to methods with non-trivial inference so FLOPs capture the full pipeline.

---

## 2026-02-25 01:16 — `ef204de` Add flops and time cost for training and inference (only inference flops before)

**Files:** 6 files (+77 −29)

Extended compute profiling to measure both training FLOPs and wall-clock time in addition to the existing inference FLOPs. Results now include `flops_training_total`, `time_training_total`, `flops_inference`, and `time_inference` — giving a complete computational cost picture for each method.

---

## 2026-02-25 01:33 — `23145c9` Capture compute for every run, magic TRAINING_FLOPS_MULTIPLIER, small fixes

**Files:** 4 files (+38 −15)

Refined compute cost measurement. Training FLOPs are estimated by profiling a single epoch and multiplying by epoch count (via `TRAINING_FLOPS_MULTIPLIER`), since profiling every epoch adds significant overhead. Ensured compute info is captured per run rather than shared. Added tests.

---

## 2026-02-25 01:44 — `d63149c` Nits to main.py - configurable number of runs

**Files:** 2 files (+51 −77)

Made the number of benchmark runs configurable via `num_runs` in `config.yaml` (default: 5) and `--num-runs` CLI flag. Previously hardcoded to 5. Also simplified test helper code.

---

## 2026-02-25 09:07 — `8c5a502` Add train/val classification metrics to json storage

**Files:** 6 files (+82 −42)

**Feature.** Extended the results JSON to include train and validation classification metrics (accuracy, F1, precision, recall) alongside the existing test metrics. Enables analysis of overfitting behavior and train-test gaps across methods.

---

## 2026-02-25 10:25 — `117bccb` Restructure JSON output for consistency and use descriptive filenames

**Files:** 11 files (+115 −138)

**Breaking change to output format.** Replaced opaque SHA1-hash filenames (e.g., `006bbdc0...json`) with human-readable directory names: `{hash8}_{dataset}_{method}_{seed}_noise-{type}-{rate}`. Restructured the JSON schema for consistent nested keys across classification, oversmoothing, and compute sections. Added `get_result_filename()` to `sweep_utils.py`.

---

## 2026-02-25 10:33 — `40a176b` Nits - formatting names, common constants, TODOs

**Files:** 10 files (+45 −39)

Consistency pass: extracted `OVERSMOOTHING_KEYS` and `ZERO_CLS` as shared constants in `evaluation.py`, cleaned up display formatting in `cli.py`, added TODOs for known improvements.

---

## 2026-02-25 11:59 — `d08ebdc` Large log expansion - log training stats, metrics, and checkpoint every epoch

**Files:** 20 files (+574 −156)

**Major feature.** Introduced comprehensive per-epoch logging via `BaseTrainer.save_training_log()`. Every epoch now records: loss, learning rate, classification metrics (train/val/test), oversmoothing metrics, and a checkpoint. All data is saved to `run_N/training_log.json`. This enables post-hoc analysis of training dynamics, learning curves, and metric evolution — and is the foundation for the later incremental-run detection feature. Updated all 13 methods and added tests.

---

## 2026-02-25 12:02 — `b45904a` Fix - all methods start from epoch=0 for consistency

**Files:** 4 files (+11 −11)

Several methods (CommunityDefense, GCOD, Positive_Eigenvalues, Standard) were using `range(1, epochs+1)` while others used `range(epochs)`. Standardized all to start from epoch 0 for consistent log indexing and early-stopping behavior.

---

## 2026-02-25 12:14 — `a399fa3` Fix CR_GNN - was missing F1 in reporting, nits

**Files:** `model/methods/CR_GNN.py` (+15 −12)

CR_GNN was not including F1 score in its per-epoch log output, making it harder to track training progress. Added F1 to the logged metrics and cleaned up formatting.

---

## 2026-02-25 12:22 — `6329749` Test nits

**Files:** 2 files (+7 −9)

Minor test cleanup: removed redundant assertions and added a missing import.

---

## 2026-02-27 15:22 — `d634a05` CRITICAL, large - Use hidden embeddings (not logits) for oversmoothing metrics

**Files:** 16 files (+305 −119)

**Impact: Metric correctness.** All oversmoothing metrics (MAD, E_dir, NumRank, etc.) were being computed on the final softmax logits instead of the hidden-layer embeddings. Logits are low-dimensional (num_classes) and already collapsed by the classification head, making oversmoothing metrics meaningless. Changed all backbone models to expose intermediate hidden embeddings via a `forward_embeddings()` method. All method trainers updated to pass embeddings to oversmoothing evaluation. Results from prior runs using logit-based oversmoothing are invalid.

---

## 2026-02-27 15:50 — `ed94833` GIN does not support edge_weights - add explicit warning

**Files:** `model/gnns.py` (+19 −1)

GIN (Graph Isomorphism Network) inherently does not support edge weights in its message passing. Added a clear warning when edge weights are provided to a GIN backbone, and silently drops them. Prevents silent incorrect behavior when GIN is combined with edge-weight-producing methods.

---

## 2026-02-27 16:21 — `7b147cf` Add plotting to all models sharing BaseTrainer, not just GCOD

**Files:** 7 files (+117 −99)

Extracted GCOD's plotting code into a shared `util/plot.py` module and wired it into `BaseTrainer`. All 13 methods now automatically generate training curves (loss, accuracy, F1, oversmoothing metrics) when plotting is enabled in config. Previously only GCOD had this capability.

---

## 2026-02-27 20:09 — `cbfad01` Refactor model checkpointing & evaluation - now complex modes covered by the same flow

**Files:** 17 files (+205 −148)

**Architecture improvement.** Unified the checkpoint save/load and final evaluation flow for all methods, including the complex Group B methods (CR_GNN, NRGNN, RTGNN, GNNGuard) that manage multiple internal models. Each method now defines `state_dict_extra()` and `load_state_dict_extra()` hooks for their auxiliary components. The base class handles the rest. Eliminated per-method evaluation duplication.

---

## 2026-02-27 21:08 — `f15bf5b` Checkpoint consistency tests, fixes for resuming complex models

**Files:** 6 files (+326 −19)

Added `tests/test_checkpoint_consistency.py` with 245 lines of tests that verify checkpoint save/load round-trip correctness for every method. Found and fixed issues with ERASE, GNNGuard, NRGNN, and RTGNN where auxiliary model states were not being fully captured or restored.

---

## 2026-02-27 21:21 — `02813f9` Fix load checkpoint for NRGNN and RTGNN

**Files:** 2 files (+9)

Follow-up fix: NRGNN and RTGNN's edge predictor models were not being moved to the correct device after loading from checkpoint, causing device mismatch errors during eval-only runs.

---

## 2026-02-27 21:57 — `9f81d78` Fix flops estimation for training & inference

**Files:** 7 files (+288 −12)

Major overhaul of FLOPs estimation. Added `training_forward()` and `inference_forward()` methods to `BaseTrainer` and all complex methods, ensuring the profiler measures the actual operations performed (including auxiliary models). Rewrote `util/profiling.py` with improved accuracy and support for multi-model methods.

---

## 2026-02-27 23:02 — `55415f6` Opt - free memory

**Files:** 3 files (+16)

Added explicit memory cleanup between runs in `main.py` (`gc.collect()` + `torch.cuda.empty_cache()`), after profiling completes, and after BaseTrainer finishes training. Reduces peak memory usage during multi-run benchmarks.

---

## 2026-02-27 23:02 — `72e211f` Opt - eliminate N×N dense matrices in oversmoothing metrics

**Files:** `model/evaluation.py` (+21 −8)

**Performance.** Oversmoothing metrics (MAD, E_dir) were constructing full N×N dense distance/similarity matrices, which is O(N²) memory and infeasible for large graphs. Rewrote to use sparse operations that only compute values along existing edges, reducing memory from O(N²) to O(E).

---

## 2026-02-27 23:03 — `e33f4ab` Opt - `optimizer.zero_grad(set_to_none=True)`

**Files:** 13 files (+17 −17)

Applied `set_to_none=True` to `optimizer.zero_grad()` across all 13 method trainers. This avoids a memset to zero and instead deallocates gradients, which is slightly faster and uses less memory per PyTorch documentation.

---

## 2026-02-27 23:41 — `cd4ae84` Add - train classification metrics for clean, noisy-factual, and noisy-corrected label subsets

**Files:** 3 files (+70 −6)

**Feature.** Added per-epoch classification metrics broken down by noise status of training nodes:
- `train_only_clean` — nodes with original correct labels
- `train_only_mislabelled_factual` — noisy nodes evaluated against their factual (true) labels
- `train_only_mislabelled_corrected` — noisy nodes evaluated against their corrupted labels

This enables analysis of how well each method learns on clean vs. noisy subsets and whether it "corrects" mislabeled nodes.

---

## 2026-02-28 00:38 — `c9040d8` Add - train class metrics for clean/noisy but for every epoch, not just final output

**Files:** 15 files (+119 −32)

Extended the noise-split classification metrics from final-only to per-epoch logging. Every trainer now reports clean/noisy subset metrics at each epoch via the shared `BaseTrainer` infrastructure. Required adding a `noisy_indices` parameter to the evaluation flow across all methods.

---

## 2026-02-28 00:53 — `1a2be99` Add - plotting for all noise/clean class metrics

**Files:** `util/plot.py` (+61 −36)

Extended the shared plotting module to visualize the noise-split classification metrics. Training curves now show separate lines for clean nodes, noisy-factual, and noisy-corrected subsets, making it easy to visually assess noise robustness.

---

## 2026-02-28 01:06 — `4b0f001` Fix metrics - Standard and UnionNet used stale train-time (before step, with dropouts); removing duplicated inference and small fixes in others

**Files:** 6 files (+30 −34)

**Impact: Metric accuracy.** Standard and UnionNET were reporting train-time metrics computed *before* the optimizer step and *with dropout active* — meaning reported training accuracy was both stale (one step behind) and noisy (dropout masking). Fixed by computing train metrics after the optimizer step in eval mode. Also removed duplicated inference calls in CR_GNN, PI_GNN, and RTGNN where metrics were being computed twice per epoch.

---

## 2026-02-28 12:13 — `39554f` Add Incremental experiment runs by increasing num_runs; add force

**Files:** `main.py`, `sweep_utils.py`, `README.md` (+127 −49)

**Feature.** Replaced the all-or-nothing experiment execution model with incremental run detection. The system now scans `run_N/training_log.json` files to identify completed runs, and only executes the missing ones. This enables scaling `num_runs: 2 → 5` without re-running the first 2.

Key changes:
- **`sweep_utils.py`**: Added `detect_completed_runs()`, removed `should_run_experiment()`
- **`main.py`**: Extracted `_accumulate_run_result()` helper, restructured run loop with detect → skip/resume → execute flow, added `--force` CLI flag, fixed `num_runs=0` truthy bug, made `compute_info` access defensive
- **`README.md`**: Documented incremental runs, skip behavior, `--force` CLI, and eval-only interaction

Behavior: `[SKIP]` when all runs complete, `[RESUME]` for partial, `[REBUILD]` when experiment.json missing, `[FORCE]` to re-run all, `[EVAL-ONLY]` always re-evaluates (writes no training_log.json).

---

## 2026-02-28 17:45 — `9496663` Fix - oversmoothing_every now consistently chooses the same epoch across all methods (and includes the last one), fix F1 in NRGNN, remove debug mode

**Files:** 12 files in `model/methods/` (+56 −94)

**Impact: Oversmoothing consistency and debug cleanup.** Three fixes applied across all 13 methods:

1. **Unified oversmoothing epoch condition** — To collect oversmoothing now all methods use the same `epoch % N == 0 or epoch == max_epochs - 1`, ensuring every method collects at the same set of epochs including epoch 0 and the last training epoch.

2. **Removed debug flags** — Standard (`debug`), GCOD (`self.debug`), NRGNN (`self.debug_mode`), and ERASE (`debug_mode`/`enable_debug_output`) gated oversmoothing computation and/or print statements behind debug flags that defaulted to True. Removed all flags; all operations now execute unconditionally.

3. **NRGNN F1 computed every epoch** — F1 was only computed inside the oversmoothing block (every N epochs), passed as `None` to `log_epoch_fn` on other epochs. Now computed unconditionally like all other methods.

---

## 2026-02-28 17:52 — `02429ca` Fix - NRGNN now writes the metrics for the last epoch (before interrupted with exception)

**Files:** `model/methods/NRGNN.py` (+13 −13)

**Impact: Code quality, capture last epoch metrics.** Replaced NRGNN's non-standard `raise StopIteration` / `except StopIteration` early stopping with a `self._should_stop` flag checked by `break` in `fit()`. 
