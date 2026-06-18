# Findings — Method Comparison under Label Noise

This is a **comparison** benchmark. The goal is to run each published robustness
method *faithfully* and report how it behaves; where a method has a weakness we
**document it as a result about that method** — we do not patch or mitigate it
(patching would report our fix, not the published method). The only changes we
make to a method's code are *faithfulness* fixes (making our implementation
match the source paper); inherent method weaknesses are left intact and recorded
here.

Scope of the numbers below: **cora**, transductive, 3 seeds, noise rate 0.30,
injected into train+val (test always clean), model selected by best **noisy**
validation accuracy. Each cell is **mean test accuracy %** with **(mean train
accuracy %)** in parentheses — train accuracy disambiguates *bad-epoch
selection* (high train, low test) from *training collapse* (train ≈ chance).
Chance ≈ 14.3% (7 classes).

---

## 1. Method → source paper (verified)

| Method | Paper | Venue / ID | Official code | Faithfulness note |
|---|---|---|---|---|
| `standard` | Kipf & Welling, *Semi-Supervised Classification with GCNs* | ICLR 2017 — [1609.02907](https://arxiv.org/abs/1609.02907) | — | plain GCN baseline. The `gcn_modified` backbone reproduces tunedGNN ([2406.08993](https://arxiv.org/abs/2406.08993)) |
| `positive_eigenvalues` | Wani et al., *Energy-Guided Smoothness…* | [2412.08419](https://arxiv.org/abs/2412.08419) (author's own) | — | negative-eigenvalue clipping; auto-appends a square readout on shallow nets (see LIMITATIONS L1) |
| `gcod` | Wani et al. (GCOD) | [2412.08419](https://arxiv.org/abs/2412.08419) (author's own) | — | graph-adapted NCOD; **deliberately tuned per-dataset**, not paper Table 4 |
| `nrgnn` | Dai et al., *NRGNN* | WSDM 2022 — [2201.00232](https://arxiv.org/abs/2201.00232) | [EnyanDai/NRGNN](https://github.com/EnyanDai/NRGNN) | official defaults (`edge_hidden=64, n_p=50, p_u=0.8`) |
| `pi_gnn` | Du et al., *PI-GNN* | TMLR — [2106.07451](https://arxiv.org/abs/2106.07451) | [TianBian95/pi-gnn](https://github.com/TianBian95/pi-gnn) | official defaults; `start_epoch=200` (needs epochs > 200 to activate) |
| `cr_gnn` | Li et al., *Contrastive learning of graphs under label noise* | Neural Networks 2024 — [doi](https://www.sciencedirect.com/science/article/abs/pii/S0893608024000273) | — | β (cross-space consistency) defaults to 0.2 per NoisyGL reference |
| `community_defense` | *Self-supervised robust GNN against noisy graphs and noisy labels* | Appl. Intelligence 2023 — [doi](https://dl.acm.org/doi/10.1007/s10489-023-04836-6) | — | **module 3 only** of a 3-module framework (see LIMITATIONS L7) |
| `rtgnn` | Qian et al., *Robust Training via Noise Governance* | WSDM 2023 — [2211.06614](https://arxiv.org/abs/2211.06614) | [GhostQ99/RobustTrainingGNN](https://github.com/GhostQ99/RobustTrainingGNN) | official defaults (`th=0.95, K=100`) |
| `graphcleaner` | Li et al., *GraphCleaner* | ICML 2023 — [2306.00015](https://arxiv.org/abs/2306.00015) | [lywww/GraphCleaner](https://github.com/lywww/GraphCleaner) | official `k=3`, binary-detector budget 500 |
| `unionnet` | Li et al., *UnionNET* | PAKDD 2021 — [2103.03414](https://arxiv.org/abs/2103.03414) | — | official `k=10, alpha=0.5, beta=1` |
| `gnn_cleaner` | Xia et al., *GNN Cleaner* | IEEE TKDE 2023 — [doi](https://ieeexplore.ieee.org/document/10049408) | — | 50 label-propagation iterations/epoch per paper |
| `erase` | Chen et al., *ERASE* | CIKM 2024 — [2312.08852](https://arxiv.org/abs/2312.08852) | [eraseai/erase](https://github.com/eraseai/erase) | official MCR2 params (`gam1=1, gam2=2, n_embedding=512`) |
| `gnnguard` | Zhang & Zitnik, *GNNGuard* | NeurIPS 2020 — [2006.08149](https://arxiv.org/abs/2006.08149) | [mims-harvard/GNNGuard](https://github.com/mims-harvard/GNNGuard) | official `P0=0.1`, cosine-attention on |

---

## 2. Observed behavior (cora, test% (train%))

### GCN backbone (2-layer, h64)
| Method | clean | uniform-0.3 | pair-0.3 |
|---|---|---|---|
| standard | 78.1 (100) | 63.9 (99.5) | 58.0 (99.5) |
| positive_eigenvalues | 78.3 (100) | 59.2 (94.0) | 59.9 (99.3) |
| gcod | 80.0 (98.1) | 64.9 (72.9) | 62.7 (71.4) |
| nrgnn | 79.7 (100) | 64.7 (97.9) | 64.5 (95.0) |
| pi_gnn | 78.5 (98.6) | 69.6 (90.7) | 63.7 (93.8) |
| cr_gnn | 79.1 (100) | 68.1 (99.5) | 63.5 (98.8) |
| community_defense | 78.4 (100) | 60.7 (99.5) | 57.7 (99.3) |
| rtgnn | 79.5 (99.5) | 72.9 (80.2) | 69.6 (72.9) |
| graphcleaner | 80.0 (100) | 72.3 (100) | 70.9 (99.7) |
| unionnet | 79.0 (95.7) | 72.7 (81.7) | 66.2 (83.6) |
| gnn_cleaner | 71.0 (90.5) | 54.7 (67.1) | 48.4 (75.0) |
| erase | 78.4 (86.9) | **78.2** (64.0) | **76.2** (65.2) |
| gnnguard | 71.8 (100) | 57.2 (99.8) | 55.0 (99.0) |

### GAT backbone (2-layer, h64, 8 heads)
| Method | clean | uniform-0.3 | pair-0.3 |
|---|---|---|---|
| standard | 80.7 (100) | 67.9 (99.0) | 60.9 (98.8) |
| positive_eigenvalues | 78.4 (98.8) | 65.1 (97.6) | 62.4 (98.1) |
| gcod | 79.9 (99.8) | 67.5 (74.5) | 66.8 (72.1) |
| nrgnn | 80.8 (99.5) | 52.9 (62.1) | 57.5 (78.6) |
| pi_gnn | 79.3 (99.5) | 67.8 (91.9) | 64.2 (94.3) |
| cr_gnn | 78.5 (99.8) | 47.2 (71.2) | 53.6 (68.1) |
| community_defense | 79.2 (100) | 62.9 (99.0) | 60.5 (98.6) |
| rtgnn | 81.5 (99.8) | 74.0 (79.0) | 66.7 (72.4) |
| graphcleaner | 79.0 (100) | 72.1 (100) | **76.2** (99.7) |
| unionnet | 81.3 (95.2) | 73.8 (80.7) | 65.9 (83.8) |
| gnn_cleaner | 70.9 (84.0) | 58.1 (66.4) | 48.8 (71.2) |
| erase | 78.3 (84.8) | **77.2** (59.8) | 73.3 (62.4) |
| gnnguard | 79.0 (100) | 64.1 (99.3) | 58.8 (98.3) |

### GCN_modified backbone (tunedGNN, 3-layer, h512)
| Method | clean | uniform-0.3 | pair-0.3 |
|---|---|---|---|
| standard | 78.0 (99.8) | 61.1 (84.0) | 57.5 (88.8) |
| positive_eigenvalues | 76.5 (99.8) | 56.3 (85.2) | 54.2 (95.7) |
| gcod | 76.9 (99.3) | 53.4 (67.1) | 57.7 (66.0) |
| nrgnn | 74.7 (97.4) | 48.6 (98.1) | 55.6 (90.2) |
| pi_gnn | 74.8 (98.6) | 61.2 (86.4) | 53.9 (89.5) |
| cr_gnn | 79.5 (99.3) | 65.8 (88.6) | 62.6 (82.4) |
| community_defense | 78.8 (100) | 61.7 (87.6) | 59.5 (89.5) |
| rtgnn | 76.3 (98.3) | 57.1 (64.0) | 63.7 (69.0) |
| graphcleaner | 78.4 (100) | 67.2 (99.3) | 61.9 (99.5) |
| unionnet | **25.7 (15.2)** ⚠ | **19.8 (16.9)** ⚠ | **11.6 (19.3)** ⚠ |
| gnn_cleaner | 62.9 (70.0) | 47.1 (39.3) | 45.4 (54.3) |
| erase | **31.9 (14.3)** ⚠ | **26.2 (16.2)** ⚠ | **19.8 (19.3)** ⚠ |
| gnnguard | 71.0 (100) | 53.8 (98.3) | 53.2 (97.4) |

⚠ = training collapse (train acc ≈ chance), discussed in F2.

---

## 3. Key findings (descriptive — not action items)

**F1 — Method robustness is backbone-dependent.** On the standard `gcn`/`gat`
backbones the benchmark behaves as intended: under noise, most robustness
methods beat `standard` (e.g. gcn uniform-0.3: erase +14.3, unionnet +8.8,
rtgnn +9.0, graphcleaner +8.4, pi_gnn +5.7). On the wide tunedGNN
`gcn_modified` backbone the same methods compress toward `standard` or fail
(see F2). Robustness reported in the source papers does not automatically
transfer to a stronger/wider backbone.

**F2 — `erase` and `unionnet` suffer rank-1 training collapse on `gcn_modified`
(h512).** Train accuracy falls to chance (~15–19%) on clean *and* noisy runs,
with embedding `NumRank ≈ 1.08`. Both methods derive their objective from
embedding geometry — erase from the MCR2 coding-rate term, unionnet from
embedding-similarity support sets — which degenerates on the 512-wide backbone.
This is an inherent property of these methods at this width (they are healthy on
the h64 `gcn`/`gat` backbones), **documented, not mitigated.** Verified to
persist on current code.

**F3 — `gnn_cleaner` chronically underfits.** It trails `standard` even on
**clean** data on every backbone (gcn −7.1, gat −9.9, gcn_modified −15.1) with
depressed train accuracy (70–90%). The self-training label-correction loop costs
fit even without noise to correct. Inherent to the method as published.

**F4 — `gnnguard` and `nrgnn` are weak under label noise.** GNNGuard is an
*adversarial-structure* defense (edge pruning); it gives little benefit against
*label* noise and trails `standard` on noisy and even clean runs. NRGNN is
high-variance (gat uniform −15.0, gcn pair +6.5) and memory-heavy (F5).

**F5 — Compute cost (cora profiling, plain gcn, ~300 epochs).** `gnnguard` ≈ 2 s
/ epoch (~170× `standard`) → ~10 min per single cora run, from recomputing
cosine-attention every epoch. `nrgnn` OOMs on a 16 GB GPU (its edge predictor is
O(|unlabeled|×|confident|), an intrinsic cost). `erase` ≈ 6× `standard`. These
costs compound across 13 methods × 3 noise types × ~3 backbones × 3 seeds × the
`gcn_modified` epoch budgets (1000–2500). Early stopping is active and working
(e.g. `standard` stops ~epoch 266), but the slow methods and long budgets
dominate wall-clock.

---

## 4. Methodology notes (for fairness)

- Noise is applied to **train + val jointly** (`util/experiment.py`); test is
  always clean. All methods see the *same* corrupted labels/splits per
  (seed, type, rate).
- Model selection is by **noisy validation accuracy**. This is the realistic
  setting (clean labels are never observed) but it mildly rewards memorization
  and compresses the robust-vs-standard gap; it is *not* the cause of the F2
  collapses (those are training failures, train acc ≈ chance, not bad-epoch
  picks).
- `standard` beating robust methods on clean / low noise is expected, not a bug —
  robust methods pay a bias cost that pays off only as the noise rate rises.
- Structural and implementation-faithfulness notes (PE shallow-net behavior,
  community_defense module scope, deterministic-noise caveat, GNNGuard on GIN,
  etc.) are recorded separately in `LIMITATIONS.md`.
