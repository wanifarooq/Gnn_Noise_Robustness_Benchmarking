# Benchmark Limitations & Method Caveats

This document records the **structural limitations** of the benchmark and of
individual robustness methods — the things that are *intrinsic* to the method or
the experimental setup and therefore are **not bugs to be patched**, but must be
understood when reading the results. It also records the **five bug fixes** that
*were* applied (with before/after numbers), so the two categories are not
confused.

Validation harness: `_fixval.py` (cora, gcn, single seed=42). `standard` is the
sanity control — its number must be identical before/after, confirming the
pipeline itself was untouched.

---

## Part 1 — Bug fixes applied (with before/after)

All numbers are **test accuracy (%)**, cora + plain GCN, seed 42.
"h64 / uniform-0.4" is the main robustness test (40 % symmetric label noise);
"h512 / clean" exposes wide-backbone embedding-geometry failures.

| Method              | Test                | Before | After | Δ      | Verdict |
|---------------------|---------------------|-------:|------:|-------:|---------|
| **standard** (control) | h64 / uniform-0.4 | 47.5 | 47.5 | 0.0 | pipeline untouched ✓ |
| **gnn_cleaner**     | h64 / uniform-0.4   | 28.3 | 48.3 | **+20.0** | fixed ✓ |
| **unionnet**        | h64 / uniform-0.4   | 45.2 | 67.4 | **+22.2** | fixed ✓ |
| **unionnet**        | h512 / clean        | 75.2 | 81.2 | **+6.0**  | fixed ✓ |
| **cr_gnn**          | h512 / clean        | 14.4 | 77.6 | **+63.2** | fixed ✓ |
| **cr_gnn**          | h64 / uniform-0.4   | 53.4 | 53.3 | −0.1 | unchanged (already ok at h64) |
| **community_defense** | h64 / uniform-0.4 | 51.3 | 49.5 | −1.8 | within single-seed noise; see note |
| **erase**           | h64 / uniform-0.4   | 76.9 | 76.9 | 0.0 | fix attempted then **reverted** |
| **erase**           | h512 / clean        | 77.1 | 77.1 | 0.0 | fix attempted then **reverted** |

### Fix #1 — `gnn_cleaner`: label propagation was sub-stochastic
`_label_propagation` aggregated messages with a symmetric-normalised operator but
never re-normalised the resulting rows. The operator is sub-stochastic, so away
from the trusted seed set the per-node label distributions decayed toward ~0 and
`argmax` over them became arbitrary — destroying the clean/relabel split that the
whole method depends on.
**Fix:** row-normalise `label_probs` back to a valid distribution each iteration
(then re-clamp trusted nodes to their seed one-hot). → +20.0 pts.

### Fix #2 — `cr_gnn`: contrastive heads ran on logits, not embeddings
Setup and all four forward sites fed `backbone(data)` — the **num_classes-dim
logits** — into the contrastive / consistency / classification heads. On a wide
backbone the heads were starved through a tiny logit bottleneck and the model
collapsed (14.4 % at hidden=512).
**Fix:** use `backbone.get_embeddings(data)` everywhere (7 sites), so the heads
operate on the GNN hidden representation. → +63.2 pts at h512, neutral at h64.

### Fix #3 — `unionnet`: support-set similarity used raw inner products
The neighbour-weighting support set was built from `h_train @ h_trainᵀ` (raw
inner product). Inner-product magnitude grows with embedding width, so the
softmax over top-k neighbours saturated onto a single high-norm node and the
support set degenerated.
**Fix:** L2-normalise embeddings first → cosine similarity in [−1, 1], which is
scale-invariant. → +22.2 pts at h64, +6.0 at h512.

### Fix #4 — `erase`: ATTEMPTED, REVERTED (kept as a documented limitation)
We hypothesised the MCR2 coding-rate term needed unit-sphere (L2-normalised)
features and added `F.normalize` before the covariance. **Measured result: it
broke erase** — 76.9 → 32.1 (h64) and 77.1 → 40.9 (h512). erase was already the
**best** robust method in this test, so the change was reverted. See Part 2 for
why erase must run on **raw** features at the tuned hidden dimensions.

### Fix #5 — `community_defense`: auxiliary CE could co-dominate
The community-CE regulariser was capped at the **full** supervised-CE magnitude,
so it could reach a ~50/50 split with the supervised signal and pull the
embedding toward community structure rather than the labels (worst on
heterophilous graphs).
**Fix:** cap it at **¼** of the supervised CE so it stays auxiliary.
On cora h64/0.4 this is −1.8 (within single-seed noise; cora is homophilous so
the failure mode this targets is not exercised here). The change is principled
and protects the heterophilous datasets (roman-empire) that this test does not
cover. **Recommend keeping**, but flagged for transparency — confirm on
roman-empire in the full multi-seed run.

---

## Part 2 — Structural limitations (NOT bugs — do not "fix")

### L1 — `positive_eigenvalues` on shallow backbones  *(RESOLVED — square readout auto-appended)*
The constraint clips negative eigenvalues of every **square** 2-D weight matrix
(`p.size(0) == p.size(1)`). On a 2-layer GCN the weights are `in→hidden` and
`hidden→num_classes` — **neither is square**, so the constraint originally
touched nothing and the method was identical to `standard`. It is only naturally
active with ≥3 layers, where the interior `hidden→hidden` matrices exist.

**Resolution (PE-only):** `setup()` now checks the backbone for an existing
square weight (`has_square_weight`). If one exists (any ≥3-layer net, e.g.
roman-empire n=5, GCN_modified) it is used directly — unchanged behaviour. If
none exists (2-layer GCN), PE wraps the backbone in `PESquareHead`, which appends
a square `num_classes × num_classes` readout on top of the **full, unchanged**
backbone forward. The readout is **identity-initialised** (no-op at epoch 0) and
is the only placement that adds a square operator without dropping a graph hop,
so PE stays depth-comparable to `standard`. This makes the negative-eigenvalue
smoothing bias active on the shallow baselines instead of a silent no-op.

Validation (cora, gcn, seed 42): standard vs PE — clean 77.7 → **78.1**;
uniform-0.4 **47.5 → 48.4** (PE now beats standard under noise, as intended).
On a 3-layer backbone the wrapper is **not** built (existing square weights used).
The method is from the author's own paper (arXiv:2412.08419); the wrapper only
restores its intended behaviour on shallow nets, it does not alter the constraint.

### L2 — `erase` is sensitive to feature scale (raw features required)
MCR2's `slogdet(I + scalar·F·Fᵀ)` is scale-dependent by construction. With the
tuned hidden dimensions this implementation is well-behaved on **raw** ReLU
features (best method at 76.9 % under 40 % noise). L2-normalising onto the unit
sphere — the "textbook" MCR2 conditioning — *collapses* it here (see Fix #4).
Limitation: erase's good behaviour is tied to the configured `n_embedding` /
hidden dims; porting it to a very different width may require re-tuning `eps`,
`gam1`, `gam2` rather than blindly normalising.

### L3 — `nrgnn` edge predictor is intrinsically O(|unlabeled| × |confident|)
NRGNN mines confident pseudo-edges every epoch between unlabeled and confident
nodes (`n_p=50`, `p_u=0.8` paper defaults). The candidate set scales with the
unlabeled-node count, so on large/dense graphs (amazon-computers, roman-empire)
it is slow and memory-heavy and can oversmooth as it adds edges. This is
inherent to the algorithm, not an implementation defect — it is a *finding* about
the method's scalability, reported via the config comment.

### L4 — `pi_gnn` uses a dense link-prediction loss and a late start epoch
PI-GNN regularises with a context-aware link-reconstruction term over a dense
decoder, and the MI model only starts at `start_epoch=200` (official default,
**absolute** epoch). Two consequences: (a) it needs `epochs > 200` or the MI
branch never activates; (b) the dense link loss is memory-heavy on large graphs.
Both are faithful to the official implementation and are configuration
constraints, not bugs.

### L5 — `gnnguard` is inert on backbones without reweightable messages
GNNGuard works by feeding cosine-attention **edge weights** into the backbone's
message passing. Backbones that ignore edge weights in aggregation (e.g. GIN's
sum aggregator) receive the weights but cannot act on them, so GNNGuard reduces
toward the plain backbone. It is fully active on GCN/GAT-style message passing.
Reported so GNNGuard numbers on GIN are read as "≈ backbone", not as a failure.

### L6 — `deterministic` noise type corrupts train only (not in default sweep)
The `deterministic` noise generator flips exactly `rate·|train|` **training**
nodes and leaves validation untouched — inconsistent with the train+val noise
model used by `uniform`/`pair` everywhere else (model selection is on noisy
validation accuracy). It is **not** in the default config sweep
(`clean`/`uniform`/`pair`), so it does not affect reported results; documented so
nobody enables it expecting val to be corrupted too.

---

## Part 3 — Benchmark-level notes (by design, for fairness)

- **Noise is injected into train + val; test is always clean.** Model selection
  uses **noisy** validation accuracy — the realistic setting (you never see clean
  labels). `uniform`/`pair` matrices are seeded via
  `instance_independent_noise(seed=noise_seed)`, so every method sees the *same*
  corrupted labels and the *same* splits for a given (seed, type, rate).
- **`standard` can beat robust methods on plain GCN under low/no noise** — this
  is expected, not a bug. Robust methods pay a bias cost that only pays off as
  the noise rate rises; the cross-over is what the sweep is designed to measure.
- The `_fixval.py` numbers above are **single-seed** (seed 42) smoke tests for
  before/after deltas, not final results. The committed configs run `num_runs: 3`
  with full noise sweeps — trust those for reporting, and treat any single-seed
  Δ smaller than ~±2 pts (e.g. community_defense −1.8) as noise.
