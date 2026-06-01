# Benchmark Limitations & Method Caveats

This document records the **structural limitations** of the benchmark and of
individual robustness methods — the things that are *intrinsic* to the method or
the experimental setup and therefore are **not bugs to be patched**, but must be
understood when reading the results. 


## Part 1 — Structural limitations (NOT bugs)

### L1 — `positive_eigenvalues` on shallow backbones  *(RESOLVED — square readout auto-appended)*
The constraint clips negative eigenvalues of every **square** 2-D weight matrix
(`p.size(0) == p.size(1)`). On a 2-layer GCN the weights are `in→hidden` and
`hidden→num_classes` — **neither is square**, so the constraint originally
touched nothing and the method should be identical to `standard`. It is only naturally
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


### L2 — `erase` is sensitive to feature scale (raw features required)
MCR2's `slogdet(I + scalar·F·Fᵀ)` is scale-dependent by construction. With the
tuned hidden dimensions this implementation is well-behaved on **raw** ReLU
features (best method at 76.9 % under 40 % noise). L2-normalising onto the unit
sphere — the "textbook" MCR2 conditioning — *collapses*.
Limitation: erase's good behaviour is tied to the configured `n_embedding` /
hidden dims; porting it to a very different width may require re-tuning `eps`,
`gam1`, `gam2` rather than blindly normalising.

### L3 — `nrgnn` edge predictor is intrinsically O(|unlabeled| × |confident|)
NRGNN mines confident pseudo-edges every epoch between unlabeled and confident
nodes (`n_p=50`, `p_u=0.8` paper defaults). The candidate set scales with the
unlabeled-node count, so on large/dense graphs (amazon-computers, roman-empire)
it is slow and memory-heavy and can oversmooth as it adds edges. This is
inherent to the algorithm, not an implementation defect — it is a *finding* about
the method's scalability.

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

## Part 2 — Benchmark-level notes (by design, for fairness)

- **Noise is injected into train + val; test is always clean.** Model selection
  uses **noisy** validation accuracy — the realistic setting (you never see clean
  labels). `uniform`/`pair` matrices are seeded via
  `instance_independent_noise(seed=noise_seed)`, so every method sees the *same*
  corrupted labels and the *same* splits for a given (seed, type, rate).
- **`standard` can beat robust methods on plain GCN under low/no noise** — this
  is expected, not a bug. Robust methods pay a bias cost that only pays off as
  the noise rate rises; the cross-over is what the sweep is designed to measure.
