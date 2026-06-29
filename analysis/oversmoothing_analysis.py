"""Oversmoothing-vs-performance analysis plots.

Relates structural metrics (NumRank, Erank, MAD, EDir, EProj) to test accuracy
across the robustness methods, to reveal which structural signature predicts
good performance for a given (dataset, backbone, noise) slice.

Produces three figure types:
  (1) scatter      -- metric (x) vs test accuracy (y), one panel per metric,
                      with Spearman rho + linear trend. Answers "is accuracy a
                      function of this metric?".
  (2) parallel     -- parallel-coordinates over the normalized metrics, one line
                      per model, coloured by test accuracy. Shows which
                      multi-metric *signature* the high-accuracy models share.
  (3) heatmap      -- Spearman rho(metric, accuracy) across models, for every
                      (backbone x noise) slice of a dataset. Summarises which
                      metric most predicts accuracy and how consistent it is.

Usage:
  python analysis/oversmoothing_analysis.py [dataset] [backbone] [noise]
  e.g.  python analysis/oversmoothing_analysis.py cora gcn_modified uniform-0.30
Outputs go to analysis/plots/.
"""
import json, glob, re, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, 'results')
OUTDIR = os.path.join(HERE, 'plots')
os.makedirs(OUTDIR, exist_ok=True)

METRICS = ['NumRank', 'Erank', 'MAD', 'EDir', 'EProj']   # test-split structural metrics
LOG_METRICS = {'EDir'}                                    # span orders of magnitude
SPLIT = 'test'                                            # oversmoothing + accuracy split

LABEL = {'standard': 'Standard', 'gcod': 'GCOD', 'positive_eigenvalues': 'PosEigen',
         'nrgnn': 'NRGNN', 'pi_gnn': 'PI-GNN', 'cr_gnn': 'CR-GNN', 'rtgnn': 'RTGNN',
         'unionnet': 'UnionNET', 'graphcleaner': 'GraphCleaner', 'gnn_cleaner': 'GNN-Cleaner',
         'community_defense': 'CommDefense', 'erase': 'ERASE', 'gnnguard': 'GNNGuard'}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load(dataset):
    """Return {(backbone, noise): {method: {'acc','acc_std', metric: mean,...}}}."""
    data = {}
    pat = re.compile(rf'[0-9a-f]+_{re.escape(dataset)}_(gcn_modified|gcn|gat|gatv2|gin|gps)_(.+?)_42_noise-(\w+)-([\d.]+)$')
    for f in glob.glob(os.path.join(RESULTS, '*', 'experiment.json')):
        m = pat.match(os.path.basename(os.path.dirname(f)))
        if not m:
            continue
        bb, meth, nt, rate = m.groups()
        noise = 'clean' if nt == 'clean' else f'{nt}-{rate}'
        try:
            d = json.load(open(f))
        except Exception:
            continue
        acc = d.get('classification', {}).get(SPLIT, {}).get('accuracy')
        osm = d.get('oversmoothing', {}).get(SPLIT, {})
        if not acc or not osm:
            continue
        rec = {'acc': np.mean(acc) * 100, 'acc_std': np.std(acc) * 100,
               'acc_runs': [a * 100 for a in acc]}
        for mk in METRICS:
            if mk in osm:
                rec[mk] = float(np.mean(osm[mk]))
                rec[mk + '_runs'] = list(osm[mk])   # per-seed values (aligned with acc_runs)
        data.setdefault((bb, noise), {})[meth] = rec
    return data


def spearman(x, y):
    """Spearman rho via Pearson on ranks (average ranks for ties)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan

    def rank(a):
        order = a.argsort()
        r = np.empty(len(a), float)
        r[order] = np.arange(1, len(a) + 1)
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- #
# (1) Scatter: metric vs accuracy
# --------------------------------------------------------------------------- #
def plot_scatter(slice_data, dataset, bb, noise):
    methods = list(slice_data)
    accs = np.array([slice_data[m]['acc'] for m in methods])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, mk in zip(axes, METRICS):
        xs = np.array([slice_data[m].get(mk, np.nan) for m in methods])
        ok = ~np.isnan(xs)
        xv, yv, mv = xs[ok], accs[ok], [methods[i] for i in range(len(methods)) if ok[i]]
        ax.scatter(xv, yv, s=70, c=yv, cmap='RdYlGn', edgecolors='k', zorder=3)
        for x, y, mm in zip(xv, yv, mv):
            ax.annotate(LABEL.get(mm, mm), (x, y), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')
        rho = spearman(xv, yv)
        # trend line (linear fit in plotting space)
        if mk in LOG_METRICS:
            ax.set_xscale('log')
            fx = np.log10(np.clip(xv, 1e-6, None))
        else:
            fx = xv
        if len(xv) > 2 and np.ptp(fx) > 0:
            a, b = np.polyfit(fx, yv, 1)
            xx = np.linspace(fx.min(), fx.max(), 50)
            ax.plot(10**xx if mk in LOG_METRICS else xx, a * xx + b,
                    '--', color='gray', lw=1.2, zorder=2)
        ax.set_xlabel(f'{mk}' + (' (log)' if mk in LOG_METRICS else ''))
        ax.set_ylabel(f'{SPLIT} accuracy (%)')
        ax.set_title(f'{mk}  (Spearman $\\rho$={rho:+.2f})', fontsize=11)
        ax.grid(alpha=0.3)
    axes[-1].axis('off')
    fig.suptitle(f'Structural metric vs accuracy — {dataset} | {bb} | {noise}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTDIR, f'scatter_metric_acc_{dataset}_{bb}_{noise}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# (2) Parallel coordinates coloured by accuracy
# --------------------------------------------------------------------------- #
def plot_parallel(slice_data, dataset, bb, noise):
    methods = list(slice_data)
    accs = np.array([slice_data[m]['acc'] for m in methods])
    # build normalized matrix [n_methods x n_metrics]
    M = np.full((len(methods), len(METRICS)), np.nan)
    for i, m in enumerate(methods):
        for j, mk in enumerate(METRICS):
            v = slice_data[m].get(mk, np.nan)
            if mk in LOG_METRICS and v is not None and v > 0:
                v = np.log10(v)
            M[i, j] = v
    # min-max normalize each column to [0,1]
    lo, hi = np.nanmin(M, 0), np.nanmax(M, 0)
    rng = np.where(hi - lo == 0, 1, hi - lo)
    Mn = (M - lo) / rng

    fig, ax = plt.subplots(figsize=(11, 6.5))
    norm = Normalize(vmin=accs.min(), vmax=accs.max())
    cmap = plt.get_cmap('RdYlGn')
    xpos = np.arange(len(METRICS))
    order = np.argsort(accs)            # draw low-acc first so high-acc on top
    for i in order:
        ax.plot(xpos, Mn[i], color=cmap(norm(accs[i])), lw=2, alpha=0.85,
                marker='o', ms=4)
        ax.annotate(f'{LABEL.get(methods[i], methods[i])} ({accs[i]:.0f})',
                    (xpos[-1], Mn[i, -1]), fontsize=6.5, xytext=(4, 0),
                    textcoords='offset points', va='center')
    ax.set_xticks(xpos)
    ax.set_xticklabels([mk + (' (log)' if mk in LOG_METRICS else '') for mk in METRICS])
    ax.set_ylabel('normalized value (min–max per metric)')
    ax.set_title(f'Structural signature per model (coloured by {SPLIT} acc.) — '
                 f'{dataset} | {bb} | {noise}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    ax.set_xlim(-0.3, len(METRICS) - 0.3 + 1.2)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.12)
    cb.set_label(f'{SPLIT} accuracy (%)')
    fig.tight_layout()
    out = os.path.join(OUTDIR, f'parallel_coords_{dataset}_{bb}_{noise}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# (3) Correlation heatmap across (backbone x noise) slices
# --------------------------------------------------------------------------- #
def plot_heatmap(data, dataset, per_run=False):
    """Spearman rho(metric, accuracy) per (backbone x noise) slice.

    per_run=False: one point per method (mean over seeds), N=#methods.
    per_run=True : one point per (method, seed), pairing each run's metric with
                   the same run's accuracy, N up to #methods x #seeds.
    """
    slices = sorted(data.keys())
    col_lbl = [f'{bb}\n{noise}' for bb, noise in slices]
    H = np.full((len(METRICS), len(slices)), np.nan)
    Ncol = [0] * len(slices)
    for j, sl in enumerate(slices):
        sd = data[sl]
        methods = list(sd)
        for i, mk in enumerate(METRICS):
            xs, ys = [], []
            for m in methods:
                if per_run:
                    mr, ar = sd[m].get(mk + '_runs'), sd[m].get('acc_runs')
                    if not mr or not ar:
                        continue
                    k = min(len(mr), len(ar))
                    xs += list(mr[:k]); ys += list(ar[:k])
                else:
                    v = sd[m].get(mk)
                    if v is None or np.isnan(v):
                        continue
                    xs.append(v); ys.append(sd[m]['acc'])
            if len(xs) >= 3:
                H[i, j] = spearman(np.array(xs), np.array(ys))
            Ncol[j] = max(Ncol[j], len(xs))
    fig, ax = plt.subplots(figsize=(1.7 * len(slices) + 2, 4.7))
    im = ax.imshow(H, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(slices)))
    ax.set_xticklabels([f'{c}\n(N={n})' for c, n in zip(col_lbl, Ncol)], fontsize=8)
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels(METRICS)
    for i in range(len(METRICS)):
        for j in range(len(slices)):
            if not np.isnan(H[i, j]):
                ax.text(j, i, f'{H[i, j]:+.2f}', ha='center', va='center',
                        fontsize=8, color='black')
    mode = 'per-run (each seed a point)' if per_run else 'per-method (mean over seeds)'
    ax.set_title(f'Spearman $\\rho$(metric, {SPLIT} accuracy) — {dataset}  [{mode}]',
                 fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Spearman $\\rho$')
    fig.tight_layout()
    suffix = '_perrun' if per_run else ''
    out = os.path.join(OUTDIR, f'corr_heatmap_{dataset}{suffix}.png')
    fig.savefig(out, dpi=95, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'cora'
    bb = sys.argv[2] if len(sys.argv) > 2 else 'gcn_modified'
    noise = sys.argv[3] if len(sys.argv) > 3 else 'uniform-0.30'
    data = load(dataset)
    if not data:
        print(f'No results found for dataset={dataset}'); return
    print('Available slices:', sorted(data.keys()))
    outs = []
    if (bb, noise) in data:
        outs.append(plot_scatter(data[(bb, noise)], dataset, bb, noise))
        outs.append(plot_parallel(data[(bb, noise)], dataset, bb, noise))
    else:
        print(f'Slice ({bb},{noise}) not found; skipping scatter/parallel.')
    outs.append(plot_heatmap(data, dataset, per_run=False))
    outs.append(plot_heatmap(data, dataset, per_run=True))
    print('Wrote:')
    for o in outs:
        print('  ' + o)


if __name__ == '__main__':
    main()
