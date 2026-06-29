"""Can test accuracy be modelled as a linear regression on the structural
metrics (NumRank, Erank, MAD, EProj -- EDir excluded as scale-confounded)?

For each (backbone x noise) slice we fit
    acc ~ b0 + b1*z(NumRank) + b2*z(Erank) + b3*z(MAD) + b4*z(EProj)
on the per-run points (each method x seed = one point, N up to 39), and report
R^2 / adjusted R^2 and standardized coefficients. We also fit one pooled model
per backbone (noise as control dummies) for more statistical power, and report
predictor collinearity (which makes individual coefficients unreliable even when
R^2 is high).

Usage:  python analysis/metric_regression.py [dataset]
Outputs a printed report + analysis/plots/regression_pred_vs_actual_<dataset>.png
"""
import json, glob, re, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, 'results')
OUTDIR = os.path.join(HERE, 'plots')
os.makedirs(OUTDIR, exist_ok=True)

PREDICTORS = ['NumRank', 'Erank', 'MAD', 'EProj']   # EDir excluded
SPLIT = 'test'


def load_points(dataset):
    """Return {(backbone,noise): {'X': [N x P], 'y': [N], 'methods':[...]}} using
    per-run points (metric run i paired with accuracy run i)."""
    pat = re.compile(rf'[0-9a-f]+_{re.escape(dataset)}_(gcn_modified|gcn|gat|gatv2|gin|gps)_(.+?)_42_noise-(\w+)-([\d.]+)$')
    raw = {}
    for f in glob.glob(os.path.join(RESULTS, '*', 'experiment.json')):
        mm = pat.match(os.path.basename(os.path.dirname(f)))
        if not mm:
            continue
        bb, meth, nt, rate = mm.groups()
        noise = 'clean' if nt == 'clean' else f'{nt}-{rate}'
        try:
            d = json.load(open(f))
        except Exception:
            continue
        acc = d.get('classification', {}).get(SPLIT, {}).get('accuracy')
        osm = d.get('oversmoothing', {}).get(SPLIT, {})
        if not acc or any(p not in osm for p in PREDICTORS):
            continue
        k = min([len(acc)] + [len(osm[p]) for p in PREDICTORS])
        for i in range(k):
            row = [osm[p][i] for p in PREDICTORS]
            raw.setdefault((bb, noise), {'X': [], 'y': [], 'm': []})
            raw[(bb, noise)]['X'].append(row)
            raw[(bb, noise)]['y'].append(acc[i] * 100)
            raw[(bb, noise)]['m'].append(meth)
    return raw


def zscore(M):
    mu, sd = M.mean(0), M.std(0)
    sd = np.where(sd == 0, 1, sd)
    return (M - mu) / sd


def ols(X, y):
    """Return (beta_with_intercept, R2, adjR2, yhat). X already z-scored."""
    n, p = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan
    return beta, r2, adj, yhat


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'cora'
    raw = load_points(dataset)
    if not raw:
        print(f'No results for {dataset}'); return

    print(f'\n=== Multiple regression: {SPLIT} acc ~ '
          f'{" + ".join("z("+p+")" for p in PREDICTORS)}  ({dataset}) ===')
    hdr = f"{'slice':26s} {'N':>4s} {'R2':>6s} {'adjR2':>7s} | " + \
          " ".join(f'{p:>8s}' for p in PREDICTORS)
    print(hdr); print('-' * len(hdr))

    fits = {}
    for sl in sorted(raw):
        X = np.array(raw[sl]['X'], float)
        y = np.array(raw[sl]['y'], float)
        if len(y) < len(PREDICTORS) + 2:
            continue
        Xz = zscore(X)
        beta, r2, adj, yhat = ols(Xz, y)
        fits[sl] = (y, yhat, r2)
        coefs = " ".join(f'{b:+8.2f}' for b in beta[1:])
        print(f"{(sl[0]+' '+sl[1]):26s} {len(y):4d} {r2:6.2f} {adj:7.2f} | {coefs}")

    # collinearity among predictors (pooled, z-scored)
    allX = zscore(np.vstack([np.array(raw[sl]['X'], float) for sl in raw]))
    C = np.corrcoef(allX, rowvar=False)
    print('\nPredictor collinearity (|r|>0.8 => coefficients unreliable):')
    print('        ' + ' '.join(f'{p:>8s}' for p in PREDICTORS))
    for i, p in enumerate(PREDICTORS):
        print(f'{p:8s}' + ' '.join(f'{C[i,j]:+8.2f}' for j in range(len(PREDICTORS))))

    # pooled-per-backbone model (noise as control dummies) for more power
    print('\nPooled per backbone (noise-level controlled via dummies):')
    bbs = sorted({bb for bb, _ in raw})
    for bb in bbs:
        sls = [sl for sl in raw if sl[0] == bb]
        X = np.vstack([np.array(raw[sl]['X'], float) for sl in sls])
        y = np.hstack([np.array(raw[sl]['y'], float) for sl in sls])
        # noise dummies
        noises = sum([[sl[1]] * len(raw[sl]['y']) for sl in sls], [])
        uniq = sorted(set(noises))[1:]   # drop first as reference
        D = np.array([[1.0 if nz == u else 0.0 for u in uniq] for nz in noises])
        Xz = zscore(X)
        Xfull = np.hstack([Xz, D]) if D.size else Xz
        beta, r2, adj, _ = ols(Xfull, y)
        print(f"  {bb:14s} N={len(y):4d}  R2={r2:.2f}  adjR2={adj:.2f}  "
              f"(metric betas: " + ", ".join(f'{p}={b:+.2f}' for p, b in zip(PREDICTORS, beta[1:1+len(PREDICTORS)])) + ")")

    # predicted vs actual figure (one panel per slice)
    sls = sorted(fits)
    ncol = 3
    nrow = int(np.ceil(len(sls) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
    axes = np.array(axes).ravel()
    for ax, sl in zip(axes, sls):
        y, yhat, r2 = fits[sl]
        ax.scatter(y, yhat, s=28, alpha=0.7, edgecolors='k', lw=0.3)
        lo, hi = min(y.min(), yhat.min()), max(y.max(), yhat.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
        ax.set_title(f'{sl[0]} | {sl[1]}  ($R^2$={r2:.2f})', fontsize=10)
        ax.set_xlabel('actual acc (%)'); ax.set_ylabel('predicted')
        ax.grid(alpha=0.3)
    for ax in axes[len(sls):]:
        ax.axis('off')
    fig.suptitle(f'Predicted vs actual accuracy from structural metrics — {dataset}',
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTDIR, f'regression_pred_vs_actual_{dataset}.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
