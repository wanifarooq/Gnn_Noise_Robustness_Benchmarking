#!/usr/bin/env python3
"""Standalone script to compare GNN robustness methods from experiment results.

Reads experiment.json (and optionally training_log.json) from the results/
directory, groups experiments by (dataset, noise, mode), and generates
comparison plots + a printed summary table.

Usage:
    python analysis/compare_methods.py [--results-dir results] [--out-dir analysis/plots]
"""

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ── Data loading ────────────────────────────────────────────────────────────

def load_experiments(results_dir):
    """Scan results_dir for experiment.json files and return parsed records."""
    experiments = []
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    for folder in sorted(results_path.iterdir()):
        if not folder.is_dir():
            continue
        exp_file = folder / 'experiment.json'
        if not exp_file.exists():
            print(f"  [skip] {folder.name} — no experiment.json")
            continue

        try:
            with open(exp_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] {folder.name} — failed to read experiment.json: {e}")
            continue

        config = data.get('config', {})
        training_cfg = config.get('training', {})
        noise_cfg = config.get('noise', {})
        dataset_cfg = config.get('dataset', {})

        record = {
            'folder': folder.name,
            'folder_path': str(folder),
            'method': training_cfg.get('method', 'unknown'),
            'dataset': dataset_cfg.get('name', 'unknown'),
            'model': config.get('model', {}).get('name', 'unknown'),
            'noise_type': noise_cfg.get('type', 'none'),
            'noise_rate': noise_cfg.get('rate', 0.0),
            'mode': training_cfg.get('mode', 'transductive'),
            'num_runs': config.get('num_runs', 1),
            'classification': data.get('classification', {}),
            'oversmoothing': data.get('oversmoothing', {}),
            'compute': data.get('compute', {}),
        }

        # Load training logs per run (for training curves)
        training_logs = []
        for run_dir in sorted(folder.glob('run_*')):
            log_file = run_dir / 'training_log.json'
            if log_file.exists():
                try:
                    with open(log_file) as f:
                        training_logs.append(json.load(f))
                except (json.JSONDecodeError, OSError):
                    pass
        record['training_logs'] = training_logs

        experiments.append(record)

    return experiments


def group_experiments(experiments):
    """Group experiments by (dataset, noise_type, noise_rate, mode, model)."""
    groups = defaultdict(list)
    for exp in experiments:
        key = (exp['dataset'], exp['noise_type'], exp['noise_rate'],
               exp['mode'], exp['model'])
        groups[key].append(exp)
    return groups


def group_label(key):
    dataset, noise_type, noise_rate, mode, model = key
    return f"{dataset} | {model} | {noise_type}-{noise_rate} | {mode}"


# ── Helper: aggregate across runs ──────────────────────────────────────────

def mean_std(values):
    """Return (mean, std) from a list of numbers."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


def fmt_mean_std(values):
    """Format as 'mean +/- std' string, or 'mean' if only one run."""
    m, s = mean_std(values)
    if len(values) <= 1 or s == 0:
        return f'{m:.4f}'
    return f'{m:.4f} +/- {s:.4f}'


def get_metric(exp, split, metric):
    """Get list of per-run values for classification[split][metric]."""
    return exp.get('classification', {}).get(split, {}).get(metric, [])


def get_oversmoothing(exp, split, metric):
    """Get list of per-run values for oversmoothing[split][metric]."""
    return exp.get('oversmoothing', {}).get(split, {}).get(metric, [])


# ── Summary table (printed to console) ────────────────────────────────────

def print_summary_table(group_exps, group_key):
    """Print a formatted comparison table for a group of experiments."""
    label = group_label(group_key)
    methods = [e['method'] for e in group_exps]

    # Collect rows of data
    rows = []
    for exp in group_exps:
        logs = exp.get('training_logs', [])
        best_ep = logs[0].get('best_epoch', '-') if logs else '-'
        stopped = logs[0].get('stopped_at_epoch', '-') if logs else '-'
        compute = exp.get('compute', {})
        time_m, _ = mean_std(compute.get('time_training_total', []))

        rows.append({
            'Method': exp['method'],
            'Test Acc': fmt_mean_std(get_metric(exp, 'test', 'accuracy')),
            'Test F1': fmt_mean_std(get_metric(exp, 'test', 'f1')),
            'Val Acc': fmt_mean_std(get_metric(exp, 'val', 'accuracy')),
            'Train Acc': fmt_mean_std(get_metric(exp, 'train', 'accuracy')),
            'Val Clean': fmt_mean_std(get_metric(exp, 'val_only_clean', 'accuracy')),
            'Val Mislbl': fmt_mean_std(get_metric(exp, 'val_only_mislabelled_corrected', 'accuracy')),
            'MAD': fmt_mean_std(get_oversmoothing(exp, 'test', 'MAD')),
            'Time(s)': f'{time_m:.1f}',
            'Best Ep': str(best_ep),
            'Stop Ep': str(stopped),
        })

    # Compute column widths
    columns = list(rows[0].keys())
    widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in columns}

    # Print
    print(f'\n{"=" * 80}')
    print(f'  {label}')
    print(f'{"=" * 80}')

    # Header
    header = ' | '.join(c.ljust(widths[c]) for c in columns)
    print(header)
    print('-+-'.join('-' * widths[c] for c in columns))

    # Rows
    for r in rows:
        print(' | '.join(r[c].ljust(widths[c]) for c in columns))

    print()


# ── Plot 1: Test Accuracy & F1 bar chart ───────────────────────────────────

def plot_test_performance(group_exps, group_key, out_dir):
    """Bar chart of test accuracy and F1 across methods."""
    methods = [e['method'] for e in group_exps]
    acc_means, acc_stds = [], []
    f1_means, f1_stds = [], []

    for exp in group_exps:
        m, s = mean_std(get_metric(exp, 'test', 'accuracy'))
        acc_means.append(m)
        acc_stds.append(s)
        m, s = mean_std(get_metric(exp, 'test', 'f1'))
        f1_means.append(m)
        f1_stds.append(s)

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), 6))
    bars1 = ax.bar(x - width / 2, acc_means, width, yerr=acc_stds,
                   label='Accuracy', capsize=4, color='#4C72B0')
    bars2 = ax.bar(x + width / 2, f1_means, width, yerr=f1_stds,
                   label='F1 Score', capsize=4, color='#DD8452')

    ax.set_ylabel('Score')
    ax.set_title(f'Test Performance — {group_label(group_key)}')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Value labels on bars
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fname = _safe_filename(group_key, 'test_performance')
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f'  Saved: {fname}')


# ── Plot 2: Train vs Val vs Test accuracy ──────────────────────────────────

def plot_split_comparison(group_exps, group_key, out_dir):
    """Grouped bar chart showing train/val/test accuracy per method."""
    methods = [e['method'] for e in group_exps]
    splits = ['train', 'val', 'test']
    colors = ['#55A868', '#C44E52', '#4C72B0']

    data = {s: [] for s in splits}
    errs = {s: [] for s in splits}
    for exp in group_exps:
        for s in splits:
            m, sd = mean_std(get_metric(exp, s, 'accuracy'))
            data[s].append(m)
            errs[s].append(sd)

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), 6))
    for i, s in enumerate(splits):
        ax.bar(x + (i - 1) * width, data[s], width, yerr=errs[s],
               label=s.capitalize(), capsize=3, color=colors[i])

    ax.set_ylabel('Accuracy')
    ax.set_title(f'Train / Val / Test Accuracy — {group_label(group_key)}')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    fname = _safe_filename(group_key, 'split_comparison')
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f'  Saved: {fname}')


# ── Plot 3: Noise robustness (clean vs mislabelled accuracy) ──────────────

def plot_noise_robustness(group_exps, group_key, out_dir):
    """Bar chart comparing clean-only vs mislabelled-corrected accuracy."""
    methods = [e['method'] for e in group_exps]

    # Train noise splits
    train_clean_means, train_clean_stds = [], []
    train_mis_means, train_mis_stds = [], []
    # Val noise splits
    val_clean_means, val_clean_stds = [], []
    val_mis_means, val_mis_stds = [], []

    has_data = False
    for exp in group_exps:
        tc = get_metric(exp, 'train_only_clean', 'accuracy')
        tm = get_metric(exp, 'train_only_mislabelled_corrected', 'accuracy')
        vc = get_metric(exp, 'val_only_clean', 'accuracy')
        vm = get_metric(exp, 'val_only_mislabelled_corrected', 'accuracy')
        if tc:
            has_data = True

        m, s = mean_std(tc)
        train_clean_means.append(m); train_clean_stds.append(s)
        m, s = mean_std(tm)
        train_mis_means.append(m); train_mis_stds.append(s)
        m, s = mean_std(vc)
        val_clean_means.append(m); val_clean_stds.append(s)
        m, s = mean_std(vm)
        val_mis_means.append(m); val_mis_stds.append(s)

    if not has_data:
        return

    x = np.arange(len(methods))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(methods) * 2), 6))

    # Train split
    ax1.bar(x - width / 2, train_clean_means, width, yerr=train_clean_stds,
            label='Clean', capsize=3, color='#55A868')
    ax1.bar(x + width / 2, train_mis_means, width, yerr=train_mis_stds,
            label='Mislabelled (corrected)', capsize=3, color='#C44E52')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Train: Clean vs Mislabelled')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha='right')
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3)

    # Val split
    ax2.bar(x - width / 2, val_clean_means, width, yerr=val_clean_stds,
            label='Clean', capsize=3, color='#55A868')
    ax2.bar(x + width / 2, val_mis_means, width, yerr=val_mis_stds,
            label='Mislabelled (corrected)', capsize=3, color='#C44E52')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Val: Clean vs Mislabelled')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha='right')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Noise Robustness — {group_label(group_key)}', fontsize=12)
    fig.tight_layout()

    fname = _safe_filename(group_key, 'noise_robustness')
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f'  Saved: {fname}')


# ── Plot 4: Training curves (val accuracy over epochs) ────────────────────

def _average_epoch_logs(training_logs, key):
    """Average a metric across multiple runs, aligned by epoch index.

    Returns (epochs, means, stds) arrays.  Runs may have different lengths;
    averaging stops at the shortest run.
    """
    all_series = []
    for log in training_logs:
        epoch_log = log.get('epoch_log', [])
        if not epoch_log:
            continue
        all_series.append([e.get(key, 0) or 0 for e in epoch_log])

    if not all_series:
        return [], [], []

    min_len = min(len(s) for s in all_series)
    arr = np.array([s[:min_len] for s in all_series])  # (num_runs, min_len)
    epochs = list(range(min_len))
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    return epochs, means, stds


def plot_training_curves(group_exps, group_key, out_dir):
    """Line plot of val accuracy and train loss over epochs, averaged across runs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for exp in group_exps:
        logs = exp.get('training_logs', [])
        if not logs:
            continue

        label = exp['method']

        # Val accuracy (averaged across runs with shaded std)
        epochs, means, stds = _average_epoch_logs(logs, 'val_acc')
        if epochs:
            line, = axes[0].plot(epochs, means, label=label, alpha=0.8)
            if len(logs) > 1:
                axes[0].fill_between(epochs, means - stds, means + stds,
                                     alpha=0.15, color=line.get_color())

        # Train loss (averaged across runs with shaded std)
        epochs, means, stds = _average_epoch_logs(logs, 'train_loss')
        if epochs:
            line, = axes[1].plot(epochs, means, label=label, alpha=0.8)
            if len(logs) > 1:
                axes[1].fill_between(epochs, means - stds, means + stds,
                                     alpha=0.15, color=line.get_color())

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Validation Accuracy over Training')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Training Loss')
    axes[1].set_title('Training Loss over Training')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    n_runs = max((len(e.get('training_logs', [])) for e in group_exps), default=1)
    fig.suptitle(
        f'Training Curves (avg over {n_runs} run{"s" if n_runs > 1 else ""}) '
        f'— {group_label(group_key)}', fontsize=12)
    fig.tight_layout()

    fname = _safe_filename(group_key, 'training_curves')
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f'  Saved: {fname}')


# ── Plot 5: Oversmoothing metrics comparison ──────────────────────────────

def plot_oversmoothing(group_exps, group_key, out_dir):
    """Bar charts for key oversmoothing metrics across methods."""
    methods = [e['method'] for e in group_exps]
    os_metrics = ['MAD', 'NumRank', 'EDir']
    split = 'test'

    fig, axes = plt.subplots(1, len(os_metrics),
                             figsize=(5 * len(os_metrics), 5))
    if len(os_metrics) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, os_metrics):
        means, stds = [], []
        for exp in group_exps:
            vals = get_oversmoothing(exp, split, metric_name)
            m, s = mean_std(vals)
            means.append(m)
            stds.append(s)

        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, capsize=4, color='#8172B2')
        ax.set_title(f'{metric_name} ({split})')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Oversmoothing Metrics — {group_label(group_key)}', fontsize=12)
    fig.tight_layout()

    fname = _safe_filename(group_key, 'oversmoothing')
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f'  Saved: {fname}')


# ── Plot 6: Compute comparison ─────────────────────────────────────────────

def plot_compute(group_exps, group_key, out_dir):
    """Bar chart comparing training time and inference FLOPs."""
    methods = [e['method'] for e in group_exps]

    time_means, time_stds = [], []
    flops_means, flops_stds = [], []

    for exp in group_exps:
        compute = exp.get('compute', {})
        m, s = mean_std(compute.get('time_training_total', []))
        time_means.append(m)
        time_stds.append(s)
        m, s = mean_std(compute.get('flops_inference', []))
        flops_means.append(m)
        flops_stds.append(s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(methods))

    ax1.bar(x, time_means, yerr=time_stds, capsize=4, color='#64B5CD')
    ax1.set_ylabel('Seconds')
    ax1.set_title('Training Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(time_means):
        if v > 0:
            ax1.text(i, v + time_stds[i] + 0.5, f'{v:.1f}s',
                     ha='center', va='bottom', fontsize=8)

    ax2.bar(x, [f / 1e6 for f in flops_means], capsize=4, color='#CCB974')
    ax2.set_ylabel('MFLOPs')
    ax2.set_title('Inference FLOPs')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Compute Comparison — {group_label(group_key)}', fontsize=12)
    fig.tight_layout()

    fname = _safe_filename(group_key, 'compute')
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f'  Saved: {fname}')


# ── Plot 7: Summary table as image ────────────────────────────────────────

def plot_summary_table(group_exps, group_key, out_dir):
    """Render the summary table as a clean image."""
    columns = ['Method', 'Test Acc', 'Test F1', 'Val Acc', 'Train Acc',
               'Val Clean', 'Val Mislbl', 'MAD', 'Time(s)']

    cell_data = []
    for exp in group_exps:
        compute = exp.get('compute', {})
        time_m, _ = mean_std(compute.get('time_training_total', []))
        cell_data.append([
            exp['method'],
            fmt_mean_std(get_metric(exp, 'test', 'accuracy')),
            fmt_mean_std(get_metric(exp, 'test', 'f1')),
            fmt_mean_std(get_metric(exp, 'val', 'accuracy')),
            fmt_mean_std(get_metric(exp, 'train', 'accuracy')),
            fmt_mean_std(get_metric(exp, 'val_only_clean', 'accuracy')),
            fmt_mean_std(get_metric(exp, 'val_only_mislabelled_corrected', 'accuracy')),
            fmt_mean_std(get_oversmoothing(exp, 'test', 'MAD')),
            f'{time_m:.1f}',
        ])

    fig, ax = plt.subplots(figsize=(max(14, len(columns) * 1.8),
                                    1.2 + len(cell_data) * 0.5))
    ax.axis('off')

    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#4C72B0')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(len(cell_data)):
        color = '#F0F0F0' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(columns)):
            table[i + 1, j].set_facecolor(color)

    # Highlight best test accuracy
    if len(cell_data) > 1:
        test_accs = [mean_std(get_metric(e, 'test', 'accuracy'))[0]
                     for e in group_exps]
        best_idx = int(np.argmax(test_accs))
        for j in range(len(columns)):
            table[best_idx + 1, j].set_text_props(fontweight='bold')
            table[best_idx + 1, j].set_edgecolor('#55A868')
            table[best_idx + 1, j].set_linewidth(2)

    ax.set_title(f'Method Comparison — {group_label(group_key)}',
                 fontsize=13, fontweight='bold', pad=20)
    fig.tight_layout()

    fname = _safe_filename(group_key, 'summary_table')
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')


# ── Utilities ──────────────────────────────────────────────────────────────

def _safe_filename(group_key, plot_name):
    dataset, noise_type, noise_rate, mode, model = group_key
    return f'{plot_name}_{dataset}_{model}_{noise_type}-{noise_rate}_{mode}.png'


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare GNN robustness methods from experiment results.')
    parser.add_argument('--results-dir', default='results',
                        help='Path to results directory (default: results)')
    parser.add_argument('--out-dir', default='analysis/plots',
                        help='Output directory for plots (default: analysis/plots)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading experiments from: {args.results_dir}')
    experiments = load_experiments(args.results_dir)
    print(f'Found {len(experiments)} experiment(s)')

    if not experiments:
        print('No experiments found. Run some experiments first.')
        sys.exit(0)

    # Group and generate comparison plots + tables
    groups = group_experiments(experiments)

    for group_key, group_exps in groups.items():
        label = group_label(group_key)
        print(f'\n--- {label} ({len(group_exps)} method(s)) ---')

        # Always print the console table
        print_summary_table(group_exps, group_key)

        if len(group_exps) < 2:
            print('  Skipping comparison plots (need >= 2 methods)')
            plot_training_curves(group_exps, group_key, args.out_dir)
            continue

        plot_summary_table(group_exps, group_key, args.out_dir)
        plot_test_performance(group_exps, group_key, args.out_dir)
        plot_split_comparison(group_exps, group_key, args.out_dir)
        plot_noise_robustness(group_exps, group_key, args.out_dir)
        plot_training_curves(group_exps, group_key, args.out_dir)
        plot_oversmoothing(group_exps, group_key, args.out_dir)
        plot_compute(group_exps, group_key, args.out_dir)

    print('\nDone!')


if __name__ == '__main__':
    main()
