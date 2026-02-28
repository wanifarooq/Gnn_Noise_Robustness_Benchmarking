"""Training visualisation utilities."""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# (suffix, label, color, linestyle)
_NOISE_SPLITS = [
    ('clean_only',                'Train clean only',              'tab:blue', '--'),
    ('only_mislabelled_factual',  'Train mislabelled (factual)',   'tab:red',  '-'),
    ('only_mislabelled_corrected','Train mislabelled (corrected)', 'tab:red',  '--'),
]


def _metric_lines(metric):
    """Build the full line list (train + val + noise splits) for *metric*."""
    lines = [
        (f'train_{metric}', 'Train', 'tab:blue',   '-', 2, 1.0),
        (f'val_{metric}',   'Val',   'tab:orange', '-', 2, 1.0),
    ]
    for suffix, label, color, ls in _NOISE_SPLITS:
        lines.append((f'train_{metric}_{suffix}', label, color, ls, 1.5, 0.8))
    return lines


def _plot_subplot(ax, epochs, epoch_log, lines, ylabel, title):
    """Draw *lines* on *ax* and apply standard formatting."""
    for key, label, color, linestyle, linewidth, alpha in lines:
        values = [e.get(key) for e in epoch_log]
        if any(v is not None for v in values):
            ax.plot(epochs, [v if v is not None else 0 for v in values],
                    color=color, linestyle=linestyle, label=label,
                    linewidth=linewidth, alpha=alpha)
    ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _save_figure(fig, output_dir, basename, label):
    """tight_layout + savefig + close."""
    fig.tight_layout()
    fname = f'{basename}_{label}.png' if label else f'{basename}.png'
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_training_plots(epoch_log: list[dict], output_dir: str,
                        *, label: str = '') -> None:
    """Save training metrics (loss, accuracy, F1) plot to output_dir."""
    if not epoch_log:
        return

    epochs = [e['epoch'] for e in epoch_log]

    subplots = [
        ('Loss', 'Loss', [
            ('train_loss', 'Train', 'tab:blue',   '-', 2, 1.0),
            ('val_loss',   'Val',   'tab:orange', '-', 2, 1.0),
        ]),
        ('Accuracy', 'Accuracy', _metric_lines('acc')),
    ]
    if any(e.get('train_f1') is not None for e in epoch_log):
        subplots.append(('F1', 'F1', _metric_lines('f1')))

    ncols = len(subplots)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    title = f'Training Metrics - {label}' if label else 'Training Metrics'
    fig.suptitle(title, fontsize=14)

    for ax, (ylabel, plot_title, lines) in zip(axes, subplots):
        _plot_subplot(ax, epochs, epoch_log, lines, ylabel, plot_title)

    _save_figure(fig, output_dir, 'training_metrics', label)


def save_oversmoothing_plots(epoch_log: list[dict], output_dir: str,
                             *, label: str = '') -> None:
    """Save oversmoothing metrics (EDir, MAD, NumRank) plot to output_dir.

    Only the three most diagnostic metrics are plotted; the full set
    (EDir_traditional, EProj, Erank) is available in training_log.json.
    """
    os_entries = [(e['epoch'], e['oversmoothing'])
                  for e in epoch_log if e.get('oversmoothing')]
    if not os_entries:
        return

    os_epochs = [ep for ep, _ in os_entries]
    metric_names = ['EDir', 'MAD', 'NumRank']
    colors = {'train': 'tab:blue', 'val': 'tab:red'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    title = f'Oversmoothing Metrics - {label}' if label else 'Oversmoothing Metrics'
    fig.suptitle(title, fontsize=14)

    for idx, name in enumerate(metric_names):
        ax = axes[idx]
        for split, color in colors.items():
            values = [os_data.get(split, {}).get(name)
                      for _, os_data in os_entries]
            if any(v is not None for v in values):
                ax.plot(os_epochs, [v if v is not None else 0 for v in values],
                        color=color, label=split.capitalize(),
                        linewidth=2, alpha=0.8)
        ax.set(xlabel='Epoch', ylabel=name, title=name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    _save_figure(fig, output_dir, 'oversmoothing_metrics', label)
