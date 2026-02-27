"""Training visualisation utilities."""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_training_plots(epoch_log: list[dict], output_dir: str,
                        *, label: str = '') -> None:
    """Save training metrics (loss, accuracy, F1) plot to output_dir."""
    if not epoch_log:
        return

    epochs = [e['epoch'] for e in epoch_log]
    has_f1 = any(e.get('train_f1') is not None for e in epoch_log)
    ncols = 3 if has_f1 else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    title = f'Training Metrics — {label}' if label else 'Training Metrics'
    fig.suptitle(title, fontsize=14)

    # Loss
    axes[0].plot(epochs, [e.get('train_loss') or 0 for e in epoch_log], label='Train', linewidth=2)
    axes[0].plot(epochs, [e.get('val_loss') or 0 for e in epoch_log], label='Val', linewidth=2)
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [e.get('train_acc') or 0 for e in epoch_log], label='Train', linewidth=2)
    axes[1].plot(epochs, [e.get('val_acc') or 0 for e in epoch_log], label='Val', linewidth=2)
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 (only if any method logged it)
    if has_f1:
        axes[2].plot(epochs, [e.get('train_f1') or 0 for e in epoch_log], label='Train', linewidth=2)
        axes[2].plot(epochs, [e.get('val_f1') or 0 for e in epoch_log], label='Val', linewidth=2)
        axes[2].set(xlabel='Epoch', ylabel='F1', title='F1')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'training_metrics_{label}.png' if label else 'training_metrics.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)


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
    title = f'Oversmoothing Metrics — {label}' if label else 'Oversmoothing Metrics'
    fig.suptitle(title, fontsize=14)

    for idx, name in enumerate(metric_names):
        ax = axes[idx]
        for split, color in colors.items():
            values = [os_data.get(split, {}).get(name)
                      for _, os_data in os_entries]
            if any(v is not None for v in values):
                ax.plot(os_epochs, [v or 0 for v in values],
                        color=color, label=split.capitalize(),
                        linewidth=2, alpha=0.8)
        ax.set(xlabel='Epoch', ylabel=name, title=name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'oversmoothing_metrics_{label}.png' if label else 'oversmoothing_metrics.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
