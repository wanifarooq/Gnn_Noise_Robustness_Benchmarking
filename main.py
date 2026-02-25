import argparse
import numpy as np
import torch
import yaml
import json
import os
from codecarbon import EmissionsTracker

from util.experiment import run_experiment
from model.evaluation import OVERSMOOTHING_KEYS
from sweep_utils import expand_yaml_sweeps, get_config_hash, should_run_experiment, json_serializer

DEFAULT_CONFIG = "config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description='Run multi-run benchmark sweeps')
    parser.add_argument('--config', '-c', default=DEFAULT_CONFIG,
                        help=f'Path to YAML config file (default: {DEFAULT_CONFIG})')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, evaluate from saved checkpoints')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Disable saving model checkpoints after training')
    parser.add_argument('--num-runs', type=int, default=None,
                        help='Number of runs per config (default: from config or 5)')
    return parser.parse_args()


def run_benchmarking(base_folder='results', config_path=DEFAULT_CONFIG,
                     eval_only=False, no_checkpoint=False, num_runs=None):
    run_codecarbon = False
    print("\n" + "-"*50)
    print("Multi-run experiment with parameter sweep")
    print("-"*50)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Loaded configuration file")

    configs = expand_yaml_sweeps(config)
    print(f"Number of sweep configurations: {len(configs)}")
    os.makedirs(base_folder, exist_ok=True)

    for idx, sweep_config in enumerate(configs, 1):
        print(f"\n=== Sweep Config {idx}/{len(configs)} ===")

        # Decide output path *before* running
        file_name = get_config_hash(sweep_config)
        file_path = os.path.join(base_folder, file_name)
        result_json_path = f"{file_path}.json"

        # Skip if already computed (unless force: true or eval-only)
        if not eval_only and not should_run_experiment(result_json_path, sweep_config):
            continue

        # Device selection (more explicit)
        requested_device = sweep_config.get("device", "cpu")
        if requested_device == "cuda":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = requested_device

        print(f"Using device: {device_str}")
        if device_str == "cuda":
            print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

        test_metrics_runs = {
            'accuracy': [], 'f1': [], 'precision': [], 'recall': [],
        }
        train_metrics_runs = {
            'accuracy': [], 'f1': [], 'precision': [], 'recall': [],
        }
        val_metrics_runs = {
            'accuracy': [], 'f1': [], 'precision': [], 'recall': [],
        }
        compute_metrics_runs = {
            'flops_inference': [], 'flops_training_total': [],
            'time_training_total': [], 'time_inference': [],
        }
        oversmoothing_metrics = {
            f'{key}{suffix}': []
            for suffix in ('', '-Train', '-Val')
            for key in OVERSMOOTHING_KEYS
        }

        # CLI flags override per-config values
        save_checkpoint = (not no_checkpoint) and sweep_config.get('save_checkpoint', True)
        run_eval_only = eval_only or sweep_config.get('eval_only', False)

        n_runs = num_runs or sweep_config.get('num_runs', 5)
        codecarbon_file_name = f"{file_name}_emissions.csv"
        for run in range(1, n_runs + 1):
            try:
                tracker = EmissionsTracker(output_dir=base_folder,
                                          output_file=codecarbon_file_name,
                                          log_level="critical",
                                          allow_multiple_runs=True)
                tracker.start()
                run_codecarbon = True
            except Exception as e:
                if run == 1:
                    print(f"[WARNING] Could not start EmissionsTracker: {e}")
                    print("[WARNING] Carbon emissions data will be missing from results.")
                    print("[WARNING] On macOS, codecarbon requires sudo access to read hardware power metrics.")
                run_codecarbon = False
            print(f"\nRun {run}/{n_runs}:")
            ckpt_path = (os.path.join(base_folder, f"{file_name}_run_{run}.pt")
                         if save_checkpoint or run_eval_only else None)
            test_metrics = run_experiment(
                sweep_config, run_id=run,
                checkpoint_path=ckpt_path, eval_only=run_eval_only,
            )
            if run_codecarbon:
                tracker.stop()
            # Convert possible numpy/torch scalars to Python floats
            for mkey in ('accuracy', 'f1', 'precision', 'recall'):
                test_metrics_runs[mkey].append(float(test_metrics['test_cls'][mkey]))
                train_metrics_runs[mkey].append(float(test_metrics['train_cls'][mkey]))
                val_metrics_runs[mkey].append(float(test_metrics['val_cls'][mkey]))
            for ckey in compute_metrics_runs:
                compute_metrics_runs[ckey].append(float(test_metrics['compute_info'][ckey]))

            for key in oversmoothing_metrics:
                if '-Train' in key:
                    base_key = key.replace('-Train', '')
                    source = test_metrics.get('train_oversmoothing', {})
                    raw_val = source.get(base_key, float('nan'))
                elif '-Val' in key:
                    base_key = key.replace('-Val', '')
                    source = test_metrics.get('val_oversmoothing', {})
                    raw_val = source.get(base_key, float('nan'))
                else:
                    raw_val = test_metrics['oversmoothing'][key]

                # Check if raw_val is a list (or iterable)
                if isinstance(raw_val, list) or (isinstance(raw_val, (np.ndarray, torch.Tensor)) and hasattr(raw_val, '__len__') and len(raw_val) > 1):
                    # It is a list (e.g., layer-wise metrics). Take the mean.
                    # We map float() first to handle lists of Tensors
                    clean_mean = np.mean([float(x) for x in raw_val])
                    oversmoothing_metrics[key].append(float(clean_mean))
                else:
                    # It is a single scalar
                    oversmoothing_metrics[key].append(float(raw_val))
            print(f"Run {run} completed - Test Acc: {float(test_metrics['test_cls']['accuracy']):.4f}, F1: {float(test_metrics['test_cls']['f1']):.4f}, "
                  f"Train: {float(test_metrics['compute_info']['time_training_total']):.2f}s, Eval: {float(test_metrics['compute_info']['time_inference']):.2f}s")

        # Compute mean ± std (store as plain floats to simplify JSON)
        mean_std_dict = {
            'Accuracy': [float(np.mean(test_metrics_runs['accuracy'])), float(np.std(test_metrics_runs['accuracy']))],
            'F1': [float(np.mean(test_metrics_runs['f1'])), float(np.std(test_metrics_runs['f1']))],
            'Precision': [float(np.mean(test_metrics_runs['precision'])), float(np.std(test_metrics_runs['precision']))],
            'Recall': [float(np.mean(test_metrics_runs['recall'])), float(np.std(test_metrics_runs['recall']))],
            'Train_Accuracy': [float(np.mean(train_metrics_runs['accuracy'])), float(np.std(train_metrics_runs['accuracy']))],
            'Train_F1': [float(np.mean(train_metrics_runs['f1'])), float(np.std(train_metrics_runs['f1']))],
            'Train_Precision': [float(np.mean(train_metrics_runs['precision'])), float(np.std(train_metrics_runs['precision']))],
            'Train_Recall': [float(np.mean(train_metrics_runs['recall'])), float(np.std(train_metrics_runs['recall']))],
            'Val_Accuracy': [float(np.mean(val_metrics_runs['accuracy'])), float(np.std(val_metrics_runs['accuracy']))],
            'Val_F1': [float(np.mean(val_metrics_runs['f1'])), float(np.std(val_metrics_runs['f1']))],
            'Val_Precision': [float(np.mean(val_metrics_runs['precision'])), float(np.std(val_metrics_runs['precision']))],
            'Val_Recall': [float(np.mean(val_metrics_runs['recall'])), float(np.std(val_metrics_runs['recall']))],
            'flops_inference': [float(np.mean(compute_metrics_runs['flops_inference'])), float(np.std(compute_metrics_runs['flops_inference']))],
            'flops_training_total': [float(np.mean(compute_metrics_runs['flops_training_total'])), float(np.std(compute_metrics_runs['flops_training_total']))],
            'time_training_total': [float(np.mean(compute_metrics_runs['time_training_total'])), float(np.std(compute_metrics_runs['time_training_total']))],
            'time_inference': [float(np.mean(compute_metrics_runs['time_inference'])), float(np.std(compute_metrics_runs['time_inference']))],
        }

        print("\nSweep Config Results:")
        for metric, (mean_val, std_val) in mean_std_dict.items():
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

        print("\nOversmoothing Metrics:")
        oversmoothing_summary = {}
        for key in oversmoothing_metrics:
            mean_val = float(np.mean(oversmoothing_metrics[key]))
            std_val = float(np.std(oversmoothing_metrics[key]))
            oversmoothing_summary[key] = [mean_val, std_val]
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")

        print("\nCompute Metrics:")
        for key in compute_metrics_runs:
            mean_val = float(np.mean(compute_metrics_runs[key]))
            std_val = float(np.std(compute_metrics_runs[key]))
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
        print("-"*50)

        all_metrics = {
            "config_hash": file_name,
            "config": sweep_config,
            "test_metrics_runs": test_metrics_runs,
            "train_metrics_runs": train_metrics_runs,
            "val_metrics_runs": val_metrics_runs,
            "test_metrics_mean_std": mean_std_dict,
            "oversmoothing_metrics_runs": oversmoothing_metrics,
            "oversmoothing_metrics_mean_std": oversmoothing_summary,
            "compute_metrics_runs": compute_metrics_runs,
        }

        # Save results
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=4, default=json_serializer)

        print(f"[SAVED] {result_json_path}")


if __name__ == "__main__":
    args = parse_args()
    run_benchmarking(config_path=args.config,
                     eval_only=args.eval_only,
                     no_checkpoint=args.no_checkpoint,
                     num_runs=args.num_runs)
