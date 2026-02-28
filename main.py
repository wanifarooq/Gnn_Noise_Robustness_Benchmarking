import argparse
import gc
import numpy as np
import torch
import yaml
import json
import os
from codecarbon import EmissionsTracker

from util.experiment import run_experiment
from util.cli import print_table, fmt_mean_std
from model.evaluation import OVERSMOOTHING_KEYS, ZERO_CLS
from sweep_utils import expand_yaml_sweeps, get_result_filename, detect_completed_runs, json_serializer

DEFAULT_CONFIG = "config.yaml"
NOISE_SPLIT_KEYS = ('train_only_clean', 'train_only_mislabelled_factual', 'train_only_mislabelled_corrected')
CLS_SPLITS = ('test', 'train', 'val') + NOISE_SPLIT_KEYS


def _accumulate_run_result(run_result, classification_runs, oversmoothing_runs, compute_runs):
    """Shared accumulation path for both disk-loaded and live run results.

    Appends the per-split classification, oversmoothing, and compute metrics
    from a single run into the corresponding accumulator dicts.
    """
    for split in CLS_SPLITS:
        src = run_result.get(f'{split}_cls', dict(ZERO_CLS))
        for mkey in classification_runs[split]:
            classification_runs[split][mkey].append(float(src[mkey]))

    for split, key_prefix in (('test', 'test_oversmoothing'), ('train', 'train_oversmoothing'), ('val', 'val_oversmoothing')):
        src = run_result.get(key_prefix, {})
        for okey in OVERSMOOTHING_KEYS:
            raw = src.get(okey, float('nan'))
            if isinstance(raw, torch.Tensor) and raw.dim() == 0:
                raw = raw.item()
            elif isinstance(raw, (list, np.ndarray, torch.Tensor)):
                raw = float(np.mean([float(x) for x in raw]))
            oversmoothing_runs[split][okey].append(float(raw))

    compute_info = run_result.get('compute_info', {})
    for ckey in compute_runs:
        compute_runs[ckey].append(float(compute_info.get(ckey, float('nan'))))


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
    parser.add_argument('--force', action='store_true',
                        help='Re-run all experiments from scratch, ignoring completed runs')
    return parser.parse_args()


def run_benchmarking(base_folder='results', config_path=DEFAULT_CONFIG,
                     eval_only=False, no_checkpoint=False, num_runs=None,
                     force=False):
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
        file_name = get_result_filename(sweep_config)
        experiment_dir = os.path.join(base_folder, file_name)
        result_json_path = os.path.join(experiment_dir, 'experiment.json')

        os.makedirs(experiment_dir, exist_ok=True)

        # CLI flags override per-config values
        save_checkpoint = (not no_checkpoint) and sweep_config.get('save_checkpoint', True)
        run_eval_only = eval_only or sweep_config.get('eval_only', False)
        n_runs = num_runs if num_runs is not None else sweep_config.get('num_runs', 5)
        force = force or bool(sweep_config.get("force", False))

        # ── Incremental run detection ──
        # Scan existing run_N/training_log.json to find already-completed runs.
        # When force=True, ignore previous results and re-execute everything.
        if force or run_eval_only:
            completed = {}
            if force:
                print(f"[FORCE] Re-running all {n_runs} runs: {experiment_dir}")
            elif run_eval_only:
                print(f"[EVAL-ONLY] Re-evaluating all {n_runs} runs: {experiment_dir}")
        else:
            completed = detect_completed_runs(experiment_dir, n_runs)

        runs_needed = [r for r in range(1, n_runs + 1) if r not in completed]

        # Skip if every requested run is already complete
        if not runs_needed and os.path.exists(result_json_path):
            print(f"[SKIP] All {n_runs} runs complete: {result_json_path}")
            continue

        if completed and runs_needed:
            print(f"[RESUME] {len(completed)}/{n_runs} runs complete, running {len(runs_needed)} more")
        elif not runs_needed:
            # All runs done but experiment.json missing — rebuild it below
            print(f"[REBUILD] All {n_runs} runs complete, rebuilding experiment.json")

        # Device selection
        requested_device = sweep_config.get("device", "cpu")
        if requested_device == "cuda":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = requested_device

        print(f"Using device: {device_str}")
        if device_str == "cuda":
            print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

        classification_runs = {
            split: {k: [] for k in ('accuracy', 'f1', 'precision', 'recall')}
            for split in CLS_SPLITS
        }
        oversmoothing_runs = {
            split: {k: [] for k in OVERSMOOTHING_KEYS}
            for split in ('test', 'train', 'val')
        }
        compute_runs = {k: [] for k in ('flops_inference', 'flops_training_total',
                                         'time_training_total', 'time_inference')}

        # Pre-populate accumulators from completed runs (in run_id order)
        for rid in sorted(completed):
            _accumulate_run_result(completed[rid], classification_runs, oversmoothing_runs, compute_runs)

        # Execute only the missing runs
        codecarbon_file_name = "emissions.csv"
        for run in runs_needed:
            try:
                tracker = EmissionsTracker(output_dir=experiment_dir,
                                          output_file=codecarbon_file_name,
                                          log_level="critical",
                                          allow_multiple_runs=True)
                tracker.start()
                run_codecarbon = True
            except Exception as e:
                if run == runs_needed[0]:
                    print(f"[WARNING] Could not start EmissionsTracker: {e}")
                    print("[WARNING] Carbon emissions data will be missing from results.")
                    print("[WARNING] On macOS, codecarbon requires sudo access to read hardware power metrics.")
                run_codecarbon = False
            print(f"\nRun {run}/{n_runs}:")
            run_dir = os.path.join(experiment_dir, f"run_{run}")
            ckpt_path = (os.path.join(experiment_dir, f"best_run_{run}.pt")
                         if save_checkpoint or run_eval_only else None)
            run_result = run_experiment(
                sweep_config, run_id=run,
                checkpoint_path=ckpt_path, eval_only=run_eval_only,
                run_dir=run_dir if not run_eval_only else None,
            )
            if run_codecarbon:
                tracker.stop()

            _accumulate_run_result(run_result, classification_runs, oversmoothing_runs, compute_runs)

            print(f"Run {run} completed - Test Acc: {float(run_result['test_cls']['accuracy']):.4f}, F1: {float(run_result['test_cls']['f1']):.4f}, "
                  f"Train: {float(run_result['compute_info']['time_training_total']):.2f}s, Eval: {float(run_result['compute_info']['time_inference']):.2f}s")
            del run_result
            if device_str == "cuda":
                torch.cuda.empty_cache()

        # ── Print summary tables ──
        cls_headers = ['split', 'accuracy', 'f1', 'precision', 'recall']
        cls_rows = []
        for split in ('test', 'train', 'val'):
            cls_rows.append([split] + [fmt_mean_std(classification_runs[split][m]) for m in ('accuracy', 'f1', 'precision', 'recall')])
        print("\nClassification Metrics:")
        print_table(cls_headers, cls_rows)

        if any(any(v != 0.0 for v in classification_runs[s]['accuracy']) for s in NOISE_SPLIT_KEYS):
            cn_rows = []
            for split in NOISE_SPLIT_KEYS:
                cn_rows.append([split] + [fmt_mean_std(classification_runs[split][m])
                                for m in ('accuracy', 'f1', 'precision', 'recall')])
            print("\nClassification Metrics (Train noise splits):")
            print_table(cls_headers, cn_rows)

        os_headers = ['split'] + list(OVERSMOOTHING_KEYS)
        os_rows = []
        for split in ('test', 'train', 'val'):
            os_rows.append([split] + [fmt_mean_std(oversmoothing_runs[split][k]) for k in OVERSMOOTHING_KEYS])
        print("\nOversmoothing Metrics:")
        print_table(os_headers, os_rows)

        print("\nCompute Metrics:")
        for ckey in compute_runs:
            print(f"  {ckey}: {fmt_mean_std(compute_runs[ckey])}")
        print("-"*50)

        all_metrics = {
            "config": sweep_config,
            "classification": classification_runs,
            "oversmoothing": oversmoothing_runs,
            "compute": compute_runs,
        }

        # Save results
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=4, default=json_serializer)

        print(f"[SAVED] {result_json_path}")
        gc.collect()
        if device_str == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    run_benchmarking(config_path=args.config,
                     eval_only=args.eval_only,
                     no_checkpoint=args.no_checkpoint,
                     num_runs=args.num_runs,
                     force=args.force)
