import numpy as np
import torch
import yaml
import json
import os
from codecarbon import EmissionsTracker

from utilities import run_experiment
from sweep_utils import *

def run_benchmarking(base_folder='results'):
    run_codecarbon = False
    print("\n" + "-"*50)
    print("Multi-run experiment with parameter sweep")
    print("-"*50)

    with open("config1.yaml", "r", encoding="utf-8") as f:
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

        # Skip if already computed (unless force: true)
        if not should_run_experiment(result_json_path, sweep_config):
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

        test_accuracies, test_f1s, test_precisions, test_recalls, test_flops = [], [], [], [], []
        oversmoothing_metrics = {
            'NumRank': [], 'Erank': [], 'EDir': [],
            'EDir_traditional': [], 'EProj': [], 'MAD': [],
            'NumRank-Train': [], 'Erank-Train': [], 'EDir-Train': [],
            'EDir_traditional-Train': [], 'EProj-Train': [], 'MAD-Train': []
        }

        for run in range(1, 6):
            codecarbon_file_name = f"{file_name}_emissions.csv"
            if not os.path.exists(os.path.join(base_folder, codecarbon_file_name)):
                tracker = EmissionsTracker(output_dir=base_folder,
                                        output_file = codecarbon_file_name,
                                        log_level = "critical", allow_multiple_runs=True)
                tracker.start()
                run_codecarbon = True
            print(f"\nRun {run}/5:")
            test_metrics = run_experiment(sweep_config, run_id=run)
            if run_codecarbon:
                tracker.stop()
                run_codecarbon = False
            # Convert possible numpy/torch scalars to Python floats
            test_accuracies.append(float(test_metrics['accuracy']))
            test_f1s.append(float(test_metrics['f1']))
            test_precisions.append(float(test_metrics['precision']))
            test_recalls.append(float(test_metrics['recall']))
            test_flops.append(float(test_metrics['flops_info']['total_flops']))

            # for key in oversmoothing_metrics:
            #     if '-Train' in key:
            #         oversmoothing_metrics[key].append(test_metrics['train_oversmoothing'][key.replace('-Train', '')])
            #     else:    
            #         oversmoothing_metrics[key].append(test_metrics['oversmoothing'][key])

            for key in oversmoothing_metrics:
                if '-Train' in key:
                    raw_val = test_metrics['train_oversmoothing'][key.replace('-Train', '')]
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
            print(f"Run {run} completed - Test Acc: {float(test_metrics['accuracy']):.4f}, F1: {float(test_metrics['f1']):.4f}")

        # Compute mean ± std (store as plain floats to simplify JSON)
        mean_std_dict = {
            'Accuracy': [float(np.mean(test_accuracies)), float(np.std(test_accuracies))],
            'F1': [float(np.mean(test_f1s)), float(np.std(test_f1s))],
            'Precision': [float(np.mean(test_precisions)), float(np.std(test_precisions))],
            'Recall': [float(np.mean(test_recalls)), float(np.std(test_recalls))],
            'FLOPS': [float(np.mean(test_flops)), float(np.std(test_flops))],
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
        print("-"*50)

        all_metrics = {
            "config_hash": file_name,
            "config": sweep_config, 
            "test_metrics_mean_std": mean_std_dict,
            "oversmoothing_metrics_runs": oversmoothing_metrics,
            "oversmoothing_metrics_mean_std": oversmoothing_summary
        }

        # Save results
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=4, default=json_serializer)

        print(f"[SAVED] {result_json_path}")


if __name__ == "__main__":
    run_benchmarking()
