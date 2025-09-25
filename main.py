import numpy as np
import torch
import yaml

from utilities import run_experiment

def run_benchmarking():
    print("\n" + "-"*50)
    print("Multi-run experiment: 5 runs")
    print("-"*50)
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Loaded configuration file")

    method_name = config['training']['method']
    dataset_name = config['dataset']['name']
    noise_rate = config['noise']['rate']
    
    print(f"Dataset: {dataset_name}")
    print(f"Method: {method_name}")
    print(f"Noise Rate: {noise_rate}")
    device_str = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.device(device_str))}")

    print(f"Running 5 experiments")
    print("-"*50)
    
    test_accuracies = []
    test_f1s = []
    test_precisions = []
    test_recalls = []

    oversmoothing_metrics = {
        'NumRank': [],
        'Erank': [],
        'EDir': [],
        'EDir_traditional': [],
        'EProj': [],
        'MAD': []
    }

    for run in range(1, 6):
        try:
            print(f"\nRun {run}/5:")
            test_metrics = run_experiment(config, run_id=run)
            test_acc = test_metrics['accuracy']
            test_f1 = test_metrics['f1']
            test_precision = test_metrics['precision']
            test_recall = test_metrics['recall']
            test_overs = test_metrics['oversmoothing']

            test_accuracies.append(test_acc.item())
            test_f1s.append(test_f1.item())
            test_precisions.append(test_precision.item())
            test_recalls.append(test_recall.item())

            for key in oversmoothing_metrics:
                oversmoothing_metrics[key].append(test_overs[key])

            print(f"Run {run} completed - Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}, "
                  f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
            
        except Exception as e:
            print(f"Run {run} failed with error: {str(e)}")
            continue

        print("-"*50)
        
    if test_accuracies:
        mean_std_dict = {
            'Accuracy': (np.mean(test_accuracies), np.std(test_accuracies)),
            'F1': (np.mean(test_f1s), np.std(test_f1s)),
            'Precision': (np.mean(test_precisions), np.std(test_precisions)),
            'Recall': (np.mean(test_recalls), np.std(test_recalls))
        }

        print("\n" + "-"*50)
        print("Final results")
        print("-"*50)
        for metric, (mean_val, std_val) in mean_std_dict.items():
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("\nOversmoothing Metrics:")
        for key in oversmoothing_metrics:
            mean_val = np.mean(oversmoothing_metrics[key])
            std_val = np.std(oversmoothing_metrics[key])
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")

        print("-"*50)
    else:
        print("\nERROR: No successful runs completed!")
        print("-"*50)

if __name__ == "__main__":
    run_benchmarking()
