import torch
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import time

from utilities import run_experiment, parse_arguments, print_table

def run_single_experiment_fixed_seed(method_name, config, fixed_run_id=1):

    try:

        experiment_config = copy.deepcopy(config)
        experiment_config['training']['method'] = method_name
        
        print(f"\n[{method_name}] Starting with fixed run_id={fixed_run_id}")
        test_metrics = run_experiment(experiment_config, run_id=fixed_run_id)
        print(f"[{method_name}] Completed - Test Acc: {test_metrics['accuracy']:.4f}, "
              f"F1: {test_metrics['f1']:.4f}")
        
        return method_name, test_metrics
    except Exception as e:
        print(f"[{method_name}] Failed with error: {str(e)}")
        return method_name, None

def run_parallel_single_benchmark():
    args = parse_arguments()
    
    print("\n" + "-"*50)
    print("Parallel single-run")
    print("-"*50)
    
    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)
    
    MAX_METHODS = 4
    methods_to_test = args.methods[:MAX_METHODS]
    
    if len(args.methods) > MAX_METHODS:
        print(f"Warning: Maximum {MAX_METHODS} methods allowed. Using first {MAX_METHODS}: {methods_to_test}")
    
    FIXED_RUN_ID = args.run_id
    
    print(f"Dataset: {base_config['dataset']['name']}")
    print(f"Noise Rate: {base_config['noise']['rate']}")
    print(f"Methods: {methods_to_test}")
    print(f"Fixed run_id: {FIXED_RUN_ID} (same seed for all)")
    
    if torch.cuda.is_available():
        max_workers = min(2, len(methods_to_test))
    else:
        max_workers = min(4, len(methods_to_test))
    
    print(f"Using {max_workers} parallel workers")
    print("-"*50)
    
    start_time = time.time()
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_method = {
            executor.submit(run_single_experiment_fixed_seed, method, base_config, FIXED_RUN_ID): method
            for method in methods_to_test
        }
        
        for future in as_completed(future_to_method):
            method_name, test_metrics = future.result()
            if test_metrics is not None:
                results[method_name] = test_metrics
    
    end_time = time.time()
    
    print("\n" + "-"*50)
    print("Results")
    print("-"*50)
    print(f"Total time: {end_time - start_time:.2f}s")
    print()
    
    if results:

        headers = ["Method", "Accuracy", "F1", "Precision", "Recall"]
        rows = []
        for method in methods_to_test:
            if method in results:
                r = results[method]
                rows.append([
                    method,
                    f"{r['accuracy']:.4f}",
                    f"{r['f1']:.4f}",
                    f"{r['precision']:.4f}",
                    f"{r['recall']:.4f}",
                ])
            else:
                rows.append([method, "FAILED", "FAILED", "FAILED", "FAILED"])

        col_widths = [max(len(str(x)) for x in col) for col in zip(*([headers] + rows))]
        print_table(headers, rows, col_widths)

        headers_os = ["Method", "NumRank", "Erank", "EDir", "EDir_trad", "EProj", "MAD"]
        rows_os = []
        for method in methods_to_test:
            if method in results:
                o = results[method]['oversmoothing']
                rows_os.append([
                    method,
                    f"{o['NumRank']:.4f}",
                    f"{o['Erank']:.4f}",
                    f"{o['EDir']:.4f}",
                    f"{o['EDir_traditional']:.2e}",
                    f"{o['EProj']:.4f}",
                    f"{o['MAD']:.4f}",
                ])
            else:
                rows_os.append([method] + ["Failed"]*6)

        col_widths_os = [max(len(str(x)) for x in col) for col in zip(*([headers_os] + rows_os))]
        print_table(headers_os, rows_os, col_widths_os)

        best_method = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest method: {best_method[0]} (Accuracy: {best_method[1]['accuracy']:.4f})")
    else:
        print("No successful runs.")
    
    print("-"*50)

if __name__ == "__main__":
    run_parallel_single_benchmark()