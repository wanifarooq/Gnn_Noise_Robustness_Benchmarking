import json
import os 

def load_json(file_name):
    with open(file_name) as f:
        d = json.load(f)
    return d

results = {}
for item in os.listdir(os.getcwd()):
    if '.json' in item:
        file = load_json(item)
        dataset_name = file["config"]["dataset"]["name"]
        backbone_name = file["config"]["model"]["name"]
        noise_type = file["config"]["noise"]["type"]
        method_type = file["config"]["training"]["method"]
        if dataset_name not in results.keys():
            results[dataset_name] = {}
        else:
            if backbone_name not in results[dataset_name].keys():
                results[dataset_name][backbone_name] = {}
            else:
                if noise_type not in results[dataset_name][backbone_name].keys():
                    results[dataset_name][backbone_name][noise_type] = []
                else:
                    results[dataset_name][backbone_name][noise_type].append(method_type)

print(results)

                

