import copy
import hashlib
import yaml
from itertools import product
from typing import Any, Dict, List, Tuple
import json
import numpy as np
import torch
import os

SWEEP_PREFIX = "£["


def get_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a SHA1 hash from a subset of configuration fields.
    Useful to uniquely identify an experiment run.
    """
    keys_to_hash = {
        "dataset": config.get("dataset", {}).get("name", ""),
        "noise": config.get("noise", {}).get("type", ""),
        "seed": config.get("seed", ""),
        "model": config.get("model", {}).get("name", ""),
        "training": config.get("training", {}).get("method", "")
    }

    # Create a deterministic string representation
    hash_input = "|".join(f"{k}:{v}" for k, v in keys_to_hash.items())

    # Compute SHA1 hash
    return hashlib.sha1(hash_input.encode("utf-8")).hexdigest()


def is_sweep_string(x: Any) -> bool:
    """
    Check whether a value is a sweep expression of the form £[...].
    """
    return isinstance(x, str) and x.strip().startswith("£[") and x.strip().endswith("]")


def parse_sweep_value(v: str) -> List[Any]:
    """
    Parse a sweep string like:
        £[cora, pubmed, 0.01, true, null]

    Each element is parsed using yaml.safe_load to preserve correct types:
    - numbers -> int / float
    - true/false -> bool
    - null -> None
    - everything else -> string
    """
    s = v.strip()
    if not (s.startswith("£[") and s.endswith("]")):
        raise ValueError(f"Invalid sweep format: {v}")

    # Remove the '£[' prefix and ']' suffix
    inner = s[2:-1].strip()
    if not inner:
        return []

    # Split elements and parse each one
    items = [item.strip() for item in inner.split(",")]
    parsed: List[Any] = []
    for item in items:
        parsed.append(yaml.safe_load(item))

    return parsed


def find_sweeps(obj: Any, prefix: Tuple[Any, ...] = ()) -> List[Tuple[Tuple[Any, ...], List[Any]]]:
    """
    Recursively traverse a Python object (dicts and lists) and
    find all sweep expressions.

    Returns:
        A list of (path, values) where:
        - path is a tuple describing how to reach the field
        - values is the list of sweep values for that field
    """
    sweeps: List[Tuple[Tuple[Any, ...], List[Any]]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            sweeps.extend(find_sweeps(v, prefix + (k,)))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            sweeps.extend(find_sweeps(v, prefix + (i,)))

    else:
        if is_sweep_string(obj):
            sweeps.append((prefix, parse_sweep_value(obj)))

    return sweeps


def set_by_path(cfg: Any, path: Tuple[Any, ...], value: Any) -> None:
    """
    Set a value inside a nested structure (dicts / lists)
    following the provided path.

    Example:
        path = ("dataset", "name")
        cfg["dataset"]["name"] = value
    """
    cur = cfg
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


def expand_yaml_sweeps(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all configurations produced by sweep expressions.

    If no sweep is found, the original configuration is returned
    as a single-element list.
    """
    # Find all sweep locations and their possible values
    sweeps = find_sweeps(base_cfg)

    if not sweeps:
        return [base_cfg]

    # Separate paths and corresponding value lists
    paths = [p for (p, _) in sweeps]
    values_lists = [vals for (_, vals) in sweeps]

    expanded_configs: List[Dict[str, Any]] = []

    # Cartesian product over all sweep values
    for combo in product(*values_lists):
        cfg_i = copy.deepcopy(base_cfg)

        # Apply the current combination to the configuration
        for path, val in zip(paths, combo):
            set_by_path(cfg_i, path, val)

        expanded_configs.append(cfg_i)

    return expanded_configs


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file into a Python dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def json_serializer(obj):
    # NumPy scalars
    if isinstance(obj, np.generic):
        return obj.item()

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Torch tensors
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    # Fallback
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def should_run_experiment(result_json_path: str, cfg: dict) -> bool:
    """
    Skip the experiment if results already exist, unless cfg['force'] is True.
    """
    force = bool(cfg.get("force", False))

    if os.path.exists(result_json_path):
        if force:
            print(f"[FORCE] Results already exist, overwriting: {result_json_path}")
            return True
        else:
            print(f"[SKIP] Results already exist: {result_json_path}")
            print("       Set `force: true` in config.yaml to overwrite.")
            return False

    return True