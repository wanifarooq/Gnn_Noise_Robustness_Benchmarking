"""Model construction (registry pattern) and FLOPs profiling."""

import threading
import torch
from torch.profiler import profile, ProfilerActivity

from model.gnns import GCN, GIN, GAT, GATv2, GPS

_profiler_lock = threading.Lock()


def get_model(model_name, in_channels, hidden_channels, out_channels, **kwargs):
    """Construct a GNN backbone by name using a registry of (class, valid_params).

    Filters ``kwargs`` so only parameters recognized by the chosen architecture
    are forwarded, preventing spurious keyword errors.
    """
    model_name = model_name.lower()
    kwargs.pop('in_channels', None)
    kwargs.pop('hidden_channels', None)
    kwargs.pop('out_channels', None)

    model_registry = {
        'gcn':    (GCN, ['n_layers', 'dropout', 'self_loop']),
        'gin':    (GIN, ['n_layers', 'dropout', 'mlp_layers', 'train_eps']),
        'gat':    (GAT, ['n_layers', 'dropout', 'heads']),
        'gatv2':  (GATv2, ['n_layers', 'dropout', 'heads']),
        'gps':    (GPS, ['n_layers', 'dropout', 'heads', 'use_pe', 'pe_dim']),
    }

    if model_name not in model_registry:
        raise ValueError(f"Model {model_name} not recognized.")

    model_cls, valid_params = model_registry[model_name]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return model_cls(in_channels, hidden_channels, out_channels, **filtered_kwargs)


def _forward_call(model, data):
    """Call model forward with the correct signature.

    Tries in this order:
    1) model(data)
    2) model(data.x)
    3) model(data.x, data.edge_index)
    """
    try:
        return model(data)
    except TypeError:
        pass

    try:
        return model(data.x)
    except TypeError:
        pass

    # Most PyG-style GNNs: forward(x, edge_index)
    return model(data.x, data.edge_index)


def profile_model_flops(model, data, device, n_warmup=1, n_iters=1):
    """Profile FLOPs with torch.profiler (forward pass).

    Runs *n_warmup* un-profiled forward passes, then *n_iters* profiled passes.
    Returns a dict with ``total_flops`` (int) and ``profiler_table`` (str).
    """
    with _profiler_lock:
        model.eval()

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        # Warmup outside profiler
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = _forward_call(model, data)
            if device.type == "cuda":
                torch.cuda.synchronize()

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                for _ in range(n_iters):
                    _ = _forward_call(model, data)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        total_flops = 0
        for e in prof.key_averages():
            fl = getattr(e, "flops", None)
            if fl:
                total_flops += fl

        table = prof.key_averages().table(
            sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
            row_limit=15
        )

        return {"total_flops": int(total_flops), "profiler_table": table}
