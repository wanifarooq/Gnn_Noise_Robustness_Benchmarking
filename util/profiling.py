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


def profile_model_flops(model, data, device, n_warmup=1, n_iters=1,
                        forward_fn=None):
    """Profile FLOPs with torch.profiler (forward pass).

    Runs *n_warmup* un-profiled forward passes, then *n_iters* profiled passes.
    Returns a dict with ``total_flops`` (int) and ``profiler_table`` (str).

    If *forward_fn* (a zero-arg callable) is provided it is used instead of
    the default ``_forward_call(model, data)`` – handy for multi-model
    pipelines.  ``model.eval()`` is still called for setup.
    """
    with _profiler_lock:
        model.eval()
        fn = forward_fn or (lambda: _forward_call(model, data))

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        # Warmup outside profiler
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = fn()
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
                    _ = fn()
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

        del prof
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {"total_flops": int(total_flops), "profiler_table": table}


def profile_training_step_flops(models, device, step_fn):
    """Profile FLOPs for one training step (forward + backward).

    Parameters
    ----------
    models : nn.Module or list[nn.Module]
        Model(s) involved — set to train mode before profiling,
        zero_grad after warmup, restored to eval mode when done.
    device : torch.device
    step_fn : () -> Tensor
        Runs the full forward pass and returns a scalar loss.
        ``backward()`` is called by this function.
    """
    import torch.nn as nn

    with _profiler_lock:
        if isinstance(models, nn.Module):
            models = [models]

        for m in models:
            m.train()
            m.zero_grad(set_to_none=True)

        # Warmup — one full step outside the profiler
        loss = step_fn()
        loss.backward()
        for m in models:
            m.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
        ) as prof:
            loss = step_fn()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()

        for m in models:
            m.zero_grad(set_to_none=True)
            m.eval()

        total_flops = 0
        for e in prof.key_averages():
            fl = getattr(e, "flops", None)
            if fl:
                total_flops += fl

        table = prof.key_averages().table(
            sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
            row_limit=15,
        )

        del prof
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {"total_flops": int(total_flops), "profiler_table": table}
