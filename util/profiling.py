"""Model construction (registry pattern) and FLOPs profiling."""

import threading
import torch
from torch.profiler import profile, ProfilerActivity

from model.gnns import GCN, GIN, GAT, GATv2, GPS, GCN_modified

_profiler_lock = threading.Lock()


_NORM_TYPES = {
    'batch': 'BatchNorm1d',
    'batchnorm': 'BatchNorm1d',
    'batchnorm1d': 'BatchNorm1d',
    'layer': 'LayerNorm',
    'layernorm': 'LayerNorm',
    'pair': 'PairNorm',
    'pairnorm': 'PairNorm',
}


def _build_norm_info(normalization):
    """Translate a ``normalization`` config value into a backbone ``norm_info`` dict.

    Accepts: None / 'none' / False  -> no normalization;
             'batch' (BatchNorm1d, the anti-oversmoothing default) or 'layer' (LayerNorm).
    """
    if normalization in (None, False, 'none', 'None', ''):
        return {'is_norm': False, 'norm_type': 'LayerNorm'}
    key = str(normalization).lower()
    if key not in _NORM_TYPES:
        raise ValueError(
            f"Unknown normalization '{normalization}'. Use one of: "
            f"none, batch, layer, pair."
        )
    return {'is_norm': True, 'norm_type': _NORM_TYPES[key]}


def get_model(model_name, in_channels, hidden_channels, out_channels, **kwargs):
    """Construct a GNN backbone by name using a registry of (class, valid_params).

    Filters ``kwargs`` so only parameters recognized by the chosen architecture
    are forwarded, preventing spurious keyword errors.  ``normalization`` is
    translated into the backbone's ``norm_info`` dict and applied to all
    architectures (BatchNorm1d between layers is the default anti-oversmoothing
    measure that keeps dense graphs from collapsing to rank-1 embeddings).
    """
    model_name = model_name.lower()
    kwargs.pop('in_channels', None)
    kwargs.pop('hidden_channels', None)
    kwargs.pop('out_channels', None)

    # normalization -> norm_info (every backbone accepts norm_info)
    norm_info = _build_norm_info(kwargs.pop('normalization', None))

    model_registry = {
        'gcn':    (GCN, ['n_layers', 'dropout', 'self_loop', 'use_residual', 'jk']),
        'gin':    (GIN, ['n_layers', 'dropout', 'mlp_layers', 'train_eps', 'use_residual', 'jk']),
        'gat':    (GAT, ['n_layers', 'dropout', 'heads', 'use_residual', 'jk']),
        'gatv2':  (GATv2, ['n_layers', 'dropout', 'heads', 'use_residual', 'jk']),
        'gps':    (GPS, ['n_layers', 'dropout', 'heads', 'use_pe', 'pe_dim']),
        'gcn_modified': (GCN_modified, ['n_layers', 'dropout', 'heads', 'self_loop',
                                        'pre_ln', 'pre_linear', 'lin_res', 'mod_norm',
                                        'jk', 'inner_gnn']),
    }

    if model_name not in model_registry:
        raise ValueError(f"Model {model_name} not recognized.")

    model_cls, valid_params = model_registry[model_name]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    filtered_kwargs['norm_info'] = norm_info

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
    import copy
    import torch.nn as nn

    with _profiler_lock:
        if isinstance(models, nn.Module):
            models = [models]

        # FLOPs depend only on tensor shapes, not weight values. Running the
        # step in train() mode would mutate state (e.g. BatchNorm running stats),
        # so snapshot every model's state_dict and restore it afterwards to keep
        # profiling side-effect free (otherwise the post-training evaluate() and a
        # reloaded checkpoint would diverge).
        _saved_states = [copy.deepcopy(m.state_dict()) for m in models]

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

        # Restore pre-profiling state (undo any BatchNorm/buffer mutation).
        for m, st in zip(models, _saved_states):
            m.load_state_dict(st)

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
