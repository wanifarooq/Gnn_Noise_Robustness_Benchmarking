import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, GATv2Conv
from torch.nn import (
    BatchNorm1d,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch_geometric.nn import GINEConv, GPSConv


def _get_edge_attr(data) -> torch.Tensor | None:
    """Extract scalar edge weights from data and unsqueeze (E,) -> (E,1) to match edge_dim=1."""
    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is None:
        return None
    return edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 else edge_weight


class PairNorm(nn.Module):
    """PairNorm (Zhao & Akoglu, ICLR 2020) — anti-oversmoothing normalization.

    Centers node features across the node dimension (removing the common
    component that all nodes drift toward under repeated message passing) and
    rescales to a constant total variance. Unlike BatchNorm it is parameter-free
    with no running statistics, so it prevents the *directional* rank-1 collapse
    on dense graphs without degrading small/sparse graphs the way BatchNorm does.

    Takes an unused ``num_features`` arg so it is drop-in compatible with the
    ``norm_type(dim)`` construction used by the backbones.
    """

    def __init__(self, num_features=None, scale: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x):
        if x.size(0) <= 1:
            return x
        x = x - x.mean(dim=0, keepdim=True)               # center across nodes
        rownorm = (x.pow(2).sum(dim=1).mean()).clamp(min=self.eps).sqrt()
        return self.scale * x / rownorm

    def reset_parameters(self):
        pass


def resolve_norm_type(name: str):
    """Map a norm name to its class. Supports torch.nn norms + custom PairNorm."""
    if name == 'PairNorm':
        return PairNorm
    return getattr(nn, name)


class MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5, use_bn: bool = False):
        super().__init__()
        self.dropout = dropout
        self.use_bn = use_bn

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))

            if use_bn:
                for _ in range(num_layers - 1):
                    self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

    def _forward_body(self, x: torch.Tensor) -> torch.Tensor:
        """Run all layers except the final projection. Returns hidden_channels dim."""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.use_bn:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[-1](self._forward_body(x))

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden_channels-dim node representations."""
        return self._forward_body(x)


class GCN(nn.Module):
    """Graph Convolutional Network.

    get_embeddings() returns the raw hidden representation (no output_norm/act/dropout).
    forward() applies the final projection and, when output_layer=True, output_norm/act/dropout.
    Default (output_layer=False): last conv is the projection (hidden->out_channels).
    output_layer=True: all convs stay at hidden_channels; output_linear is the projection.
    input_layer=True: input_linear projects in_channels -> hidden_channels before convs.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 n_layers: int = 2, dropout: float = 0.5,
                 with_bias: bool = True, self_loop: bool = True, norm_info: dict | None = None,
                 act: str = 'F.relu', input_layer: bool = False, output_layer: bool = False,
                 use_residual: bool = False, jk: str = 'none'):
        super().__init__()

        self.dropout = dropout
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.use_residual = use_residual
        self.jk = (jk or 'none').lower()

        norm_info = norm_info or {'is_norm': False, 'norm_type': 'LayerNorm'}
        self.is_norm = norm_info['is_norm']
        self.norm_type = resolve_norm_type(norm_info['norm_type']) if self.is_norm else None
        self.act = eval(act)

        if self.jk in ('cat', 'max'):
            # Jumping-Knowledge: every conv stays at hidden_channels, all layer
            # outputs are aggregated, then a final linear projects to out_channels.
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList() if self.is_norm else None
            for i in range(n_layers):
                in_dim = in_channels if i == 0 else hidden_channels
                self.convs.append(GCNConv(in_dim, hidden_channels, bias=with_bias,
                                          add_self_loops=self_loop))
                if self.is_norm:
                    self.norms.append(self.norm_type(hidden_channels))
            jk_in = hidden_channels * n_layers if self.jk == 'cat' else hidden_channels
            self.jk_lin = nn.Linear(jk_in, out_channels)
            return

        if input_layer:
            self.input_linear = nn.Linear(in_channels, hidden_channels)
        if output_layer:
            self.output_linear = nn.Linear(hidden_channels, out_channels)
            if self.is_norm:
                assert self.norm_type is not None
                self.output_norm = self.norm_type(hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if self.is_norm else None

        for i in range(n_layers):
            in_dim = in_channels if i == 0 and not input_layer else hidden_channels
            out_dim = out_channels if i == n_layers - 1 and not output_layer else hidden_channels
            self.convs.append(GCNConv(in_dim, out_dim, bias=with_bias, add_self_loops=self_loop))
            if self.is_norm and i != n_layers - 1:
                assert self.norms is not None
                assert self.norm_type is not None
                self.norms.append(self.norm_type(out_dim))

    def _forward_body(self, data):
        """Run inter-layer convs with norm/act/dropout between them.

        With Jumping-Knowledge, run ALL convs at hidden width and aggregate every
        layer's output (cat or max) — this is what lets deep GCNs avoid
        oversmoothing on heterophilous graphs. Otherwise: run convs[:-1] (and
        convs[-1] when output_layer=True). Returns the representation fed to the
        final projection (jk_lin / output_linear / last conv).
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)

        if self.jk in ('cat', 'max'):
            layer_outs = []
            for i, conv in enumerate(self.convs):
                h = conv(x, edge_index, edge_weight)
                if self.is_norm:
                    h = self.norms[i](h)
                h = self.act(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if self.use_residual and h.size(-1) == x.size(-1):
                    x = h + x
                else:
                    x = h
                layer_outs.append(x)
            if self.jk == 'cat':
                return torch.cat(layer_outs, dim=-1)
            return torch.stack(layer_outs, dim=-1).max(dim=-1).values

        if self.input_layer:
            x = self.input_linear(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index, edge_weight)
            if self.is_norm:
                h = self.norms[i](h)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_residual and h.size(-1) == x.size(-1):
                x = h + x
            else:
                x = h

        if self.output_layer:
            h = self.convs[-1](x, edge_index, edge_weight)
            if self.use_residual and h.size(-1) == x.size(-1):
                x = h + x
            else:
                x = h

        return x

    def forward(self, data):
        x = self._forward_body(data)
        edge_weight = getattr(data, 'edge_weight', None)
        if self.jk in ('cat', 'max'):
            return self.jk_lin(x)
        if self.output_layer:
            if self.is_norm:
                x = self.output_norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.output_linear(x)
        return self.convs[-1](x, data.edge_index, edge_weight)

    def get_embeddings(self, data):
        """Return raw hidden representation before the final projection and its transforms."""
        return self._forward_body(data)

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        if hasattr(self, 'input_linear'):
            self.input_linear.reset_parameters()
        if hasattr(self, 'output_linear'):
            self.output_linear.reset_parameters()
        if self.norms:
            for norm in self.norms:
                norm.reset_parameters()
        if hasattr(self, 'jk_lin'):
            self.jk_lin.reset_parameters()


class GIN(nn.Module):
    """Graph Isomorphism Network.

    Each GINConv wraps an MLP. Last conv's MLP projects hidden_channels -> out_channels.
    get_embeddings() runs convs[:-1], returning hidden_channels dim.
    forward() runs all convs, returning out_channels dim.

    NOTE: GINConv does not support edge weights. Any edge_weight on the input
    data is silently ignored. This is by design — GIN's theoretical power
    (injective multiset aggregation) requires unweighted sum. Methods that
    produce adaptive edge weights (RTGNN, NRGNN, GNNGuard) will not benefit
    from them when using GIN as backbone.
    """
    _edge_weight_warned = False

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 n_layers: int = 3, mlp_layers: int = 2, dropout: float = 0.5,
                 train_eps: bool = True, norm_info: dict | None = None,
                 use_residual: bool = False, jk: str = 'none'):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.use_residual = use_residual
        self.jk = (jk or 'none').lower()
        self.convs = nn.ModuleList()

        norm_info = norm_info or {'is_norm': False, 'norm_type': 'LayerNorm'}
        self.is_norm = norm_info['is_norm']
        self.norm_type = resolve_norm_type(norm_info['norm_type']) if self.is_norm else None
        self.norms = nn.ModuleList() if self.is_norm else None

        jk_mode = self.jk in ('cat', 'max')
        for i in range(n_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels if (jk_mode or i != n_layers - 1) else out_channels
            mlp = MLP(in_dim, hidden_channels, out_dim, mlp_layers, dropout)
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            if self.is_norm and (jk_mode or i != n_layers - 1):
                assert self.norms is not None and self.norm_type is not None
                self.norms.append(self.norm_type(hidden_channels))
        if jk_mode:
            jk_in = hidden_channels * n_layers if self.jk == 'cat' else hidden_channels
            self.jk_lin = nn.Linear(jk_in, out_channels)

    def _forward_body(self, data):
        """Run convs[:-1] (last conv is the projection). Returns hidden_channels dim.
        With Jumping-Knowledge, run all convs and aggregate every layer's output."""
        if not GIN._edge_weight_warned and getattr(data, 'edge_weight', None) is not None:
            warnings.warn(
                "GIN backbone ignores edge_weight (GINConv uses unweighted sum "
                "to preserve injective aggregation). Consider using GCN, GAT, "
                "or GATv2 if edge weights are important.",
                stacklevel=2,
            )
            GIN._edge_weight_warned = True
        x, edge_index = data.x, data.edge_index
        if self.jk in ('cat', 'max'):
            layer_outs = []
            for i, conv in enumerate(self.convs):
                h = conv(x, edge_index)
                if self.is_norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if self.use_residual and h.size(-1) == x.size(-1):
                    x = h + x
                else:
                    x = h
                layer_outs.append(x)
            if self.jk == 'cat':
                return torch.cat(layer_outs, dim=-1)
            return torch.stack(layer_outs, dim=-1).max(dim=-1).values
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            if self.is_norm:
                h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.use_residual and h.size(-1) == x.size(-1):
                x = h + x
            else:
                x = h
        return x

    def forward(self, data):
        x = self._forward_body(data)
        if self.jk in ('cat', 'max'):
            return self.jk_lin(x)
        return self.convs[-1](x, data.edge_index)

    def get_embeddings(self, data):
        """Return hidden_channels-dim node representations."""
        return self._forward_body(data)

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.norms:
            for norm in self.norms:
                norm.reset_parameters()
        if hasattr(self, 'jk_lin'):
            self.jk_lin.reset_parameters()


class GAT(nn.Module):
    """Graph Attention Network (v1).

    Intermediate layers concatenate heads (dim = hidden_channels * heads).
    Last conv uses 1 head projecting to out_channels.
    get_embeddings() runs convs[:-1], returning hidden_channels * heads dim.
    forward() runs all convs, returning out_channels dim.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 n_layers: int = 3, heads: int = 4, dropout: float = 0.6,
                 with_bias: bool = True, self_loop: bool = True,
                 norm_info: dict | None = None, act: str = 'F.elu',
                 use_residual: bool = False, jk: str = 'none'):
        super().__init__()

        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout
        self.act = eval(act)
        self.use_residual = use_residual
        self.jk = (jk or 'none').lower()

        norm_info = norm_info or {'is_norm': False, 'norm_type': 'LayerNorm'}
        self.is_norm = norm_info['is_norm']
        self.norm_type = resolve_norm_type(norm_info['norm_type']) if self.is_norm else None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if self.is_norm else None

        if self.jk in ('cat', 'max'):
            # JK mode: every layer outputs hidden_channels (heads averaged via
            # concat=False, so no head-divisibility constraint), aggregated below.
            for i in range(n_layers):
                in_dim = in_channels if i == 0 else hidden_channels
                self.convs.append(
                    GATConv(in_dim, hidden_channels, heads=heads, concat=False,
                            dropout=dropout, bias=with_bias, add_self_loops=self_loop,
                            edge_dim=1)
                )
                if self.is_norm:
                    self.norms.append(self.norm_type(hidden_channels))
            jk_in = hidden_channels * n_layers if self.jk == 'cat' else hidden_channels
            self.jk_lin = nn.Linear(jk_in, out_channels)
        else:
            for i in range(n_layers):
                in_dim = in_channels if i == 0 else hidden_channels * heads
                out_dim = out_channels if i == n_layers - 1 else hidden_channels
                layer_heads = 1 if i == n_layers - 1 else heads
                concat = i != n_layers - 1
                # edge_dim=1: all edge weights in our experiments are scalar (1-dim),
                # produced dynamically by methods like GNNGuard, NRGNN, RTGNN.
                # Without edge_dim, GATConv silently ignores edge_attr.
                self.convs.append(
                    GATConv(in_dim, out_dim, heads=layer_heads, concat=concat,
                            dropout=dropout, bias=with_bias, add_self_loops=self_loop,
                            edge_dim=1)
                )
                if self.is_norm and i != n_layers - 1:
                    assert self.norms is not None
                    assert self.norm_type is not None
                    norm_dim = out_dim * layer_heads if concat else out_dim
                    self.norms.append(self.norm_type(norm_dim))

    def _forward_body(self, data):
        """Run convs[:-1] (last conv is the projection). Returns hidden_channels * heads dim.
        With Jumping-Knowledge, run all convs and aggregate every layer's output."""
        x, edge_index = data.x, data.edge_index
        edge_attr = _get_edge_attr(data)
        if self.jk in ('cat', 'max'):
            layer_outs = []
            for i, conv in enumerate(self.convs):
                h = conv(x, edge_index, edge_attr=edge_attr)
                if self.is_norm:
                    h = self.norms[i](h)
                h = self.act(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if self.use_residual and h.size(-1) == x.size(-1):
                    x = h + x
                else:
                    x = h
                layer_outs.append(x)
            if self.jk == 'cat':
                return torch.cat(layer_outs, dim=-1)
            return torch.stack(layer_outs, dim=-1).max(dim=-1).values
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index, edge_attr=edge_attr)
            if self.is_norm:
                h = self.norms[i](h)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_residual and h.size(-1) == x.size(-1):
                x = h + x
            else:
                x = h
        return x

    def forward(self, data):
        if self.jk in ('cat', 'max'):
            return self.jk_lin(self._forward_body(data))
        return self.convs[-1](self._forward_body(data), data.edge_index, edge_attr=_get_edge_attr(data))


    def get_embeddings(self, data):
        """Return hidden representation before the final projection conv."""
        return self._forward_body(data)

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.norms:
            for norm in self.norms:
                norm.reset_parameters()
        if hasattr(self, 'jk_lin'):
            self.jk_lin.reset_parameters()


class GATv2(nn.Module):
    """Graph Attention Network v2 (head-normalised variant).

    Intermediate layers: per-head dim = hidden_channels // heads, concat=True -> hidden_channels.
    Last conv uses heads=1, concat=False projecting to out_channels.
    get_embeddings() runs convs[:-1], returning hidden_channels dim.
    forward() runs all convs, returning out_channels dim.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 n_layers: int = 3, heads: int = 4, dropout: float = 0.6,
                 with_bias: bool = True, self_loop: bool = True,
                 norm_info: dict | None = None, act: str = 'F.elu',
                 use_residual: bool = False, jk: str = 'none'):
        super().__init__()

        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout
        self.act = eval(act)
        self.use_residual = use_residual
        self.jk = (jk or 'none').lower()

        norm_info = norm_info or {'is_norm': False, 'norm_type': 'LayerNorm'}
        self.is_norm = norm_info['is_norm']
        self.norm_type = resolve_norm_type(norm_info['norm_type']) if self.is_norm else None

        if hidden_channels % heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"heads ({heads}) for GATv2"
            )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if self.is_norm else None
        jk_mode = self.jk in ('cat', 'max')

        for i in range(n_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            if i == n_layers - 1 and not jk_mode:
                out_dim = out_channels
                layer_heads = 1
                concat = False
            else:
                out_dim = hidden_channels // heads
                layer_heads = heads
                concat = True
            # edge_dim=1: all edge weights in our experiments are scalar (1-dim),
            # produced dynamically by methods like GNNGuard, NRGNN, RTGNN.
            # Without edge_dim, GATv2Conv silently ignores edge_attr.
            self.convs.append(
                GATv2Conv(in_dim, out_dim, heads=layer_heads, concat=concat,
                          dropout=dropout, bias=with_bias, add_self_loops=self_loop,
                          share_weights=False, edge_dim=1)
            )
            if self.is_norm and (jk_mode or i != n_layers - 1):
                assert self.norms is not None
                assert self.norm_type is not None
                self.norms.append(self.norm_type(hidden_channels))
        if jk_mode:
            jk_in = hidden_channels * n_layers if self.jk == 'cat' else hidden_channels
            self.jk_lin = nn.Linear(jk_in, out_channels)

    def _forward_body(self, data):
        """Run convs[:-1] (last conv is the projection). Returns hidden_channels dim.
        With Jumping-Knowledge, run all convs and aggregate every layer's output."""
        x, edge_index = data.x, data.edge_index
        edge_attr = _get_edge_attr(data)
        if self.jk in ('cat', 'max'):
            layer_outs = []
            for i, conv in enumerate(self.convs):
                h = conv(x, edge_index, edge_attr=edge_attr)
                if self.is_norm:
                    h = self.norms[i](h)
                h = self.act(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if self.use_residual and h.size(-1) == x.size(-1):
                    x = h + x
                else:
                    x = h
                layer_outs.append(x)
            if self.jk == 'cat':
                return torch.cat(layer_outs, dim=-1)
            return torch.stack(layer_outs, dim=-1).max(dim=-1).values
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index, edge_attr=edge_attr)
            if self.is_norm:
                h = self.norms[i](h)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_residual and h.size(-1) == x.size(-1):
                x = h + x
            else:
                x = h
        return x

    def forward(self, data):
        if self.jk in ('cat', 'max'):
            return self.jk_lin(self._forward_body(data))
        return self.convs[-1](self._forward_body(data), data.edge_index, edge_attr=_get_edge_attr(data))

    def get_embeddings(self, data):
        """Return hidden_channels-dim node representations."""
        return self._forward_body(data)

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.norms:
            for norm in self.norms:
                norm.reset_parameters()
        if hasattr(self, 'jk_lin'):
            self.jk_lin.reset_parameters()

class GPS(nn.Module):
    """Graph Transformer (GPS: General, Powerful, Scalable).

    Projection scheme (A — required by residual connections): lin_in projects
    in_channels → hidden_channels; all GPSConv layers operate at hidden_channels;
    lin_out projects hidden_channels → out_channels.
    get_embeddings() returns hidden_channels dim (before lin_out); forward() returns out_channels dim.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 n_layers: int = 3, dropout: float = 0.5, heads: int = 4,
                 attn_type: str = 'multihead', use_pe: bool = False, pe_dim: int = 8,
                 with_bias: bool = True, norm_info: dict | None = None, act: str = 'F.relu'):
        super().__init__()

        self.n_layers = n_layers
        self.dropout = dropout
        self.heads = heads
        self.act = eval(act)
        self.use_pe = use_pe
        self.attn_type = attn_type

        norm_info = norm_info or {'is_norm': False, 'norm_type': 'LayerNorm'}
        self.is_norm = norm_info['is_norm']
        self.norm_type = resolve_norm_type(norm_info['norm_type']) if self.is_norm else None

        if use_pe:
            self.pe_norm = BatchNorm1d(pe_dim)
            self.pe_lin = Linear(pe_dim, pe_dim)
            effective_in_channels = in_channels + pe_dim
        else:
            effective_in_channels = in_channels

        # GPSConv uses residual connections (h + x), so input dim must equal
        # output dim (= channels).  We project in/out to keep all GPS layers
        # at a uniform hidden_channels width.
        self.lin_in = Linear(effective_in_channels, hidden_channels)
        self.lin_out = Linear(hidden_channels, out_channels)

        self.convs = ModuleList()
        self.norms = ModuleList() if self.is_norm else None

        for i in range(n_layers):
            nn_module = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )

            attn_kwargs = {'dropout': dropout}

            conv = GPSConv(
                hidden_channels,
                GINEConv(nn_module),
                heads=heads,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs
            )
            self.convs.append(conv)

            if self.is_norm and i != n_layers - 1:
                assert self.norms is not None
                assert self.norm_type is not None
                self.norms.append(self.norm_type(hidden_channels))

    def _forward_body(self, data):
        """Run lin_in + all GPS convs. Returns hidden_channels dim (before lin_out)."""
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)

        if self.use_pe and hasattr(data, 'pe'):
            x_pe = self.pe_norm(data.pe)
            x_pe = self.pe_lin(x_pe)
            x = torch.cat([x, x_pe], dim=1)

        x = self.lin_in(x)

        # GINEConv requires (E, D) edge features matching hidden_channels.
        # We use edge_weight (produced by robustness methods like RTGNN/GNNGuard) 
        # to scale these features.
        edge_weight = getattr(data, 'edge_weight', None)
        edge_attr = x.new_zeros(edge_index.size(1), x.size(1))
        
        if edge_weight is not None:
            # Scale the implicit edge attributes by the learned weights
            edge_attr = edge_attr + edge_weight.view(-1, 1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

            if i < self.n_layers - 1:
                if self.is_norm:
                    x = self.norms[i](x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def forward(self, data):
        return self.lin_out(self._forward_body(data))

    def get_embeddings(self, data):
        """Return hidden_channels-dim node representations."""
        return self._forward_body(data)

    def initialize(self):
        self.lin_in.reset_parameters()
        self.lin_out.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.norms:
            for norm in self.norms:
                norm.reset_parameters()
        if hasattr(self, 'jk_lin'):
            self.jk_lin.reset_parameters()

class GCN_modified(nn.Module):
    """Faithful port of the "Classic GNNs are Strong Baselines" tuned backbone
    (Luo, Shi, Wu, NeurIPS 2024; LUOyk1999/tunedGNN ``MPNNs``).

    Key ingredients over a vanilla GCN:
      - ``pre_linear``: project features (lin_in) before message passing, so every
        conv is hidden->hidden.
      - learned LINEAR residual per layer: ``x = conv(x) + lins[i](x)`` (ego/neighbor
        separation — the main reason it works on heterophilous graphs).
      - LayerNorm (``ln``) or BatchNorm (``bn``) per layer.
      - additive Jumping-Knowledge (``jk``): sum of every layer's output.
      - inner conv switchable: gcn / gat / sage.

    Interface matches the other backbones: ``forward(data)`` -> logits,
    ``get_embeddings(data)`` -> pre-classifier representation.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 n_layers: int = 3, dropout: float = 0.5, heads: int = 1,
                 pre_ln: bool = False, pre_linear: bool = False, lin_res: bool = False,
                 mod_norm: str = 'none', jk: bool = False,
                 inner_gnn: str = 'gcn', self_loop: bool = True, **_ignored):
        super().__init__()

        def _flag(v):
            if isinstance(v, bool):
                return v
            return str(v).lower() in ('true', '1', 'yes', 'sum', 'cat', 'max', 'jk')

        self.dropout = dropout
        self.pre_ln = _flag(pre_ln)
        self.pre_linear = _flag(pre_linear)
        self.res = _flag(lin_res)
        norm = str(mod_norm).lower()
        self.ln = norm in ('ln', 'layer', 'layernorm')
        self.bn = norm in ('bn', 'batch', 'batchnorm', 'batchnorm1d')
        self.jk = _flag(jk)
        self.inner_gnn = str(inner_gnn).lower()

        self.local_convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.bns = nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = nn.ModuleList()

        self.lin_in = nn.Linear(in_channels, hidden_channels)

        def make_conv(ic, oc):
            if self.inner_gnn == 'gat':
                # concat=False (average heads) keeps the output at `oc` so the
                # per-layer linear residual matches for any `heads`. Identical to
                # tunedGNN's GAT (which uses heads=1) at heads=1; add_self_loops
                # False matches tunedGNN.
                return GATConv(ic, oc, heads=heads, concat=False,
                               add_self_loops=False, bias=False)
            if self.inner_gnn == 'sage':
                from torch_geometric.nn import SAGEConv
                return SAGEConv(ic, oc)
            return GCNConv(ic, oc, cached=False, normalize=True, add_self_loops=self_loop)

        local_layers = n_layers
        if not self.pre_linear:
            self.local_convs.append(make_conv(in_channels, hidden_channels))
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.lns.append(nn.LayerNorm(hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(nn.LayerNorm(in_channels))
            local_layers -= 1

        for _ in range(local_layers):
            self.local_convs.append(make_conv(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(nn.LayerNorm(hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(nn.LayerNorm(hidden_channels))

        self.pred_local = nn.Linear(hidden_channels, out_channels)

    def _forward_body(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)

        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            # GCNConv accepts edge_weight; GAT/SAGE in this port do not.
            if self.inner_gnn not in ('gat', 'sage') and edge_weight is not None:
                conv_out = local_conv(x, edge_index, edge_weight)
            else:
                conv_out = local_conv(x, edge_index)
            if self.res:
                x = conv_out + self.lins[i](x)
            else:
                x = conv_out
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x
        return x_final

    def forward(self, data):
        return self.pred_local(self._forward_body(data))

    def get_embeddings(self, data):
        return self._forward_body(data)

    def initialize(self):
        for c in self.local_convs:
            c.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p in self.pre_lns:
                p.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()
