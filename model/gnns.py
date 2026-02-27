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
                 act: str = 'F.relu', input_layer: bool = False, output_layer: bool = False):
        super().__init__()

        self.dropout = dropout
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer

        norm_info = norm_info or {'is_norm': False, 'norm_type': 'LayerNorm'}
        self.is_norm = norm_info['is_norm']
        self.norm_type = getattr(nn, norm_info['norm_type']) if self.is_norm else None
        self.act = eval(act)

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

        Always runs convs[:-1]. When output_layer=True, also runs convs[-1] (hidden->hidden).
        Returns the raw hidden representation before any projection-specific transforms
        (output_norm/act/dropout/output_linear or the final projection conv).
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)

        if self.input_layer:
            x = self.input_linear(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if self.is_norm:
                x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.output_layer:
            x = self.convs[-1](x, edge_index, edge_weight)

        return x

    def forward(self, data):
        x = self._forward_body(data)
        edge_weight = getattr(data, 'edge_weight', None)
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


class GIN(nn.Module):
    """Graph Isomorphism Network.

    Each GINConv wraps an MLP. Last conv's MLP projects hidden_channels -> out_channels.
    get_embeddings() runs convs[:-1], returning hidden_channels dim.
    forward() runs all convs, returning out_channels dim.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 n_layers: int = 3, mlp_layers: int = 2, dropout: float = 0.5, 
                 train_eps: bool = True):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        for i in range(n_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == n_layers - 1 else hidden_channels
            mlp = MLP(in_dim, hidden_channels, out_dim, mlp_layers, dropout)
            self.convs.append(GINConv(mlp, train_eps=train_eps))

    def _forward_body(self, data):
        """Run convs[:-1] (last conv is the projection). Returns hidden_channels dim."""
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, data):
        x = self._forward_body(data)
        return self.convs[-1](x, data.edge_index)

    def get_embeddings(self, data):
        """Return hidden_channels-dim node representations."""
        return self._forward_body(data)

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()


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
                 norm_info: dict | None = None, act: str = 'F.elu'):
        super().__init__()

        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout
        self.act = eval(act)

        norm_info = norm_info or {'is_norm': False, 'norm_type': 'LayerNorm'}
        self.is_norm = norm_info['is_norm']
        self.norm_type = getattr(nn, norm_info['norm_type']) if self.is_norm else None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if self.is_norm else None

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
        """Run convs[:-1] (last conv is the projection). Returns hidden_channels * heads dim."""
        x, edge_index = data.x, data.edge_index
        edge_attr = _get_edge_attr(data)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if self.is_norm:
                x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, data):
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
                 norm_info: dict | None = None, act: str = 'F.elu'):
        super().__init__()

        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout
        self.act = eval(act)

        norm_info = norm_info or {'is_norm': False, 'norm_type': 'LayerNorm'}
        self.is_norm = norm_info['is_norm']
        self.norm_type = getattr(nn, norm_info['norm_type']) if self.is_norm else None

        if hidden_channels % heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"heads ({heads}) for GATv2"
            )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if self.is_norm else None

        for i in range(n_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            if i == n_layers - 1:
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
            if self.is_norm and i != n_layers - 1:
                assert self.norms is not None
                assert self.norm_type is not None
                self.norms.append(self.norm_type(hidden_channels))

    def _forward_body(self, data):
        """Run convs[:-1] (last conv is the projection). Returns hidden_channels dim."""
        x, edge_index = data.x, data.edge_index
        edge_attr = _get_edge_attr(data)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if self.is_norm:
                x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, data):
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
        self.norm_type = getattr(nn, norm_info['norm_type']) if self.is_norm else None

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
        # Unlike GATConv/GATv2Conv where edge_dim=1 cleanly skips when
        # edge_attr=None, GINEConv always uses edge_attr (x_j + edge_attr).
        # Adding edge_dim=1 to project scalar weights would introduce a
        # Linear(1, hidden_channels, bias=True) whose bias term turns zero
        # edge features into non-zero vectors — silently changing behavior
        # for all experiments that don't use edge weights. Zeros are a safe
        # no-op: x_j + 0 = x_j.
        edge_attr = x.new_zeros(edge_index.size(1), x.size(1))

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