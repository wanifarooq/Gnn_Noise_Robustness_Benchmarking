import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, GATv2Conv

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.use_bn:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 n_layers: int = 2, dropout: float = 0.5, with_relu: bool = True, 
                 with_bias: bool = True, self_loop: bool = True, norm_info: dict = None, 
                 act: str = 'F.relu', input_layer: bool = False, output_layer: bool = False):
        super().__init__()

        self.with_relu = with_relu
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
                self.output_norm = self.norm_type(hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if self.is_norm else None

        for i in range(n_layers):
            in_dim = in_channels if i == 0 and not input_layer else hidden_channels
            out_dim = out_channels if i == n_layers - 1 and not output_layer else hidden_channels
            self.convs.append(GCNConv(in_dim, out_dim, bias=with_bias, add_self_loops=self_loop))
            if self.is_norm:
                self.norms.append(self.norm_type(out_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        
        if self.input_layer:
            x = self.input_linear(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < self.n_layers - 1:
                if self.is_norm:
                    x = self.norms[i](x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.output_layer:
            if self.is_norm:
                x = self.output_norm(x)
            x = self.output_linear(x)

        return x

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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()


class GAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 n_layers: int = 3, heads: int = 4, dropout: float = 0.6, 
                 with_bias: bool = True, self_loop: bool = True, 
                 norm_info: dict = None, act: str = 'F.elu'):
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
            self.convs.append(
                GATConv(in_dim, out_dim, heads=layer_heads, concat=concat,
                        dropout=dropout, bias=with_bias, add_self_loops=self_loop)
            )
            if self.is_norm and i != n_layers - 1:
                norm_dim = out_dim * layer_heads if concat else out_dim
                self.norms.append(self.norm_type(norm_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.n_layers - 1:
                if self.is_norm:
                    x = self.norms[i](x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.norms:
            for norm in self.norms:
                norm.reset_parameters()


class GATv2(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 n_layers: int = 3, heads: int = 4, dropout: float = 0.6, 
                 with_bias: bool = True, self_loop: bool = True, 
                 norm_info: dict = None, act: str = 'F.elu'):
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
            in_dim = in_channels if i == 0 else hidden_channels
            if i == n_layers - 1:
                out_dim = out_channels
                layer_heads = 1
                concat = False
            else:
                out_dim = hidden_channels // heads
                layer_heads = heads
                concat = True
            self.convs.append(
                GATv2Conv(in_dim, out_dim, heads=layer_heads, concat=concat,
                          dropout=dropout, bias=with_bias, add_self_loops=self_loop,
                          share_weights=False)
            )
            if self.is_norm and i != n_layers - 1:
                self.norms.append(self.norm_type(hidden_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        for i, conv in enumerate(self.convs):
            if edge_weight is not None:
                x = conv(x, edge_index, edge_attr=edge_weight)
            else:
                x = conv(x, edge_index)
            if i < self.n_layers - 1:
                if self.is_norm:
                    x = self.norms[i](x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.norms:
            for norm in self.norms:
                norm.reset_parameters()
