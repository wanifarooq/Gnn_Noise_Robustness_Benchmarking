import torch.nn as nn
import torch.nn.functional as F
import torch
from util.laplacian_builder import GeneralLaplacianBuilder
import torch
import torch_sparse
from util.householder import torch_householder_orgqr
from model.gnns import MLP


class SheafNN_Bodnar(nn.Module):
    """
    An implementation which follows directly the orthogonal maps implementation of the Sheaf by Bodnar.
    1. It has a first2 layer MLP which makes an embedding of the input features, dim in_channels --> hidden_channels which must be divisible by stalk;
    2. initialization of the linear layer for generation of the restriction maps on the forward, based on the concatenation of the nodes' features;
    3. generate the Laplacian and apply diffusion for n_layer times;
    4. finally I apply the linear layer out to generate the probability vectors from the final embeddings, dim hidden_channels --> out_channels;
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 n_layers: int = 2, 
                 dropout_in: float = 0.5,
                 dropout: float = 0.5, 
                 stalk: int = 2, 
                 non_linear: bool = False,
                 ego: bool = False,
                 act: str = 'F.elu',
                 norm_info: dict = {},
                 attention :bool = False):
        
        super().__init__()
        assert hidden_channels % stalk == 0, "Hidden channels must be divisible by the stalk dimension"
        self.hidden_channels = hidden_channels
        self.stalk = stalk
        self.dropout_in = dropout_in  
        self.dropout = dropout
        self.n_layers = n_layers  
        self.non_linear = non_linear
        self.ego = ego
        self.attention = attention

        self.attention_layer = None
        if self.attention:
            self.attention_layer = nn.Linear(2 * hidden_channels, 1)

        norm_info = norm_info or {'is_norm': False, 'norm_type': 'LayerNorm'}
        self.is_norm = norm_info['is_norm']
        self.norm_type = getattr(nn, norm_info['norm_type']) if self.is_norm else None

        self.act = eval(act)
        self.laplacian_builder = None  # Will be initialized in the forward pass
        self.edge_index = None

        self.MLP_in = MLP(in_channels, hidden_channels, hidden_channels, num_layers=2, dropout=dropout)
        if self.ego:
            dim_out = hidden_channels + ego * hidden_channels
            self.emb_out_1 = nn.Linear(dim_out, hidden_channels)
            self.emb_out_2 = nn.Linear(hidden_channels, out_channels)
        else:
            self.emb_out = nn.Linear(hidden_channels, out_channels)

        num_gen_maps = n_layers if non_linear else 1
        self.gen_maps = nn.ModuleList()
        for _ in range(num_gen_maps):
            self.gen_maps.append(nn.Linear(2 * hidden_channels, stalk**2, bias=False))

        self.linear_layers = nn.ModuleList()
        self.epsilons = nn.ParameterList()
        f = hidden_channels // stalk
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.linear_layers.append(nn.Linear(stalk, stalk, bias=False))
            nn.init.eye_(self.linear_layers[-1].weight)

            self.linear_layers.append(nn.Linear(f, f, bias=False))
            nn.init.orthogonal_(self.linear_layers[-1].weight)

            self.epsilons.append(torch.nn.Parameter(torch.zeros(self.stalk, 1)))  # Learnable step size for diffusion

            if self.is_norm:
                self.norms.append(self.norm_type(f))
    
    def left_right_linear(self, x, left, right):
        x = x.t().reshape(-1, self.stalk)
        x = left(x)
        x = x.reshape(-1, self.N * self.stalk).t()

        x = right(x)

        return x


    def _init_maps(self, edge_index, x, layer = 0):
            """
            Genera mappe di restrizione ORTOGONALI usando le riflessioni di Householder.
            x: tensore degli embedding [N, hidden_channels] (output di mlp_in)
            """
            num_edges = edge_index.size(1)
            source, destination = edge_index[0, :], edge_index[1, :]

            embed = torch.cat((x[source], x[destination]), dim=1)

            params = self.gen_maps[layer](embed).reshape(num_edges, self.stalk, self.stalk)
            params = F.tanh(params)  
            
            # Inizializziamo le mappe finali come matrici Identità 
            eye = torch.eye(self.stalk, device=x.device).unsqueeze(0).repeat(num_edges, 1, 1)
            A = params.tril(diagonal=-1) + eye
            
            # Chiamata all'implementazione di Householder in util
            A = torch_householder_orgqr(A)
            
            if self.attention:
                attn_scores = self.attention_layer(embed)
                attn_scores = 2 * torch.sigmoid(attn_scores)
                A = A * attn_scores.reshape(-1, 1, 1)
            
            return A


    def _diffusion(self, x): 
        x0 = x
        for layer in range(self.n_layers):
            if self.non_linear or layer == 0:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0, training=self.training)
                x_maps = x_maps.reshape(self.N, -1)
                maps = self._init_maps(self.edge_index, x_maps, layer)
                laplacian, _ = self.laplacian_builder(maps)
                index, value = laplacian

            W1 = self.linear_layers[2 * layer]
            W2 = self.linear_layers[2 * layer + 1]

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, W1, W2)

            x = torch_sparse.spmm(index, value, x.size(0), x.size(0), x)
            
            if self.is_norm:
                # normalizzazione in parallelo sugli stalks
                x = self.norms[layer](x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.N,1)) 
            x0 = coeff* x0 - self.act(x)
            x = x0
                
        return x


    def _forward_body(self, data):

        x = data.x
        if self.edge_index is None:
            self.edge_index = data.edge_index
        
        self.N = x.size(0)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        x = self.MLP_in(x)

        if self.laplacian_builder is None:
            # The normalizatin is the exact same one of the Sheaf Diffusion paper, but it might be instable
            self.laplacian_builder = GeneralLaplacianBuilder(size = self.N, edge_index = self.edge_index, d = self.stalk, normalised = False, deg_normalised = True)

        x = x.reshape((self.N * self.stalk, -1)) 
        
        if self.ego:
            x_diffuse = self._diffusion(x)
            x = torch.cat((x.reshape(self.N, -1), x_diffuse.reshape(self.N, -1)), dim=1)
        else:
            x = self._diffusion(x)
            x = x.reshape(self.N, -1)

        return x
    
    def get_embeddings(self, data):
        """Return raw hidden representation before the final projection and its transforms."""
        return self._forward_body(data)

    def forward(self, data):
        x = self._forward_body(data)

        if self.ego:
            x = self.emb_out_1(x)
            x = self.act(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.emb_out_2(x)
        else:
            x = self.emb_out(x)
        return x

    def initialize(self):
            
            if self.attention and self.attention_layer is not None:
                self.attention_layer.reset_parameters()

            for module in self.MLP_in.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            if self.ego:
                self.emb_out_1.reset_parameters()
                self.emb_out_2.reset_parameters()
            else:
                self.emb_out.reset_parameters()

            for i, layer in enumerate(self.linear_layers):
                if i % 2 == 0:
                    nn.init.eye_(layer.weight)  
                else:
                    nn.init.orthogonal_(layer.weight)
            for layer in self.gen_maps:
                layer.reset_parameters()
            if self.is_norm:
                for norm in self.norms:
                    norm.reset_parameters()

            self.laplacian_builder = None  # Clear laplacian builder to ensure it is re-initialized on the next forward pass