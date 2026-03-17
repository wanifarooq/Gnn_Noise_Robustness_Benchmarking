import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseTrainer
from model.registry import register


class GraphLinkDecoder(nn.Module):
    #Decoder for reconstructing graph adjacency matrix using inner product
    
    def __init__(self, activation_function=lambda x: x):
        super().__init__()
        self.activation_function = activation_function
    
    def forward(self, node_embeddings):
        adjacency_reconstruction = self.activation_function(
            torch.mm(node_embeddings, node_embeddings.t())
        )
        return adjacency_reconstruction


class PiGnnModel(nn.Module):

    def __init__(self, backbone_gnn, supplementary_decoder=None):
        super().__init__()
        self.backbone_gnn = backbone_gnn
        self.supplementary_decoder = supplementary_decoder
    
    def forward(self, graph_data):

        node_embeddings = self.backbone_gnn(graph_data)
        supplementary_output = (
            self.supplementary_decoder(node_embeddings) 
            if self.supplementary_decoder is not None else None
        )
        return F.log_softmax(node_embeddings, dim=1), supplementary_output


@register('pi_gnn')
class PiGnnMethodTrainer(BaseTrainer):
    def train(self):
        from methods.registry import get_helper
        from training.training_loop import TrainingLoop

        d = self.init_data
        helper = get_helper('pi_gnn')
        loop = TrainingLoop(helper, log_epoch_fn=self.log_epoch)
        result = loop.run(
            d['backbone_model'], d['data_for_training'],
            self.config, d['device'], d,
        )
        self._loop = loop
        self._helper = helper
        return result