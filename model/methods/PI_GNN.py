import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import time
from collections import defaultdict

from model.evaluation import (OversmoothingMetrics, ClassificationMetrics,
                              compute_oversmoothing_for_mask)
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


class PiGnnTrainer:
    
    def __init__(self, device, epochs=400, mutual_info_start_epoch=200, 
                 use_self_mi=False, main_learning_rate=0.01, mi_learning_rate=0.01, 
                 weight_decay=5e-4, normalization_factor=None, use_vanilla_training=False, 
                 early_stopping_patience=50, improvement_threshold=1e-4,
                 oversmoothing_every=20):

        self.device = device
        self.total_epochs = epochs
        self.mi_regularization_start_epoch = mutual_info_start_epoch
        self.use_self_mutual_information = use_self_mi
        self.main_model_lr = main_learning_rate
        self.mi_model_lr = mi_learning_rate
        self.l2_regularization = weight_decay
        self.loss_normalization_factor = normalization_factor
        self.vanilla_training_mode = use_vanilla_training
        self.early_stop_patience = early_stopping_patience
        self.min_improvement_delta = improvement_threshold
        self.oversmoothing_every = oversmoothing_every

        self.oversmoothing_evaluator = OversmoothingMetrics(device=device)
        self.cls_evaluator = ClassificationMetrics(average='macro')
        self.training_history = {
            'train': [],
            'val': [],
            'test': []
        }

    def _determine_learning_rate(self, config, model_type):

        if model_type.lower() in ['gat', 'gatv2']:
            return config.get('lr', 0.005)
        else:
            return config.get('lr', 0.01)

    def _evaluate_model_performance(self, model, graph_data):
        model.eval()
        with torch.no_grad():
            classification_output, _ = model(graph_data)
            predicted_labels = classification_output.argmax(dim=1)
            
            performance_metrics = {}
            
            # Training metrics
            train_loss = F.nll_loss(
                classification_output[graph_data.train_mask], 
                graph_data.y[graph_data.train_mask]
            ).item()
            train_accuracy = (
                predicted_labels[graph_data.train_mask]
                .eq(graph_data.y[graph_data.train_mask])
                .sum().item() / graph_data.train_mask.sum().item()
            )
            train_f1 = self.cls_evaluator.compute_f1(
                predicted_labels[graph_data.train_mask], 
                graph_data.y[graph_data.train_mask], 
                average='macro'
            )
            
            performance_metrics.update({
                'train_loss': train_loss,
                'train_acc': train_accuracy,
                'train_f1': train_f1
            })
            
            val_loss = F.nll_loss(
                classification_output[graph_data.val_mask], 
                graph_data.y[graph_data.val_mask]
            ).item()
            val_accuracy = (
                predicted_labels[graph_data.val_mask]
                .eq(graph_data.y[graph_data.val_mask])
                .sum().item() / graph_data.val_mask.sum().item()
            )
            val_f1 = self.cls_evaluator.compute_f1(
                predicted_labels[graph_data.val_mask], 
                graph_data.y[graph_data.val_mask], 
                average='macro'
            )
            
            performance_metrics.update({
                'val_loss': val_loss,
                'val_acc': val_accuracy,
                'val_f1': val_f1
            })

            hidden_representations = model.backbone_gnn.get_embeddings(graph_data)
            train_oversmoothing_metrics = compute_oversmoothing_for_mask(
                self.oversmoothing_evaluator, hidden_representations, graph_data.edge_index, graph_data.train_mask
            )
            val_oversmoothing_metrics = compute_oversmoothing_for_mask(
                self.oversmoothing_evaluator, hidden_representations, graph_data.edge_index, graph_data.val_mask
            )
            
            return performance_metrics, train_oversmoothing_metrics, val_oversmoothing_metrics


    def train_model(self, model, graph_data, config=None, model_factory_function=None,
                    log_epoch_fn=None):
        per_epochs_oversmoothing = defaultdict(list)
        per_epochs_val_oversmoothing = defaultdict(list)
        training_start_time = time.time()
        
        graph_data = graph_data.to(self.device)
        model = model.to(self.device)
        num_output_classes = graph_data.y.max().item() + 1
        # Create mutual information model
        if model_factory_function and config:
            mi_backbone_gnn = model_factory_function(
                model_name=config['model']['name'],
                in_channels=graph_data.num_features,
                hidden_channels=config.get('hidden_channels', 64),
                out_channels=num_output_classes,
                n_layers=config.get('n_layers', 2),
                dropout=config.get('dropout', 0.5),
                mlp_layers=config.get('mlp_layers', 2),
                train_eps=config.get('train_eps', True),
                heads=config.get('heads', 8),
                self_loop=config.get('self_loop', True)
            )
            mi_link_decoder = GraphLinkDecoder()
            mutual_information_model = PiGnnModel(
                backbone_gnn=mi_backbone_gnn, 
                supplementary_decoder=mi_link_decoder
            ).to(self.device)
            main_lr = self._determine_learning_rate(config, config['model']['name'])
            mi_lr = self._determine_learning_rate(config, config['model']['name'])
        else:
            mutual_information_model = PiGnnModel(
                backbone_gnn=type(model.backbone_gnn)(
                    in_channels=graph_data.num_features,
                    hidden_channels=64,
                    out_channels=num_output_classes
                ),
                supplementary_decoder=GraphLinkDecoder()
            ).to(self.device)
            main_lr = self.main_model_lr
            mi_lr = self.mi_model_lr

        # Initialize optimizers
        main_optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=main_lr, 
            weight_decay=self.l2_regularization
        )
        mi_optimizer = torch.optim.Adam(
            mutual_information_model.parameters(), 
            lr=mi_lr, 
            weight_decay=self.l2_regularization
        )

        # Prepare adjacency matrix for link prediction
        training_edges = graph_data.edge_index.t().cpu().numpy()
        edge_weights = np.ones(training_edges.shape[0])
        adjacency_matrix_csr = sp.csr_matrix(
            (edge_weights, (training_edges[:, 0], training_edges[:, 1])),
            shape=(graph_data.num_nodes, graph_data.num_nodes)
        )
        symmetric_adjacency = (adjacency_matrix_csr + adjacency_matrix_csr.T) / 2
        adjacency_with_self_loops = symmetric_adjacency + sp.eye(symmetric_adjacency.shape[0])
        adjacency_target = torch.FloatTensor(adjacency_with_self_loops.toarray()).to(self.device)
        
        positive_edge_weight = torch.tensor(
            [float(graph_data.num_nodes ** 2 - len(edge_weights)) / len(edge_weights)], 
            device=self.device
        )
        
        loss_normalization = (
            self.loss_normalization_factor 
            if self.loss_normalization_factor is not None and self.loss_normalization_factor != 10000 
            else graph_data.num_nodes ** 2 / float((graph_data.num_nodes ** 2 - len(edge_weights)) * 2)
        )

        training_node_labels = graph_data.y[graph_data.train_mask]

        # Early stopping variables
        best_validation_loss = float('inf')
        best_training_epoch = 0
        patience_counter = 0

        for current_epoch in range(self.total_epochs):
            model.train()
            mutual_information_model.train()
            main_optimizer.zero_grad(set_to_none=True)
            mi_optimizer.zero_grad(set_to_none=True)

            main_classification_output, main_link_prediction = model(graph_data)
            mi_classification_output, mi_link_prediction = mutual_information_model(graph_data)

            # Classification loss
            classification_loss = F.nll_loss(main_classification_output[graph_data.train_mask], training_node_labels)
            
            # Mutual information loss
            mi_reconstruction_loss = loss_normalization * F.binary_cross_entropy_with_logits(
                mi_link_prediction, adjacency_target, pos_weight=positive_edge_weight
            )
            mi_reconstruction_loss.backward()
            mi_optimizer.step()

            # Context-aware regularization loss
            context_regularization_loss = 0
            if not self.vanilla_training_mode:
                if current_epoch > self.mi_regularization_start_epoch:
                    # Create importance mask based on MI predictions
                    importance_mask = torch.zeros_like(mi_link_prediction).view(-1).to(self.device)
                    positive_edge_positions = adjacency_target.view(-1).bool()
                    negative_edge_positions = (~adjacency_target.view(-1).bool())
                    
                    if self.use_self_mutual_information:
                        mask_predictions = main_link_prediction
                        importance_mask[positive_edge_positions] = torch.sigmoid(mask_predictions).view(-1)[positive_edge_positions]
                        importance_mask[negative_edge_positions] = 1 - torch.sigmoid(mask_predictions).view(-1)[negative_edge_positions]
                    else:
                        importance_mask[positive_edge_positions] = torch.sigmoid(mi_link_prediction).view(-1)[positive_edge_positions]
                        importance_mask[negative_edge_positions] = 1 - torch.sigmoid(mi_link_prediction).view(-1)[negative_edge_positions]
                    
                    importance_mask = importance_mask.view(adjacency_target.size(0), adjacency_target.size(1))
                    weighted_reconstruction_loss = F.binary_cross_entropy_with_logits(
                        main_link_prediction, adjacency_target, 
                        pos_weight=positive_edge_weight, reduction='none'
                    ) * importance_mask.detach()
                    context_regularization_loss = loss_normalization * weighted_reconstruction_loss.mean()
                else:
                    context_regularization_loss = loss_normalization * F.binary_cross_entropy_with_logits(
                        main_link_prediction, adjacency_target, pos_weight=positive_edge_weight
                    )

            # Total loss
            total_training_loss = classification_loss + context_regularization_loss
            total_training_loss.backward()
            main_optimizer.step()

            # Evaluate current performance
            current_metrics, train_oversmoothing, val_oversmoothing = self._evaluate_model_performance(model, graph_data)

            if train_oversmoothing is not None:
                self.training_history['train'].append(train_oversmoothing)
            if val_oversmoothing is not None:
                self.training_history['val'].append(val_oversmoothing)

            # logging
            os_entry = None
            if current_epoch % self.oversmoothing_every == 0:
                logged_train_os, logged_val_os = self._log_training_progress(current_epoch, current_metrics, train_oversmoothing, val_oversmoothing)
                for key, value in logged_train_os.items():
                    per_epochs_oversmoothing[key].append(value)
                for key, value in logged_val_os.items():
                    per_epochs_val_oversmoothing[key].append(value)
                os_entry = {'train': dict(logged_train_os), 'val': dict(logged_val_os)}

            # Early stopping
            is_best = current_metrics['val_loss'] < best_validation_loss
            if log_epoch_fn is not None:
                with torch.no_grad():
                    classification_output, _ = model(graph_data)
                    pred = classification_output.argmax(dim=1)
                log_epoch_fn(current_epoch, current_metrics['train_loss'], current_metrics['val_loss'],
                             current_metrics['train_acc'], current_metrics['val_acc'],
                             train_f1=current_metrics['train_f1'], val_f1=current_metrics['val_f1'],
                             oversmoothing=os_entry, is_best=is_best,
                             train_predictions=pred)
            if is_best:
                best_validation_loss = current_metrics['val_loss']
                best_training_epoch = current_epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {current_epoch}, best epoch {best_training_epoch}")
                break

        total_training_time = time.time() - training_start_time
        print(f"\nTraining completed in {total_training_time:.2f}s")

        return {
            'train_oversmoothing': dict(per_epochs_oversmoothing),
            'val_oversmoothing': dict(per_epochs_val_oversmoothing),
            'stopped_at_epoch': current_epoch,
        }

    def _log_training_progress(self, epoch, metrics, train_oversmoothing, val_oversmoothing):

        train_edir = train_oversmoothing['EDir'] if train_oversmoothing else 0.0
        train_edir_traditional = train_oversmoothing['EDir_traditional'] if train_oversmoothing else 0.0
        train_eproj = train_oversmoothing['EProj'] if train_oversmoothing else 0.0
        train_mad = train_oversmoothing['MAD'] if train_oversmoothing else 0.0
        train_num_rank = train_oversmoothing['NumRank'] if train_oversmoothing else 0.0
        train_effective_rank = train_oversmoothing['Erank'] if train_oversmoothing else 0.0
        
        val_edir = val_oversmoothing['EDir'] if val_oversmoothing else 0.0
        val_edir_traditional = val_oversmoothing['EDir_traditional'] if val_oversmoothing else 0.0
        val_eproj = val_oversmoothing['EProj'] if val_oversmoothing else 0.0
        val_mad = val_oversmoothing['MAD'] if val_oversmoothing else 0.0
        val_num_rank = val_oversmoothing['NumRank'] if val_oversmoothing else 0.0
        val_effective_rank = val_oversmoothing['Erank'] if val_oversmoothing else 0.0

        print(f"Epoch {epoch:03d} | Train Loss: {metrics['train_loss']:.4f}, Val Loss: {metrics['val_loss']:.4f} | "
              f"Train Acc: {metrics['train_acc']:.4f}, Val Acc: {metrics['val_acc']:.4f} | "
              f"Train F1: {metrics['train_f1']:.4f}, Val F1: {metrics['val_f1']:.4f}")
        print(f"Train DE: {train_edir:.4f}, Val DE: {val_edir:.4f} | "
              f"Train DE_trad: {train_edir_traditional:.4f}, Val DE_trad: {val_edir_traditional:.4f} | "
              f"Train EProj: {train_eproj:.4f}, Val EProj: {val_eproj:.4f} | "
              f"Train MAD: {train_mad:.4f}, Val MAD: {val_mad:.4f} | "
              f"Train NumRank: {train_num_rank:.4f}, Val NumRank: {val_num_rank:.4f} | "
              f"Train Erank: {train_effective_rank:.4f}, Val Erank: {val_effective_rank:.4f}")
        return train_oversmoothing, val_oversmoothing

    def get_training_history(self):
        return self.training_history


@register('pi_gnn')
class PiGnnMethodTrainer(BaseTrainer):
    def train(self):
        d = self.init_data
        pi_params = self.config.get('pi_gnn_params', {})

        trainer = PiGnnTrainer(
            device=d['device'],
            epochs=d['epochs'],
            main_learning_rate=d['lr'],
            mi_learning_rate=d['lr'],
            weight_decay=d['weight_decay'],
            early_stopping_patience=d['patience'],
            mutual_info_start_epoch=int(pi_params.get('start_epoch', 200)),
            use_self_mi=bool(pi_params.get('miself', False)),
            normalization_factor=pi_params.get('norm', None),
            use_vanilla_training=bool(pi_params.get('vanilla', False)),
            oversmoothing_every=d['oversmoothing_every'],
        )

        link_decoder = GraphLinkDecoder()
        pi_gnn_model = PiGnnModel(
            backbone_gnn=d['backbone_model'],
            supplementary_decoder=link_decoder,
        )

        return trainer.train_model(
            pi_gnn_model, d['data_for_training'],
            self.config, d['get_model'],
            log_epoch_fn=self.log_epoch,
        )