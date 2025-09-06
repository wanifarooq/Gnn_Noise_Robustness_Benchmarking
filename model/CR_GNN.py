import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, mask_feature
from torch_geometric.data import Data
from copy import deepcopy
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model.evaluation import OversmoothingMetrics


class ContrastiveProjectionHead(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.first_layer = torch.nn.Linear(input_dim, output_dim)
        self.second_layer = torch.nn.Linear(output_dim, output_dim)

    def forward(self, embeddings):
        embeddings = F.relu(self.first_layer(embeddings))
        return self.second_layer(embeddings)


class NodeClassificationHead(torch.nn.Module):
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = torch.nn.Linear(input_dim, num_classes)

    def forward(self, embeddings):
        return self.classifier(embeddings)


class ContrastiveLossFunction:
    #Handles contrastive loss
    
    @staticmethod
    def compute_symmetric_contrastive_loss(embeddings_view1, embeddings_view2, temperature):
        #Compute symmetric contrastive loss

        normalized_emb1 = F.normalize(embeddings_view1, p=2, dim=1)
        normalized_emb2 = F.normalize(embeddings_view2, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = (normalized_emb1 @ normalized_emb2.t()) / temperature
        batch_labels = torch.arange(normalized_emb1.size(0), device=normalized_emb1.device)

        # Symmetric loss computation
        forward_loss = F.cross_entropy(similarity_matrix, batch_labels)
        backward_loss = F.cross_entropy(similarity_matrix.t(), batch_labels)
        return 0.5 * (forward_loss + backward_loss)

    @staticmethod
    def compute_dynamic_classification_loss(predictions_view1, predictions_view2, true_labels):
        #Compute dynamic cross-entropy loss
        loss_view1 = F.cross_entropy(predictions_view1, true_labels)
        loss_view2 = F.cross_entropy(predictions_view2, true_labels)
        return (loss_view1 + loss_view2) / 2

    @staticmethod
    def compute_cross_space_consistency_loss(metric_embeddings, prediction_embeddings):
        #Compute consistency loss
        return F.mse_loss(metric_embeddings, prediction_embeddings)


class DataAugmentationManager:
    #Handles data augmentation for contrastive learning
    
    @staticmethod
    def create_augmented_views(node_features, edge_index, dropout_rate=0.3, feature_mask_rate=0.3):
        #Create two augmented views of the graph data

        edge_index_view1, _ = dropout_adj(edge_index, p=dropout_rate)
        edge_index_view2, _ = dropout_adj(edge_index, p=dropout_rate)
        
        features_view1, _ = mask_feature(node_features, p=feature_mask_rate)
        features_view2, _ = mask_feature(node_features, p=feature_mask_rate)
        
        return (features_view1, edge_index_view1), (features_view2, edge_index_view2)


class OversmoothingnessEvaluator:
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.metric_calculator = OversmoothingMetrics(device=device)
        
        self.training_history = {
            'train': [],
            'validation': [],
            'test': []
        }
    
    def compute_metrics_for_node_subset(self, node_embeddings, edge_index, node_mask, node_labels=None):

        try:
            masked_node_indices = torch.where(node_mask)[0]
            subset_embeddings = node_embeddings[node_mask]
            
            index_mapping_set = set(masked_node_indices.cpu().numpy())
            valid_edge_mask = torch.tensor([
                source_idx.item() in index_mapping_set and target_idx.item() in index_mapping_set
                for source_idx, target_idx in edge_index.t()
            ], device=edge_index.device)
            
            if not valid_edge_mask.any():
                return self._create_default_metrics(subset_embeddings.shape[0])
            
            filtered_edges = edge_index[:, valid_edge_mask]
            index_mapping = {orig_idx.item(): local_idx for local_idx, orig_idx in enumerate(masked_node_indices)}
            
            remapped_edges = torch.stack([
                torch.tensor([index_mapping[src.item()] for src in filtered_edges[0]], device=edge_index.device),
                torch.tensor([index_mapping[tgt.item()] for tgt in filtered_edges[1]], device=edge_index.device)
            ])
            
            graph_data_list = [{
                'X': subset_embeddings,
                'edge_index': remapped_edges,
                'edge_weight': None
            }]
            
            return self.metric_calculator.compute_all_metrics(
                X=subset_embeddings,
                edge_index=remapped_edges,
                graphs_in_class=graph_data_list
            )
            
        except Exception as error:
            print(f"Warning: Could not compute oversmoothing metrics for node subset: {error}")
            return None
    
    def _create_default_metrics(self, embedding_dim):
        return {
            'NumRank': float(min(embedding_dim, embedding_dim)),
            'Erank': float(min(embedding_dim, embedding_dim)),
            'EDir': 0.0,
            'EDir_traditional': 0.0,
            'EProj': 0.0,
            'MAD': 0.0
        }
    
    def get_training_history(self):
        return self.training_history


class CRGNNModel:
    
    def __init__(self, device='cuda', **training_config):
        self.device = torch.device(device)
        self.hyperparameters = self._initialize_hyperparameters(**training_config)
        
        self.best_validation_loss = float('inf')
        self.early_stopping_counter = 0
        self.best_model_weights = None
        
        self.oversmoothing_evaluator = OversmoothingnessEvaluator(device=device)
        self.loss_calculator = ContrastiveLossFunction()
        self.data_augmentor = DataAugmentationManager()
    
    def _initialize_hyperparameters(self, **config):

        return {
            'embedding_dim': config.get('hidden_channels', 64),
            'learning_rate': config.get('lr', 0.001),
            'weight_decay': config.get('weight_decay', 5e-4),
            'max_epochs': config.get('epochs', 200),
            'early_stopping_patience': config.get('patience', 10),
            'temperature_T': config.get('T', 0.5),
            'contrastive_temperature': config.get('tau', 0.5),
            'consistency_threshold': config.get('p', 0.5),
            'contrastive_weight': config.get('alpha', 1.0),
            'consistency_weight': config.get('beta', 0.0),
            'enable_debug': config.get('debug', True),
            'oversmoothing_eval_frequency': config.get('eval_oversmoothing_freq', 10)
        }
    
    def train_model(self, backbone_model, graph_data, model_config, model_factory_function):

        from sklearn.metrics import precision_score, recall_score
        
        print(f"Training CR-GNN with {model_config['model_name'].upper()} backbone")
        
        if graph_data.num_nodes > 10000:
            print(f"Warning: large dataset with {graph_data.num_nodes} nodes.")
        
        graph_data = graph_data.to(self.device)
        num_classes = graph_data.y.max().item() + 1
        
        # Initialize backbone encoder
        backbone_encoder = backbone_model.to(self.device)
        
        with torch.no_grad():
            sample_output = backbone_encoder(graph_data)
        
        backbone_output_dim = sample_output.size(1)
        target_embedding_dim = self.hyperparameters['embedding_dim']
        
        if backbone_output_dim != target_embedding_dim:
            dimension_adapter = nn.Linear(backbone_output_dim, target_embedding_dim).to(self.device)
            final_embedding_dim = target_embedding_dim
        else:
            dimension_adapter = nn.Identity().to(self.device)
            final_embedding_dim = backbone_output_dim
        
        # Initialize heads
        contrastive_head = ContrastiveProjectionHead(final_embedding_dim, final_embedding_dim).to(self.device)
        classification_head = NodeClassificationHead(final_embedding_dim, num_classes).to(self.device)
        
        trainable_parameters = (list(backbone_encoder.parameters()) + 
                              list(dimension_adapter.parameters()) +
                              list(contrastive_head.parameters()) +
                              list(classification_head.parameters()))
        
        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=self.hyperparameters['learning_rate'],
            weight_decay=self.hyperparameters['weight_decay']
        )
        
        if hasattr(graph_data, 'train_mask') and hasattr(graph_data, 'val_mask') and hasattr(graph_data, 'test_mask'):
            train_mask = graph_data.train_mask
            validation_mask = graph_data.val_mask
            test_mask = graph_data.test_mask
        else:
            total_nodes = graph_data.x.size(0)
            train_size = int(0.6 * total_nodes)
            val_size = int(0.2 * total_nodes)
            
            random_indices = torch.randperm(total_nodes)
            train_mask = torch.zeros(total_nodes, dtype=torch.bool)
            validation_mask = torch.zeros(total_nodes, dtype=torch.bool)
            test_mask = torch.zeros(total_nodes, dtype=torch.bool)
            
            train_mask[random_indices[:train_size]] = True
            validation_mask[random_indices[train_size:train_size + val_size]] = True
            test_mask[random_indices[train_size + val_size:]] = True
            
            train_mask = train_mask.to(self.device)
            validation_mask = validation_mask.to(self.device)
            test_mask = test_mask.to(self.device)
        
        clean_labels = getattr(graph_data, 'y_original', graph_data.y)
        noisy_labels = graph_data.y
        
        # Training loop
        training_start_time = time.time()
        total_training_time = 0
        
        oversmoothing_frequency = 20
        
        for epoch in range(self.hyperparameters['max_epochs']):
            backbone_encoder.train()
            dimension_adapter.train()
            contrastive_head.train()
            classification_head.train()
            
            optimizer.zero_grad()
            
            calculate_oversmoothing = (epoch + 1) % oversmoothing_frequency == 0 or epoch == 0
            
            if calculate_oversmoothing:
                embeddings, training_loss, train_accuracy, train_f1_metric, train_oversmoothing = self._forward_pass_with_augmentation(
                    backbone_encoder, dimension_adapter, contrastive_head, classification_head,
                    graph_data.x, graph_data.edge_index, noisy_labels, train_mask
                )
            else:
                embeddings, training_loss, train_accuracy, train_f1_metric, _ = self._forward_pass_with_augmentation_no_oversmoothing(
                    backbone_encoder, dimension_adapter, contrastive_head, classification_head,
                    graph_data.x, graph_data.edge_index, noisy_labels, train_mask
                )
                train_oversmoothing = None
            
            if training_loss is not None and torch.isfinite(training_loss):
                training_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            val_loss, val_accuracy, val_f1_metric, val_oversmoothing = self._evaluate_model_performance(
                backbone_encoder, dimension_adapter, classification_head,
                graph_data.x, graph_data.edge_index, noisy_labels, validation_mask,
                compute_oversmoothing=calculate_oversmoothing
            )
            
            if calculate_oversmoothing:
                if train_oversmoothing is not None:
                    self.oversmoothing_evaluator.training_history['train'].append(train_oversmoothing)
                if val_oversmoothing is not None:
                    self.oversmoothing_evaluator.training_history['validation'].append(val_oversmoothing)
            
            if self.hyperparameters['enable_debug']:
                if calculate_oversmoothing:

                    self._print_epoch_metrics(epoch + 1, train_accuracy, val_accuracy, train_f1_metric, val_f1_metric,
                                            train_oversmoothing, val_oversmoothing)
                else:

                    self._print_epoch_metrics_simple(epoch + 1, train_accuracy, val_accuracy, train_f1_metric, val_f1_metric)
            
            # Early stopping
            if val_loss < self.best_validation_loss:
                self.best_validation_loss = val_loss
                self.early_stopping_counter = 0
                total_training_time = time.time() - training_start_time
                
                self.best_model_weights = {
                    'backbone_encoder': deepcopy(backbone_encoder.state_dict()),
                    'dimension_adapter': deepcopy(dimension_adapter.state_dict()),
                    'contrastive_head': deepcopy(contrastive_head.state_dict()),
                    'classification_head': deepcopy(classification_head.state_dict())
                }
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.hyperparameters['early_stopping_patience']:
                    break
        
        if self.best_model_weights is not None:
            backbone_encoder.load_state_dict(self.best_model_weights['backbone_encoder'])
            dimension_adapter.load_state_dict(self.best_model_weights['dimension_adapter'])
            contrastive_head.load_state_dict(self.best_model_weights['contrastive_head'])
            classification_head.load_state_dict(self.best_model_weights['classification_head'])
        
        final_results = self._compute_final_evaluation(
            backbone_encoder, dimension_adapter, classification_head,
            graph_data, clean_labels, train_mask, validation_mask, test_mask,
            total_training_time
        )
        
        return final_results

    def _forward_pass_with_augmentation(self, base_encoder, dimension_adapter, projection_head, 
                                      classification_head, node_features, edge_index, 
                                      node_labels=None, training_mask=None, compute_oversmoothing=True):

        standard_embeddings = base_encoder(Data(x=node_features, edge_index=edge_index))
        adapted_embeddings = dimension_adapter(standard_embeddings)
        final_embeddings = adapted_embeddings
        
        total_loss, accuracy, f1_metric, oversmoothing_metrics = None, None, None, None
        
        if node_labels is not None and training_mask is not None:

            view1_data, view2_data = self.data_augmentor.create_augmented_views(
                node_features, edge_index, dropout_rate=0.3, feature_mask_rate=0.3
            )
            
            view1_embeddings = base_encoder(Data(x=view1_data[0], edge_index=view1_data[1]))
            view1_embeddings = dimension_adapter(view1_embeddings)
            
            view2_embeddings = base_encoder(Data(x=view2_data[0], edge_index=view2_data[1]))
            view2_embeddings = dimension_adapter(view2_embeddings)
            
            # Contrastive learning components
            view1_projections = projection_head(view1_embeddings)
            view2_projections = projection_head(view2_embeddings)
            contrastive_loss = self.loss_calculator.compute_symmetric_contrastive_loss(
                view1_projections, view2_projections, self.hyperparameters['contrastive_temperature']
            )
            
            # Classification components
            view1_predictions = classification_head(view1_embeddings)
            view2_predictions = classification_head(view2_embeddings)
            classification_loss = self.loss_calculator.compute_dynamic_classification_loss(
                view1_predictions[training_mask], view2_predictions[training_mask], node_labels[training_mask]
            )
            
            if self.hyperparameters['consistency_weight'] > 0:
                metric_similarities = torch.exp(
                    F.cosine_similarity(view1_projections.unsqueeze(1), view2_projections.unsqueeze(0), dim=2) 
                    / self.hyperparameters['temperature_T']
                ).mean(dim=1)
                
                prediction_similarities = torch.exp(
                    F.cosine_similarity(view1_predictions.unsqueeze(1), view2_predictions.unsqueeze(0), dim=2) 
                    / self.hyperparameters['temperature_T']
                ).mean(dim=1)
                
                thresholded_pred_sim = torch.where(
                    prediction_similarities > self.hyperparameters['consistency_threshold'], 
                    prediction_similarities, 
                    torch.zeros_like(prediction_similarities)
                )
                
                consistency_loss = self.loss_calculator.compute_cross_space_consistency_loss(
                    metric_similarities, thresholded_pred_sim
                )
                
                total_loss = (self.hyperparameters['contrastive_weight'] * contrastive_loss + 
                            classification_loss + 
                            self.hyperparameters['consistency_weight'] * consistency_loss)
            else:
                total_loss = (self.hyperparameters['contrastive_weight'] * contrastive_loss + 
                            classification_loss)
            
            with torch.no_grad():
                final_predictions = classification_head(final_embeddings)
                predicted_labels = final_predictions[training_mask].argmax(dim=1)
                
                accuracy = accuracy_score(
                    node_labels[training_mask].cpu().numpy(), 
                    predicted_labels.cpu().numpy()
                )
                f1_metric = f1_score(
                    node_labels[training_mask].cpu().numpy(), 
                    predicted_labels.cpu().numpy(), 
                    average='macro'
                )
                
                if compute_oversmoothing:
                    oversmoothing_metrics = self.oversmoothing_evaluator.compute_metrics_for_node_subset(
                        final_embeddings, edge_index, training_mask, node_labels
                    )
        
        return final_embeddings, total_loss, accuracy, f1_metric, oversmoothing_metrics


    def _evaluate_model_performance(self, base_encoder, dimension_adapter, classification_head, 
                                  node_features, edge_index, node_labels, evaluation_mask, compute_oversmoothing=True):

        base_encoder.eval()
        dimension_adapter.eval()
        classification_head.eval()
        
        with torch.no_grad():
            embeddings = base_encoder(Data(x=node_features, edge_index=edge_index))
            embeddings = dimension_adapter(embeddings)
            predictions = classification_head(embeddings)
            
            evaluation_loss = F.cross_entropy(predictions[evaluation_mask], node_labels[evaluation_mask])
            predicted_labels = predictions[evaluation_mask].argmax(dim=1)
            
            accuracy = accuracy_score(
                node_labels[evaluation_mask].cpu().numpy(), 
                predicted_labels.cpu().numpy()
            )
            f1_metric = f1_score(
                node_labels[evaluation_mask].cpu().numpy(), 
                predicted_labels.cpu().numpy(), 
                average='macro'
            )
            
            oversmoothing_metrics = None
            if compute_oversmoothing:
                oversmoothing_metrics = self.oversmoothing_evaluator.compute_metrics_for_node_subset(
                    embeddings, edge_index, evaluation_mask, node_labels
                )
        
        return evaluation_loss, accuracy, f1_metric, oversmoothing_metrics

    def _forward_pass_with_augmentation_no_oversmoothing(self, base_encoder, dimension_adapter, projection_head, 
                                                        classification_head, node_features, edge_index, 
                                                        node_labels=None, training_mask=None):

        standard_embeddings = base_encoder(Data(x=node_features, edge_index=edge_index))
        adapted_embeddings = dimension_adapter(standard_embeddings)
        final_embeddings = adapted_embeddings
        
        total_loss, accuracy, f1_metric = None, None, None
        
        if node_labels is not None and training_mask is not None:

            view1_data, view2_data = self.data_augmentor.create_augmented_views(
                node_features, edge_index, dropout_rate=0.3, feature_mask_rate=0.3
            )
            
            view1_embeddings = base_encoder(Data(x=view1_data[0], edge_index=view1_data[1]))
            view1_embeddings = dimension_adapter(view1_embeddings)
            
            view2_embeddings = base_encoder(Data(x=view2_data[0], edge_index=view2_data[1]))
            view2_embeddings = dimension_adapter(view2_embeddings)
            
            # Contrastive learning components
            view1_projections = projection_head(view1_embeddings)
            view2_projections = projection_head(view2_embeddings)
            contrastive_loss = self.loss_calculator.compute_symmetric_contrastive_loss(
                view1_projections, view2_projections, self.hyperparameters['contrastive_temperature']
            )
            
            # Classification components
            view1_predictions = classification_head(view1_embeddings)
            view2_predictions = classification_head(view2_embeddings)
            classification_loss = self.loss_calculator.compute_dynamic_classification_loss(
                view1_predictions[training_mask], view2_predictions[training_mask], node_labels[training_mask]
            )
            
            if self.hyperparameters['consistency_weight'] > 0:
                metric_similarities = torch.exp(
                    F.cosine_similarity(view1_projections.unsqueeze(1), view2_projections.unsqueeze(0), dim=2) 
                    / self.hyperparameters['temperature_T']
                ).mean(dim=1)
                
                prediction_similarities = torch.exp(
                    F.cosine_similarity(view1_predictions.unsqueeze(1), view2_predictions.unsqueeze(0), dim=2) 
                    / self.hyperparameters['temperature_T']
                ).mean(dim=1)
                
                thresholded_pred_sim = torch.where(
                    prediction_similarities > self.hyperparameters['consistency_threshold'], 
                    prediction_similarities, 
                    torch.zeros_like(prediction_similarities)
                )
                
                consistency_loss = self.loss_calculator.compute_cross_space_consistency_loss(
                    metric_similarities, thresholded_pred_sim
                )
                
                total_loss = (self.hyperparameters['contrastive_weight'] * contrastive_loss + 
                            classification_loss + 
                            self.hyperparameters['consistency_weight'] * consistency_loss)
            else:
                total_loss = (self.hyperparameters['contrastive_weight'] * contrastive_loss + 
                            classification_loss)
            
            with torch.no_grad():
                final_predictions = classification_head(final_embeddings)
                predicted_labels = final_predictions[training_mask].argmax(dim=1)
                
                accuracy = accuracy_score(
                    node_labels[training_mask].cpu().numpy(), 
                    predicted_labels.cpu().numpy()
                )
                f1_metric = f1_score(
                    node_labels[training_mask].cpu().numpy(), 
                    predicted_labels.cpu().numpy(), 
                    average='macro'
                )
        
        return final_embeddings, total_loss, accuracy, f1_metric, None


    def _evaluate_model_performance_no_oversmoothing(self, base_encoder, dimension_adapter, classification_head, 
                                                    node_features, edge_index, node_labels, evaluation_mask):
        base_encoder.eval()
        dimension_adapter.eval()
        classification_head.eval()
        
        with torch.no_grad():
            embeddings = base_encoder(Data(x=node_features, edge_index=edge_index))
            embeddings = dimension_adapter(embeddings)
            predictions = classification_head(embeddings)
            
            evaluation_loss = F.cross_entropy(predictions[evaluation_mask], node_labels[evaluation_mask])
            predicted_labels = predictions[evaluation_mask].argmax(dim=1)
            
            accuracy = accuracy_score(
                node_labels[evaluation_mask].cpu().numpy(), 
                predicted_labels.cpu().numpy()
            )
            f1_metric = f1_score(
                node_labels[evaluation_mask].cpu().numpy(), 
                predicted_labels.cpu().numpy(), 
                average='macro'
            )
        
        return evaluation_loss, accuracy, f1_metric, None


    def _print_epoch_metrics_simple(self, epoch, train_acc, val_acc, train_f1_metric, val_f1_metric):

        print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
              f"Train F1: {train_f1_metric:.4f}, Val F1: {val_f1_metric:.4f}")
        
    def _print_epoch_metrics(self, epoch, train_acc, val_acc, train_f1_metric, val_f1_metric, 
                           train_oversmoothing, val_oversmoothing):

        print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
              f"Train F1: {train_f1_metric:.4f}, Val F1: {val_f1_metric:.4f}")
        
        if train_oversmoothing and val_oversmoothing:
            train_metrics = train_oversmoothing
            val_metrics = val_oversmoothing
            
            print(f"Train DE: {train_metrics['EDir']:.4f}, Val DE: {val_metrics['EDir']:.4f} | "
                  f"Train DE_trad: {train_metrics['EDir_traditional']:.4f}, Val DE_trad: {val_metrics['EDir_traditional']:.4f} | "
                  f"Train EProj: {train_metrics['EProj']:.4f}, Val EProj: {val_metrics['EProj']:.4f} | "
                  f"Train MAD: {train_metrics['MAD']:.4f}, Val MAD: {val_metrics['MAD']:.4f} | "
                  f"Train NumRank: {train_metrics['NumRank']:.4f}, Val NumRank: {val_metrics['NumRank']:.4f} | "
                  f"Train Erank: {train_metrics['Erank']:.4f}, Val Erank: {val_metrics['Erank']:.4f}")
    
    def _compute_final_evaluation(self, backbone_encoder, dimension_adapter, classification_head,
                                graph_data, clean_labels, train_mask, validation_mask, test_mask,
                                training_time):

        from sklearn.metrics import precision_score, recall_score
        
        final_train_loss, final_train_acc, final_train_f1_metric, final_train_oversmoothing = self._evaluate_model_performance(
            backbone_encoder, dimension_adapter, classification_head,
            graph_data.x, graph_data.edge_index, clean_labels, train_mask
        )
        
        final_val_loss, final_val_acc, final_val_f1_metric, final_val_oversmoothing = self._evaluate_model_performance(
            backbone_encoder, dimension_adapter, classification_head,
            graph_data.x, graph_data.edge_index, clean_labels, validation_mask
        )
        
        test_loss, test_acc, test_f1_metric, test_oversmoothing = self._evaluate_model_performance(
            backbone_encoder, dimension_adapter, classification_head,
            graph_data.x, graph_data.edge_index, clean_labels, test_mask
        )
        
        backbone_encoder.eval()
        dimension_adapter.eval()
        classification_head.eval()
        
        with torch.no_grad():
            embeddings = backbone_encoder(Data(x=graph_data.x, edge_index=graph_data.edge_index))
            embeddings = dimension_adapter(embeddings)
            test_predictions = classification_head(embeddings)
            
            true_test_labels = clean_labels[test_mask].cpu().numpy()
            predicted_test_labels = test_predictions[test_mask].argmax(dim=1).cpu().numpy()
            
            test_precision = precision_score(true_test_labels, predicted_test_labels, average='macro', zero_division=0)
            test_recall = recall_score(true_test_labels, predicted_test_labels, average='macro', zero_division=0)
        
        if test_oversmoothing is not None:
            self.oversmoothing_evaluator.training_history['test'].append(test_oversmoothing)
        
        if self.hyperparameters['enable_debug']:
            self._print_final_results(training_time, test_loss, test_acc, test_f1_metric, 
                                    test_precision, test_recall, final_train_oversmoothing,
                                    final_val_oversmoothing, test_oversmoothing)
        
        return {
            'accuracy': float(test_acc),
            'f1': float(test_f1_metric),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'oversmoothing': test_oversmoothing if test_oversmoothing is not None else {
                'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0
            }
        }
        
    def _print_final_results(self, training_time, test_loss, test_acc, test_f1_metric, 
                           test_precision, test_recall, train_oversmoothing, 
                           val_oversmoothing, test_oversmoothing):
  
        print(f"\nTraining completed in {training_time:.2f}s")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1_metric:.4f}")
        print(f"Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f}")
        print("Final Oversmoothing Metrics:")
        
        for split_name, metrics in [("Train", train_oversmoothing), ("Val", val_oversmoothing), ("Test", test_oversmoothing)]:
            if metrics is not None:
                print(f"{split_name}: EDir: {metrics['EDir']:.4f}, EDir_traditional: {metrics['EDir_traditional']:.4f}, "
                      f"EProj: {metrics['EProj']:.4f}, MAD: {metrics['MAD']:.4f}, "
                      f"NumRank: {metrics['NumRank']:.4f}, Erank: {metrics['Erank']:.4f}")
    
    def get_oversmoothing_training_history(self):
        return self.oversmoothing_evaluator.get_training_history()