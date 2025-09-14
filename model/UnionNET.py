import time
from copy import deepcopy
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model.evaluation import OversmoothingMetrics


class UnionNET:
    
    def __init__(self, gnn_model, graph_data, num_classes, training_config):

        self.gnn_model = gnn_model.to(graph_data.x.device)
        self.graph_data = graph_data
        self.device = graph_data.x.device
        
        # Training hyperparameters
        self.num_epochs = training_config.get('n_epochs', 200)
        self.learning_rate = training_config.get('lr', 0.01)
        self.weight_decay_coef = float(training_config.get('weight_decay', 5e-4))
        self.early_stop_patience = training_config.get('patience', 100)
        
        # UnionNET specific parameters
        self.support_size_k = training_config.get('k', 5)
        self.reweight_alpha = training_config.get('alpha', 0.5)
        self.kl_beta = training_config.get('beta', 1.0)
        
        self.num_nodes = graph_data.x.shape[0]
        self.feature_dim = graph_data.x.shape[1]
        self.num_classes = num_classes

        self.node_features = graph_data.x.to(torch.float)
        if training_config.get('feat_norm', True):
            self.node_features = self._normalize_node_features(self.node_features)

        self.clean_node_labels = graph_data.y
        self.noisy_node_labels = graph_data.y_noisy

        self.edge_connections = graph_data.edge_index
        self.adjacency_matrix = self._build_adjacency_matrix(
            graph_data.edge_index, self.num_nodes, self.device
        )
 
        self.train_node_mask = graph_data.train_mask
        self.val_node_mask = graph_data.val_mask
        self.test_node_mask = graph_data.test_mask

        self.optimizer = torch.optim.Adam(
            self.gnn_model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay_coef
        )
        self.best_validation_loss = float('inf')
        self.patience_counter = 0
        self.best_model_weights = None
        self.training_results = {'train': -1, 'val': -1, 'test': -1}

        self.oversmoothing_evaluator = OversmoothingMetrics(device=self.device)
    
    def _normalize_node_features(self, features):
        row_sum = torch.sum(features, dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return features / row_sum

    def _build_support_set_for_nodes(self, features, train_mask, labels, edge_index, k):
        #Construct support sets for each node using k most similar neighbors.
        device = features.device
        n_nodes = features.size(0)
        feat_dim = features.size(1)
        
        support_features = torch.zeros(n_nodes, k, feat_dim, device=device)
        support_node_labels = torch.zeros(n_nodes, k, dtype=torch.long, device=device)
        
        for node_idx in range(n_nodes):
            if train_mask[node_idx]:
                neighbor_nodes = edge_index[1][edge_index[0] == node_idx]
                
                if len(neighbor_nodes) >= k:
                    anchor_feature = features[node_idx].unsqueeze(0)
                    similarity_scores = torch.mm(features[neighbor_nodes], anchor_feature.T).squeeze()
                    _, top_k_indices = torch.topk(similarity_scores, k=k)
                    
                    support_features[node_idx] = features[neighbor_nodes[top_k_indices]]
                    support_node_labels[node_idx] = labels[neighbor_nodes[top_k_indices]]
        
        return support_features, support_node_labels

    def _aggregate_labels_from_support(self, support_features, support_labels, node_features, n_classes):
        #Aggregate labels from support set

        device = node_features.device
        n_nodes = node_features.size(0)
        class_probabilities = torch.zeros(n_nodes, n_classes, device=device)
        
        for node_idx in range(n_nodes):
            if torch.sum(support_features[node_idx]) != 0:
                similarity_scores = torch.exp(torch.mm(
                    support_features[node_idx], 
                    node_features[node_idx:node_idx+1].T
                )).squeeze()
                similarity_weights = similarity_scores / torch.sum(similarity_scores)
                
                for support_idx, class_label in enumerate(support_labels[node_idx]):
                    class_probabilities[node_idx, class_label] += similarity_weights[support_idx]
        
        return class_probabilities

    def _build_adjacency_matrix(self, edge_index, n_nodes, device):
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=device),
            [n_nodes, n_nodes]
        ).coalesce()

        identity_matrix = torch.eye(n_nodes, device=device).to_sparse()
        return (adj + identity_matrix).coalesce()

    def _compute_kl_divergence_loss(self, logits, one_hot_labels):
        #Compute KL divergence loss between predictions and labels
        log_probabilities = F.log_softmax(logits, dim=1)
        kl_loss = F.kl_div(log_probabilities, one_hot_labels.float(), reduction='batchmean')
        return kl_loss
    
    def _forward_pass(self, data_split=None, use_clean_labels=False):

        if data_split != 'train':
            self.gnn_model.eval()
        else:
            self.gnn_model.train()
        
        context = torch.no_grad() if data_split != 'train' else torch.enable_grad()
        with context:
            model_output = self.gnn_model(self.graph_data)

            if data_split is not None:
                target_labels = self.clean_node_labels if use_clean_labels else self.noisy_node_labels
                split_mask = getattr(self, f'{data_split}_node_mask')
                
                loss = F.cross_entropy(model_output[split_mask], target_labels[split_mask])
                true_labels = target_labels[split_mask].cpu().numpy()
                predicted_labels = model_output[split_mask].detach().cpu().numpy().argmax(1)
                accuracy = accuracy_score(true_labels, predicted_labels)
                return model_output, loss, accuracy
        
        return model_output
    
    def _compute_unionnet_training_loss(self, model_output):

        support_node_features, support_node_labels = self._build_support_set_for_nodes(
            self.node_features, self.train_node_mask, self.noisy_node_labels,
            self.edge_connections, self.support_size_k
        )

        train_node_features = self.node_features[self.train_node_mask]
        train_noisy_labels = self.noisy_node_labels[self.train_node_mask]

        class_probabilities = self._aggregate_labels_from_support(
            support_node_features[self.train_node_mask], 
            support_node_labels[self.train_node_mask], 
            train_node_features, 
            self.num_classes
        )

        confidence_weights = class_probabilities[range(len(train_noisy_labels)), train_noisy_labels]
        instance_losses = F.cross_entropy(
            model_output[self.train_node_mask], 
            train_noisy_labels, 
            reduction='none'
        )
        reweighted_loss = torch.mean(confidence_weights * instance_losses)

        # Standard correction loss component
        correction_loss = F.cross_entropy(model_output[self.train_node_mask], train_noisy_labels)

        # KL divergence loss component
        one_hot_train_labels = F.one_hot(train_noisy_labels, num_classes=self.num_classes).to(self.device)
        kl_regularization_loss = self._compute_kl_divergence_loss(
            model_output[self.train_node_mask], 
            one_hot_train_labels
        )

        # Combine all loss components
        combined_loss = (
            self.reweight_alpha * reweighted_loss + 
            (1 - self.reweight_alpha) * correction_loss + 
            self.kl_beta * kl_regularization_loss
        )
        
        return combined_loss

    def train_model(self, enable_debug=True):

        training_start_time = time.time()
        
        for current_epoch in range(self.num_epochs):
            self.gnn_model.train()
            self.optimizer.zero_grad()
            
            model_predictions = self.gnn_model(self.graph_data)
            training_loss = self._compute_unionnet_training_loss(model_predictions)
            training_loss.backward()
            self.optimizer.step()
            
            train_true_labels = self.noisy_node_labels[self.train_node_mask].cpu().numpy()
            train_pred_labels = model_predictions[self.train_node_mask].detach().cpu().numpy().argmax(1)
            train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
            train_f1_score = f1_score(train_true_labels, train_pred_labels, average='macro')
            
            _, validation_loss, _ = self._forward_pass('val')
            val_true_labels = self.noisy_node_labels[self.val_node_mask].cpu().numpy()
            val_pred_labels = model_predictions[self.val_node_mask].detach().cpu().numpy().argmax(1)
            val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
            val_f1_score = f1_score(val_true_labels, val_pred_labels, average='macro')
            
            # Compute oversmoothing metrics
            if current_epoch % 20 == 0:
                train_oversmooth_metrics = self._evaluate_oversmoothing_metrics(
                    model_predictions, mask=self.train_node_mask
                )
                val_oversmooth_metrics = self._evaluate_oversmoothing_metrics(
                    model_predictions, mask=self.val_node_mask
                )
                
                if enable_debug:
                    print(f"Epoch {current_epoch+1:03d} | Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f} | "
                          f"Train F1: {train_f1_score:.4f}, Val F1: {val_f1_score:.4f}")
                    print(f"Train EDir: {train_oversmooth_metrics['EDir']:.4f}, Val EDir: {val_oversmooth_metrics['EDir']:.4f} | "
                          f"Train EDir_trad: {train_oversmooth_metrics['EDir_traditional']:.4f}, Val EDir_trad: {val_oversmooth_metrics['EDir_traditional']:.4f} | "
                          f"Train EProj: {train_oversmooth_metrics['EProj']:.4f}, Val EProj: {val_oversmooth_metrics['EProj']:.4f} | "
                          f"Train MAD: {train_oversmooth_metrics['MAD']:.4f}, Val MAD: {val_oversmooth_metrics['MAD']:.4f} | "
                          f"Train NumRank: {train_oversmooth_metrics['NumRank']:.4f}, Val NumRank: {val_oversmooth_metrics['NumRank']:.4f} | "
                          f"Train Erank: {train_oversmooth_metrics['Erank']:.4f}, Val Erank: {val_oversmooth_metrics['Erank']:.4f}")
            else:
                if enable_debug and current_epoch % 5 == 0:
                    print(f"Epoch {current_epoch+1:03d} | Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f} | "
                          f"Train F1: {train_f1_score:.4f}, Val F1: {val_f1_score:.4f}")
            
            # Early stopping
            if validation_loss.item() < self.best_validation_loss:
                self.best_validation_loss = validation_loss.item()
                self.patience_counter = 0
                self.training_results['train'] = train_accuracy
                self.training_results['val'] = val_accuracy
                self.best_model_weights = deepcopy(self.gnn_model.state_dict())
            else:
                self.patience_counter += 1

            if self.early_stop_patience and self.patience_counter >= self.early_stop_patience:
                if enable_debug:
                    print(f"Early stopping at epoch {current_epoch}")
                break
        
        if self.best_model_weights is not None:
            self.gnn_model.load_state_dict(self.best_model_weights)

        self.gnn_model.eval()
        with torch.no_grad():
            final_predictions = self.gnn_model(self.graph_data)

            test_true_labels = self.clean_node_labels[self.test_node_mask].cpu().numpy()
            test_pred_labels = final_predictions[self.test_node_mask].cpu().numpy().argmax(1)
            test_accuracy = accuracy_score(test_true_labels, test_pred_labels)
            test_f1_score = f1_score(test_true_labels, test_pred_labels, average='macro')
            test_precision = precision_score(test_true_labels, test_pred_labels, average='macro')
            test_recall = recall_score(test_true_labels, test_pred_labels, average='macro')
            
            test_loss = F.cross_entropy(
                final_predictions[self.test_node_mask], 
                self.clean_node_labels[self.test_node_mask]
            )
            test_oversmooth_metrics = self._evaluate_oversmoothing_metrics(
                final_predictions, mask=self.test_node_mask
            )

        total_training_time = time.time() - training_start_time
        
        if enable_debug:
            print(f"\nUnionNET Training completed!")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test F1: {test_f1_score:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print(f"Training completed in {total_training_time:.2f}s")
            print("Final Oversmoothing Metrics:")
            if test_oversmooth_metrics:
                print(f"Test: EDir: {test_oversmooth_metrics['EDir']:.4f}, EDir_traditional: {test_oversmooth_metrics['EDir_traditional']:.4f}, "
                      f"EProj: {test_oversmooth_metrics['EProj']:.4f}, MAD: {test_oversmooth_metrics['MAD']:.4f}, "
                      f"NumRank: {test_oversmooth_metrics['NumRank']:.4f}, Erank: {test_oversmooth_metrics['Erank']:.4f}")

        return {
            'accuracy': test_accuracy,
            'f1': test_f1_score,
            'precision': test_precision,
            'recall': test_recall,
            'oversmoothing': test_oversmooth_metrics
        }

    def _evaluate_oversmoothing_metrics(self, node_embeddings, mask=None):
        try:
            if mask is not None:
                masked_embeddings = node_embeddings[mask]

                edge_mask = mask[self.edge_connections[0]] & mask[self.edge_connections[1]]
                masked_edge_index = self.edge_connections[:, edge_mask]

                index_mapping = torch.full((mask.size(0),), -1, device=masked_edge_index.device)
                index_mapping[mask] = torch.arange(mask.sum(), device=masked_edge_index.device)
                masked_edge_index = index_mapping[masked_edge_index]
                subgraph_list = [{
                    'X': masked_embeddings,
                    'edge_index': masked_edge_index,
                    'edge_weight': None
                }]
                return self.oversmoothing_evaluator.compute_all_metrics(
                    X=masked_embeddings,
                    edge_index=masked_edge_index,
                    graphs_in_class=subgraph_list
                )
            else:
                full_graph_list = [{
                    'X': node_embeddings,
                    'edge_index': self.edge_connections,
                    'edge_weight': None
                }]
                return self.oversmoothing_evaluator.compute_all_metrics(
                    X=node_embeddings,
                    edge_index=self.edge_connections,
                    graphs_in_class=full_graph_list
                )
        except Exception as e:
            print(f"Warning: Could not compute oversmoothing metrics: {e}")
            return {
                'NumRank': 0.0, 'Erank': 0.0, 'EDir': 0.0,
                'EDir_traditional': 0.0, 'EProj': 0.0, 'MAD': 0.0
            }
        