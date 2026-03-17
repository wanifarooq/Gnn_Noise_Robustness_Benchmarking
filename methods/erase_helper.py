"""ERASE method helper — MCR2 (Maximal Coding Rate Reduction) training.

ERASE learns node representations via a self-supervised MCR2 loss that
encourages maximal coding rate reduction.  At inference time a linear probe
(LogisticRegression) is fitted on the learned features to produce class
predictions, because the model output is an embedding, not logits.

Key differences from standard training:
    - Model is an EnhancedGNNWrapper around a fresh backbone (not the shared
      backbone) with out_channels = n_embedding.
    - Loss is MCR2 with label propagation and cosine-similarity updates.
    - Predictions use sklearn LogisticRegression on L2-normalized features.
    - Checkpointing saves both model weights and predicted_labels.
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from methods.base_helper import MethodHelper
from methods.registry import register_helper
from model.methods.ERASE import (
    AdjacencyMatrixProcessor,
    EnhancedGNNWrapper,
    MaximalCodingRateReductionLoss,
)


@register_helper('erase')
class ERASEHelper(MethodHelper):
    """MCR2-based self-supervised training with linear-probe evaluation."""

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup(self, backbone_model, data, config, device, init_data):
        model_cfg = config.get('model', {})
        training_cfg = config.get('training', {})
        erase_params = config.get('erase_params', {})

        # ── Build the ERASE-specific config (mirrors ERASEMethodTrainer) ──
        erase_gnn_type = model_cfg.get('name', 'gcn')
        in_channels = data.num_features
        hidden_channels = model_cfg.get('hidden_channels', 128)
        n_embedding = erase_params.get('n_embedding', 512)
        n_layers = model_cfg.get('n_layers', 2)
        dropout = model_cfg.get('dropout', 0.5)
        self_loop = model_cfg.get('self_loop', True)
        mlp_layers = model_cfg.get('mlp_layers', 2)
        train_eps = model_cfg.get('train_eps', True)
        n_heads = erase_params.get('n_heads', model_cfg.get('heads', 8))

        lr = float(training_cfg.get('lr', 0.001))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))

        gam1 = erase_params.get('gam1', 1.0)
        gam2 = erase_params.get('gam2', 2.0)
        eps = erase_params.get('eps', 0.05)
        alpha = erase_params.get('alpha', 0.6)
        beta = erase_params.get('beta', 0.6)
        T = erase_params.get('T', 3)
        seed = init_data.get('seed', 42)

        use_layer_norm = erase_params.get('use_layer_norm', False)
        use_residual = erase_params.get('use_residual', False)
        use_residual_linear = erase_params.get('use_residual_linear', False)

        # ── Create the enhanced GNN model ──────────────────────────────────
        get_model = init_data['get_model']
        base_gnn = get_model(
            model_name=erase_gnn_type.lower(),
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=n_embedding,
            n_layers=n_layers,
            dropout=dropout,
            self_loop=self_loop,
            mlp_layers=mlp_layers,
            train_eps=train_eps,
            heads=n_heads,
        )

        enhancement_config = {
            'use_layer_normalization': use_layer_norm,
            'use_residual_connections': use_residual,
            'use_learnable_residual_projection': use_residual_linear,
            'final_activation_function': 'relu',
            'final_feature_normalization': 'l1',
        }

        if any(enhancement_config.values()):
            enhanced_model = EnhancedGNNWrapper(
                base_gnn_model=base_gnn,
                **enhancement_config,
            )
        else:
            enhanced_model = base_gnn

        enhanced_model = enhanced_model.to(device)

        # ── Preprocess: label propagation for semantic labels ──────────────
        noisy_labels = data.y
        initial_predicted_labels, semantic_labels_matrix, adjacency_matrix = (
            AdjacencyMatrixProcessor.preprocess_semantic_labels_with_propagation(
                data, noisy_labels, alpha, T, str(device),
            )
        )

        # ── Loss + optimizer ───────────────────────────────────────────────
        loss_fn = MaximalCodingRateReductionLoss(
            compression_weight=gam1,
            discrimination_weight=gam2,
            eps=eps,
        )

        optimizer = torch.optim.Adam(
            enhanced_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )

        return {
            'models': [enhanced_model],
            'optimizers': [optimizer],
            'model': enhanced_model,
            'optimizer': optimizer,
            'loss_fn': loss_fn,
            'adjacency_matrix': adjacency_matrix,
            'semantic_labels_matrix': semantic_labels_matrix,
            'predicted_labels': initial_predicted_labels,
            'alpha': alpha,
            'beta': beta,
            'T': T,
            'seed': seed,
            'device': device,
        }

    # ── Training step ──────────────────────────────────────────────────────

    def train_step(self, state, data, epoch):
        model = state['model']
        optimizer = state['optimizer']
        loss_fn = state['loss_fn']
        adjacency_matrix = state['adjacency_matrix']
        semantic_labels = state['semantic_labels_matrix']
        alpha = state['alpha']
        beta = state['beta']
        T = state['T']

        model.train()
        optimizer.zero_grad(set_to_none=True)

        node_features = model(data)

        loss_result = loss_fn(
            node_features, data, adjacency_matrix, semantic_labels,
            alpha, beta, T, data.train_mask,
        )

        # loss_fn.forward returns (total_loss, [disc, comp], [disc_th, comp_th], predicted_labels)
        if len(loss_result) == 3:
            total_loss, _components, updated_labels = loss_result
        else:
            total_loss = loss_result[0]
            updated_labels = loss_result[3]

        total_loss.backward()
        optimizer.step()

        # Update predicted labels in state for next epoch
        state['predicted_labels'] = updated_labels

        return {'train_loss': total_loss.item()}

    # ── Validation loss ────────────────────────────────────────────────────

    def compute_val_loss(self, state, data):
        """CE loss on val_mask (matching ERASETrainer._compute_cross_entropy_loss_for_split)."""
        model = state['model']
        model.eval()
        with torch.no_grad():
            features = model(data)
            val_idx = data.val_mask.nonzero(as_tuple=True)[0]
            return F.cross_entropy(features[val_idx], data.y[val_idx]).item()

    # ── Predictions (linear probe) ─────────────────────────────────────────

    def get_predictions(self, state, data):
        """Fit LogisticRegression on L2-normalised train features, predict all nodes."""
        model = state['model']
        predicted_labels = state['predicted_labels']
        seed = state.get('seed', 42)

        model.eval()
        with torch.no_grad():
            features = model(data)

        features_np = normalize(features.detach().cpu().numpy(), norm='l2')

        train_mask_np = data.train_mask.cpu().numpy()
        train_features = features_np[train_mask_np]
        train_labels = predicted_labels[data.train_mask].cpu().numpy()

        clf = LogisticRegression(
            solver='lbfgs',
            multi_class='auto',
            max_iter=1000,
            random_state=seed,
        ).fit(train_features, train_labels.ravel())

        all_predictions = clf.predict(features_np)
        return torch.tensor(all_predictions, device=state['device'], dtype=torch.long)

    # ── Embeddings ─────────────────────────────────────────────────────────

    def get_embeddings(self, state, data):
        model = state['model']
        model.eval()
        with torch.no_grad():
            return model.get_embeddings(data)

    # ── Checkpointing ──────────────────────────────────────────────────────

    def get_checkpoint_state(self, state):
        checkpoint = {
            'trained_model': deepcopy(state['model'].state_dict()),
        }
        if state.get('predicted_labels') is not None:
            checkpoint['predicted_labels'] = state['predicted_labels'].clone()
        return checkpoint

    def load_checkpoint_state(self, state, checkpoint):
        state['model'].load_state_dict(checkpoint['trained_model'])
        if 'predicted_labels' in checkpoint:
            state['predicted_labels'] = checkpoint['predicted_labels']
