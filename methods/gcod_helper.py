"""GCOD method helper — Graph Centroid Outlier Discounting.

Wraps the GraphCentroidOutlierDiscounting loss module from
model.methods.GCOD_loss into the shared TrainingLoop + MethodHelper system.

GCOD uses mini-batch training with NeighborLoader, dual optimizers
(model parameters + uncertainty parameters), and a three-component loss
(L1 soft-label CE, L2 uncertainty alignment, L3 KL divergence).
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from methods.base_helper import MethodHelper
from methods.registry import register_helper
from model.methods.GCOD_loss import GraphCentroidOutlierDiscounting, evaluate_ce_only


@register_helper('gcod')
class GCODHelper(MethodHelper):
    """MethodHelper for the GCOD robustness method."""

    def supports_batched_training(self):
        return True

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup(self, backbone_model, data, config, device, init_data):
        training_cfg = config.get('training', {})
        gcod_params = config.get('gcod_params', {})

        lr = float(training_cfg.get('lr', 0.01))
        weight_decay = float(training_cfg.get('weight_decay', 5e-4))
        uncertainty_lr = float(gcod_params.get('uncertainty_lr', 0.001))
        kl_start_epoch = int(gcod_params.get('kl_start_epoch', 2))
        momentum = float(gcod_params.get('momentum', 0.9))
        temperature = float(gcod_params.get('temperature', 1.0))
        similarity_mode = str(gcod_params.get('similarity_mode', 'correction'))
        batch_size = int(gcod_params.get('batch_size', training_cfg.get('batch_size', 64)))
        # Optional full-graph (full-batch) training: one forward over the whole
        # graph per epoch instead of NeighborLoader mini-batches. Much faster on
        # large transductive graphs and consistent with the other (full-batch)
        # methods, but it changes GCOD's dynamics (u updated once/epoch; full vs
        # sampled neighbourhoods), so it is OPT-IN and defaults to the original
        # mini-batch behaviour.
        full_batch = bool(gcod_params.get('full_batch', False))
        num_classes = int(data.y.max().item()) + 1

        backbone_model = backbone_model.to(device)

        # Determine embedding dim by probing the backbone
        backbone_model.eval()
        with torch.no_grad():
            _probe = backbone_model.get_embeddings(data.to(device))
            embedding_dim = _probe.shape[1]
        del _probe

        # GCOD loss module (holds uncertainty u + centroid buffers)
        gcod_loss = GraphCentroidOutlierDiscounting(
            num_classes=num_classes,
            device=device,
            num_samples=data.num_nodes,
            embedding_dim=embedding_dim,
            sample_labels=data.y,
            train_mask=data.train_mask,
            kl_start_epoch=kl_start_epoch,
            momentum=momentum,
            temperature=temperature,
            similarity_mode=similarity_mode,
        ).to(device)

        # Dual optimizers
        model_optimizer = torch.optim.Adam(
            backbone_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        uncertainty_optimizer = torch.optim.Adam(
            [gcod_loss.u], lr=uncertainty_lr
        )

        # NeighborLoaders for mini-batch training / evaluation. Skipped entirely
        # in full-batch mode (which trains/evaluates on the full graph).
        train_loader = val_loader = test_loader = None
        if not full_batch:
            train_indices = data.train_mask.nonzero(as_tuple=True)[0]
            val_indices = data.val_mask.nonzero(as_tuple=True)[0]
            test_indices = data.test_mask.nonzero(as_tuple=True)[0]

            train_loader = NeighborLoader(
                data, num_neighbors=[15, 10], batch_size=batch_size,
                input_nodes=train_indices, shuffle=True,
            )
            val_loader = NeighborLoader(
                data, num_neighbors=[15, 10], batch_size=batch_size,
                input_nodes=val_indices, shuffle=False,
            )
            test_loader = NeighborLoader(
                data, num_neighbors=[15, 10], batch_size=batch_size,
                input_nodes=test_indices, shuffle=False,
            )

        return {
            'models': [backbone_model],
            'optimizers': [model_optimizer, uncertainty_optimizer],
            'backbone': backbone_model,
            'gcod_loss': gcod_loss,
            'model_optimizer': model_optimizer,
            'uncertainty_optimizer': uncertainty_optimizer,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'device': device,
            'num_classes': num_classes,
            'training_accuracy': 0.1,
            'full_batch': full_batch,
        }

    # ── Per-epoch training ─────────────────────────────────────────────────

    def train_step(self, state, data, epoch):
        if state.get('full_batch'):
            return self._train_step_full(state, data, epoch)
        model = state['backbone']
        gcod_loss_fn = state['gcod_loss']
        model_optimizer = state['model_optimizer']
        uncertainty_optimizer = state['uncertainty_optimizer']
        train_loader = state['train_loader']
        device = state['device']
        num_classes = state['num_classes']

        # --- Compute epoch training accuracy (model in eval mode) ---
        current_acc = self._compute_epoch_training_accuracy(
            model, train_loader, device
        )
        smooth_factor = 0.9 if epoch > 0 else 0.5
        state['training_accuracy'] = (
            smooth_factor * state['training_accuracy']
            + (1 - smooth_factor) * current_acc
        )
        training_accuracy = state['training_accuracy']

        # --- Recompute centroids at start of each epoch ---
        gcod_loss_fn.recompute_centroids()

        # --- Mini-batch training ---
        model.train()
        gcod_loss_fn.train()

        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Get embeddings (detached — for similarity/centroid storage)
            with torch.no_grad():
                embeddings = model.get_embeddings(batch)

            # Get logits (with gradient — for L1 and L3)
            model_logits = model(batch)

            batch_train_mask = batch.train_mask[:batch.batch_size]
            target_nodes = batch_train_mask.nonzero(as_tuple=True)[0]

            if len(target_nodes) == 0:
                continue

            original_indices = batch.n_id[target_nodes]
            true_labels = batch.y[target_nodes]
            true_labels_onehot = F.one_hot(
                true_labels, num_classes=num_classes
            ).float()

            # GCOD loss components
            gcod_total_loss, loss_l1, loss_l2, loss_l3 = gcod_loss_fn(
                batch_indices=original_indices,
                model_logits=model_logits[target_nodes],
                label_onehot=true_labels_onehot,
                embeddings_detached=embeddings[target_nodes],
                training_accuracy=training_accuracy,
                epoch=epoch,
            )

            # Model backward + step (L1 + L3 — u is detached in both)
            model_optimizer.zero_grad(set_to_none=True)
            model_loss = loss_l1 + loss_l3
            model_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model_optimizer.step()

            # Uncertainty backward + step (L2 — model output detached)
            uncertainty_optimizer.zero_grad(set_to_none=True)
            loss_l2.backward()
            torch.nn.utils.clip_grad_norm_([gcod_loss_fn.u], max_norm=1.0)
            uncertainty_optimizer.step()

            total_loss += gcod_total_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_loss}

    # ── Full-graph (full-batch) training step ──────────────────────────────
    def _train_step_full(self, state, data, epoch):
        """One forward over the WHOLE graph per epoch (opt-in via gcod_params.full_batch).

        Algorithmically identical to the mini-batch step but over all training
        nodes at once: u (per-node) and prevSimilarity are indexed by the global
        train-node ids, exactly as the batched path uses batch.n_id."""
        model = state['backbone']
        gcod_loss_fn = state['gcod_loss']
        model_optimizer = state['model_optimizer']
        uncertainty_optimizer = state['uncertainty_optimizer']
        device = state['device']
        num_classes = state['num_classes']

        data = data.to(device)
        train_idx = data.train_mask.nonzero(as_tuple=True)[0]

        # --- Epoch training accuracy (full graph, eval) ---
        model.eval()
        with torch.no_grad():
            eval_logits = model(data)
            current_acc = ((eval_logits[train_idx].argmax(dim=1) == data.y[train_idx])
                           .float().mean().item()) if train_idx.numel() else 0.0
        smooth_factor = 0.9 if epoch > 0 else 0.5
        state['training_accuracy'] = (smooth_factor * state['training_accuracy']
                                      + (1 - smooth_factor) * current_acc)
        training_accuracy = state['training_accuracy']

        # --- Recompute centroids from last epoch's stored embeddings ---
        gcod_loss_fn.recompute_centroids()

        if train_idx.numel() == 0:
            return {'train_loss': 0.0}

        model.train()
        gcod_loss_fn.train()

        with torch.no_grad():
            embeddings = model.get_embeddings(data)   # detached, for similarity/centroids
        logits = model(data)                          # with gradient, for L1/L3

        true_labels_onehot = F.one_hot(data.y[train_idx], num_classes=num_classes).float()
        gcod_total_loss, loss_l1, loss_l2, loss_l3 = gcod_loss_fn(
            batch_indices=train_idx,
            model_logits=logits[train_idx],
            label_onehot=true_labels_onehot,
            embeddings_detached=embeddings[train_idx],
            training_accuracy=training_accuracy,
            epoch=epoch,
        )

        # Model step (L1 + L3 — u detached inside)
        model_optimizer.zero_grad(set_to_none=True)
        (loss_l1 + loss_l3).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model_optimizer.step()

        # Uncertainty step (L2 — model output detached inside)
        uncertainty_optimizer.zero_grad(set_to_none=True)
        loss_l2.backward()
        torch.nn.utils.clip_grad_norm_([gcod_loss_fn.u], max_norm=1.0)
        uncertainty_optimizer.step()

        return {'train_loss': gcod_total_loss.item()}

    # ── Validation ─────────────────────────────────────────────────────────

    def compute_val_loss(self, state, data):
        model = state['backbone']
        device = state['device']

        if state.get('full_batch'):
            model.eval()
            with torch.no_grad():
                out = model(data.to(device))
                val_idx = data.val_mask.nonzero(as_tuple=True)[0]
                return F.cross_entropy(out[val_idx], data.y[val_idx].to(out.device)).item()

        val_loader = state['val_loader']
        result = evaluate_ce_only(model, val_loader, device=device, mask_name='val')
        return result['ce_loss']

    # ── Predictions / Embeddings (full-graph) ──────────────────────────────

    def get_predictions(self, state, data):
        model = state['backbone']
        model.eval()
        with torch.no_grad():
            return model(data).argmax(dim=1)

    def get_probabilities(self, state, data):
        model = state['backbone']
        model.eval()
        with torch.no_grad():
            return F.softmax(model(data), dim=1)

    def get_embeddings(self, state, data):
        model = state['backbone']
        model.eval()
        with torch.no_grad():
            return model.get_embeddings(data)

    # ── Checkpointing ──────────────────────────────────────────────────────

    def get_checkpoint_state(self, state):
        return {
            'backbone': deepcopy(state['backbone'].state_dict()),
            'gcod_loss': deepcopy(state['gcod_loss'].state_dict()),
        }

    def load_checkpoint_state(self, state, checkpoint):
        if 'backbone' in checkpoint:
            state['backbone'].load_state_dict(checkpoint['backbone'])
        if 'gcod_loss' in checkpoint:
            state['gcod_loss'].load_state_dict(checkpoint['gcod_loss'])

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _compute_epoch_training_accuracy(model, train_loader, device):
        """Iterate over the train loader in eval mode to compute accuracy."""
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                model_outputs = model(batch)

                batch_train_mask = batch.train_mask[:batch.batch_size]
                target_nodes = batch_train_mask.nonzero(as_tuple=True)[0]

                if len(target_nodes) > 0:
                    predictions = model_outputs[target_nodes].argmax(dim=1)
                    total_correct += (
                        predictions == batch.y[target_nodes]
                    ).sum().item()
                    total_samples += len(target_nodes)

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return max(0.01, min(0.99, accuracy))
