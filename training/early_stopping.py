"""Early stopping tracker for the shared training loop."""


class EarlyStopping:
    """Track validation accuracy and signal when to stop training.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum increase in val_acc to count as improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0,
                 warmup_epochs: int = 50):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.best_val_acc: float = -float('inf')
        self.counter: int = 0
        self.best_epoch: int | None = None

    def step(self, val_acc: float, epoch: int) -> bool:
        """Update state and return True if training should stop.

        Args:
            val_acc: Current epoch's validation accuracy.
            epoch: Current epoch index (0-based).

        Returns:
            True if patience is exhausted, False otherwise.
        """
        if epoch < self.warmup_epochs:
            self._in_warmup = True
            return False
        self._in_warmup = False
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.counter = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        return self.counter >= self.patience

    @property
    def is_best(self) -> bool:
        """True if the most recent step() was a new best."""
        return self.counter == 0 and not getattr(self, '_in_warmup', True)
