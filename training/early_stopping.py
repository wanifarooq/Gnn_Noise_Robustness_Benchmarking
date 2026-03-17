"""Early stopping tracker for the shared training loop."""


class EarlyStopping:
    """Track validation loss and signal when to stop training.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum decrease in val_loss to count as improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss: float = float('inf')
        self.counter: int = 0
        self.best_epoch: int | None = None

    def step(self, val_loss: float, epoch: int) -> bool:
        """Update state and return True if training should stop.

        Args:
            val_loss: Current epoch's validation loss.
            epoch: Current epoch index (0-based).

        Returns:
            True if patience is exhausted, False otherwise.
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        return self.counter >= self.patience

    @property
    def is_best(self) -> bool:
        """True if the most recent step() was a new best."""
        return self.counter == 0
