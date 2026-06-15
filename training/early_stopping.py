"""Early stopping tracker for the shared training loop."""


class EarlyStopping:
    """Track a validation metric and signal when to stop training.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum change in metric to count as improvement.
        mode: 'max' if higher is better (accuracy, f1), 'min' if lower
              is better (loss).
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0,
                 warmup_epochs: int = 50, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.mode = mode
        if mode == 'min':
            self.best_value: float = float('inf')
        else:
            self.best_value: float = -float('inf')
        self.counter: int = 0
        self.best_epoch: int | None = None

    def _is_improvement(self, current: float) -> bool:
        if self.mode == 'min':
            return current < self.best_value - self.min_delta
        return current > self.best_value + self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Update state and return True if training should stop.

        Args:
            value: Current epoch's monitored metric value.
            epoch: Current epoch index (0-based).

        Returns:
            True if patience is exhausted, False otherwise.
        """
        if epoch < self.warmup_epochs:
            self._in_warmup = True
            return False
        self._in_warmup = False
        if self._is_improvement(value):
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        return self.counter >= self.patience

    @property
    def is_best(self) -> bool:
        """True if the most recent step() was a new best."""
        return self.counter == 0 and not getattr(self, '_in_warmup', True)
