"""
Early stopping implementation for training.
"""


class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait after last improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for metrics like F1
        verbose: Print status messages
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
    
    def __call__(self, epoch: int, val_metric: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number
            val_metric: Validation metric value
            
        Returns:
            True if training should stop
        """
        score = -val_metric if self.mode == 'min' else val_metric
        
        if self.best_score is None:
            self._update_best(epoch, score, val_metric)
        elif score < self.best_score + self.min_delta:
            self._increment_counter()
        else:
            self._update_best(epoch, score, val_metric)
        
        return self.early_stop
    
    def _update_best(self, epoch: int, score: float, val_metric: float) -> None:
        """Update best score and reset counter."""
        if self.verbose and self.best_score is not None:
            improvement = score - self.best_score
            print(f"[EarlyStopping] Improvement: {improvement:.4f}")
        
        self.best_score = score
        self.best_epoch = epoch
        self.counter = 0
    
    def _increment_counter(self) -> None:
        """Increment patience counter."""
        self.counter += 1
        
        if self.verbose:
            print(f"[EarlyStopping] No improvement: {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"[EarlyStopping] Triggered at epoch {self.best_epoch}")
