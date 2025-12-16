"""
Early Stopping Module for MLO Landmark Detection

Stops training when validation loss stops improving to prevent overfitting.
"""

import numpy as np
import torch


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait after last improvement
        verbose: If True, prints message for each validation loss improvement
        delta: Minimum change to qualify as an improvement
        path: Path to save the best model checkpoint
        trace_func: Function to use for printing (default: print)
    """
    
    def __init__(
        self, 
        patience: int = 15, 
        verbose: bool = False, 
        delta: float = 0.001, 
        path: str = 'checkpoint.pth', 
        trace_func=print
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save if validation improves
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        """Save model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                f'Saving model...'
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
