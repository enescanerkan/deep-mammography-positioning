# validate.py - MLO Landmark Detection

import torch
from tqdm import tqdm
from .loss import MultifacetedLoss

class Validator:
    def __init__(self, config, val_loader, model):
        self.config = config
        self.val_loader = val_loader
        self.model = model.to(config['device'])
        self.device = config['device']
        self.criterion = MultifacetedLoss(
            w=config['w'],
            epsilon=config['epsilon'],
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            adaptive_weights=False  # Validation uses fixed weights for consistent evaluation
        ).to(self.device)
        self.best_val_loss = float('inf')

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        sum_pec1, sum_pec2, sum_nipple = 0.0, 0.0, 0.0
        valid_batches = 0
        nan_inf_count = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validating', leave=False)
            for batch_idx, (images, landmarks) in enumerate(progress_bar):
                images, landmarks = images.to(self.device), landmarks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, landmarks)
                
                # Check for NaN/Inf in validation loss
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_inf_count += 1
                    continue
                
                total_loss += loss.item()
                valid_batches += 1
                
                # Accumulate component losses
                sum_pec1 += self.criterion.last_pec1_loss
                sum_pec2 += self.criterion.last_pec2_loss
                sum_nipple += self.criterion.last_nipple_loss

        # Use valid_batches count to avoid division by zero
        if valid_batches == 0:
            print("  ERROR: No valid validation batches! Returning high loss.")
            return float('inf'), False
        
        avg_loss = total_loss / valid_batches
        avg_pec1 = sum_pec1 / valid_batches
        avg_pec2 = sum_pec2 / valid_batches
        avg_nipple = sum_nipple / valid_batches
        
        print(f"  Val Loss: {avg_loss:.6f}", end='')
        
        # Determine if this is the best model so far based on validation loss
        is_best = avg_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = avg_loss
            print(f" <- Best model!")
        else:
            print()
        
        # Print warnings summary if any
        if nan_inf_count > 0:
            print(f"  WARNING: {nan_inf_count} validation batch(es) skipped due to NaN/Inf")

        return avg_loss, is_best
