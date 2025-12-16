# validate.py - CC Nipple Detection

import torch
import gc
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
            gamma=config['gamma']
        ).to(self.device)
        self.best_val_loss = float('inf')

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_nipple_x = 0.0
        total_nipple_y = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, (images, landmarks) in enumerate(self.val_loader):
                try:
                    images, landmarks = images.to(self.device), landmarks.to(self.device)
                    outputs = self.model(images)
                    loss, nipple_x, nipple_y = self.criterion(outputs, landmarks)
                    
                    # Skip batch if loss is NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        del images, landmarks, outputs, loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    
                    total_loss += loss.item()
                    total_nipple_x += nipple_x
                    total_nipple_y += nipple_y
                    batch_count += 1
                    
                    # Clean up tensors
                    del images, landmarks, outputs, loss
                    
                    # Periodic memory cleanup every 20 batches
                    if batch_idx % 20 == 0 and batch_idx > 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"⚠️  WARNING: OOM in validation batch {batch_idx}. Clearing cache and skipping.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
        
        # Use batch_count instead of len(val_loader) in case some batches were skipped
        if batch_count == 0:
            print("⚠️  WARNING: All validation batches were skipped!")
            return float('inf'), 0.0, 0.0, False

        avg_loss = total_loss / batch_count
        avg_nipple_x = total_nipple_x / batch_count
        avg_nipple_y = total_nipple_y / batch_count

        # Determine if this is the best model so far based on validation loss
        is_best = avg_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = avg_loss

        return avg_loss, avg_nipple_x, avg_nipple_y, is_best
