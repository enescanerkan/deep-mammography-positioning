# validate.py - CC Nipple Detection

import torch
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
        with torch.no_grad():
            print('Validation başladı')
            for images, landmarks in self.val_loader:
                images, landmarks = images.to(self.device), landmarks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, landmarks)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        print(f'Validation bitti - Loss: {avg_loss:.6f}')

        # Determine if this is the best model so far based on validation loss
        is_best = avg_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = avg_loss

        return avg_loss, is_best
