# train.py - MLO Landmark Detection

import torch
import gc
from tqdm import tqdm
from .loss import MultifacetedLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


class Trainer:
    def __init__(self, config, train_loader, model=None):
        self.config = config
        self.train_loader = train_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.criterion = MultifacetedLoss(
            w=config['w'],
            epsilon=config['epsilon'],
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma']
        ).to(self.device)
        
        # Add weight decay for regularization
        weight_decay = config.get('weight_decay', 1e-5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=config['learning_rate'],
                                          weight_decay=weight_decay)
        
        # Use Cosine Annealing with Warmup for stable training
        warmup_epochs = config.get('warmup_epochs', 5)
        min_lr = config.get('min_lr', 1e-6)
        total_epochs = config['num_epochs']
        batches_per_epoch = len(train_loader)
        
        # Create cosine annealing scheduler
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=(total_epochs - warmup_epochs) * batches_per_epoch,
            eta_min=min_lr
        )
        
        # Warmup scheduler
        warmup_steps = warmup_epochs * batches_per_epoch
        
        def warmup_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def train(self, epoch):
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        progress_bar = tqdm(self.train_loader, 
                           desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}',
                           leave=True)
        
        for batch_idx, (images, landmarks) in enumerate(progress_bar):
            try:
                images, landmarks = images.to(self.device), landmarks.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = self.criterion(outputs, landmarks)
                
                # Check for NaN/Inf loss before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Step scheduler based on warmup phase
                if self.current_step < self.warmup_steps:
                    self.warmup_scheduler.step()
                else:
                    self.cosine_scheduler.step()
                
                self.current_step += 1 
                
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clean up tensors
                del images, landmarks, outputs, loss
                
                # Periodic memory cleanup
                if batch_idx % 50 == 0 and batch_idx > 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        if batch_count == 0:
            return 0.0
            
        avg_loss = total_loss / batch_count
        current_lr = self.optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
        
        return avg_loss
