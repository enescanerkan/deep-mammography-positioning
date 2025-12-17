# loss.py - MLO Landmark Detection

import torch
import torch.nn as nn
import torch.nn.functional as F

class WingLoss(nn.Module):
    """
    Wing Loss for robust detection, adapted for tasks with dynamic landmark counts.
    Transitions from logarithmic to linear form outside a defined pixel width.
    """

    def __init__(self, w=3.0, epsilon=1.5):
        super(WingLoss, self).__init__()
        self.w = torch.tensor(w)
        self.epsilon = torch.tensor(epsilon)
        self.C = self.w - self.w * torch.log(1.0 + self.w / self.epsilon)

    def forward(self, prediction, target):
        y = torch.abs(prediction - target)
        # Clamp to prevent extreme values
        y = torch.clamp(y, min=0.0, max=1000.0)
        loss = torch.where(y < self.w, self.w * torch.log(1.0 + y / self.epsilon), y - self.C)
        
        # Check for NaN or Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Warning: NaN or Inf detected in WingLoss, replacing with zero")
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        num_landmarks = prediction.size(1) // 2
        loss_per_coordinate = loss.view(-1, num_landmarks, 2)
        mean_loss_per_landmark = torch.mean(loss_per_coordinate, dim=0)
        return mean_loss_per_landmark


class MultifacetedLoss(nn.Module):
    """
    Multifaceted loss for MLO landmark detection.
    Combines Wing Loss for pectoral muscle and nipple landmarks.
    """
    def __init__(self, w=5.0, epsilon=2, alpha=1.0, 
                 beta=1.0, gamma=1.0, adaptive_weights=False):
        super(MultifacetedLoss, self).__init__()
        self.wing_loss = WingLoss(w, epsilon)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.adaptive_weights = adaptive_weights
        
        # For adaptive weighting: track historical losses
        self.pec1_history = []
        self.pec2_history = []
        self.nipple_history = []
        self.update_counter = 0
 
    def forward(self, prediction, target):
        wing_loss_values = self.wing_loss(prediction, target)
        pec1_loss_value, pec2_loss_value, nipple_loss_value = wing_loss_values.mean(dim=1)
        
        # Adaptive weighting based on component performance
        if self.adaptive_weights and self.update_counter > 50:
            self.pec1_history.append(pec1_loss_value.item())
            self.pec2_history.append(pec2_loss_value.item())
            self.nipple_history.append(nipple_loss_value.item())
            
            if len(self.pec1_history) > 100:
                self.pec1_history.pop(0)
                self.pec2_history.pop(0)
                self.nipple_history.pop(0)
            
            if len(self.pec1_history) >= 10:
                pec1_avg = sum(self.pec1_history[-10:]) / 10
                pec2_avg = sum(self.pec2_history[-10:]) / 10
                nipple_avg = sum(self.nipple_history[-10:]) / 10
                
                total_avg = pec1_avg + pec2_avg + nipple_avg
                if total_avg > 0:
                    adaptive_alpha = (pec1_avg / total_avg) * 3.0
                    adaptive_beta = (pec2_avg / total_avg) * 3.0
                    adaptive_gamma = (nipple_avg / total_avg) * 3.0
                    
                    blend_factor = 0.1
                    self.alpha = (1 - blend_factor) * self.alpha + blend_factor * adaptive_alpha
                    self.beta = (1 - blend_factor) * self.beta + blend_factor * adaptive_beta
                    self.gamma = (1 - blend_factor) * self.gamma + blend_factor * adaptive_gamma
        
        self.update_counter += 1
        
        total_loss = self.alpha * pec1_loss_value + \
                     self.beta * pec2_loss_value + \
                     self.gamma * nipple_loss_value
        
        # Store component losses for epoch-level reporting
        self.last_pec1_loss = (pec1_loss_value * self.alpha).item()
        self.last_pec2_loss = (pec2_loss_value * self.beta).item()
        self.last_nipple_loss = (nipple_loss_value * self.gamma).item()
        
        return total_loss
    
    def get_current_weights(self):
        """Return current adaptive weights for monitoring"""
        return {
            'alpha': self.alpha,
            'beta': self.beta, 
            'gamma': self.gamma
        }
