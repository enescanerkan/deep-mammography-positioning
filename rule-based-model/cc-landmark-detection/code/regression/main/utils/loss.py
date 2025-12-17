# loss.py - CC Nipple Detection

import torch
import torch.nn as nn
import torch.nn.functional as F

class WingLoss(nn.Module):
    """
    Wing Loss for robust detection, adapted for CC nipple detection.
    Transitions from logarithmic to linear form outside a defined pixel width.
    """

    def __init__(self, w=3.0, epsilon=1.5):
        super(WingLoss, self).__init__()
        self.w = torch.tensor(w)
        self.epsilon = torch.tensor(epsilon)
        self.C = self.w - self.w * torch.log(1.0 + self.w / self.epsilon)

    def forward(self, prediction, target):
        y = torch.abs(prediction - target)
        loss = torch.where(y < self.w, self.w * torch.log(1.0 + y / self.epsilon), y - self.C)
        return torch.mean(loss)


class MultifacetedLoss(nn.Module):
    """
    Multifaceted loss for CC nipple detection, simplified for single landmark.
    Uses Wing Loss for robust nipple position prediction.
    """
    def __init__(self, w=5.0, epsilon=2, alpha=1.0, 
                 beta=1.0, gamma=1.0):
        super(MultifacetedLoss, self).__init__()
        self.wing_loss = WingLoss(w, epsilon)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
 
    def forward(self, prediction, target):
        # For CC, we only have nipple coordinates (x, y)
        wing_loss_value = self.wing_loss(prediction, target)
        
        # Split into x and y components for detailed analysis
        x_loss = torch.mean(torch.abs(prediction[:, 0] - target[:, 0]))
        y_loss = torch.mean(torch.abs(prediction[:, 1] - target[:, 1]))

        total_loss = self.gamma * wing_loss_value
        return total_loss
