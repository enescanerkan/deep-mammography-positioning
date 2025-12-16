import torch
import torch.nn as nn
import torch.nn.functional as F

class WingLoss(nn.Module):
    """
    Wing Loss for robust nipple detection in CC images.
    Transitions from logarithmic to linear form outside a defined pixel width.

    Parameters:
        w (float): Width of the piecewise function's linear region.
        epsilon (float): Adjusts the curvature of the logarithmic part.
    """

    def __init__(self, w=3.0, epsilon=1.5):
        super(WingLoss, self).__init__()
        # Register as buffers so they move with the module to correct device
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32))
        self.register_buffer('epsilon', torch.tensor(epsilon, dtype=torch.float32))
        self.register_buffer('C', self.w - self.w * torch.log(1.0 + self.w / self.epsilon))

    def forward(self, prediction, target):
        """
        Computes the WingLoss between predicted and target nipple coordinates.

        Parameters:
            prediction (torch.Tensor): Predicted coordinates (batch_size, 2) - nipple (x, y).
            target (torch.Tensor): Ground truth coordinates, same shape as prediction.

        Returns:
            torch.Tensor: The loss for nipple x and y coordinates separately.
        """
        y = torch.abs(prediction - target)
        # Clamp y to prevent extreme values that could cause NaN
        y = torch.clamp(y, min=0.0, max=10.0)
        
        # Compute loss with numerical stability
        log_term = self.w * torch.log(1.0 + y / self.epsilon)
        loss = torch.where(y < self.w, log_term, y - self.C)
        
        # Check for NaN/Inf and replace with large finite value
        loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.tensor(10.0, device=loss.device), loss)
        
        # For CC: prediction shape is (batch_size, 2) -> nipple_x, nipple_y
        # Return mean loss per coordinate (x and y)
        mean_loss_per_coord = torch.mean(loss, dim=0)  # Shape: (2,) -> [x_loss, y_loss]
        return mean_loss_per_coord


class MultifacetedLoss(nn.Module):
    """
    Multifaceted loss for CC nipple detection.
    Uses Wing Loss for robust nipple position prediction.
    
    Parameters:
        w (float): Width of the piecewise function's linear region in Wing Loss.
        epsilon (float): Curvature adjustment for the logarithmic part in Wing Loss.
        alpha (float): Weight for the nipple x-coordinate loss component.
        beta (float): Weight for the nipple y-coordinate loss component.
        gamma (float): Overall weight multiplier (kept for config compatibility).
    """
    def __init__(self, w=5.0, epsilon=2, alpha=1.0, beta=1.0, gamma=1.0):
        super(MultifacetedLoss, self).__init__()
        self.wing_loss = WingLoss(w, epsilon)
        self.alpha = alpha  # Weight for nipple_x
        self.beta = beta    # Weight for nipple_y
        self.gamma = gamma  # Overall multiplier
 
    def forward(self, prediction, target):
        wing_loss_values = self.wing_loss(prediction, target)
        nipple_x_loss, nipple_y_loss = wing_loss_values[0], wing_loss_values[1]

        # Weighted total loss
        total_loss = self.gamma * (self.alpha * nipple_x_loss + self.beta * nipple_y_loss)
        
        # Return individual losses for tracking (nipple_x, nipple_y)
        return total_loss, nipple_x_loss.item(), nipple_y_loss.item()
