import torch
import torch.nn as nn
import torch.nn.functional as F

class WingLoss(nn.Module):
    """
    Wing Loss for robust detection, adapted for tasks with dynamic landmark counts.
    Transitions from logarithmic to linear form outside a defined pixel width, tailored for precision in key point prediction.
    Automatically adjusts to the number of landmarks based on input tensor shape.

    Parameters:
        w (float): Width of the piecewise function's linear region, beyond which errors are considered large.
        epsilon (float): Adjusts the curvature of the logarithmic part, affecting sensitivity to small errors.
    """

    def __init__(self, w=3.0, epsilon=1.5):
        """
        Initializes the WingLoss with specified parameters.
        Parameters:
            w (float): Controls the width of the non-linear region.
            epsilon (float): Defines the curvature of the non-linear region.
        """
        super(WingLoss, self).__init__()
        # Register as buffers so they move with the module to correct device
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32))
        self.register_buffer('epsilon', torch.tensor(epsilon, dtype=torch.float32))
        self.register_buffer('C', self.w - self.w * torch.log(1.0 + self.w / self.epsilon))

    def forward(self, prediction, target):
        """
        Computes the WingLoss between predicted and target landmark coordinates.

        Parameters:
            prediction (torch.Tensor): Predicted coordinates (batch_size, num_landmarks*2).
                                       Each landmark represented by an (x, y) pair.
            target (torch.Tensor): Ground truth coordinates, same shape as `prediction`.

        Returns:
            torch.Tensor: The loss for each landmark, facilitating detailed analysis and backpropagation.
        """
        y = torch.abs(prediction - target)
        # Clamp y to prevent extreme values that could cause NaN
        y = torch.clamp(y, min=0.0, max=10.0)
        
        # Compute loss with numerical stability
        log_term = self.w * torch.log(1.0 + y / self.epsilon)
        loss = torch.where(y < self.w, log_term, y - self.C)
        
        # Check for NaN/Inf and replace with large finite value
        loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.tensor(10.0, device=loss.device), loss)
        
        num_landmarks = prediction.size(1) // 2  # Dynamically infer the number of landmarks
        loss_per_coordinate = loss.view(-1, num_landmarks, 2)  # Adjust to the inferred number of landmarks
        mean_loss_per_landmark = torch.mean(loss_per_coordinate, dim=0)
        return mean_loss_per_landmark

class MultifacetedLoss(nn.Module):
    """
    This multifaceted approach ensures the model's focus on key aspects of landmark positioning, including the precise
    delineation of pectoralis muscle and nipple points. 
    
    Parameters:
        w (float): Width of the piecewise function's linear region in Wing Loss.
        epsilon (float): Curvature adjustment for the logarithmic part in Wing Loss.
        alpha (float): Weight for the pec1 loss component.
        beta (float): Weight for the pec2 loss component.
        gamma (float): Weight for the nipple loss component.
    """
    def __init__(self, w=5.0, epsilon=2, alpha=1.2, 
                 beta=1.2, gamma=0.8):
        super(MultifacetedLoss, self).__init__()
        self.wing_loss = WingLoss(w, epsilon)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
 
    def forward(self, prediction, target):
        wing_loss_values = self.wing_loss(prediction, target)
        pec1_loss_value, pec2_loss_value, nipple_loss_value = wing_loss_values.mean(dim=1)

        # Balanced weights: higher weight for pec1/pec2, lower for nipple
        total_loss = self.alpha * pec1_loss_value + \
                     self.beta * pec2_loss_value + \
                     self.gamma * nipple_loss_value
        
        # Return individual losses for tracking
        return total_loss, pec1_loss_value.item(), pec2_loss_value.item(), nipple_loss_value.item()
    
