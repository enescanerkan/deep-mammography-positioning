"""
CRAUNet Model Architecture for CC Nipple Detection

Coordinate-aware Residual Attention UNet for precise nipple localization
in CC (Cranio-Caudal) mammography images.

Architecture Components:
    - CoordConv: Adds spatial coordinate awareness
    - Attention Gates: Focus on relevant anatomical regions
    - Encoder-Decoder: U-Net style with skip connections
    - Regression Head: Outputs normalized (x, y) coordinates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


# =============================================================================
# Base Classes (Single Responsibility Principle)
# =============================================================================

class BaseConvBlock(nn.Module, ABC):
    """Abstract base class for convolutional blocks."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BaseModel(nn.Module, ABC):
    """Abstract base class for landmark detection models."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def get_num_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Building Blocks (Open/Closed Principle - extensible without modification)
# =============================================================================

class ConvBlock(BaseConvBlock):
    """
    Standard convolutional block with batch normalization.
    
    Architecture: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    """
    Upsampling block with convolution.
    
    Architecture: Upsample(2x) -> Conv3x3 -> BN -> ReLU
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# =============================================================================
# Coordinate Convolution (Spatial Awareness)
# =============================================================================

class AddCoordinates(nn.Module):
    """
    Adds normalized X,Y coordinate channels to input tensor.
    
    Appends two channels containing normalized (0-1) x and y coordinates,
    enabling the network to learn position-aware features.
    
    Input shape: (B, C, H, W)
    Output shape: (B, C+2, H, W)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.size()
        
        # Create coordinate grids
        x_coords = torch.linspace(0, 1, width, device=x.device)
        y_coords = torch.linspace(0, 1, height, device=x.device)
        
        # Expand to batch size
        x_coords = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        y_coords = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        
        return torch.cat([x, x_coords, y_coords], dim=1)


class CoordConv(nn.Module):
    """
    CoordConv layer that adds coordinate channels before convolution.
    
    Enhances spatial awareness by concatenating normalized x,y
    coordinates as additional input channels before convolution.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        **kwargs: Additional arguments for Conv2d
    """
    
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.add_coords = AddCoordinates()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_coords(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


# =============================================================================
# Attention Mechanism
# =============================================================================

class AttentionGate(nn.Module):
    """
    Attention gate for focusing on relevant spatial regions.
    
    Computes attention weights based on gating signal and skip connection,
    allowing the network to focus on important anatomical features.
    
    Args:
        gate_channels: Number of channels in gating signal
        skip_channels: Number of channels in skip connection
        inter_channels: Number of intermediate channels
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Gating signal from decoder path
            skip: Skip connection from encoder path
            
        Returns:
            Attention-weighted skip connection
        """
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attention = self.relu(g + s)
        attention = self.psi(attention)
        return skip * attention


# =============================================================================
# CRAUNet Model (Liskov Substitution Principle - can replace BaseModel)
# =============================================================================

class CRAUNet(BaseModel):
    """
    Coordinate-aware Residual Attention UNet for CC nipple detection.
    
    Combines coordinate convolution with attention gates for precise
    nipple localization in CC mammography images.
    
    Architecture:
        - Encoder: CoordConv + 4 ConvBlocks with MaxPool
        - Bottleneck: ConvBlock with 1024 channels
        - Decoder: 4 UpConv + AttentionGate + ConvBlock stages
        - Head: Conv1x1 -> AdaptiveAvgPool -> FC
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        out_features: Number of output coordinates (2 for nipple x, y)
        dropout_rate: Dropout rate for regularization
    """
    
    # Channel configuration for each encoder/decoder level
    CHANNELS = [64, 128, 256, 512, 1024]
    
    def __init__(
        self, 
        in_channels: int = 1, 
        out_features: int = 2, 
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.out_features = out_features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Build encoder and decoder
        self._build_encoder(in_channels)
        self._build_decoder()
        self._build_head(out_features, dropout_rate)
    
    def _build_encoder(self, in_channels: int) -> None:
        """Builds encoder path with CoordConv first layer."""
        ch = self.CHANNELS
        
        # First layer with coordinate awareness
        self.enc1 = CoordConv(in_channels, ch[0], kernel_size=3, padding=1)
        
        # Remaining encoder blocks
        self.enc2 = ConvBlock(ch[0], ch[1])
        self.enc3 = ConvBlock(ch[1], ch[2])
        self.enc4 = ConvBlock(ch[2], ch[3])
        self.enc5 = ConvBlock(ch[3], ch[4])  # Bottleneck
    
    def _build_decoder(self) -> None:
        """Builds decoder path with attention gates."""
        ch = self.CHANNELS
        
        # Decoder stage 5 -> 4
        self.up5 = UpConv(ch[4], ch[3])
        self.att5 = AttentionGate(gate_channels=ch[3], skip_channels=ch[3], inter_channels=ch[2])
        self.dec5 = ConvBlock(ch[4], ch[3])
        
        # Decoder stage 4 -> 3
        self.up4 = UpConv(ch[3], ch[2])
        self.att4 = AttentionGate(gate_channels=ch[2], skip_channels=ch[2], inter_channels=ch[1])
        self.dec4 = ConvBlock(ch[3], ch[2])
        
        # Decoder stage 3 -> 2
        self.up3 = UpConv(ch[2], ch[1])
        self.att3 = AttentionGate(gate_channels=ch[1], skip_channels=ch[1], inter_channels=ch[0])
        self.dec3 = ConvBlock(ch[2], ch[1])
        
        # Decoder stage 2 -> 1
        self.up2 = UpConv(ch[1], ch[0])
        self.dec2 = ConvBlock(ch[1], ch[0])
    
    def _build_head(self, out_features: int, dropout_rate: float) -> None:
        """Builds regression head."""
        self.final_conv = nn.Conv2d(self.CHANNELS[0], out_features, kernel_size=1)
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(out_features * 16 * 16, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
            
        Returns:
            Predicted coordinates of shape (B, 2) for nipple (x, y)
        """
        # Encoder path
        e1 = self.enc1(x)
        
        e2 = self.pool(e1)
        e2 = self.enc2(e2)
        
        e3 = self.pool(e2)
        e3 = self.enc3(e3)
        e3 = self.dropout(e3)  # Regularization at mid-level
        
        e4 = self.pool(e3)
        e4 = self.enc4(e4)
        
        e5 = self.pool(e4)
        e5 = self.enc5(e5)  # Bottleneck
        
        # Decoder path with attention
        d5 = self.up5(e5)
        s4 = self.att5(gate=d5, skip=e4)
        d5 = torch.cat([s4, d5], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
        s3 = self.att4(gate=d4, skip=e3)
        d4 = torch.cat([s3, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        s2 = self.att3(gate=d3, skip=e2)
        d3 = torch.cat([s2, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e1, d2], dim=1)
        d2 = self.dec2(d2)
        
        # Regression head
        out = self.final_conv(d2)
        out = F.adaptive_avg_pool2d(out, (16, 16))
        out = torch.flatten(out, 1)
        out = self.fc_dropout(out)
        out = self.fc(out)
        
        return out


# =============================================================================
# Factory Function (Dependency Inversion Principle)
# =============================================================================

def create_model(
    in_channels: int = 1,
    out_features: int = 2,
    dropout_rate: float = 0.3
) -> CRAUNet:
    """
    Factory function to create CRAUNet model for CC nipple detection.
    
    Args:
        in_channels: Number of input channels (1 for grayscale mammography)
        out_features: Number of output coordinates (2 for nipple x, y)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Configured CRAUNet model instance
    """
    return CRAUNet(
        in_channels=in_channels,
        out_features=out_features,
        dropout_rate=dropout_rate
    )
