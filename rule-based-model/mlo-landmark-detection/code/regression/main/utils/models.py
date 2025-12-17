"""
CRAUNet Model Architecture for MLO Landmark Detection

Coordinate-aware Residual Attention UNet for precise landmark localization
in MLO (Medio-Lateral Oblique) mammography images.

Architecture Components:
    - CoordConv: Adds spatial coordinate awareness
    - Attention Gates: Focus on relevant anatomical regions
    - Encoder-Decoder: U-Net style with skip connections
    - Regression Head: Outputs normalized coordinates

Output Landmarks:
    - Pectoral muscle line: 2 points (x1, y1, x2, y2)
    - Nipple position: center coordinates (x, y)
    - Total: 6 coordinates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AddCoordinates(nn.Module):
    """Layer to add X,Y coordinates as channels at runtime."""
    def forward(self, input_tensor):
        batch_size, _, height, width = input_tensor.size()
        # Transform coordinates to range from 0 to 1
        x_coords = torch.linspace(0, 1, width, device=input_tensor.device).repeat(batch_size, height, 1)
        y_coords = torch.linspace(0, 1, height, device=input_tensor.device).repeat(batch_size, width, 1).transpose(1, 2)
        coords = torch.stack((x_coords, y_coords), dim=1)
        return torch.cat((input_tensor, coords), dim=1)


class CoordConv(nn.Module):
    """CoordConv layer replacing the initial Conv2d in ConvBlock."""
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CoordConv, self).__init__()
        self.add_coords = AddCoordinates()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.add_coords(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class CRAUNet(nn.Module):
    """
    Coordinate-aware Attention UNet for landmark detection.
    
    It uses attention gates to refine the feature maps at each level of the U-Net.
    Predicts the coordinates of the landmarks.
    
    Args:
        in_channels (int): number of input channels.
        out_features (int): number of output features (coordinates).
    """
    def __init__(self, in_channels=1, out_features=6):
        super(CRAUNet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.Conv1 = CoordConv(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        # Decoder
        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        # Final layers for regression
        self.last = nn.Conv2d(64, out_features, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(out_features * 16 * 16, out_features)

    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)
        e2 = self.MaxPool(e1); e2 = self.Conv2(e2)
        e3 = self.MaxPool(e2); e3 = self.Conv3(e3)
        e4 = self.MaxPool(e3); e4 = self.Conv4(e4)
        e5 = self.MaxPool(e4); e5 = self.Conv5(e5)

        # Decoder
        d5 = self.Up5(e5); s4 = self.Att5(gate=d5, skip_connection=e4); d5 = torch.cat((s4, d5), dim=1); d5 = self.UpConv5(d5)
        d4 = self.Up4(d5); s3 = self.Att4(gate=d4, skip_connection=e3); d4 = torch.cat((s3, d4), dim=1); d4 = self.UpConv4(d4)
        d3 = self.Up3(d4); s2 = self.Att3(gate=d3, skip_connection=e2); d3 = torch.cat((s2, d3), dim=1); d3 = self.UpConv3(d3)
        d2 = self.Up2(d3); d2 = torch.cat((e1, d2), dim=1); d2 = self.UpConv2(d2)

        # Regression output
        out = self.last(d2)
        out = F.adaptive_avg_pool2d(out, (16, 16))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def create_model(in_channels=1, out_features=6, dropout_rate=0.3):
    """
    Factory function to create CRAUNet model for MLO landmark detection.
    
    Args:
        in_channels: Number of input channels (1 for grayscale mammography)
        out_features: Number of output coordinates
        dropout_rate: Not used (kept for compatibility)
        
    Returns:
        Configured CRAUNet model instance
    """
    return CRAUNet(in_channels=in_channels, out_features=out_features)
