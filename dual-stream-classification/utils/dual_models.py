"""
Dual-stream classification models for mammography quality assessment.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


def _create_resnet_encoder(model_fn, pretrained: bool = True) -> nn.Module:
    """Create ResNet encoder with single channel input."""
    encoder = model_fn(pretrained=pretrained)
    encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    encoder.fc = nn.Identity()
    return encoder


def _create_efficientnet_encoder(pretrained: bool = True) -> nn.Module:
    """Create EfficientNet encoder with single channel input."""
    encoder = models.efficientnet_b0(pretrained=pretrained)
    encoder.features[0][0] = nn.Conv2d(
        1, encoder.features[0][0].out_channels,
        kernel_size=3, stride=2, padding=1, bias=False
    )
    encoder.classifier = nn.Identity()
    return encoder


def _create_mobilenet_encoder(pretrained: bool = True) -> nn.Module:
    """Create MobileNet encoder with single channel input."""
    encoder = models.mobilenet_v2(pretrained=pretrained)
    encoder.features[0][0] = nn.Conv2d(
        1, encoder.features[0][0].out_channels,
        kernel_size=3, stride=2, padding=1, bias=False
    )
    encoder.classifier = nn.Identity()
    return encoder


def get_feature_dim(backbone: str) -> int:
    """Get feature dimension for backbone."""
    dims = {
        'resnet18': 512,
        'resnet50': 2048,
        'efficientnet_b0': 1280,
        'mobilenet_v2': 1280
    }
    if backbone not in dims:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return dims[backbone]


def create_encoder(backbone: str, pretrained: bool = True) -> nn.Module:
    """Create encoder network based on backbone name."""
    if backbone == 'resnet18':
        return _create_resnet_encoder(models.resnet18, pretrained)
    elif backbone == 'resnet50':
        return _create_resnet_encoder(models.resnet50, pretrained)
    elif backbone == 'efficientnet_b0':
        return _create_efficientnet_encoder(pretrained)
    elif backbone == 'mobilenet_v2':
        return _create_mobilenet_encoder(pretrained)
    raise ValueError(f"Unsupported backbone: {backbone}")


class DualStreamClassifier(nn.Module):
    """
    Dual-stream classifier for mammography quality assessment.
    
    Processes MLO and CC images through separate encoders,
    fuses features, and classifies as Good/Bad quality.
    """
    
    FUSION_METHODS = {'concat', 'add', 'attention'}
    
    def __init__(
        self,
        num_classes: int = 2,
        fusion_method: str = 'concat',
        pretrained: bool = True,
        backbone: str = 'resnet18',
        dropout_rate1: float = 0.5,
        dropout_rate2: float = 0.4
    ):
        super().__init__()
        
        if fusion_method not in self.FUSION_METHODS:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        self.fusion_method = fusion_method
        self.backbone = backbone
        
        # Create encoders directly (no wrapper class)
        self.mlo_encoder = create_encoder(backbone, pretrained)
        self.cc_encoder = create_encoder(backbone, pretrained)
        
        feature_dim = get_feature_dim(backbone)
        
        if fusion_method == 'concat':
            fusion_dim = feature_dim * 2
        elif fusion_method == 'attention':
            fusion_dim = feature_dim
            self.attention_mlo = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, feature_dim),
                nn.Sigmoid()
            )
            self.attention_cc = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, feature_dim),
                nn.Sigmoid()
            )
        else:
            fusion_dim = feature_dim
        
        # Classifier directly as Sequential (no wrapper class)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(128, num_classes)
        )
    
    def freeze_encoders(self) -> None:
        """Freeze encoder parameters for transfer learning."""
        for param in self.mlo_encoder.parameters():
            param.requires_grad = False
        for param in self.cc_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self) -> None:
        """Unfreeze encoder parameters for fine-tuning."""
        for param in self.mlo_encoder.parameters():
            param.requires_grad = True
        for param in self.cc_encoder.parameters():
            param.requires_grad = True
    
    def _fuse_features(self, mlo_features: torch.Tensor, cc_features: torch.Tensor) -> torch.Tensor:
        """Fuse MLO and CC features."""
        if self.fusion_method == 'concat':
            return torch.cat([mlo_features, cc_features], dim=1)
        elif self.fusion_method == 'add':
            return mlo_features + cc_features
        elif self.fusion_method == 'attention':
            attn_mlo = self.attention_mlo(mlo_features)
            attn_cc = self.attention_cc(cc_features)
            return attn_mlo * mlo_features + attn_cc * cc_features
    
    def forward(self, mlo_img: torch.Tensor, cc_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            mlo_img: MLO images [B, 1, H, W]
            cc_img: CC images [B, 1, H, W]
            
        Returns:
            Classification logits [B, num_classes]
        """
        mlo_features = self.mlo_encoder(mlo_img)
        cc_features = self.cc_encoder(cc_img)
        combined = self._fuse_features(mlo_features, cc_features)
        return self.classifier(combined)


class SingleStreamClassifier(nn.Module):
    """Single-stream classifier for comparison experiments."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        backbone: str = 'resnet18'
    ):
        super().__init__()
        self.encoder = create_encoder(backbone, pretrained)
        self.fc = nn.Linear(get_feature_dim(backbone), num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.fc(features)


def get_dual_model(
    model_type: str = 'dual',
    num_classes: int = 2,
    fusion_method: str = 'concat',
    pretrained: bool = True,
    backbone: str = 'resnet18',
    dropout_rate1: float = 0.5,
    dropout_rate2: float = 0.4
) -> nn.Module:
    """
    Factory function for creating models.
    
    Args:
        model_type: 'dual' or 'single'
        num_classes: Number of output classes
        fusion_method: Feature fusion method for dual-stream
        pretrained: Use pretrained weights
        backbone: Backbone architecture name
        dropout_rate1: First dropout rate
        dropout_rate2: Second dropout rate
        
    Returns:
        Model instance
    """
    if model_type == 'dual':
        return DualStreamClassifier(
            num_classes, fusion_method, pretrained, 
            backbone, dropout_rate1, dropout_rate2
        )
    elif model_type == 'single':
        return SingleStreamClassifier(num_classes, pretrained, backbone)
    raise ValueError(f"Unsupported model type: {model_type}")
