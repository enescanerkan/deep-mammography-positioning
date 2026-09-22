"""
Dual-stream classification models for mammography quality assessment.
"""

import os
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models

# RadImageNet ResNet-50 weights (radiology-pretrained): a torchvision ResNet-50
# trained for 90 epochs on RadImageNet's 165 classes. Keys carry a torch.compile
# '_orig_mod.' prefix and a 165-way fc, both stripped on load.
#
# An earlier candidate (huggingface.co/Lab-Rasool/RadImageNet, ResNet50.pt) is
# deliberately NOT used: its layer4 maps every input to nearly the same feature
# vector (between-sample std 0.005, against 0.26 for ImageNet ResNet-18), so
# training never leaves chance level at any learning rate.
#
# Default location: <repo>/weights/RadImageNet_ResNet50_alt.pth (see weights/README.md
# for the download and checksum). Override with the RADIMAGENET_RESNET50_PATH
# environment variable.
RADIMAGENET_RESNET50_PATH = os.environ.get(
    'RADIMAGENET_RESNET50_PATH',
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        'weights', 'RadImageNet_ResNet50_alt.pth'
    )
)


def _to_single_channel(conv: nn.Conv2d, grayscale_init: str = 'mean') -> nn.Conv2d:
    """
    Replace a 3-channel first convolution with a 1-channel equivalent.

    With grayscale_init='mean' the pretrained RGB filters are averaged, which
    preserves the pretrained first layer. 'random' reinitializes it from
    scratch and therefore discards that part of the pretraining.
    """
    new_conv = nn.Conv2d(
        1, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, bias=conv.bias is not None
    )
    if grayscale_init == 'mean':
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
    elif grayscale_init != 'random':
        raise ValueError(f"Unsupported grayscale_init: {grayscale_init}")
    return new_conv


def _load_radimagenet_resnet50() -> nn.Module:
    """Build a torchvision ResNet-50 carrying RadImageNet pretrained weights."""
    if not os.path.exists(RADIMAGENET_RESNET50_PATH):
        raise FileNotFoundError(
            f"RadImageNet weights not found at {RADIMAGENET_RESNET50_PATH}. "
            "Download resnet50.pth from huggingface.co/convergedmachine/RadImagenet."
        )
    checkpoint = torch.load(RADIMAGENET_RESNET50_PATH, map_location='cpu',
                            weights_only=False)
    raw = checkpoint['model']

    # Drop the torch.compile prefix and the 165-way RadImageNet head.
    remapped = {
        key.replace('_orig_mod.', ''): value
        for key, value in raw.items()
        if not key.startswith('_orig_mod.fc')
    }

    encoder = models.resnet50(weights=None)
    encoder.fc = nn.Identity()
    missing, unexpected = encoder.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected RadImageNet key mismatch (missing={missing}, unexpected={unexpected})"
        )
    return encoder


def _create_resnet_encoder(model_fn, pretrained: bool = True,
                           grayscale_init: str = 'mean') -> nn.Module:
    """Create ResNet encoder with single channel input."""
    encoder = model_fn(pretrained=pretrained)
    encoder.conv1 = _to_single_channel(encoder.conv1, grayscale_init)
    encoder.fc = nn.Identity()
    return encoder


def _create_radimagenet_resnet50_encoder(grayscale_init: str = 'mean') -> nn.Module:
    """Create a RadImageNet-pretrained ResNet-50 encoder with single channel input."""
    encoder = _load_radimagenet_resnet50()
    encoder.conv1 = _to_single_channel(encoder.conv1, grayscale_init)
    encoder.fc = nn.Identity()
    return encoder


def _create_efficientnet_encoder(pretrained: bool = True,
                                 grayscale_init: str = 'mean') -> nn.Module:
    """Create EfficientNet encoder with single channel input."""
    encoder = models.efficientnet_b0(pretrained=pretrained)
    encoder.features[0][0] = _to_single_channel(encoder.features[0][0], grayscale_init)
    encoder.classifier = nn.Identity()
    return encoder


def _create_convnext_encoder(pretrained: bool = True,
                             grayscale_init: str = 'mean') -> nn.Module:
    """Create ConvNeXt-Tiny encoder with single channel input."""
    encoder = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    )
    encoder.features[0][0] = _to_single_channel(encoder.features[0][0], grayscale_init)
    # Keep the norm/flatten head, drop only the 1000-way linear layer.
    encoder.classifier[2] = nn.Identity()
    return encoder


def _create_mobilenet_encoder(pretrained: bool = True,
                              grayscale_init: str = 'mean') -> nn.Module:
    """Create MobileNet encoder with single channel input."""
    encoder = models.mobilenet_v2(pretrained=pretrained)
    encoder.features[0][0] = _to_single_channel(encoder.features[0][0], grayscale_init)
    encoder.classifier = nn.Identity()
    return encoder


def get_feature_dim(backbone: str) -> int:
    """Get feature dimension for backbone."""
    dims = {
        'resnet18': 512,
        'resnet50': 2048,
        'resnet50_radimagenet': 2048,
        'efficientnet_b0': 1280,
        'convnext_tiny': 768,
        'mobilenet_v2': 1280
    }
    if backbone not in dims:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return dims[backbone]


def create_encoder(backbone: str, pretrained: bool = True,
                   grayscale_init: str = 'mean') -> nn.Module:
    """Create encoder network based on backbone name."""
    if backbone == 'resnet18':
        return _create_resnet_encoder(models.resnet18, pretrained, grayscale_init)
    elif backbone == 'resnet50':
        return _create_resnet_encoder(models.resnet50, pretrained, grayscale_init)
    elif backbone == 'resnet50_radimagenet':
        return _create_radimagenet_resnet50_encoder(grayscale_init)
    elif backbone == 'efficientnet_b0':
        return _create_efficientnet_encoder(pretrained, grayscale_init)
    elif backbone == 'convnext_tiny':
        return _create_convnext_encoder(pretrained, grayscale_init)
    elif backbone == 'mobilenet_v2':
        return _create_mobilenet_encoder(pretrained, grayscale_init)
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
        dropout_rate2: float = 0.4,
        grayscale_init: str = 'mean'
    ):
        super().__init__()

        if fusion_method not in self.FUSION_METHODS:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

        self.fusion_method = fusion_method
        self.backbone = backbone

        # Create encoders directly (no wrapper class)
        self.mlo_encoder = create_encoder(backbone, pretrained, grayscale_init)
        self.cc_encoder = create_encoder(backbone, pretrained, grayscale_init)
        
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
    dropout_rate2: float = 0.4,
    grayscale_init: str = 'mean'
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
        grayscale_init: 'mean' to average pretrained RGB filters into the
            single-channel first conv, 'random' to reinitialize it

    Returns:
        Model instance
    """
    if model_type == 'dual':
        return DualStreamClassifier(
            num_classes, fusion_method, pretrained,
            backbone, dropout_rate1, dropout_rate2, grayscale_init
        )
    elif model_type == 'single':
        return SingleStreamClassifier(num_classes, pretrained, backbone)
    raise ValueError(f"Unsupported model type: {model_type}")
