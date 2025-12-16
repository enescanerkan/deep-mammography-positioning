"""
Data augmentation module for mammography images.
"""

from typing import List, Callable
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter, map_coordinates


class MammographyAugmentation:
    """
    Augmentation pipeline for mammography images.
    
    Applies conservative transformations to preserve diagnostic features:
    - Small rotations
    - Brightness/contrast adjustments
    - Gaussian noise
    - Gaussian blur
    - Elastic deformation
    """
    
    def __init__(
        self,
        rotation_degree: float = 10,
        brightness_factor: float = 0.2,
        contrast_factor: float = 0.2,
        noise_std: float = 0.03,
        blur_sigma: float = 0.5,
        elastic_alpha: float = 50,
        elastic_sigma: float = 5,
        p_rotation: float = 0.5,
        p_brightness: float = 0.5,
        p_contrast: float = 0.5,
        p_noise: float = 0.4,
        p_blur: float = 0.3,
        p_elastic: float = 0.2
    ):
        self.rotation_degree = rotation_degree
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.noise_std = noise_std
        self.blur_sigma = blur_sigma
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.p_rotation = p_rotation
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_elastic = p_elastic
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to image."""
        if random.random() < self.p_rotation:
            img = self._apply_rotation(img)
        
        if random.random() < self.p_brightness:
            img = self._apply_brightness(img)
        
        if random.random() < self.p_contrast:
            img = self._apply_contrast(img)
        
        if random.random() < self.p_noise:
            img = self._apply_noise(img)
        
        if random.random() < self.p_blur:
            img = self._apply_blur(img)
        
        if random.random() < self.p_elastic:
            img = self._apply_elastic(img)
        
        return img
    
    def _apply_rotation(self, img: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.rotation_degree, self.rotation_degree)
        return TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
    
    def _apply_brightness(self, img: torch.Tensor) -> torch.Tensor:
        factor = 1 + random.uniform(-self.brightness_factor, self.brightness_factor)
        return TF.adjust_brightness(img, factor)
    
    def _apply_contrast(self, img: torch.Tensor) -> torch.Tensor:
        factor = 1 + random.uniform(-self.contrast_factor, self.contrast_factor)
        return TF.adjust_contrast(img, factor)
    
    def _apply_noise(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img) * self.noise_std
        return torch.clamp(img + noise, 0, 1)
    
    def _apply_blur(self, img: torch.Tensor) -> torch.Tensor:
        kernel_size = int(2 * np.ceil(2 * self.blur_sigma) + 1)
        return TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[self.blur_sigma])
    
    def _apply_elastic(self, img: torch.Tensor) -> torch.Tensor:
        img_np = img.squeeze(0).cpu().numpy()
        shape = img_np.shape
        
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        deformed = map_coordinates(img_np, indices, order=1, mode='reflect').reshape(shape)
        return torch.from_numpy(deformed).unsqueeze(0).to(img.device)


class NoAugmentation:
    """Identity transform for validation/test sets."""
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img


class TestTimeAugmentation:
    """Test-time augmentation for robust inference."""
    
    def __init__(self, n_augmentations: int = 5):
        self.n_augmentations = n_augmentations
        self.augment = MammographyAugmentation(
            rotation_degree=3,
            brightness_factor=0.1,
            contrast_factor=0.1,
            noise_std=0.01,
            p_rotation=0.5,
            p_brightness=0.5,
            p_contrast=0.5,
            p_noise=0.3,
            p_blur=0.0,
            p_elastic=0.0
        )
    
    def __call__(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Generate augmented versions including original."""
        augmented = [img]
        for _ in range(self.n_augmentations - 1):
            augmented.append(self.augment(img.clone()))
        return augmented


def get_augmentation(
    split_type: str = 'Train',
    use_augmentation: bool = True,
    **kwargs
) -> Callable:
    """
    Get augmentation pipeline for specified split.
    
    Args:
        split_type: Data split ('Train', 'Validation', 'Test')
        use_augmentation: Whether to apply augmentation
        **kwargs: Additional augmentation parameters
        
    Returns:
        Augmentation callable
    """
    if split_type == 'Train' and use_augmentation:
        return MammographyAugmentation(**kwargs)
    return NoAugmentation()
