import random
import torch
from .augmentation_interface import AugmentationStrategy

class BrightnessContrastAugmentation(AugmentationStrategy):
    """
    Aplica cambios aleatorios de brillo y contraste a la imagen.
    No modifica la máscara.
    """
    def __init__(self, brightness_range: tuple = (0.7, 1.3), contrast_range: tuple = (0.7, 1.3)):
        """
        Args:
            brightness_range: (min, max) factor multiplicativo para el brillo.
            contrast_range: (min, max) factor multiplicativo para el contraste.
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple:
        # image: (C, H, W) o (B, C, H, W) - asumimos (C, H, W) para una imagen
        # Si es batch, aplicamos el mismo factor a todo el batch
        if image.dim() == 4:  # (B, C, H, W)
            batch_size = image.size(0)
            # Generamos factores por batch (no por imagen individual para consistencia)
            brightness_factor = random.uniform(*self.brightness_range)
            contrast_factor = random.uniform(*self.contrast_range)
            # Aplicamos a todo el batch
            image = image * brightness_factor
            # Contraste: restamos la media (por canal) y escalamos
            # Para simplificar, escalamos la imagen completa restando la media global del batch
            mean = image.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
            image = (image - mean) * contrast_factor + mean
        else:  # (C, H, W)
            brightness_factor = random.uniform(*self.brightness_range)
            contrast_factor = random.uniform(*self.contrast_range)
            image = image * brightness_factor
            mean = image.mean(dim=[1, 2], keepdim=True)  # (C, 1, 1)
            image = (image - mean) * contrast_factor + mean

        # Recortamos valores para evitar overflow (asumiendo que la imagen está en [0,1])
        image = torch.clamp(image, 0.0, 1.0)
        return image, mask