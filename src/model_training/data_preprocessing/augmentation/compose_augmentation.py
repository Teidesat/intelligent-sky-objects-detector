import torch
from .augmentation_interface import AugmentationStrategy


class ComposeAugmentation(AugmentationStrategy):
    """
    Aplica una lista de augmentations secuencialmente.
    """
    def __init__(self, augmentations: list):
        self.augmentations = augmentations

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        for aug in self.augmentations:
            image, mask = aug(image, mask)
        return image, mask