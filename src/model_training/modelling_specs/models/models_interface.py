from abc import ABC, abstractmethod
import torch.nn as nn


class ModelStrategy(ABC):

    @abstractmethod
    def build(self, input_shape: tuple, num_classes: int, apply_activation: bool = True) -> nn.Module:
        """
        Args:
            input_shape: (H, W, C) tuple e.g. (256, 256, 1).
            num_classes: number of segmentation classes.
            apply_activation: whether to apply activation function to the output.

        Returns:
            nn.Module instance.
        """
        ...