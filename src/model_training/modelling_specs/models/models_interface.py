from abc import ABC, abstractmethod
import torch.nn as nn


class ModelStrategy(ABC):

    @abstractmethod
    def build(self, input_shape: tuple, num_classes: int) -> nn.Module:
        """
        Args:
            input_shape: (H, W, C) tuple e.g. (256, 256, 1).
            num_classes: number of segmentation classes.

        Returns:
            nn.Module instance.
        """
        ...