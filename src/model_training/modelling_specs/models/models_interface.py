from abc import ABC, abstractmethod
import torch.nn as nn


class ModelStrategy(ABC):
    """
    ModelStrategy is an abstract base class that defines the interface for building different model architectures.
    Each concrete implementation of this class should provide a specific model architecture by implementing the build method.
    This allows for flexibility and extensibility in the model training pipeline, enabling the use of various architectures without changing the training code.
    """
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