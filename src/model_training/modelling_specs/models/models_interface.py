from abc import ABC, abstractmethod

from tensorflow import keras


class ModelStrategy(ABC):

    @abstractmethod
    def build(self, input_shape: tuple, num_classes: int) -> keras.Model:
        """
        Args:
            input_shape: (H, W, C) tuple e.g. (256, 256, 1).
            num_classes: number of segmentation classes.

        Returns:
            Uncompiled keras.Model instance.
        """
        ...