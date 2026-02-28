from typing import Callable

from tensorflow import keras

from .losses_interface import LossStrategy


class CrossEntropyLoss(LossStrategy):
    """
    Cross-entropy loss
    """

    def get_loss(self) -> Callable:
        return keras.losses.SparseCategoricalCrossentropy()