from typing import Callable

import tensorflow as tf

from .losses_interface import LossStrategy


class CombinedLoss(LossStrategy):
    """
    Combines categorical cross-entropy (stability) and Dice loss (sensibility)
    """

    NUM_CLASSES = 2

    def __init__(self, cce_weight: float = 0.5, dice_weight: float = 1.0):
        self.cce_weight = cce_weight
        self.dice_weight = dice_weight

    def get_loss(self) -> Callable:
        num_classes = self.NUM_CLASSES
        cce_weight = self.cce_weight
        dice_weight = self.dice_weight

        def combined_loss(y_true, y_pred):
            y_true_one_hot = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), num_classes), tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            # Cross-entropy
            cce = tf.keras.losses.CategoricalCrossentropy()(y_true_one_hot, y_pred)

            # Dice
            smooth = 1e-6
            intersection = tf.reduce_sum(y_true_one_hot[..., 1] * y_pred[..., 1])
            union = tf.reduce_sum(y_true_one_hot[..., 1]) + tf.reduce_sum(y_pred[..., 1])
            dice = 1 - (2. * intersection + smooth) / (union + smooth)

            return cce_weight * cce + dice_weight * dice

        return combined_loss