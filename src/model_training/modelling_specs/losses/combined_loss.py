from typing import Callable
import tensorflow as tf
from .losses_interface import LossStrategy

class CombinedLoss(LossStrategy):
    """
    Combines binary cross-entropy and Dice loss for binary segmentation (1 output channel).
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def get_loss(self) -> Callable:
        bce_weight = self.bce_weight
        dice_weight = self.dice_weight

        def combined_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            if len(y_true.shape) == 3:
                y_true = tf.expand_dims(y_true, axis=-1)
            if len(y_pred.shape) == 3:
                y_pred = tf.expand_dims(y_pred, axis=-1)

            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            bce = tf.reduce_mean(bce)  

            smooth = 1e-6
            intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
            union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
            dice = 1 - (2. * intersection + smooth) / (union + smooth)
            dice = tf.reduce_mean(dice)

            return bce_weight * bce + dice_weight * dice

        return combined_loss