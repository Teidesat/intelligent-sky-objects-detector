from typing import Callable

import tensorflow as tf

from .losses_interface import LossStrategy


class DiceLoss(LossStrategy):
    """
    Avoids high accuracy illusion due to class imbalance.
    """

    NUM_CLASSES = 2 # Either sky or non-sky (bright points)

    def get_loss(self) -> Callable:
        num_classes = self.NUM_CLASSES

        def dice_loss(y_true, y_pred):
            y_true_one_hot = tf.cast(
                tf.one_hot(tf.cast(y_true, tf.int32), num_classes), tf.float32
            )
            y_pred = tf.cast(y_pred, tf.float32)
            numerator = 2 * tf.reduce_sum(y_true_one_hot * y_pred, axis=(1, 2))
            denominator = tf.reduce_sum(y_true_one_hot + y_pred, axis=(1, 2))
            return 1 - tf.reduce_mean(numerator / (denominator + 1e-6))

        return dice_loss