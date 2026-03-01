from typing import Callable
import tensorflow as tf
from .losses_interface import LossStrategy

class DiceLoss(LossStrategy):
    """
    Dice loss for binary segmentation.
    Handles both masks and predictions with or without channel dimension.
    """
    def get_loss(self) -> Callable:
        def dice_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            if len(y_true.shape) == 3:
                y_true = tf.expand_dims(y_true, axis=-1)
            if len(y_pred.shape) == 3:
                y_pred = tf.expand_dims(y_pred, axis=-1)
            
            smooth = 1e-6
            intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
            union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
            
            dice = (2. * intersection + smooth) / (union + smooth)
            return 1 - tf.reduce_mean(dice)
        
        return dice_loss