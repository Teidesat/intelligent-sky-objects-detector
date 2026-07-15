from typing import Callable

import torch                          
import torch.nn as nn
import torch.nn.functional as F

from .losses_interface import LossStrategy


class BCELoss(LossStrategy):
    def get_loss(self) -> Callable:
        return nn.BCELoss()


class WeightedBCELoss(LossStrategy):
    """
    Weighted Binary Cross-Entropy Loss for handling class imbalance in binary classification tasks.
    Commented out the `needs_activation` property, as it is not currently used in the implementation.
    """
    def __init__(self, pos_weight: float = 10.0):
        self.pos_weight = pos_weight

#     @property
#     def needs_activation(self) -> bool:
#         return False

    def get_loss(self) -> Callable:
        def loss_fn(y_pred, y_true):
            y_pred = torch.clamp(y_pred, 1e-6, 1 - 1e-6)
            bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
            #  bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
            weights = y_true * self.pos_weight + (1 - y_true)
            return (bce * weights).mean()
        return loss_fn
