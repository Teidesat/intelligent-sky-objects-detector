from typing import Callable

import torch                          # FALTABA
import torch.nn as nn
import torch.nn.functional as F

from .losses_interface import LossStrategy


class BCELoss(LossStrategy):
    def get_loss(self) -> Callable:
        return nn.BCELoss()


class WeightedBCELoss(LossStrategy):
    def __init__(self, pos_weight: float = 10.0):
        self.pos_weight = pos_weight

    def get_loss(self) -> Callable:
        def loss_fn(y_pred, y_true):
            y_pred = torch.clamp(y_pred, 1e-6, 1 - 1e-6)
            bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
            weights = y_true * self.pos_weight + (1 - y_true)
            return (bce * weights).mean()
        return loss_fn

# class BCELoss(LossStrategy):
#     "Binary Cross-Entropy loss"
#     def get_loss(self) -> Callable:
#         return nn.BCELoss()


# class WeightedBCELoss(LossStrategy):
#     def __init__(self, pos_weight: float = 10.0):
#         self.pos_weight = pos_weight

#     @property
#     def needs_activation(self) -> bool:
#         return False
    
#     def get_loss(self) -> Callable:
#         def loss_fn(y_pred, y_true):
#             # CAMBIO AQUÍ: Usar con _with_logits para que acepte la salida sin Sigmoid
#             bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
#             weights = y_true * self.pos_weight + (1 - y_true)
#             return (bce * weights).mean()
#         return loss_fn