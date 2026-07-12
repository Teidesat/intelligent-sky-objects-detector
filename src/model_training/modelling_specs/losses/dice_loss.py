from typing import Callable

import torch
import torch.nn as nn

from .losses_interface import LossStrategy


class DiceLossModule(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()
        y_pred = y_pred.float()

        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)

        intersection = (y_true * y_pred).sum(dim=[1, 2, 3])
        union = (y_true + y_pred).sum(dim=[1, 2, 3])

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceLoss(LossStrategy):
    """Dice loss for binary segmentation."""

    def get_loss(self) -> Callable:
        return DiceLossModule()

# class DiceLossModule(nn.Module):
#     def __init__(self, smooth: float = 1e-6):
#         super().__init__()
#         self.smooth = smooth

#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#         y_true = y_true.float()
#         # CAMBIO AQUÍ: Convertimos los logits de entrada en probabilidades de forma interna
#         y_pred = torch.sigmoid(y_pred.float())

#         if y_true.dim() == 3:\
#             y_true = y_true.unsqueeze(1)
#         if y_pred.dim() == 3:\
#             y_pred = y_pred.unsqueeze(1)

#         intersection = (y_true * y_pred).sum(dim=[1, 2, 3])
#         union = (y_true + y_pred).sum(dim=[1, 2, 3])

#         dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
#         return 1 - dice.mean()


# class DiceLoss(LossStrategy):
#     """Dice loss for binary segmentation."""

#     @property
#     def needs_activation(self) -> bool:
#         return False  # CAMBIO AQUÍ: Ahora se alinea con WeightedBCELoss al esperar Logits

#     def get_loss(self) -> Callable:
#         module = DiceLossModule()
#         return module