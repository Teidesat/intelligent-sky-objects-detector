from typing import Callable

import torch

from .losses_interface import LossStrategy


class CombinedLoss(LossStrategy):
    """Weighted combination of two loss functions."""

    def __init__(self, loss_a: LossStrategy, loss_b: LossStrategy, weight_a: float = 0.5):
        self.loss_a = loss_a.get_loss()
        self.loss_b = loss_b.get_loss()
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    def get_loss(self) -> Callable:
        def combined(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            return self.weight_a * self.loss_a(y_pred, y_true) + \
                   self.weight_b * self.loss_b(y_pred, y_true)

        return combined