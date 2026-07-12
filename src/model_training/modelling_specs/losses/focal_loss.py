from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses_interface import LossStrategy


class FocalLossModule(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):  # FALTABA ESTO
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()
        y_pred = torch.clamp(y_pred, 1e-6, 1 - 1e-6)
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class FocalLoss(LossStrategy):
    """Focal loss for binary segmentation with class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    def get_loss(self) -> Callable:
        return FocalLossModule(alpha=self.alpha, gamma=self.gamma)