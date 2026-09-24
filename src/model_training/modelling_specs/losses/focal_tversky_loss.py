from typing import Callable

import torch
import torch.nn as nn

from .losses_interface import LossStrategy


class FocalTverskyLossModule(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.0, smooth: float = 1e-6):
        """ Initialize the Focal Tversky Loss module with given parameters. 
            - alpha: weight for false positives
            - beta: weight for false negatives (increasing it prioritizes recall)
            - gamma: focal parameter ( >1 enfatiza ejemplos difíciles)
            - smooth: smoothing factor 
        """
        super().__init__()
        self.alpha = alpha   
        self.beta = beta     
        self.gamma = gamma 
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()
        y_pred = y_pred.float()
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)

        tp = (y_pred * y_true).sum(dim=[1, 2, 3])
        fp = (y_pred * (1 - y_true)).sum(dim=[1, 2, 3])
        fn = ((1 - y_pred) * y_true).sum(dim=[1, 2, 3])

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return ((1 - tversky) ** self.gamma).mean()


class FocalTverskyLoss(LossStrategy):
    """ Variant of Dice that allows prioritizing recall over precision (beta > alpha). """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_loss(self) -> Callable:
        return FocalTverskyLossModule(alpha=self.alpha, beta=self.beta, gamma=self.gamma)