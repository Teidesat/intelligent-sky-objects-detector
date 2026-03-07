from typing import Callable

import torch.nn as nn

from .losses_interface import LossStrategy


class CrossEntropyLoss(LossStrategy):
    """Cross-entropy loss (multi-class, sparse targets)."""

    def get_loss(self) -> Callable:
        # nn.CrossEntropyLoss es Callable: loss(y_pred, y_true)
        # Espera preds: (B, C, H, W) y targets: (B, H, W) con dtype long
        return nn.CrossEntropyLoss()