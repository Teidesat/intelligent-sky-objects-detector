from typing import Callable

import torch.nn as nn

from .losses_interface import LossStrategy

class BCELoss(LossStrategy):
    "Binary Cross-Entropy loss"
    def get_loss(self) -> Callable:
        return nn.BCELoss()