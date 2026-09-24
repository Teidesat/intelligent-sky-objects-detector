from typing import Callable

import torch
import torch.nn as nn

from .losses_interface import LossStrategy


class CrossEntropyLoss(LossStrategy):
    """
    Tends to converge with many false positives due to pixel imbalance, 
    as the average loss is dominated by the background (easy to predict) 
    and does not sufficiently penalize each individual false positive.
    """

    def __init__(self, num_classes: int = 2, star_class_index: int = 1,
                 class_weights: list[float] | None = None):
        self.num_classes = num_classes
        self.star_class_index = star_class_index
        self.class_weights = class_weights

    @property
    def output_channels(self) -> int:
        return self.num_classes

    @property
    def needs_activation(self) -> bool:
        return False

    def get_loss(self) -> Callable:
        weight = torch.tensor(self.class_weights, dtype=torch.float32) if self.class_weights else None
        return nn.CrossEntropyLoss(weight=weight)

    def format_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        return preds

    def format_targets(self, masks: torch.Tensor) -> torch.Tensor:
        return masks.long()

    def predictions_to_probability(self, preds: torch.Tensor) -> torch.Tensor:
        return torch.softmax(preds, dim=1)[:, self.star_class_index]