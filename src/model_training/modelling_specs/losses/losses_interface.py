from abc import ABC, abstractmethod
from typing import Callable

import torch


class LossStrategy(ABC):
    """
    Default case: binary segmentation with a single-channel Sigmoid head (BCE, Dice, Focal, Tversky...).
    Only a loss that requires another output representation (e.g., CrossEntropyLoss) overrides these
    """

    @property
    def output_channels(self) -> int:
        return 1

    @property
    def needs_activation(self) -> bool:
        return True 

    @abstractmethod
    def get_loss(self) -> Callable:
        ...

    def format_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        """preds: Raw output from the model (B, C, H, W) -> format that the loss expects."""
        return preds.squeeze(1) if preds.shape[1] == 1 else preds

    def format_targets(self, masks: torch.Tensor) -> torch.Tensor:
        """masks: (B, H, W) long, values {0,1} -> format that the loss expects."""
        return masks.float()

    def predictions_to_probability(self, preds: torch.Tensor) -> torch.Tensor:
        """Only for metrics/inference: always (B, H, W), probability of 'star'."""
        return preds[:, 0] if preds.shape[1] == 1 else torch.softmax(preds, dim=1)[:, 1]