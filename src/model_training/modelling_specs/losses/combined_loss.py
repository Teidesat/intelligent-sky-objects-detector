from typing import Callable

import torch

from .losses_interface import LossStrategy


class CombinedLoss(LossStrategy):
    """ Combines two loss functions (loss_a and loss_b) with specified weights. Both losses must have the same output shape and activation requirements. """
    def __init__(self, loss_a: LossStrategy, loss_b: LossStrategy, weight_a: float = 0.5):
        if (loss_a.output_channels != loss_b.output_channels or
                loss_a.needs_activation != loss_b.needs_activation):
            raise ValueError("CombinedLoss requiere que ambas losses usen la misma forma de salida del modelo.")

        self._reference_strategy = loss_a
        self.loss_a = loss_a.get_loss()
        self.loss_b = loss_b.get_loss()
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    @property
    def output_channels(self) -> int:
        return self._reference_strategy.output_channels

    @property
    def needs_activation(self) -> bool:
        return self._reference_strategy.needs_activation

    def format_predictions(self, preds):
        return self._reference_strategy.format_predictions(preds)

    def format_targets(self, masks):
        return self._reference_strategy.format_targets(masks)

    def predictions_to_probability(self, preds):
        return self._reference_strategy.predictions_to_probability(preds)

    def get_loss(self) -> Callable:
        def combined(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            return self.weight_a * self.loss_a(y_pred, y_true) + self.weight_b * self.loss_b(y_pred, y_true)
        return combined