from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses_interface import LossStrategy

def lovasz_grad(gt_sorted):
    """Computes the Lovasz gradient for sorted labels."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """Computes the Lovász-Hinge loss for flattened predictions (logits and labels)."""
    labels = labels.float()
    signs = 2. * labels - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    labels_sorted = labels[perm]
    grad = lovasz_grad(labels_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits, labels, per_image=True):
    """
    logits: (B, H, W) o (B, 1, H, W) - output of the model without Sigmoid (raw logits).
    labels: (B, H, W) - valores {0, 1}.
    """
    if logits.dim() == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if labels.dim() == 4 and labels.shape[1] == 1:
        labels = labels.squeeze(1)

    if per_image:
        losses = [lovasz_hinge_flat(logit.flatten(), label.flatten()) 
                  for logit, label in zip(logits, labels)]
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=logits.device)
    else:
        return lovasz_hinge_flat(logits.flatten(), labels.flatten())


class LovaszHingeLossModule(nn.Module):
    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred: (B, 1, H, W) - LOGITS (sin Sigmoid)
        # y_true: (B, H, W) - float 0/1
        return lovasz_hinge(y_pred, y_true, per_image=self.per_image)


class LovaszHingeLoss(LossStrategy):
    """
    Lovasz-Hinge: directly optimizes the IoU. The model output should be raw logits (without Sigmoid).
    """

    @property
    def needs_activation(self) -> bool:
        return False

    def get_loss(self) -> Callable:
        return LovaszHingeLossModule(per_image=True)

    def predictions_to_probability(self, preds: torch.Tensor) -> torch.Tensor:
        logits = preds[:, 0] if preds.shape[1] == 1 else preds
        return torch.sigmoid(logits)