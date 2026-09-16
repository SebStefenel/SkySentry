"""Classification losses: sigmoid focal loss and balanced BCE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["sigmoid_focal_loss", "FocalLoss"]


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sigmoid focal loss (Lin et al., ICCV 2017).

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * BCE(p, p*)`` computed elementwise
    on independent sigmoid outputs (multi-label style, one channel per class).

    Args:
        logits: raw scores, any shape.
        targets: binary targets, same shape as ``logits``.
        alpha: class-imbalance weight on positives (paper: 0.25).
        gamma: focusing parameter; ``gamma > 0`` down-weights easy examples.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
    """
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class FocalLoss(nn.Module):
    """Module wrapper around :func:`sigmoid_focal_loss`."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(logits, targets, self.alpha, self.gamma, self.reduction)
