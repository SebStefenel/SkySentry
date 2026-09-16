"""Normalized Wasserstein Distance (NWD) loss for tiny-object bounding boxes.

Background
----------
IoU-based losses are pathological for tiny targets: two boxes that are
*almost* identical still have IoU = 0 the moment they stop overlapping, and
even when they overlap by 1–2 pixels the IoU value (and its gradient) swings
violently because the intersection area is comparable to each box's area.
For objects of 10–30 px in aerial imagery (VisDrone, drone-vs-bird), this
makes regression supervision sparse and noisy.

The NWD metric (Wang et al., arXiv:2110.13389, "A Normalized Gaussian
Wasserstein Distance for Tiny Object Detection") fixes this by modelling each
bounding box as a 2-D Gaussian and measuring distribution distance, which is
smooth and informative at *any* displacement.

Mathematical formulation
------------------------
1. **Gaussian modelling.** A box ``(cx, cy, w, h)`` defines the Gaussian

   .. math::
       \\mathcal{N}(\\mu, \\Sigma) \\quad\\text{with}\\quad
       \\mu = (c_x, c_y)^\\top,\\quad
       \\Sigma = \\mathrm{diag}\\!\\left(\\tfrac{w}{2}, \\tfrac{h}{2}\\right)^2

   i.e. the box is the 2-sigma contour of its own Gaussian (half-widths as
   standard deviations).

2. **Closed-form 2-Wasserstein distance.** For two diagonal Gaussians the
   2-Wasserstein (Fréchet/Bures) distance has the closed form

   .. math::
       W_2^2(\\mathcal{N}_1, \\mathcal{N}_2)
       = \\lVert \\mu_1 - \\mu_2 \\rVert_2^2
       + \\bigl\\lVert \\Sigma_1^{1/2} - \\Sigma_2^{1/2} \\bigr\\rVert_F^2

   With diagonal covariances this is simply

   .. math::
       W_2^2 = (c_{x,1}-c_{x,2})^2 + (c_{y,1}-c_{y,2})^2
       + \\left(\\tfrac{w_1-w_2}{2}\\right)^2
       + \\left(\\tfrac{h_1-h_2}{2}\\right)^2

3. **Scale normalization.** Raw W2 is in pixels, so its dynamic range depends
   on the dataset's object scale. It is squashed through an exponential with
   a dataset constant ``C`` (typically the mean absolute object size, e.g.
   ≈ 12.8 px on VisDrone):

   .. math::
       \\mathrm{NWD}(\\mathcal{N}_1, \\mathcal{N}_2)
       = \\exp\\!\\left(-\\frac{\\sqrt{W_2^2}}{C}\\right) \\in (0, 1]

   ``NWD = 1`` for identical boxes and decays smoothly to 0 — a metric that
   is differentiable everywhere and treats a 1-px miss on a 6-px bird the
   same way IoU treats a 30-px miss on a 300-px car.

4. **Loss.** ``L_NWD = 1 - NWD`` on matched (prediction, target) pairs.
   Empirically NWD blends well with CIoU, which remains a better metric for
   large objects and near-perfect localization:

   .. math::
       L_{\\text{box}} = \\alpha \\, (1 - \\mathrm{NWD})
       + (1 - \\alpha) \\, L_{\\text{CIoU}}

This module implements exactly that, with an operator exposing both the
pairwise similarity matrix (for label assignment, where NWD shines vs IoU)
and the matched-pair loss (for regression supervision).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .iou_losses import ciou_loss_matched

__all__ = [
    "xyxy_to_cxcywh",
    "cxcywh_to_xyxy",
    "wasserstein_sq_distance",
    "nwd_similarity",
    "NWDLoss",
]


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 4)`` boxes from ``(x1, y1, x2, y2)`` to ``(cx, cy, w, h)``."""
    wh = boxes[..., 2:] - boxes[..., :2]
    cxcy = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    return torch.cat([cxcy, wh], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 4)`` boxes from ``(cx, cy, w, h)`` to ``(x1, y1, x2, y2)``."""
    xy1 = boxes[..., :2] - boxes[..., 2:] * 0.5
    xy2 = boxes[..., :2] + boxes[..., 2:] * 0.5
    return torch.cat([xy1, xy2], dim=-1)


def wasserstein_sq_distance(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Squared 2-Wasserstein distance between the Gaussians of two box sets.

    Implements
    ``W2^2 = ||mu1 - mu2||^2 + ||Sigma1^{1/2} - Sigma2^{1/2}||_F^2``
    for diagonal covariances ``Sigma_i = diag((w_i/2)^2, (h_i/2)^2)``, i.e.

    ``(cx1-cx2)^2 + (cy1-cy2)^2 + ((w1-w2)/2)^2 + ((h1-h2)/2)^2``.

    Args:
        boxes1: ``(N, 4)`` boxes in xyxy format (values may require grad).
        boxes2: ``(M, 4)`` boxes in xyxy format.
    Returns:
        ``(N, M)`` tensor of squared distances in pixel^2 units.
    """
    b1 = xyxy_to_cxcywh(boxes1)                 # (N, 4)
    b2 = xyxy_to_cxcywh(boxes2)                 # (M, 4)
    mu1, half_wh1 = b1[:, :2], b1[:, 2:] * 0.5  # std devs
    mu2, half_wh2 = b2[:, :2], b2[:, 2:] * 0.5

    # ||mu1 - mu2||^2  -> (N, M)
    center_term = ((mu1[:, None, :] - mu2[None, :, :]) ** 2).sum(dim=-1)
    # ||Sigma1^{1/2} - Sigma2^{1/2}||_F^2 = sum over dims of (s1 - s2)^2
    shape_term = ((half_wh1[:, None, :] - half_wh2[None, :, :]) ** 2).sum(dim=-1)
    return center_term + shape_term


def nwd_similarity(boxes1: torch.Tensor, boxes2: torch.Tensor, constant: float = 12.8) -> torch.Tensor:
    """Pairwise NWD similarity matrix in ``(0, 1]``.

    ``NWD(i, j) = exp(-sqrt(W2^2(i, j)) / C)``.

    Args:
        boxes1: ``(N, 4)`` xyxy boxes.
        boxes2: ``(M, 4)`` xyxy boxes.
        constant: Normalizing constant ``C`` in pixels. Should be set to the
            dataset's mean absolute object size (paper uses ~12.8 for VisDrone).
            Larger ``C`` → slower decay → more forgiving at a given pixel offset.
    Returns:
        ``(N, M)`` similarity matrix; 1.0 on the diagonal of identical boxes.
    """
    w2_sq = wasserstein_sq_distance(boxes1, boxes2)
    return torch.exp(-torch.sqrt(w2_sq.clamp(min=0)) / max(constant, 1e-6))


class NWDLoss(nn.Module):
    """Normalized Wasserstein Distance regression loss for tiny boxes.

    Args:
        constant: Scale-normalizing constant ``C`` (pixels). Calibrate to the
            dataset's mean object size; see :func:`nwd_similarity`.
        alpha: Blend weight between the NWD term and the CIoU term:
            ``L = alpha * (1 - NWD) + (1 - alpha) * CIoU_loss``.
            ``alpha = 1.0`` → pure NWD; ``alpha = 0.0`` → pure CIoU.
            The paper's ablations favor a strong NWD weight for tiny objects.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.

    Shapes:
        * ``forward(pred, target)``: ``(N, 4)`` xyxy each (aligned pairs), or
          ``(B, N, 4)`` — the batch dimension is flattened.
        * With optional ``weights`` ``(N,)``: weighted per-pair mean.
    """

    def __init__(self, constant: float = 12.8, alpha: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be mean|sum|none, got {reduction!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.constant = float(constant)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the blended NWD + CIoU loss on aligned box pairs."""
        if pred_boxes.shape != target_boxes.shape:
            raise ValueError(f"shape mismatch: {tuple(pred_boxes.shape)} vs {tuple(target_boxes.shape)}")
        if pred_boxes.numel() == 0:
            return pred_boxes.sum() * 0.0  # keep the graph on the right device

        pred_flat = pred_boxes.reshape(-1, 4)
        target_flat = target_boxes.reshape(-1, 4)

        nwd = nwd_similarity(pred_flat, target_flat, self.constant).diagonal()
        nwd_loss = 1.0 - nwd
        ciou = ciou_loss_matched(pred_flat, target_flat)

        loss = self.alpha * nwd_loss + (1.0 - self.alpha) * ciou

        if weights is not None:
            w = weights.reshape(-1).to(loss.dtype)
            loss = loss * w
            denom = w.sum().clamp(min=1e-7)
            if self.reduction == "mean":
                return loss.sum() / denom
            if self.reduction == "sum":
                return loss.sum()
            return loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
