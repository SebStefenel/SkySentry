"""IoU-family losses (IoU / GIoU / CIoU) with pairwise and matched modes.

All functions take boxes in ``xyxy`` pixel format and are fully differentiable.

Conventions
-----------
* ``boxes_a``: ``(N, 4)`` or ``(B, N, 4)`` tensor.
* Pairwise functions broadcast to ``(N, M)`` (or ``(B, N, M)``) similarity
  matrices — used for label assignment.
* Matched functions return a per-pair tensor ``(N,)`` — used inside losses.
"""

from __future__ import annotations

import torch


def _area(boxes: torch.Tensor) -> torch.Tensor:
    """Area of ``(..., 4)`` xyxy boxes; degenerate boxes contribute 0."""
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU matrix between two sets of xyxy boxes.

    Args:
        boxes1: ``(N, 4)``.
        boxes2: ``(M, 4)``.
    Returns:
        ``(N, M)`` IoU values in ``[0, 1]``.
    """
    area1 = _area(boxes1).unsqueeze(1)          # (N, 1)
    area2 = _area(boxes2).unsqueeze(0)          # (1, M)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])   # (N, M, 2) top-left
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])   # (N, M, 2) bottom-right
    wh = (rb - lt).clamp(min=0)                 # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-7)


def _matched_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Elementwise IoU between two ``(N, 4)`` sets of aligned xyxy boxes."""
    area1 = _area(boxes1)
    area2 = _area(boxes2)
    lt = torch.max(boxes1[..., :2], boxes2[..., :2])
    rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-7)


def giou_loss_matched(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU loss for aligned boxes (Rezatofighi et al., CVPR 2019).

    ``L_GIoU = 1 - IoU + (C - union) / C`` where ``C`` is the smallest enclosing
    box. Adds gradient signal for disjoint boxes, but the enclosing-box term
    depends only on *where* the boxes are, not their size — weak for tiny targets.
    """
    iou = _matched_iou(boxes1, boxes2)
    lt = torch.min(boxes1[..., :2], boxes2[..., :2])
    rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    c_wh = (rb - lt).clamp(min=0)
    c_area = (c_wh[..., 0] * c_wh[..., 1]).clamp(min=1e-7)
    lt2 = torch.max(boxes1[..., :2], boxes2[..., :2])
    rb2 = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    wh2 = (rb2 - lt2).clamp(min=0)
    union = _area(boxes1) + _area(boxes2) - wh2[..., 0] * wh2[..., 1]
    giou = iou - (c_area - union) / c_area
    return 1.0 - giou


def ciou_loss_matched(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete IoU loss for aligned boxes (Zheng et al., AAAI 2020).

    ``L_CIoU = 1 - IoU + rho^2(b, b_gt)/c^2 + alpha * v`` with
    ``v = (4/pi^2) (arctan(w_gt/h_gt) - arctan(w/h))^2`` and
    ``alpha = v / (1 - IoU + v)``. Penalizes center offset, aspect ratio and
    IoU jointly.
    """
    iou = _matched_iou(boxes1, boxes2)
    # Centers (xyxy -> center x, y)
    c1 = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
    c2 = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
    lt = torch.min(boxes1[..., :2], boxes2[..., :2])
    rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    c_wh = (rb - lt).clamp(min=eps)
    center_dist_sq = ((c1 - c2) ** 2).sum(dim=-1) / (c_wh ** 2).sum(dim=-1).clamp(min=eps)
    w1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=eps)
    h1 = (boxes1[..., 3] - boxes1[..., 1]).clamp(min=eps)
    w2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=eps)
    h2 = (boxes2[..., 3] - boxes2[..., 1]).clamp(min=eps)
    v = (4.0 / (torch.pi ** 2)) * (torch.arctan(w2 / h2) - torch.arctan(w1 / h1)) ** 2
    alpha = v / (1.0 - iou + v).clamp(min=eps)
    return 1.0 - iou + center_dist_sq + alpha * v.detach()


def iou_similarity(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU similarity matrix, ``(N, M)``."""
    return _pairwise_iou(boxes1, boxes2)
