"""Inference utilities: decoding model outputs into final detections.

Pure-torch NMS (no torchvision dependency — see repo requirements notes).
Class-aware NMS by default; pass ``class_agnostic=True`` for the transfer
evaluation where all VisDrone categories collapse into one "object" class.
"""

from __future__ import annotations

import torch

from models.detector_p2 import TinyDetector

__all__ = ["nms", "postprocess"]


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> list[int]:
    """Greedy NMS. ``boxes`` (N, 4) xyxy; returns kept indices, score order."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(x1[i], x1[rest])
        yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest])
        yy2 = torch.min(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-7)
        order = rest[iou <= iou_threshold]
    return keep


@torch.no_grad()
def postprocess(
    model: TinyDetector,
    images: torch.Tensor,
    conf_threshold: float = 0.01,
    nms_iou: float = 0.5,
    class_agnostic: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Model batch → per-image detections.

    Args:
        images: (B, 3, H, W) letterboxed input.
        conf_threshold: keep cells with obj·max_cls_prob above this. Use a low
            value (~0.01) for AP evaluation, higher (~0.25) for deployment.
        nms_iou: NMS overlap threshold.
        class_agnostic: if True, every detection becomes class 0 (used when
            evaluating synthetic-trained models on VisDrone, whose class space
            differs — localization transfer only).
    Returns:
        list (len B) of dicts with ``boxes`` (N, 4) xyxy, ``scores`` (N,),
        ``labels`` (N,) int64.
    """
    model.eval()
    outputs = model(images)
    detections: list[dict[str, torch.Tensor]] = []
    raw = model.decode(outputs, conf_threshold=conf_threshold)   # (B, N, 6), padded -1
    for i in range(raw.shape[0]):
        d = raw[i][raw[i][:, 4] > conf_threshold]
        if d.shape[0] == 0:
            detections.append({"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)})
            continue
        if class_agnostic:
            d = torch.cat([d[:, :5], torch.zeros(d.shape[0], 1)], dim=1)
        keep = nms(d[:, :4], d[:, 4], nms_iou)
        d = d[keep]
        detections.append({"boxes": d[:, :4].cpu(), "scores": d[:, 4].cpu(), "labels": d[:, 5].long().cpu()})
    return detections
