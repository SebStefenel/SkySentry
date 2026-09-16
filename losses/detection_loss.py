"""Multi-level detection loss: focal classification + objectness BCE + NWD box regression.

Label assignment
----------------
A tiny-object-friendly assignment (FCOS-style scale partitioning + center
sampling):

* Each ground-truth box is routed to **one pyramid level** based on its longer
  side: level with stride ``s`` owns boxes with ``max(w, h)`` in
  ``[scale_ranges[s][0], scale_ranges[s][1])`` pixels. For a P2 (stride-4)
  head the default ranges send anything under 64 px to the two finest levels,
  which is where tiny aerial targets live.
* Within the owning level, every grid cell whose **center point** falls inside
  the GT box (optionally expanded by a center-sampling radius) is a positive.

This keeps supervision dense for small boxes (a 6-px bird still owns a 2x2
block of stride-4 cells) and guarantees at least the cell containing the
center is positive.

Regression supervision on positives is the blended NWD + CIoU loss from
:class:`losses.nwd_loss.NWDLoss`; classification uses sigmoid focal loss on
``num_classes`` channels; objectness is BCE on every cell with target 1 for
positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal_loss import sigmoid_focal_loss
from .nwd_loss import NWDLoss, cxcywh_to_xyxy

__all__ = ["DetectionLossConfig", "TinyDetectionLoss"]


@dataclass
class DetectionLossConfig:
    """Hyperparameters of the composite detection loss."""

    num_classes: int = 2
    strides: tuple[int, ...] = (4, 8, 16)
    # Per-stride ownership ranges on the longer GT side, in input pixels.
    # (0, 64) for stride 4, [64, 128) for stride 8, [128, inf) for stride 16.
    scale_ranges: tuple[tuple[float, float], ...] = ((0.0, 64.0), (64.0, 128.0), (128.0, float("inf")))
    center_sample_radius: float = 1.5   # in cells; 0 disables center sampling
    weight_box: float = 5.0             # lambda_box for NWD+CIoU regression
    weight_cls: float = 1.0             # lambda_cls for focal loss
    weight_obj: float = 1.0             # lambda_obj for objectness BCE
    nwd_constant: float = 12.8          # dataset-scale constant C for NWD
    nwd_alpha: float = 0.5              # NWD vs CIoU blend
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class _LevelTargets:
    """Per-image buffers of assignment results for one pyramid level."""

    cls: list[torch.Tensor] = field(default_factory=list)   # (nc, H, W) one-hot
    obj: list[torch.Tensor] = field(default_factory=list)   # (H, W) binary
    box: list[torch.Tensor] = field(default_factory=list)   # (4, H, W) cxcywh pixels (xyxy rows in targets)
    pos: list[torch.Tensor] = field(default_factory=list)   # (H, W) bool


class TinyDetectionLoss(nn.Module):
    """Composite loss over the P2 detector's per-level outputs.

    Forward signature::

        loss, stats = criterion(outputs, targets)

    Args:
        outputs: list (one entry per pyramid level, ordered by stride) of
            dicts with keys ``cls`` ``(B, nc, H, W)``, ``obj`` ``(B, 1, H, W)``
            and ``box`` ``(B, 4, H, W)`` — box predictions are
            ``(cx, cy, w, h)`` in input-image pixel coordinates.
        targets: list (length B) of ``(P, 5)`` tensors with rows
            ``(class_id, x1, y1, x2, y2)`` in input-image pixels. P = 0 is
            allowed (pure background image).

    Returns:
        ``loss`` scalar; ``stats`` dict with the loss components and counts.
    """

    def __init__(self, config: DetectionLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or DetectionLossConfig()
        if len(self.cfg.scale_ranges) != len(self.cfg.strides):
            raise ValueError("scale_ranges must have one (lo, hi) per stride")
        self.box_loss = NWDLoss(constant=self.cfg.nwd_constant, alpha=self.cfg.nwd_alpha, reduction="none")
        self.obj_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # ------------------------------------------------------------------ #
    # Assignment                                                          #
    # ------------------------------------------------------------------ #

    def _assign_level(self, boxes_xyxy: torch.Tensor, stride: int, feat_hw: tuple[int, int]) -> torch.Tensor:
        """Boolean positives mask ``(H, W)`` for one GT box on one level.

        A cell is positive if its center lies inside the GT box expanded by
        ``center_sample_radius`` cells around the GT center.
        """
        device = boxes_xyxy.device
        h, w = feat_hw
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")           # (H, W) cell centers in pixels

        x1, y1, x2, y2 = boxes_xyxy.tolist()
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        r = self.cfg.center_sample_radius * stride
        inside = (
            (xx > x1) & (xx < x2) & (yy > y1) & (yy < y2)
        ) | (
            ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
        )
        return inside

    def _build_targets(
        self,
        outputs: list[dict[str, torch.Tensor]],
        targets: list[torch.Tensor],
        input_hw: tuple[int, int],
        device: torch.device,
    ) -> list[_LevelTargets]:
        """Route GT boxes to their owning level and rasterize targets there."""
        levels: list[_LevelTargets] = []
        for out in outputs:
            _, _, h, w = out["cls"].shape
            levels.append(_LevelTargets())

        for img_targets in targets:
            if img_targets.numel() == 0:
                for li, out in enumerate(outputs):
                    _, _, h, w = out["cls"].shape
                    levels[li].cls.append(torch.zeros(self.cfg.num_classes, h, w, device=device))
                    levels[li].obj.append(torch.zeros(h, w, device=device))
                    levels[li].box.append(torch.zeros(4, h, w, device=device))
                    levels[li].pos.append(torch.zeros(h, w, dtype=torch.bool, device=device))
                continue

            gt_cls = img_targets[:, 0].long()
            gt_xyxy = img_targets[:, 1:5].float()
            longer_side = (gt_xyxy[:, 2:4] - gt_xyxy[:, 0:2]).max(dim=1).values  # (P,)

            for li, stride in enumerate(self.cfg.strides):
                out = outputs[li]
                _, _, h, w = out["cls"].shape
                cls_buf = torch.zeros(self.cfg.num_classes, h, w, device=device)
                obj_buf = torch.zeros(h, w, device=device)
                box_buf = torch.zeros(4, h, w, device=device)
                pos_buf = torch.zeros(h, w, dtype=torch.bool, device=device)

                lo, hi = self.cfg.scale_ranges[li]
                owner = (longer_side >= lo) & (longer_side < hi)
                for gt_idx in owner.nonzero(as_tuple=True)[0]:
                    pos = self._assign_level(gt_xyxy[gt_idx], stride, (h, w))
                    pos_buf |= pos
                    x1, y1, x2, y2 = gt_xyxy[gt_idx]
                    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                    cls_buf[gt_cls[gt_idx]][pos] = 1.0
                    obj_buf[pos] = 1.0
                    box_buf[0][pos] = cx
                    box_buf[1][pos] = cy
                    box_buf[2][pos] = (x2 - x1).clamp(min=1.0)
                    box_buf[3][pos] = (y2 - y1).clamp(min=1.0)

                levels[li].cls.append(cls_buf)
                levels[li].obj.append(obj_buf)
                levels[li].box.append(box_buf)
                levels[li].pos.append(pos_buf)
        return levels

    # ------------------------------------------------------------------ #
    # Loss                                                                #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        outputs: list[dict[str, torch.Tensor]],
        targets: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = outputs[0]["cls"].device
        b = outputs[0]["cls"].shape[0]
        input_hw = (int(outputs[0]["__input_h__"]), int(outputs[0]["__input_w__"])) if "__input_h__" in outputs[0] else self._infer_input_hw(outputs)
        levels = self._build_targets(outputs, targets, input_hw, device)
        # FCOS convention: sum-style losses are normalized by the TOTAL number
        # of positive samples in the batch (not per level, not by cells).
        n_pos = sum(int(torch.stack(lt.pos).sum()) for lt in levels)

        cls_losses, obj_losses, box_losses = [], [], []
        for li, out in enumerate(outputs):
            lt = levels[li]
            pred_cls = out["cls"]    # (B, nc, H, W) logits
            pred_obj = out["obj"]    # (B, 1, H, W) logits
            pred_box = out["box"]    # (B, 4, H, W) cxcywh pixels

            tgt_cls = torch.stack(lt.cls, dim=0)                    # (B, nc, H, W)
            tgt_obj = torch.stack(lt.obj, dim=0)                    # (B, H, W)
            tgt_box = torch.stack(lt.box, dim=0)                    # (B, 4, H, W)
            pos = torch.stack(lt.pos, dim=0)                        # (B, H, W)

            cls_losses.append(sigmoid_focal_loss(pred_cls, tgt_cls, self.cfg.focal_alpha, self.cfg.focal_gamma, reduction="sum") / max(n_pos, 1))
            obj_losses.append(self.obj_loss_fn(pred_obj[:, 0], tgt_obj).mean())

            if pos.any():
                p = pred_box.permute(0, 2, 3, 1)[pos]               # (K, 4) cxcywh
                t = tgt_box.permute(0, 2, 3, 1)[pos]                # (K, 4) cxcywh
                p_xyxy = cxcywh_to_xyxy(p)
                t_xyxy = cxcywh_to_xyxy(t)
                # IoU-based losses are meaningless for the degenerate
                # "center-only" positives on boxes smaller than one cell;
                # weight them by relative size so they still get NWD signal.
                area_t = (t_xyxy[:, 2] - t_xyxy[:, 0]) * (t_xyxy[:, 3] - t_xyxy[:, 1])
                w = torch.clamp(area_t / self.cfg.nwd_constant**2, max=1.0) + 1e-3
                per_pair = self.box_loss(p_xyxy, t_xyxy)            # reduction='none'
                box_losses.append((per_pair * w).sum() / w.sum())

        cls_loss = torch.stack(cls_losses).sum()
        obj_loss = torch.stack(obj_losses).sum()
        box_loss = torch.stack(box_losses).sum() if box_losses else torch.zeros((), device=device)
        if n_pos == 0:
            box_loss = torch.zeros((), device=device, requires_grad=True) + box_loss * 0.0

        total = (
            self.cfg.weight_box * box_loss
            + self.cfg.weight_cls * cls_loss
            + self.cfg.weight_obj * obj_loss
        )
        stats = {
            "loss": total.detach(),
            "loss_box": box_loss.detach(),
            "loss_cls": cls_loss.detach(),
            "loss_obj": obj_loss.detach(),
            "num_positives": torch.tensor(float(n_pos), device=device),
        }
        return total, stats

    @staticmethod
    def _infer_input_hw(outputs: list[dict[str, torch.Tensor]]) -> tuple[int, int]:
        """Recover input size from stride-4 feature map: H_in ≈ H_feat * 4."""
        _, _, h, w = outputs[0]["cls"].shape
        return h * 4, w * 4
