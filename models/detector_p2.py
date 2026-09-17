"""Lightweight YOLO-style detector with a P2 (stride-4) high-resolution head.

Why P2 matters
--------------
Standard YOLO heads start at P3 (stride 8): after 8x downsampling, a 16-px
object occupies a 2x2 cell patch and a 6-px bird is *smaller than one
receptive cell* — classification and regression signals alias onto the
background. Aerial tiny-object detectors (drone/bird/clutter at 6-32 px)
therefore add a **P2 level**: features at 1/4 input resolution, where a 6-px
bird spans a 1.5x1.5-cell neighborhood and keeps its exact location.

Architecture (compact CSP variant, ~1.9 M params at width 0.50)
---------------------------------------------------------------
* **Backbone**: conv stem (s2) → CSP stages at strides 4/8/16/32 producing
  ``C2 (s4), C3 (s8), C4 (s16), C5 (s32)``.
* **Neck**: PAN/FPN. Top-down path fuses C5→P4→P3→P2; bottom-up path
  re-fines P2→N3→N4→N5. Detection taps levels in ``strides`` — default
  ``(4, 8, 16)``, the standard tiny-object configuration (add 32 for large
  vehicles/aircraft).
* **Heads**: anchor-free decoupled heads — separate 3x3 conv stems for
  classification, objectness and box regression. Boxes decode anchor-free as
  ``cx = (gx + sigmoid(tx)) * s``, ``w = s * exp(tw)`` (per-cell offset +
  log-size), giving NWD-friendly cxcywh outputs in input pixel coordinates.

Outputs are a list of per-level dicts (``cls``, ``obj``, ``box``) consumed by
:class:`losses.detection_loss.TinyDetectionLoss`; :meth:`TinyDetector.decode`
converts them to ``(B, N, 6)`` ``(x1, y1, x2, y2, score, class)`` detections
for inference/tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DetectorConfig", "TinyDetector"]


def make_divisible(x: float, divisor: int = 8) -> int:
    return max(divisor, int(math.ceil(x / divisor) * divisor))


class ConvBnSiLU(nn.Module):
    """Conv2d + BatchNorm + SiLU, the atomic block of the network."""

    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard CSP bottleneck: 3x3 reduce → 3x3 expand, optional shortcut."""

    def __init__(self, c_in: int, c_out: int, shortcut: bool = True, expansion: float = 0.5) -> None:
        super().__init__()
        c_mid = int(c_out * expansion)
        self.cv1 = ConvBnSiLU(c_in, c_mid, 3, 1)
        self.cv2 = ConvBnSiLU(c_mid, c_out, 3, 1)
        self.add = shortcut and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """CSP bottleneck stage (3 convs, ``n`` bottleneck branches)."""

    def __init__(self, c_in: int, c_out: int, n: int = 1, shortcut: bool = True) -> None:
        super().__init__()
        c_mid = c_out // 2
        self.cv1 = ConvBnSiLU(c_in, c_mid, 1)
        self.cv2 = ConvBnSiLU(c_in, c_mid, 1)
        self.cv3 = ConvBnSiLU(2 * c_mid, c_out, 1)
        self.m = nn.Sequential(*(Bottleneck(c_mid, c_mid, shortcut) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling (fast): serial 5x5 max-pools, concatenated."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        c_mid = c_in // 2
        self.cv1 = ConvBnSiLU(c_in, c_mid, 1)
        self.cv2 = ConvBnSiLU(c_mid * 4, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv1(x)
        y2 = F.max_pool2d(y1, 5, 1, 2)
        y3 = F.max_pool2d(y2, 5, 1, 2)
        return self.cv2(torch.cat((y1, y2, y3, F.max_pool2d(y3, 5, 1, 2)), dim=1))


@dataclass
class DetectorConfig:
    """Hyperparameters of the P2 detector."""

    num_classes: int = 2
    width: float = 0.50                   # channel width multiplier
    depth: float = 0.33                   # bottleneck depth multiplier
    strides: tuple[int, ...] = (4, 8, 16)  # pyramid levels used for detection
    in_chans: int = 3
    # Base channel counts of the backbone stages (before width multiplier).
    base_channels: tuple[int, ...] = (32, 64, 128, 256, 512)

    @property
    def channels(self) -> list[int]:
        return [make_divisible(c * self.width) for c in self.base_channels]

    @property
    def neck_channels(self) -> int:
        return self.channels[3]           # width of neck/decode features (256 * width)


class Backbone(nn.Module):
    """CSP-darknet-style backbone producing C2 (s4), C3 (s8), C4 (s16), C5 (s32)."""

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__()
        ch = cfg.channels
        n1 = max(round(1 * cfg.depth), 1)
        n2 = max(round(2 * cfg.depth), 1)
        n3 = max(round(3 * cfg.depth), 1)

        self.stem = ConvBnSiLU(cfg.in_chans, ch[0], 3, 2)                                      # s2
        self.stage1 = nn.Sequential(ConvBnSiLU(ch[0], ch[1], 3, 2), C3(ch[1], ch[1], n1))      # s4  -> C2
        self.stage2 = nn.Sequential(ConvBnSiLU(ch[1], ch[2], 3, 2), C3(ch[2], ch[2], n2))      # s8  -> C3
        self.stage3 = nn.Sequential(ConvBnSiLU(ch[2], ch[3], 3, 2), C3(ch[3], ch[3], n3))      # s16 -> C4
        self.stage4 = nn.Sequential(
            ConvBnSiLU(ch[3], ch[4], 3, 2), C3(ch[4], ch[4], n3), SPPF(ch[4], ch[4])
        )                                                                                       # s32 -> C5
        self.out_channels = ch[1:5]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c2, c3, c4, c5


class P2Neck(nn.Module):
    """PAN neck that builds only the pyramid levels a config detects on.

    Top-down (coarse→fine): C5 → up⊕C4 → t16 → up⊕C3 → t8 → up⊕C2 → t4.
    Bottom-up (fine→coarse): from the finest top-down level back down,
    ``n_s = C3(cat(down(prev), t_s))``. Levels absent from ``cfg.strides``
    are not built (no dead parameters, one code path for the P2 ablation).

    ``forward`` returns ``{stride: feature}`` for exactly ``cfg.strides``.
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__()
        ch = cfg.channels
        c = cfg.neck_channels
        n = max(round(1 * cfg.depth), 1)
        self.strides = tuple(cfg.strides)
        self.min_stride = min(self.strides)

        # Laterals: project every backbone level from the finest detected
        # stride up to C5 to neck width (coarser ones never enter the neck).
        self.lat = nn.ModuleDict({
            str(s): ConvBnSiLU(c_back, c, 1)
            for s, c_back in zip((4, 8, 16, 32), (ch[1], ch[2], ch[3], ch[4]))
            if s >= self.min_stride or s == 32
        })
        # Top-down smoothing blocks for the levels the chain passes through
        # (every level from 16 down to the finest detected stride).
        self.td = nn.ModuleDict({
            str(s): C3(2 * c, c, n, shortcut=False)
            for s in (16, 8, 4) if s >= self.min_stride
        })
        # Bottom-up: one strided conv + fusion per detected stride coarser
        # than the finest one; keys are the OUTPUT stride of the conv.
        self.down = nn.ModuleDict({
            str(s): ConvBnSiLU(c, c, 3, 2) for s in self.strides if s > self.min_stride
        })
        self.n = nn.ModuleDict({
            str(s): C3(2 * c, c, n, shortcut=False) for s in self.strides if s > self.min_stride
        })

    def forward(
        self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        c2, c3, c4, c5 = feats
        backbone = {4: c2, 8: c3, 16: c4, 32: c5}

        # Top-down: coarsest (32) first, then down to the finest detected stride.
        top_down: dict[int, torch.Tensor] = {32: self.lat["32"](c5)}
        for s in (16, 8, 4):
            if s < self.min_stride:
                break
            lat = self.lat[str(s)](backbone[s])
            up = F.interpolate(top_down[s * 2], scale_factor=2, mode="nearest")
            if lat.shape[-2:] != up.shape[-2:]:     # odd input dims: match exactly
                up = F.interpolate(up, size=lat.shape[-2:], mode="nearest")
            top_down[s] = self.td[str(s)](torch.cat((lat, up), dim=1))

        # Bottom-up: from the finest detected level back to the coarsest.
        out: dict[int, torch.Tensor] = {}
        finest = min(s for s in top_down if s in self.strides)
        out[finest] = top_down[finest]
        prev = top_down[finest]
        for s in (8, 16, 32):
            if s not in self.strides or s == finest:
                continue
            fused = self.n[str(s)](torch.cat((self.down[str(s)](prev), top_down[s]), dim=1))
            out[s] = fused
            prev = fused
        return out
        return out


class DecoupledHead(nn.Module):
    """Anchor-free decoupled detection head for one pyramid level.

    Three parallel 3x3 conv stems branch into classification logits, objectness
    logit and raw box offsets ``(tx, ty, tw, th)``. The final 1x1 convs get
    YOLO-style init (weights std=0.01 + bias priors) so training starts from a
    sane prior — objectness ≈ -4 → p≈0.018, box offsets ≈ 0 — instead of the
    ±O(1) logits that kaiming weights on BN outputs would produce.
    """

    def __init__(self, c_in: int, num_classes: int) -> None:
        super().__init__()
        c_mid = c_in
        self.cls = nn.Sequential(ConvBnSiLU(c_in, c_mid, 3), nn.Conv2d(c_mid, num_classes, 1))
        self.obj = nn.Sequential(ConvBnSiLU(c_in, c_mid, 3), nn.Conv2d(c_mid, 1, 1))
        self.box = nn.Sequential(ConvBnSiLU(c_in, c_mid, 3), nn.Conv2d(c_mid, 4, 1))
        self.num_classes = num_classes
        self._init_final_layers()

    def _init_final_layers(self) -> None:
        """Small-weight + bias-prior init of the final 1x1 convs.

        Must run *after* any global kaiming pass (call again if re-initializing
        the whole network) — the priors only mean anything if the final layers
        start near-zero-output.
        """
        for branch in (self.cls, self.obj, self.box):
            nn.init.normal_(branch[-1].weight, std=0.01)
        nn.init.constant_(self.obj[-1].bias, -4.0)
        prior_prob = 0.01
        nn.init.constant_(self.cls[-1].bias, -math.log((1 - prior_prob) / prior_prob))
        nn.init.zeros_(self.box[-1].bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"cls": self.cls(x), "obj": self.obj(x), "box": self.box(x)}


class TinyDetector(nn.Module):
    """End-to-end tiny-object detector with a stride-4 P2 detection level.

    Args:
        cfg: :class:`DetectorConfig`.

    Input: ``(B, 3, H, W)`` image batch, H/W divisible by 32.
    Output: list (len == len(cfg.strides)) of per-level dicts with
        ``cls`` ``(B, nc, H_l, W_l)`` logits,
        ``obj`` ``(B, 1, H_l, W_l)`` logits,
        ``box`` ``(B, 4, H_l, W_l)`` decoded as ``(cx, cy, w, h)`` in
        input-image pixel coordinates (gradients flow through the decode).
    """

    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DetectorConfig()
        self.backbone = Backbone(self.cfg)
        self.neck = P2Neck(self.cfg)
        neck_feats = {4: self.cfg.neck_channels, 8: self.cfg.neck_channels,
                      16: self.cfg.neck_channels, 32: self.cfg.neck_channels}
        self.heads = nn.ModuleList(
            DecoupledHead(neck_feats[s], self.cfg.num_classes) for s in self.cfg.strides
        )
        self.strides = tuple(self.cfg.strides)
        self._init_weights()
        # Head priors must win over the generic pass above.
        for head in self.heads:
            head._init_final_layers()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch's calculate_gain has no 'silu'; 'relu' gain (~sqrt2)
                # is the standard approximation used by YOLO for SiLU nets.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        input_hw = x.shape[-2:]
        neck_feats = self.neck(self.backbone(x))    # {stride: feature}
        outputs = []
        for stride, head in zip(self.strides, self.heads):
            out = head(neck_feats[stride])
            out["box"] = self._decode_boxes(out["box"], stride, input_hw)
            outputs.append(out)
        return outputs

    @staticmethod
    def _decode_boxes(raw: torch.Tensor, stride: int, input_hw: tuple[int, int]) -> torch.Tensor:
        """Decode raw head offsets to cxcywh boxes in input pixel coordinates.

        ``cx = (gx + 2*sigmoid(tx) - 0.5) * s`` (YOLOv5-style stabilized offset)
        ``w  = s * exp(clamp(tw, max=4))``   (log-size, clamped for stability)
        """
        h_in, w_in = input_hw
        h, w = raw.shape[-2:]
        ys = torch.arange(h, device=raw.device, dtype=raw.dtype).view(1, 1, h, 1)
        xs = torch.arange(w, device=raw.device, dtype=raw.dtype).view(1, 1, 1, w)
        tx, ty, tw, th = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3], raw[:, 3:4]
        cx = (xs + 2.0 * torch.sigmoid(tx) - 0.5) * stride
        cy = (ys + 2.0 * torch.sigmoid(ty) - 0.5) * stride
        bw = stride * torch.exp(tw.clamp(max=4.0))
        bh = stride * torch.exp(th.clamp(max=4.0))
        boxes = torch.cat((cx.clamp(0, w_in), cy.clamp(0, h_in), bw, bh), dim=1)  # (B, 4, H, W)
        return boxes

    @torch.no_grad()
    def decode(self, outputs: list[dict[str, torch.Tensor]], conf_threshold: float = 0.25) -> torch.Tensor:
        """Convert per-level outputs to flat detections for tracking/eval.

        Args:
            outputs: forward() outputs.
            conf_threshold: keep cells with ``obj * max_cls_prob`` above this.
        Returns:
            ``(B, N, 6)`` tensor of ``(x1, y1, x2, y2, score, class_id)``;
            ``N`` varies per image (padded with score = -1 rows to N_max).
        """
        all_boxes: list[list[torch.Tensor]] = []
        for _ in range(outputs[0]["cls"].shape[0]):
            all_boxes.append([])
        for stride, out in zip(self.strides, outputs):
            cls_prob = out["cls"].sigmoid()                       # (B, nc, H, W)
            obj_prob = out["obj"].sigmoid()                       # (B, 1, H, W)
            score = obj_prob * cls_prob.max(dim=1, keepdim=True).values
            best_cls = cls_prob.argmax(dim=1)                     # (B, H, W)
            boxes = out["box"]                                    # (B, 4, H, W) cxcywh
            mask = score[:, 0] > conf_threshold                   # (B, H, W)
            for i in range(boxes.shape[0]):
                if mask[i].any():
                    b = boxes[i].permute(1, 2, 0)[mask[i]]        # (K, 4)
                    x1y1 = b[:, :2] - b[:, 2:] * 0.5
                    x2y2 = b[:, :2] + b[:, 2:] * 0.5
                    s = score[i, 0][mask[i]]
                    c = best_cls[i][mask[i]].float()
                    all_boxes[i].append(torch.cat((x1y1, x2y2, s.unsqueeze(1), c.unsqueeze(1)), dim=1))
        max_n = max((sum(len(x) for x in per_img) for per_img in all_boxes), default=0)
        max_n = max(max_n, 1)
        out = torch.full((len(all_boxes), max_n, 6), -1.0, device=outputs[0]["cls"].device)
        for i, per_img in enumerate(all_boxes):
            if per_img:
                merged = torch.cat(per_img, dim=0)           # concat across levels
                out[i, : merged.shape[0]] = merged
        return out
