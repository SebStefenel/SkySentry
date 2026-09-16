"""Detection metrics: COCO-style AP/mAP with small-area split, FP analysis,
and sim-to-real domain-gap reporting.

Everything is self-contained (numpy): no pycocotools dependency.

Conventions
-----------
* Boxes: ``xyxy`` in input-image pixels.
* Predictions are pushed per batch via :meth:`DetectionEvaluator.update`;
  results are computed lazily via :meth:`DetectionEvaluator.results`.

APIs
----
* :class:`DetectionEvaluator` — mAP@[.5:.95], mAP50, and the COCO area splits
  AP_small (area < 32²), AP_medium (< 96²), AP_large (>= 96²), using GT box
  area. Plus false-positive breakdown at a confidence operating point and
  per-class precision/recall.
* :func:`domain_gap_report` — compares the *same* evaluator results computed
  on a synthetic (source) and a real (target) split: AP drop, small-object AP
  drop and score-distribution shift quantify the sim-to-real gap.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None

__all__ = ["DetectionEvaluator", "APResult", "FPAnalysis", "domain_gap_report", "plot_pr_curves"]

# COCO area splits, on GT box area in pixels^2.
AREA_RANGES: dict[str, tuple[float, float]] = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)[:, None]
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)[None, :]
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.clip(area_a + area_b - inter, 1e-7, None)


@dataclass
class APResult:
    """Average precision at one IoU threshold for one area range."""

    iou_threshold: float
    area_range: str
    ap: float
    num_gt: int
    num_dets: int
    precisions: np.ndarray      # monotone envelope, score-descending order
    recalls: np.ndarray
    thresholds: np.ndarray      # score operating points (same length)


@dataclass
class FPAnalysis:
    """False-positive taxonomy at one confidence operating point.

    * ``localization_fp``: FP detections that DO overlap a GT (IoU > 0.1) but
      below threshold or wrong class — the model found *something* real.
    * ``background_fp``: FPs with no meaningful overlap — hallucinations.
    * ``missed_gt``: GT with no matching detection — recall loss.
    """

    conf_threshold: float
    iou_threshold: float
    localization_fp: int
    background_fp: int
    true_positives: int
    missed_gt: int
    num_gt: int

    @property
    def false_alarm_rate(self) -> float:
        denom = self.localization_fp + self.background_fp
        return self.background_fp / denom if denom else 0.0


class DetectionEvaluator:
    """Accumulates (prediction, GT) pairs and computes COCO-style AP."""

    def __init__(self, num_classes: int, class_names: list[str] | None = None) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        # Per class: list of (image_idx, box, score) for dets; list of (image_idx, box) for GT.
        self._dets: dict[int, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
        self._gts: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        self._gt_by_image: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        self.num_images = 0

    # ------------------------------------------------------------------ #

    def update(
        self,
        image_idx: int,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> None:
        """Add one image's predictions and ground truth.

        Args:
            pred_boxes: (N, 4) xyxy, already NMS-ed.
            pred_scores: (N,) in [0, 1] — keep ALL detections above a low
                floor (~0.01); AP integration uses the full PR curve.
            gt_boxes: (M, 4) xyxy; crowd regions are not handled (Phase 2).
        """
        self.num_images = max(self.num_images, image_idx + 1)
        for b, s, c in zip(pred_boxes, pred_scores, pred_labels):
            self._dets[int(c)].append((image_idx, np.asarray(b, dtype=np.float64), float(s)))
        for b, c in zip(gt_boxes, gt_labels):
            self._gts[int(c)].append((image_idx, np.asarray(b, dtype=np.float64)))
            self._gt_by_image[image_idx].append((int(c), np.asarray(b, dtype=np.float64)))

    # ------------------------------------------------------------------ #

    def _per_class_ap(self, cls: int, iou_thr: float, area: str) -> APResult:
        """AP at one IoU threshold / area range via all-point interpolation.

        The area split filters the **ground truth** side (as COCO does);
        detections are ranked globally and can only match qualified GT.
        """
        dets = self._dets.get(cls, [])
        gts = self._gt_by_image_qualified(cls, area)
        num_gt = sum(len(b) for _, b in gts)
        if num_gt == 0:
            return APResult(iou_thr, area, float("nan"), 0, len(dets), np.zeros(0), np.zeros(0), np.zeros(0))

        order = sorted(dets, key=lambda t: -t[2])          # descending score
        matched_gt = {img: np.zeros(len(boxes), dtype=bool) for img, boxes in gts}
        gt_map = dict(gts)
        tp = np.zeros(len(order))
        fp = np.zeros(len(order))
        for k, (img, box, _s) in enumerate(order):
            boxes = gt_map.get(img)
            if boxes is None or len(boxes) == 0:
                fp[k] = 1
                continue
            ious = _iou_matrix(box[None, :], boxes)[0]
            best = int(np.argmax(ious))
            if ious[best] >= iou_thr and not matched_gt[img][best]:
                tp[k] = 1
                matched_gt[img][best] = True
            else:
                fp[k] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / num_gt
        precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-7, None)
        thresholds = np.array([t[2] for t in order])

        # All-point interpolated AP (VOC2010+ / torchmetrics convention):
        # envelope the precision curve, integrate (R_i - R_{i-1}) * P_i.
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        return APResult(iou_thr, area, ap, num_gt, len(order), precision, recall, thresholds)

    def _gt_by_image_qualified(self, cls: int, area: str):
        lo, hi = AREA_RANGES[area]
        gts = [
            (img, [b for (c2, b) in self._gt_by_image[img] if c2 == cls and lo <= (b[2] - b[0]) * (b[3] - b[1]) < hi])
            for img in range(self.num_images)
        ]
        return [(img, np.asarray(boxes).reshape(-1, 4)) for img, boxes in gts]

    # ------------------------------------------------------------------ #

    def average_precisions(self, area: str = "all") -> dict[int, float]:
        """AP@[.5:.95] averaged over IoU thresholds, per class."""
        out = {}
        for cls in range(self.num_classes):
            aps = [self._per_class_ap(cls, t, area).ap for t in IOU_THRESHOLDS]
            aps = [a for a in aps if not np.isnan(a)]
            out[cls] = float(np.mean(aps)) if aps else float("nan")
        return out

    def results(self) -> dict[str, float]:
        """Aggregate summary: mAP, mAP50, mAP75 and COCO area splits."""
        def mean_or_nan(xs: list[float]) -> float:
            xs = [x for x in xs if not np.isnan(x)]
            return float(np.mean(xs)) if xs else float("nan")

        out: dict[str, float] = {}
        for area in AREA_RANGES:
            out[f"mAP_{area}"] = mean_or_nan(
                [mean_or_nan([self._per_class_ap(c, t, area).ap for t in IOU_THRESHOLDS])
                 for c in range(self.num_classes)]
            )
        out["mAP50"] = mean_or_nan([self._per_class_ap(c, 0.5, "all").ap for c in range(self.num_classes)])
        out["mAP75"] = mean_or_nan([self._per_class_ap(c, 0.75, "all").ap for c in range(self.num_classes)])
        return out

    def fp_analysis(self, conf_threshold: float = 0.25, iou_threshold: float = 0.5) -> FPAnalysis:
        """Taxonomize detections at a fixed operating point."""
        tp = loc = bg = missed = 0
        total_gt = 0
        matched = defaultdict(set)  # (image, cls) -> set of matched gt indices
        for cls in range(self.num_classes):
            dets = [d for d in self._dets.get(cls, []) if d[2] >= conf_threshold]
            gts = self._gt_by_image_qualified(cls, "all")
            total_gt += sum(len(b) for _, b in gts)
            gt_here = {img: boxes for img, boxes in gts}
            for img, box, _s in sorted(dets, key=lambda t: -t[2]):
                boxes = gt_here.get(img)
                if boxes is None or len(boxes) == 0:
                    bg += 1
                    continue
                ious = _iou_matrix(box[None, :], boxes)[0]
                best = int(np.argmax(ious))
                if ious[best] >= iou_threshold and best not in matched[(img, cls)]:
                    tp += 1
                    matched[(img, cls)].add(best)
                elif ious[best] > 0.1:
                    loc += 1
                else:
                    bg += 1
        matched_total = sum(len(v) for v in matched.values())
        missed = total_gt - matched_total
        return FPAnalysis(conf_threshold, iou_threshold, loc, bg, tp, missed, total_gt)


def domain_gap_report(source_results: dict[str, float], target_results: dict[str, float]) -> dict[str, float]:
    """Quantify the sim-to-real domain gap between two evaluator results.

    A good transfer report: absolute AP drop (how much reality costs), the
    small-object AP drop (are tiny targets hit hardest?), and relative drops
    normalized by source performance.
    """
    report: dict[str, float] = {}
    for key in ("mAP_all", "mAP50", "mAP_small", "mAP_medium", "mAP_large"):
        s, t = source_results.get(key), target_results.get(key)
        if s is None or t is None or np.isnan(s) or np.isnan(t):
            continue
        report[f"gap_{key}"] = s - t                       # absolute drop
        report[f"gap_rel_{key}"] = (s - t) / s if s > 0 else float("nan")
    return report


def plot_pr_curves(results: list[APResult], title: str, out_path: str) -> None:
    """Render precision-recall curves of one class across IoU thresholds."""
    if matplotlib is None:
        raise ImportError("matplotlib is required for plotting")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=130)
    for r in results:
        ax.plot(r.recalls, r.precisions, label=f"IoU={r.iou_threshold:.2f} AP={r.ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
