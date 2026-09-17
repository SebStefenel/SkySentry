"""Occlusion benchmarking for the tracker (Phase 4).

Protocol
--------
Synthetic clips carry ground-truth track IDs (:func:`data.synthetic_composer.generate_clip`).
A *simulated detector* renders the GT boxes with small jitter, then detections
are artificially dropped for a contiguous window of ``D`` frames (D ∈ {0, 5,
10, 15}) — the tracker must coast through on its motion model. Identity
quality is scored with MOT metrics (MOTA, IDF1, ID switches) via ``motmetrics``,
plus a simple ID-preservation rate (a GT object counts as preserved if ≥90 %
of its visible frames carry one dominant hypothesis ID).

The whole sweep runs for **both** association metrics (NWD and IoU) on the
same clips, giving the Phase-4 comparison. Every number lands in
``results/tracking/occlusion_benchmark.csv`` — measured, never hand-written.

Usage::

    python -m tracking.benchmark_occlusion --clips 8 --out results/tracking
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import motmetrics as mm
except ImportError:  # pragma: no cover
    mm = None

from data.synthetic_composer import generate_clip

__all__ = ["score_clip", "run_benchmark"]


def score_clip(
    frames: list[tuple[np.ndarray, list]],
    drop_len: int,
    drop_start_frac: float,
    assoc_metric: str,
    tracker_cfg_kwargs: dict | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Run the tracker over one clip with an artificial dropout window.

    Returns MOTA / IDF1 / id_switches / idp / idr (motmetrics) and the simple
    dominant-ID preservation rate.
    """
    from tracking.kalman_tracker import SORTTracker, SORTTrackerConfig

    tracker_cfg_kwargs = tracker_cfg_kwargs or {}
    size = frames[0][0].shape[0]
    n_frames = len(frames)
    drop_start = int(n_frames * drop_start_frac)
    drop_set = set(range(drop_start, min(drop_start + drop_len, n_frames)))

    cfg = SORTTrackerConfig(assoc_metric=assoc_metric, assoc_threshold=0.2 if assoc_metric == "nwd" else 0.1,
                            max_age=max(drop_len + 2, 5), min_hits=1, **tracker_cfg_kwargs)
    tracker = SORTTracker(cfg)
    rng = np.random.default_rng(seed)

    # Collect per-frame (gt_ids, hyp_ids, iou_matrix) for motmetrics.
    frame_events: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    hyp_track_of_gt: dict[int, dict[int, int]] = {}   # gt_id -> {hyp_id: count}

    def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)))
        aa = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)[:, None]
        ab = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)[None, :]
        lt = np.maximum(a[:, None, :2], b[None, :, :2])
        rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
        wh = np.clip(rb - lt, 0, None)
        return np.clip(wh[..., 0] * wh[..., 1], 0, None) / np.clip(aa + ab - wh[..., 0] * wh[..., 1], 1e-7, None)

    for f_idx, (img, labels) in enumerate(frames):
        if f_idx in drop_set:
            dets = np.zeros((0, 6))
        else:
            rows = []
            for lbl in labels:
                tid, cls, cx, cy, w, h = lbl
                jx, jy = rng.normal(0, 1.0, 2)
                x1, y1 = (cx - w / 2) * size + jx, (cy - h / 2) * size + jy
                rows.append([x1, y1, x1 + w * size, y1 + h * size, rng.uniform(0.7, 0.99), float(cls)])
            dets = np.asarray(rows).reshape(-1, 6)

        tracks = tracker.update(dets, frame_idx=f_idx)

        gt_ids = np.array([lbl[0] for lbl in labels], dtype=int)
        gt_boxes = np.array([[(_cx - _w / 2) * size, (_cy - _h / 2) * size,
                              (_cx + _w / 2) * size, (_cy + _h / 2) * size] for _tid, _c, _cx, _cy, _w, _h in labels]).reshape(-1, 4)
        hyp_ids = np.array([t.track_id for t in tracks], dtype=int)
        hyp_boxes = np.array([t.box_xyxy for t in tracks]).reshape(-1, 4)
        dist = 1.0 - iou_xyxy(gt_boxes, hyp_boxes)   # motmetrics wants distances
        frame_events.append((gt_ids, hyp_ids, dist))

        # Dominant-ID bookkeeping for the preservation rate (only on visible frames).
        for gi, gt_id in enumerate(gt_ids):
            best_h, best_iou = -1, 0.0
            for hi, hyp_id in enumerate(hyp_ids):
                iou = 1.0 - float(dist[gi, hi])
                if iou > best_iou:
                    best_iou, best_h = iou, hyp_id
            if best_iou >= 0.5:
                hyp_track_of_gt.setdefault(gt_id, {}).setdefault(best_h, 0)
                hyp_track_of_gt[gt_id][best_h] += 1

    preserved = 0
    total_gt = len(hyp_track_of_gt)
    for _gt, hyp_counts in hyp_track_of_gt.items():
        total_frames = sum(hyp_counts.values())
        dominant = max(hyp_counts.values())
        if total_frames and dominant / total_frames >= 0.9:
            preserved += 1

    out: dict[str, float] = {"id_preservation": preserved / total_gt if total_gt else float("nan"),
                             "drop_len": float(drop_len), "assoc": assoc_metric}
    if mm is not None:
        acc = mm.MOTAccumulator(auto_id=True)
        for gt_ids, hyp_ids, dist in frame_events:
            acc.update(gt_ids, hyp_ids, dist)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches", "idp", "idr"])

        def _scalar(v) -> float:
            arr = np.asarray(getattr(v, "values", v), dtype=float)
            return float(arr.squeeze())

        out.update({"mota": _scalar(summary["mota"]), "idf1": _scalar(summary["idf1"]),
                    "id_switches": _scalar(summary["num_switches"]),
                    "idp": _scalar(summary["idp"]), "idr": _scalar(summary["idr"])})
    else:  # pragma: no cover
        out.update({"mota": float("nan"), "idf1": float("nan"), "id_switches": float("nan"),
                    "idp": float("nan"), "idr": float("nan")})
    return out


def benchmark_false_positives(clips: int, out_dir: Path, fps_per_clip: int = 2, seed: int = 7) -> "Path":
    """Track-level precision before vs after motion filtering.

    Injects erratic random-walk false tracks (2 per clip) alongside the true
    detections, runs the tracker, then scores: a confirmed track is a TRUE
    positive if its mean best-IoU against GT exceeds 0.3, else a false
    positive. ``ErraticMotionFilter`` verdicts give the 'after' precision.
    """
    from tracking.kalman_tracker import SORTTracker, SORTTrackerConfig
    from tracking.track_filter import ErraticMotionFilter

    rng = np.random.default_rng(seed)
    tp_before = fp_before = tp_after = fp_after = 0
    for c in range(clips):
        frames = generate_clip(num_frames=40, size=320, seed=seed + c, num_objects=4)
        size = frames[0][0].shape[0]
        tracker = SORTTracker(SORTTrackerConfig(assoc_metric="nwd", assoc_threshold=0.2, max_age=8, min_hits=2))
        motion = ErraticMotionFilter(window=7, min_points=4)
        # Random-walk false tracks: born anywhere, erratic steps — the classic
        # hallucination pattern (reflections, sensor speckle).
        fp_state = [{"pos": rng.uniform(30, 290, 2)} for _ in range(fps_per_clip)]
        for f_idx, (img, labels) in enumerate(frames):
            rows = []
            for _tid, cls, cx, cy, w, h in labels:
                jx, jy = rng.normal(0, 1.0, 2)
                rows.append([(cx - w / 2) * size + jx, (cy - h / 2) * size + jy,
                             (cx + w / 2) * size + jx, (cy + h / 2) * size + jy,
                             rng.uniform(0.7, 0.99), float(cls)])
            for fp in fp_state:
                fp["pos"] += rng.uniform(-15, 15, 2)
                fp["pos"] = np.clip(fp["pos"], 5, 315)
                rows.append([fp["pos"][0], fp["pos"][1], fp["pos"][0] + 10, fp["pos"][1] + 10,
                             rng.uniform(0.6, 0.9), 2.0])
            tracks = tracker.update(np.asarray(rows).reshape(-1, 6), frame_idx=f_idx)
            gt_boxes = np.array([[(_cx - _w / 2) * 320, (_cy - _h / 2) * 320,
                                  (_cx + _w / 2) * 320, (_cy + _h / 2) * 320]
                                 for _t, _c, _cx, _cy, _w, _h in labels]).reshape(-1, 4)
            for t in tracks:
                if t.time_since_update > 0 or t.hits < 3:
                    continue
                bx = np.asarray(t.box_xyxy)
                center = np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2])
                motion_ok = motion.update(t.track_id, center[0], center[1])
                # True track: mean best-IoU against GT (over window length) > 0.3.
                size_box = np.array([10.0, 10.0])
                corners = np.concatenate([center - size_box / 2, center + size_box / 2])
                lt = np.maximum(corners[:2], gt_boxes[:, :2])
                rb = np.minimum(corners[2:], gt_boxes[:, 2:])
                inter = np.prod(np.clip(rb - lt, 0, None), axis=1)
                area = np.prod(size_box)
                best_iou = float(np.max(inter / (area + np.prod(np.clip(gt_boxes[:, 2:] - gt_boxes[:, :2], 0, None), axis=1) - inter))) if len(gt_boxes) else 0.0
                is_tp = best_iou > 0.3
                tp_before, fp_before = (tp_before + is_tp, fp_before + (not is_tp))
                if motion_ok:
                    tp_after, fp_after = (tp_after + is_tp, fp_after + (not is_tp))
    precision_before = tp_before / max(tp_before + fp_before, 1)
    precision_after = tp_after / max(tp_after + fp_after, 1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fp_precision.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows([
            ["true_tracks", tp_before], ["false_tracks_before", fp_before],
            ["precision_before", round(precision_before, 4)],
            ["true_tracks_after", tp_after], ["false_tracks_after", fp_after],
            ["precision_after", round(precision_after, 4)],
        ])
    print(f"wrote {path}: precision {precision_before:.3f} -> {precision_after:.3f}")
    return path


def run_benchmark(clips: int, out_dir: str | Path, seed: int = 42) -> "Path":
    """Full sweep: {no-drop, 5, 10, 15} × {nwd, iou} over ``clips`` clips."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "occlusion_benchmark.csv"
    fields = ["clip", "assoc", "drop_len", "mota", "idf1", "id_switches", "idp", "idr", "id_preservation"]
    drop_lens = [0, 5, 10, 15]
    rows = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in range(clips):
            frames = generate_clip(num_frames=40, size=320, seed=seed + c, num_objects=4)
            for assoc in ("nwd", "iou"):
                for drop in drop_lens:
                    res = score_clip(frames, drop, drop_start_frac=0.3, assoc_metric=assoc, seed=seed + c)
                    res.update({"clip": c})
                    rows.append(res)
                    writer.writerow({k: res[k] for k in fields})
        # Aggregate mean per (assoc, drop_len) — the comparison table.
        summary_path = out / "occlusion_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as sf:
            sfields = ["assoc", "drop_len", "mota", "idf1", "id_switches", "id_preservation"]
            sw = csv.DictWriter(sf, fieldnames=sfields)
            sw.writeheader()
            for assoc in ("nwd", "iou"):
                for drop in drop_lens:
                    sel = [r for r in rows if r["assoc"] == assoc and r["drop_len"] == drop]
                    sw.writerow({
                        "assoc": assoc, "drop_len": drop,
                        "mota": round(float(np.nanmean([r["mota"] for r in sel])), 4),
                        "idf1": round(float(np.nanmean([r["idf1"] for r in sel])), 4),
                        "id_switches": round(float(np.nanmean([r["id_switches"] for r in sel])), 2),
                        "id_preservation": round(float(np.nanmean([r["id_preservation"] for r in sel])), 4),
                    })
        print(f"wrote {csv_path} and {summary_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Occlusion / association benchmark")
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/tracking")
    args = parser.parse_args()
    run_benchmark(args.clips, args.out, args.seed)
    benchmark_false_positives(args.clips, Path(args.out))


if __name__ == "__main__":
    main()
