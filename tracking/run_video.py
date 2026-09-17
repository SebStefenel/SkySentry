"""Detector → tracker video pipeline (Phase 4).

Two input modes:

* ``--weights checkpoints/exp_c_p2_nwd/best.pt`` — real detector on a video
  file (letterboxed inference at the experiment's img_size → NMS → tracker).
* default (no weights) — **GT-sim mode**: a synthetic clip's ground truth
  becomes noisy detections, isolating tracker behavior from detector quality
  (useful while the full training runs are pending).

Outputs (per run):
    results/tracking/video/<name>.mp4      boxes + track IDs + trajectory trails + FPS
    results/tracking/video/<name>_mot.csv  frame, id, x1, y1, x2, y2 (tracker output)
    results/tracking/video/<name>_filter.json  motion-filter kept/rejected counts

Note: lives in ``tracking/`` (not a new top-level ``tracker/`` package) —
same concern, no parallel duplicate package.

Usage::

    python -m tracking.run_video --mode gt-sim --frames 40 --assoc nwd
    python -m tracking.run_video --mode video --source clip.mp4 --weights best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from tracking.kalman_tracker import SORTTracker, SORTTrackerConfig
from tracking.track_filter import ErraticMotionFilter

__all__ = ["run_gt_sim", "run_video_file"]

COLORS = [(66, 186, 78), (52, 152, 219), (155, 89, 182), (241, 196, 15), (230, 126, 34)]


def _draw(frame: np.ndarray, tracks: list, fps: float) -> np.ndarray:
    """Boxes + IDs + trajectory trails + FPS overlay."""
    for t in tracks:
        if t.state.name != "CONFIRMED" and t.time_since_update > 1:
            continue
        x1, y1, x2, y2 = (int(v) for v in t.box_xyxy)
        color = COLORS[t.track_id % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        if len(t.trajectory) >= 2:
            pts = [(int(cx), int(cy)) for _f, cx, cy in t.trajectory[-30:]]
            cv2.polylines(frame, [np.asarray(pts)], False, color, 1)
        cv2.putText(frame, f"id{t.track_id}", (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.putText(frame, f"{fps:5.1f} FPS", (8, frame.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def run_gt_sim(out_dir: Path, frames_n: int = 40, size: int = 320, assoc: str = "nwd",
               seed: int = 7, filter_cfg: dict | None = None) -> dict:
    """GT-sim mode: synthetic clip GT (+1px jitter) → tracker → annotated video."""
    from data.synthetic_composer import generate_clip

    frames = generate_clip(num_frames=frames_n, size=size, seed=seed, num_objects=4)
    tracker = SORTTracker(SORTTrackerConfig(assoc_metric=assoc, assoc_threshold=0.2 if assoc == "nwd" else 0.1,
                                            max_age=8, min_hits=1))
    motion = ErraticMotionFilter(**(filter_cfg or {}))
    rng = np.random.default_rng(seed)
    out_dir = out_dir / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"gt_sim_{assoc}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (size, size))

    mot_rows, latencies = [], []
    for f_idx, (img, labels) in enumerate(frames):
        rows = []
        for tid, cls, cx, cy, w, h in labels:
            jx, jy = rng.normal(0, 1.0, 2)
            x1, y1 = (cx - w / 2) * size + jx, (cy - h / 2) * size + jy
            rows.append([x1, y1, x1 + w * size, y1 + h * size, rng.uniform(0.7, 0.99), float(cls)])
        t0 = time.perf_counter()
        tracks = tracker.update(np.asarray(rows).reshape(-1, 6), frame_idx=f_idx)
        latencies.append(time.perf_counter() - t0)
        for t in tracks:
            if t.time_since_update == 0:
                bx = np.asarray(t.box_xyxy)
                motion.update(t.track_id, (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
            mot_rows.append([f_idx, t.track_id, *t.box_xyxy])
        writer.write(_draw(img, tracks, 15.0 / max(np.mean(latencies[-10:]), 1e-6)))
    writer.release()

    with open(out_dir / f"gt_sim_{assoc}_mot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "id", "x1", "y1", "x2", "y2"])
        w.writerows(mot_rows)
    report = {"mode": "gt-sim", "assoc": assoc, "frames": frames_n,
              "avg_latency_ms": round(float(np.mean(latencies)) * 1000, 2),
              "fps": round(1.0 / float(np.mean(latencies)), 1),
              "motion_filter": motion.summary()}
    (out_dir / f"gt_sim_{assoc}_filter.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run_video_file(source: str, weights: str, out_dir: Path, img_size: int = 640,
                   conf: float = 0.25, nms_iou: float = 0.45, assoc: str = "nwd") -> dict:
    """Real-detector mode: video file → letterboxed inference → tracker → video."""
    import torch
    from data.yolo_dataset import letterbox
    from infer import postprocess
    from models.detector_p2 import DetectorConfig, TinyDetector

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    # The experiment name rides in the checkpoint; recover its strides/classes.
    exp_name = ckpt.get("exp", "exp_c_p2_nwd")
    stride_map = {"exp_a_baseline": (8, 16, 32), "exp_b_p2": (4, 8, 16),
                  "exp_c_p2_nwd": (4, 8, 16), "exp_d_p2_nwd_aug": (4, 8, 16)}
    model = TinyDetector(DetectorConfig(num_classes=3, strides=stride_map.get(exp_name, (4, 8, 16))))
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = SORTTracker(SORTTrackerConfig(assoc_metric=assoc, assoc_threshold=0.2 if assoc == "nwd" else 0.1,
                                            max_age=12, min_hits=2))
    motion = ErraticMotionFilter()
    out_dir = out_dir / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source).stem
    writer = cv2.VideoWriter(str(out_dir / f"{stem}_tracked.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))
    mot_rows, latencies = [], []
    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lb, _boxes = letterbox(frame, img_size, np.zeros((0, 4)))
        tensor = torch.from_numpy(lb.transpose(2, 0, 1).copy()).float() / 255.0
        t0 = time.perf_counter()
        dets = postprocess(model, tensor[None], conf_threshold=conf, nms_iou=nms_iou)[0]
        scale = min(img_size / height, img_size / width)
        pad_top = (img_size - int(round(height * scale))) // 2
        pad_left = (img_size - int(round(width * scale))) // 2
        det_arr = dets["boxes"].numpy().copy()
        if len(det_arr):   # undo letterbox for the tracker (input-image coords)
            det_arr[:, [0, 2]] = (det_arr[:, [0, 2]] - pad_left) / scale
            det_arr[:, [1, 3]] = (det_arr[:, [1, 3]] - pad_top) / scale
        full = np.concatenate([det_arr, dets["scores"].numpy()[:, None], dets["labels"].numpy()[:, None].astype(float)], axis=1) \
            if len(det_arr) else np.zeros((0, 6))
        tracks = tracker.update(full, frame_idx=f_idx)
        latencies.append(time.perf_counter() - t0)
        for t in tracks:
            if t.time_since_update == 0:
                bx = np.asarray(t.box_xyxy)
                motion.update(t.track_id, (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
            mot_rows.append([f_idx, t.track_id, *t.box_xyxy])
        writer.write(_draw(frame, tracks, 1.0 / max(np.mean(latencies[-10:]), 1e-6)))
        f_idx += 1
    cap.release()
    writer.release()

    with open(out_dir / f"{stem}_mot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "id", "x1", "y1", "x2", "y2"])
        w.writerows(mot_rows)
    report = {"mode": "video", "source": source, "weights": weights, "frames": f_idx,
              "avg_latency_ms": round(float(np.mean(latencies)) * 1000, 2),
              "fps": round(1.0 / float(np.mean(latencies)), 1),
              "motion_filter": motion.summary()}
    (out_dir / f"{stem}_filter.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector → tracker video pipeline")
    parser.add_argument("--mode", choices=["gt-sim", "video"], default="gt-sim")
    parser.add_argument("--source", default=None, help="video path (video mode)")
    parser.add_argument("--weights", default=None, help="checkpoint path (video mode)")
    parser.add_argument("--assoc", choices=["nwd", "iou"], default="nwd")
    parser.add_argument("--frames", type=int, default=40, help="gt-sim clip length")
    parser.add_argument("--out", default="results/tracking")
    args = parser.parse_args()
    out_dir = Path(args.out)
    if args.mode == "gt-sim":
        run_gt_sim(out_dir, frames_n=args.frames, assoc=args.assoc)
    else:
        if not (args.source and args.weights):
            raise SystemExit("video mode needs --source and --weights")
        run_video_file(args.source, args.weights, out_dir)


if __name__ == "__main__":
    main()
