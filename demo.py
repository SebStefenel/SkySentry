"""Side-by-side demo video: raw detections vs tracked output.

Left panel: raw per-frame detections (boxes + confidence, no memory).
Right panel: tracker output — boxes, confidence, track IDs, trajectory trails.
Overlay: per-frame latency and FPS of the full detect→track step.

Modes:
* ``--gt-sim`` (works with no trained model): a synthetic clip's ground truth
  becomes noisy detections; isolates tracking from detection quality.
* ``--source clip.mp4 --weights checkpoints/exp_c_p2_nwd/best.pt``: real
  detector on a video file.

Outputs under ``results/demo/``: ``<name>.mp4``, ``<name>.gif`` (downscaled,
README-friendly) and ``<name>_report.json`` (latency/FPS — measured).

Usage::

    python demo.py --gt-sim
    python demo.py --source clip.mp4 --weights checkpoints/exp_c_p2_nwd/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from tracking.kalman_tracker import SORTTracker, SORTTrackerConfig
from tracking.track_filter import ErraticMotionFilter
from tracking.run_video import COLORS

OUT_DIR = Path("results/demo")


def draw_detections(img: np.ndarray, det_rows: np.ndarray) -> np.ndarray:
    """Left panel: this frame's raw detections only."""
    for row in det_rows:
        x1, y1, x2, y2, score = (int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
        cv2.putText(img, f"{score:.2f}", (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (160, 160, 160), 1)
    cv2.putText(img, "raw detections", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return img


def draw_tracks(img: np.ndarray, tracks: list, min_hits: int) -> np.ndarray:
    """Right panel: tracked boxes + confidence + ID + trajectory trail."""
    for t in tracks:
        if t.time_since_update > 1:
            continue
        x1, y1, x2, y2 = (int(v) for v in t.box_xyxy)
        color = COLORS[t.track_id % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        if len(t.trajectory) >= 2:
            pts = [(int(cx), int(cy)) for _f, cx, cy in t.trajectory[-40:]]
            cv2.polylines(img, [np.asarray(pts)], False, color, 1)
        cv2.putText(img, f"id{t.track_id} {t.score:.2f}", (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    cv2.putText(img, f"tracked (min_hits={min_hits})", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def compose(left: np.ndarray, right: np.ndarray, latency_ms: float, fps: float) -> np.ndarray:
    """Side-by-side canvas with the latency/FPS overlay."""
    canvas = np.concatenate([left, right], axis=1)
    cv2.putText(canvas, f"{latency_ms:5.1f} ms/frame  {fps:5.1f} FPS",
                (8, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


def save_gif(frames: list[np.ndarray], path: Path, width: int = 480, fps: int = 10) -> None:
    """Assemble a downscaled GIF with Pillow (no imageio dependency)."""
    from PIL import Image

    imgs = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h = int(rgb.shape[0] * width / rgb.shape[1])
        imgs.append(Image.fromarray(cv2.resize(rgb, (width, h))))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def sim_frame_detections(labels: list, size: int, rng: np.random.Generator) -> np.ndarray:
    """GT clip labels → noisy detector rows (the gt-sim stand-in for a model)."""
    rows = []
    for _tid, cls, cx, cy, w, h in labels:
        jx, jy = rng.normal(0, 1.0, 2)
        x1, y1 = (cx - w / 2) * size + jx, (cy - h / 2) * size + jy
        rows.append([x1, y1, x1 + w * size, y1 + h * size, rng.uniform(0.7, 0.99), float(cls)])
    return np.asarray(rows).reshape(-1, 6)


def demo_gt_sim(out: Path, frames_n: int = 40, size: int = 320, seed: int = 7) -> dict:
    from data.synthetic_composer import generate_clip

    frames = generate_clip(num_frames=frames_n, size=size, seed=seed, num_objects=4)
    tracker = SORTTracker(SORTTrackerConfig(assoc_metric="nwd", assoc_threshold=0.2, max_age=8, min_hits=1))
    motion = ErraticMotionFilter()
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out / "gt_sim.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (size * 2, size))
    gif_frames, latencies = [], []

    for f_idx, (img, labels) in enumerate(frames):
        det_rows = sim_frame_detections(labels, size, rng)
        t0 = time.perf_counter()
        tracks = tracker.update(det_rows, frame_idx=f_idx)
        latency = time.perf_counter() - t0
        latencies.append(latency)
        for t in tracks:
            if t.time_since_update == 0:
                bx = np.asarray(t.box_xyxy)
                motion.update(t.track_id, (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
        fps = 1.0 / max(float(np.mean(latencies[-10:])), 1e-6)
        frame = compose(draw_detections(img.copy(), det_rows), draw_tracks(img.copy(), tracks, 1),
                        latency * 1000, fps)
        writer.write(frame)
        gif_frames.append(frame)
    writer.release()
    save_gif(gif_frames[::2], out / "gt_sim.gif")

    report = {"mode": "gt-sim", "frames": frames_n, "size_px": size,
              "avg_latency_ms": round(float(np.mean(latencies)) * 1000, 2),
              "fps": round(1.0 / float(np.mean(latencies)), 1),
              "motion_filter": motion.summary()}
    (out / "gt_sim_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def demo_video(source: str, weights: str, out: Path, img_size: int = 640,
               conf: float = 0.25, nms_iou: float = 0.45) -> dict:
    import torch
    from data.yolo_dataset import letterbox
    from models.detector_p2 import DetectorConfig, TinyDetector

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    exp_name = ckpt.get("exp", "exp_c_p2_nwd")
    strides = {"exp_a_baseline": (8, 16, 32)}.get(exp_name, (4, 8, 16))
    model = TinyDetector(DetectorConfig(num_classes=3, strides=strides))
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 15
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = SORTTracker(SORTTrackerConfig(assoc_metric="nwd", assoc_threshold=0.2, max_age=12, min_hits=2))
    motion = ErraticMotionFilter()
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(source).stem
    writer = cv2.VideoWriter(str(out / f"{stem}_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width * 2, height))
    latencies, gif_frames = [], []
    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lb, _ = letterbox(frame, img_size, np.zeros((0, 4)))
        tensor = torch.from_numpy(lb.transpose(2, 0, 1).copy()).float() / 255.0
        t0 = time.perf_counter()
        dets = postprocess_frames(model, tensor[None], conf, nms_iou)
        det_rows = undo_letterbox(dets, height, width, img_size)
        tracks = tracker.update(det_rows, frame_idx=f_idx)
        latency = time.perf_counter() - t0
        latencies.append(latency)
        for t in tracks:
            if t.time_since_update == 0:
                bx = np.asarray(t.box_xyxy)
                motion.update(t.track_id, (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
        fps = 1.0 / max(float(np.mean(latencies[-10:])), 1e-6)
        vis = frame.copy()
        frame_out = compose(draw_detections(vis.copy(), det_rows[:, [0, 1, 2, 3, 4]] if len(det_rows) else np.zeros((0, 6))),
                            draw_tracks(vis, tracks, 2), latency * 1000, fps)
        writer.write(frame_out)
        gif_frames.append(frame_out)
        f_idx += 1
    cap.release()
    writer.release()
    if gif_frames:
        save_gif(gif_frames[::2], out / f"{stem}_demo.gif", width=640)

    report = {"mode": "video", "source": source, "weights": weights, "frames": f_idx,
              "avg_latency_ms": round(float(np.mean(latencies)) * 1000, 2),
              "fps": round(1.0 / float(np.mean(latencies)), 1),
              "motion_filter": motion.summary()}
    (out / f"{stem}_demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def postprocess_frames(model, tensor, conf, nms_iou):
    from infer import postprocess
    return postprocess(model, tensor, conf_threshold=conf, nms_iou=nms_iou)[0]


def undo_letterbox(dets, h_orig: int, w_orig: int, img_size: int) -> np.ndarray:
    """Model-space detections (x1,y1,x2,y2,score,cls) → original-image coords."""
    scale = min(img_size / h_orig, img_size / w_orig)
    pad_top = (img_size - int(round(h_orig * scale))) // 2
    pad_left = (img_size - int(round(w_orig * scale))) // 2
    out = dets["boxes"].numpy().copy()
    if len(out):
        out[:, [0, 2]] = (out[:, [0, 2]] - pad_left) / scale
        out[:, [1, 3]] = (out[:, [1, 3]] - pad_top) / scale
    scores = dets["scores"].numpy()[:, None]
    labels = dets["labels"].numpy()[:, None].astype(float)
    return np.concatenate([out, scores, labels], axis=1) if len(out) else np.zeros((0, 6))


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side detection vs tracking demo")
    parser.add_argument("--gt-sim", action="store_true", help="synthetic-clip demo (no model needed)")
    parser.add_argument("--source", default=None, help="video path")
    parser.add_argument("--weights", default=None, help="checkpoint path")
    parser.add_argument("--out", default="results/demo")
    args = parser.parse_args()
    out = Path(args.out)
    if args.gt_sim or not (args.source and args.weights):
        if not args.gt_sim:
            print("[info] --source/--weights not both given; falling back to gt-sim demo")
        demo_gt_sim(out)
    else:
        demo_video(args.source, args.weights, out)


if __name__ == "__main__":
    main()
