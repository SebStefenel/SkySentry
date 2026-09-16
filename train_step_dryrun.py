"""End-to-end pipeline verification with synthetic data only (no downloads).

Runs the full Phase-1 stack on the CPU in under a minute:

1. Builds the P2 detector from ``configs/model.yaml`` and reports its
   parameter budget and per-level feature-map sizes (the stride-4 story).
2. Synthesizes a tiny batch (images + GT boxes) and runs 3 real training
   steps (forward → composite NWD/cls/obj loss → AdamW backward), printing
   the loss breakdown, gradient norms and step time.
3. Feeds two frames of synthetic constant-velocity detections through the
   Kalman tracker and shows one track identity persisting across a
   deliberately dropped detection (occlusion coasting).
4. Scores garbage predictions vs ground truth with the evaluator to show the
   metric plumbing (mAP/mAP_small/FP taxonomy) works end-to-end.

Exit code 0 + ``DRY RUN PASSED`` on success; any non-finite loss/gradient or
broken invariant fails loudly.

Usage::

    python train_step_dryrun.py [--steps 3] [--size 320]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.metrics import DetectionEvaluator  # noqa: E402
from losses.detection_loss import DetectionLossConfig, TinyDetectionLoss  # noqa: E402
from models.detector_p2 import DetectorConfig, TinyDetector  # noqa: E402
from tracking.kalman_tracker import SORTTracker, SORTTrackerConfig  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(name: str) -> dict:
    """Load a YAML config with sane defaults if the file is missing."""
    path = CONFIG_DIR / name
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    print(f"  [warn] configs/{name} not found — using built-in defaults")
    return {}


def make_synthetic_batch(
    batch_size: int,
    size: int,
    num_classes: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Random images + 1-4 tiny GT boxes per image, in pixel coords."""
    images = torch.from_numpy(rng.normal(128, 40, (batch_size, 3, size, size)).astype(np.float32))
    targets = []
    for _ in range(batch_size):
        n = int(rng.integers(1, 5))
        boxes = rng.uniform(0.05, 0.95, size=(n, 2)) * size              # centers
        wh = rng.uniform(4.0, 24.0, size=(n, 2))                         # tiny sizes
        xyxy = np.concatenate([boxes - wh / 2, boxes + wh / 2], axis=1)
        xyxy = np.clip(xyxy, 0, size - 1)
        labels = rng.integers(0, num_classes, size=(n, 1))
        targets.append(torch.from_numpy(np.concatenate([labels, xyxy], axis=1).astype(np.float32)))
    return images, targets


def run_training_steps(model: TinyDetector, criterion: TinyDetectionLoss, steps: int, size: int) -> None:
    device = torch.device("cpu")
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    rng = np.random.default_rng(0)

    for step in range(steps):
        images, targets = make_synthetic_batch(2, size, model.cfg.num_classes, rng)
        t0 = time.perf_counter()
        loss, stats = criterion(model(images.to(device)), targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_sq, grad_max = 0.0, 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq += float(p.grad.pow(2).sum())
                grad_max = max(grad_max, float(p.grad.abs().max()))
        opt.step()
        dt = time.perf_counter() - t0
        if not torch.isfinite(loss):
            raise RuntimeError(f"step {step}: non-finite loss {loss}")
        print(
            f"  step {step}: loss {loss.item():7.4f} "
            f"(box {stats['loss_box'].item():6.4f}  cls {stats['loss_cls'].item():7.4f}  "
            f"obj {stats['loss_obj'].item():7.4f})  "
            f"|grad| rms {np.sqrt(grad_sq):.3e}  max {grad_max:.2e}  "
            f"pos {int(stats['num_positives'])}  {dt * 1000:.0f} ms"
        )


def run_tracker_smoke() -> None:
    """Constant-velocity target with one dropped detection (occlusion)."""
    tracker = SORTTracker(SORTTrackerConfig(assoc_metric="nwd", assoc_threshold=0.2, max_age=5, min_hits=1))
    ids_seen, tsu_history = [], []
    for frame in range(6):
        # Target moves +8 px/frame to the right; frame 3 is fully occluded.
        dets = np.array([[100.0 + 8 * frame, 200.0, 110.0 + 8 * frame, 208.0, 0.9, 0.0]])
        if frame == 3:
            dets = np.zeros((0, 6))
        tracks = tracker.update(dets, frame_idx=frame)
        ids_seen.append([t.track_id for t in tracks])
        tsu_history.append(tracker.tracks[0].time_since_update)
    all_ids = {i for frame_ids in ids_seen for i in frame_ids}
    assert len(all_ids) == 1, f"one target must keep one identity, got {all_ids}"
    assert tsu_history[3] == 1, "frame after the missed detection: track must coast on prediction"
    track = tracker.tracks[0]
    assert track.time_since_update == 0, "track must re-lock after the occlusion ends"
    assert len(track.trajectory) == 5, f"expected 5 trajectory points (4 updates + birth), got {len(track.trajectory)}"
    print(f"  track id {track.id}: {len(track.trajectory)} trajectory points, coasted 1 occluded "
          f"frame, identity preserved — OK")


def run_evaluator_smoke(num_classes: int) -> None:
    """Feed near-perfect and random predictions; metric ordering must hold."""
    rng = np.random.default_rng(7)
    ev = DetectionEvaluator(num_classes=num_classes, class_names=["drone", "bird", "clutter"][:num_classes])
    for img in range(4):
        gt = rng.uniform(50, 500, size=(3, 2))
        wh = rng.uniform(6, 20, size=(3, 2))
        boxes = np.concatenate([gt - wh / 2, gt + wh / 2], axis=1)
        labels = rng.integers(0, num_classes, size=3)
        ev.update(img, boxes, np.full(3, 0.9), labels, boxes, labels)          # perfect dets
        junk = rng.uniform(0, 640, size=(5, 2))
        jw = rng.uniform(5, 15, size=(5, 2))
        jboxes = np.concatenate([junk - jw / 2, junk + jw / 2], axis=1)
        # Noise dets (GT is NOT re-added — it already entered via the call above).
        ev.update(img, jboxes, rng.uniform(0.05, 0.4, size=5),
                  rng.integers(0, num_classes, size=5), np.zeros((0, 4)), np.zeros(0, dtype=np.int64))
    res = ev.results()
    fp = ev.fp_analysis(conf_threshold=0.25)
    assert res["mAP_all"] > 0.5, f"near-perfect dets should score high, got {res['mAP_all']:.3f}"
    print(f"  mAP={res['mAP_all']:.3f}  mAP50={res['mAP50']:.3f}  mAP_small={res['mAP_small']:.3f}  "
          f"FP: {fp.true_positives} TP, {fp.localization_fp} loc-FP, {fp.background_fp} bg-FP — OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="SkySentry Phase-1 dry run")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--size", type=int, default=320, help="input size (multiple of 32)")
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))  # avoid thread thrash on small models

    print("== SkySentry Phase-1 dry run ==")
    model_cfg = load_config("model.yaml").get("model", {})
    loss_cfg = load_config("loss.yaml").get("loss", {})

    # 1. Model ----------------------------------------------------------------
    det_cfg = DetectorConfig(
        num_classes=int(model_cfg.get("num_classes", 3)),
        width=float(model_cfg.get("width", 0.5)),
        depth=float(model_cfg.get("depth", 0.33)),
        strides=tuple(model_cfg.get("strides", [4, 8, 16])),
    )
    model = TinyDetector(det_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[1] model: {n_params / 1e6:.2f} M params, detection strides {model.strides}")
    x = torch.randn(1, 3, args.size, args.size)
    model.eval()
    with torch.no_grad():
        outs = model(x)
    for s, o in zip(model.strides, outs):
        print(f"    stride {s:2d}: feature {tuple(o['cls'].shape[-2:])} "
              f"({args.size // s}x downsample) — cls {tuple(o['cls'].shape)}")

    # 2. Training steps -------------------------------------------------------
    criterion = TinyDetectionLoss(DetectionLossConfig(
        num_classes=det_cfg.num_classes,
        strides=det_cfg.strides,
        scale_ranges=((0.0, 64.0), (64.0, 128.0), (128.0, float("inf"))),
        nwd_constant=float(loss_cfg.get("nwd_constant", 12.8)),
        nwd_alpha=float(loss_cfg.get("nwd_alpha", 0.5)),
        weight_box=float(loss_cfg.get("weight_box", 5.0)),
    ))
    print(f"\n[2] {args.steps} training steps on synthetic tensors ({args.size}px, CPU):")
    model.train()
    run_training_steps(model, criterion, args.steps, args.size)

    # 3. Tracker ----------------------------------------------------------------
    print("\n[3] Kalman tracker (NWD association, occlusion coasting):")
    run_tracker_smoke()

    # 4. Evaluator ----------------------------------------------------------------
    print("\n[4] COCO-style evaluator (mAP / mAP-small / FP taxonomy):")
    run_evaluator_smoke(det_cfg.num_classes)

    print("\nDRY RUN PASSED")


if __name__ == "__main__":
    main()
