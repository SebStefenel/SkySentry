# SkySentry
High-resolution small-object detection and spatiotemporal trajectory tracking pipeline in PyTorch, featuring a P2 stride-4 feature head, Normalized Wasserstein Distance (NWD) loss, and constant-velocity Kalman state estimation.

## Status — Phase 1 (foundation): implemented & verified

```
SkySentry/
├── configs/               YAML: model / loss / data / tracking hyperparameters
├── data/
│   ├── dataset_downloader.py   CLI: synthetic sample (default) + VisDrone zip → YOLO conversion
│   └── synthetic_composer.py   Seeded aerial-scene renderer (sim) + moving-object clips
├── models/
│   └── detector_p2.py     Compact CSP backbone + PAN neck with P2 (stride-4) + anchor-free decoupled heads
├── losses/
│   ├── nwd_loss.py        Normalized Wasserstein Distance loss (closed-form Gaussian W2)
│   ├── iou_losses.py      IoU / GIoU / CIoU (pairwise + matched)
│   ├── focal_loss.py      Sigmoid focal loss
│   └── detection_loss.py  Multi-level assignment + focal/obj/NWD composite
├── tracking/
│   └── kalman_tracker.py  SORT-style Kalman tracker, IoU/NWD association, occlusion coasting
├── evaluation/
│   └── metrics.py         COCO-style mAP (+ small split), FP taxonomy, sim-to-real gap report
├── tests/                 pytest suites (NWD math, detector shapes/gradients, tracker, metrics)
└── train_step_dryrun.py   End-to-end pipeline verification on synthetic tensors
```

## Quickstart

```powershell
pip install -r requirements.txt
python tests/test_nwd_loss.py          # or: pytest
python train_step_dryrun.py            # full-stack verification, CPU, ~1 min
python -m data.dataset_downloader --dataset synthetic --out data/raw --num-train 64 --num-val 16
```

## Why each piece exists

**P2 (stride-4) head.** Standard YOLO heads start at stride 8; a 6-px bird is
then smaller than one grid cell. The neck here lifts the top-down path to
stride 4, where tiny aerial targets retain 1.5–6 cells of spatial detail.
Default detection strides: `(4, 8, 16)` (configure `32` for large objects).

**NWD loss.** IoU is pathological for tiny boxes: near-identical boxes that
stop overlapping get IoU = 0 and zero gradient. NWD models each box
`(cx, cy, w, h)` as a Gaussian `N(mu, diag((w/2, h/2)^2))` and uses the
closed-form 2-Wasserstein distance

```
W2^2 = (dcx)^2 + (dcy)^2 + ((w1-w2)/2)^2 + ((h1-h2)/2)^2
NWD  = exp(-sqrt(W2^2) / C)          # C ≈ mean object size (12.8 px for VisDrone)
L    = alpha * (1 - NWD) + (1 - alpha) * CIoU
```

which is smooth, bounded in (0, 1], and differentiable everywhere — see
`losses/nwd_loss.py` for the full derivation and `tests/test_nwd_loss.py`
for the mathematical properties under test.

**Kalman tracking.** SORT-style constant-velocity filter on
`(cx, cy, area, aspect)` with Hungarian association. For tiny targets the
association metric is **NWD** (not IoU): a 3-px frame-to-frame jump destroys
IoU on a 6-px bird while barely moving NWD. Tracks coast through up to
`max_age` missed frames (occlusions) without losing identity.

**Sim-to-real.** Phase 1 trains/verifies on the seeded synthetic composer;
`evaluation/metrics.py` already computes the gap report (AP drop,
small-object AP drop, FP taxonomy) that will quantify synthetic→real
transfer once real VisDrone/Drone-vs-Bird splits are wired in Phase 2.

## Roadmap

- **Phase 1 (this repo)** — architecture, losses, tracking, metrics, dry run. ✅
- **Phase 2** — real-dataset integration (VisDrone / Drone-vs-Bird), training loop,
  NMS + inference, video tracking demo, domain-gap baselines.
- **Phase 3** — augmentation-based domain adaptation (haze/gamma/compression),
  self-training, ablation studies (P2 on/off, NWD vs IoU).
