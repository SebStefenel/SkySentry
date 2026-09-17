"""Tests for Phase-3 pipeline pieces: NMS, letterbox, config toggles, dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synthetic_composer import SyntheticAerialGenerator, SyntheticConfig  # noqa: E402
from data.yolo_dataset import YoloDetDataset, letterbox  # noqa: E402
from infer import nms, postprocess  # noqa: E402
from losses.detection_loss import DetectionLossConfig, TinyDetectionLoss  # noqa: E402
from models.detector_p2 import DetectorConfig, TinyDetector  # noqa: E402


def test_nms_suppresses_overlaps_keeps_distinct() -> None:
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],     # A
        [1.0, 1.0, 11.0, 11.0],     # B: overlaps A heavily
        [50.0, 50.0, 60.0, 60.0],   # C: distinct
    ])
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep == [0, 2]


def test_letterbox_transforms_boxes_consistently() -> None:
    img = np.full((100, 200, 3), 128, dtype=np.uint8)      # 2:1 aspect
    boxes = np.array([[10.0, 10.0, 60.0, 60.0]])
    out, tb = letterbox(img, 200, boxes)
    assert out.shape == (200, 200, 3)
    # scale = 200/200 = 1.0; pad left = 0, top = 50
    assert np.allclose(tb[0], [10.0, 60.0, 60.0, 110.0])
    # roundtrip: invert the transform
    inv = tb.copy()
    inv[:, [1, 3]] -= 50.0
    assert np.allclose(inv, boxes, atol=1e-6)


def test_baseline_strides_path_trains() -> None:
    """strides=(8,16,32), pure CIoU — the (a) variant must forward/backward."""
    cfg = DetectorConfig(num_classes=3, strides=(8, 16, 32))
    model = TinyDetector(cfg)
    x = torch.randn(1, 3, 320, 320)
    outs = model(x)
    assert [tuple(o["cls"].shape[-2:]) for o in outs] == [(40, 40), (20, 20), (10, 10)]
    criterion = TinyDetectionLoss(DetectionLossConfig(
        num_classes=3, strides=(8, 16, 32), nwd_alpha=0.0))
    # One image, one target per scale range; levels are indexed by position:
    # stride 8 owns (0,64), stride 16 owns [64,128), stride 32 owns [128,inf).
    targets = [torch.tensor([
        [0.0, 40.0, 40.0, 48.0, 46.0],      # 8 px  -> stride 8
        [1.0, 80.0, 80.0, 160.0, 160.0],    # 80 px -> stride 16
        [2.0, 60.0, 60.0, 260.0, 260.0],    # 200px -> stride 32
    ])]
    loss, _ = criterion(model(x), targets)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not no_grad, f"parameters without gradient: {no_grad[:5]}"


def test_param_counts_differ_by_variant() -> None:
    n_p2 = sum(p.numel() for p in TinyDetector(DetectorConfig(strides=(4, 8, 16))).parameters())
    n_base = sum(p.numel() for p in TinyDetector(DetectorConfig(strides=(8, 16, 32))).parameters())
    assert n_p2 != n_base  # the toggle actually changes the architecture


def test_yolo_dataset_loads_with_class_map(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    gen = SyntheticAerialGenerator(SyntheticConfig(seed=3, size=96))
    gen.write_dataset(root, 3, split="train")
    ds = YoloDetDataset(root / "train" / "images", root / "train" / "labels",
                        img_size=128, class_map={0: 0, 1: 0, 2: 0})
    assert len(ds) == 3
    img, tgt = ds[0]
    assert img.shape == (3, 128, 128) and tgt["boxes"].shape[1] == 5
    # class-agnostic map: every label row must carry class 0
    if len(tgt["boxes"]):
        assert (tgt["boxes"][:, 0] == 0).all()


def test_postprocess_shapes_and_agnostic() -> None:
    model = TinyDetector(DetectorConfig(num_classes=3, strides=(4, 8, 16)))
    imgs = torch.randn(2, 3, 320, 320)
    dets = postprocess(model, imgs, conf_threshold=0.005, class_agnostic=True)
    assert len(dets) == 2
    for d in dets:
        assert set(d) == {"boxes", "scores", "labels"}
        if len(d["labels"]):
            assert (d["labels"] == 0).all()  # agnostic: single class


if __name__ == "__main__":
    import inspect
    import tempfile

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        if "tmp_path" in inspect.signature(fn).parameters:
            with tempfile.TemporaryDirectory() as td:
                fn(Path(td))
        else:
            fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
