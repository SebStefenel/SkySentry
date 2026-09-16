"""Smoke tests for the P2 detector: shapes, decode, and a full loss backward."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.detection_loss import DetectionLossConfig, TinyDetectionLoss  # noqa: E402
from models.detector_p2 import DetectorConfig, TinyDetector  # noqa: E402


def test_forward_shapes() -> None:
    cfg = DetectorConfig(num_classes=2, strides=(4, 8, 16))
    model = TinyDetector(cfg)
    x = torch.randn(2, 3, 320, 320)
    outs = model(x)
    assert len(outs) == 3
    expected_hw = [(80, 80), (40, 40), (20, 20)]  # 320 / [4, 8, 16]
    for out, (h, w) in zip(outs, expected_hw):
        assert out["cls"].shape == (2, 2, h, w)
        assert out["obj"].shape == (2, 1, h, w)
        assert out["box"].shape == (2, 4, h, w)
        assert (out["box"] >= 0).all()


def test_loss_backward_reaches_all_heads() -> None:
    model = TinyDetector(DetectorConfig(num_classes=2, strides=(4, 8, 16)))
    criterion = TinyDetectionLoss(DetectionLossConfig(num_classes=2, strides=(4, 8, 16)))
    x = torch.randn(2, 3, 320, 320)
    # Image 1: one target per scale range so every head's box branch gets
    # positives — 8px+10px -> stride 4, 80x70 -> stride 8, 140x120 -> stride 16.
    # Image 2: background-only.
    targets = [
        torch.tensor([
            [0.0, 100.0, 100.0, 108.0, 106.0],
            [1.0, 200.0, 150.0, 210.0, 160.0],
            [0.0, 50.0, 50.0, 190.0, 170.0],
            [1.0, 220.0, 40.0, 300.0, 110.0],
        ]),
        torch.zeros(0, 5),
    ]
    loss, stats = criterion(model(x), targets)
    assert torch.isfinite(loss)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    # Everything must receive gradient (decode keeps graph; assignment has positives).
    assert not no_grad, f"parameters without gradient: {no_grad[:5]}"


def test_decode_output() -> None:
    model = TinyDetector(DetectorConfig(num_classes=2, strides=(4, 8, 16)))
    outs = model(torch.randn(1, 3, 320, 320))
    dets = model.decode(outs, conf_threshold=0.5)
    assert dets.shape[0] == 1 and dets.shape[2] == 6
    assert torch.isfinite(dets).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
