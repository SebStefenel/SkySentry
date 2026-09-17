"""Phase-4 tests: migrated Kalman state [x, y, vx, vy, w, h], motion filter,
and occlusion-benchmark plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracking.benchmark_occlusion import score_clip  # noqa: E402
from data.synthetic_composer import generate_clip  # noqa: E402
from tracking.kalman_tracker import KalmanBoxTracker  # noqa: E402
from tracking.track_filter import ErraticMotionFilter  # noqa: E402


def test_kalman_state_uses_size_parameterization() -> None:
    """State [x, y, vx, vy, w, h]: constant-velocity center, stable size.

    Follows the real SORT protocol — predict() then update() each frame —
    which is what builds the position–velocity cross-covariance the filter
    needs to learn velocity.
    """
    t = KalmanBoxTracker(np.array([100.0, 100.0, 110.0, 110.0]), 0.9, 0)
    # Object moves +8 px/frame right for 8 frames (10px box stays constant).
    for f in range(1, 9):
        t.predict()
        box = (100 + 8 * f, 100.0, 110 + 8 * f, 110.0)
        t.update(np.array(box, dtype=float), 0.9, 0, frame_idx=f)
    # Predict 3 frames ahead: center must extrapolate the +8 px/frame motion.
    for _ in range(3):
        predicted = t.predict()
    cx = (predicted[0] + predicted[2]) / 2
    expected_cx = 105 + 8 * 11  # center(f=8)=169, +3 frames of learned velocity
    assert abs(cx - expected_cx) < 8.0, f"center {cx} vs expected {expected_cx}"
    # A zero-velocity model would predict 169; the tolerance above fails that.
    w = predicted[2] - predicted[0]
    assert 8.0 <= w <= 12.0, f"size drifted: {w}"


def test_kalman_legacy_behaviors_survive_migration() -> None:
    """Zero-area input and long coasts stay finite after the state change."""
    t = KalmanBoxTracker(np.array([50.0, 50.0, 50.0, 50.0]), 0.9, 0)
    for f in range(5):
        assert np.all(np.isfinite(t.predict()))
    t.update(np.array([50.0, 50.0, 50.0, 50.0]), 0.9, 0, frame_idx=5)
    assert np.all(np.isfinite(t.box))


def test_motion_filter_rejects_jitter_keets_straight() -> None:
    filt = ErraticMotionFilter(window=7, min_points=4, accel_thresh=6.0, flip_thresh=0.5)
    # Straight smooth track: kept.
    for f in range(8):
        ok = filt.update(1, 100.0 + 6.0 * f, 200.0 + 1.0 * f)
    assert ok, "smooth track must be kept"
    # Random-walk jitter track: rejected.
    rng = np.random.default_rng(0)
    rejected = False
    for f in range(8):
        ok = filt.update(2, 100.0 + rng.uniform(-30, 30), 200.0 + rng.uniform(-30, 30))
        rejected = rejected or not ok
    assert rejected, "jittery track must be rejected"
    summary = filt.summary()
    assert summary == {"kept": 1, "rejected": 1}


def test_occlusion_benchmark_scoring(tmp_path: Path) -> None:
    """One clip, one dropout length: metrics must be finite and sensible."""
    frames = generate_clip(num_frames=30, size=256, seed=5, num_objects=3)
    res = score_clip(frames, drop_len=5, drop_start_frac=0.3, assoc_metric="nwd", seed=5)
    for key in ("mota", "idf1", "id_switches", "id_preservation"):
        assert np.isfinite(res[key]), f"{key} not finite: {res}"
    assert 0.0 <= res["idf1"] <= 1.0
    assert res["id_preservation"] > 0.5, "near-perfect detections should preserve most IDs"


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
