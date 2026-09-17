"""Smoke tests for tracking, evaluation, and the synthetic composer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synthetic_composer import SyntheticAerialGenerator, SyntheticConfig, generate_clip  # noqa: E402
from evaluation.metrics import DetectionEvaluator  # noqa: E402
from tracking.kalman_tracker import KalmanBoxTracker, SORTTracker, SORTTrackerConfig  # noqa: E402


def test_tracker_identity_survives_occlusion() -> None:
    tracker = SORTTracker(SORTTrackerConfig(assoc_metric="nwd", assoc_threshold=0.2, max_age=5, min_hits=1))
    ids = set()
    for frame in range(6):
        dets = np.array([[100.0 + 8 * frame, 200.0, 110.0 + 8 * frame, 208.0, 0.9, 0.0]])
        if frame == 3:
            dets = np.zeros((0, 6))  # occluded frame: no detections
        for t in tracker.update(dets, frame_idx=frame):
            ids.add(t.track_id)
    assert len(ids) == 1, f"identity broke across occlusion: {ids}"


def test_tracker_uses_iou_metric_too() -> None:
    tracker = SORTTracker(SORTTrackerConfig(assoc_metric="iou", assoc_threshold=0.1, max_age=5, min_hits=1))
    for frame in range(5):
        dets = np.array([[50.0 + 2 * frame, 60.0, 70.0 + 2 * frame, 80.0, 0.8, 1.0]])
        tracker.update(dets, frame_idx=frame)
    assert len(tracker.tracks) == 1


def test_two_targets_two_ids() -> None:
    tracker = SORTTracker(SORTTrackerConfig(assoc_metric="nwd", assoc_threshold=0.3, min_hits=1))
    for frame in range(4):
        dets = np.array([
            [100.0, 100.0, 110.0, 110.0, 0.9, 0.0],
            [300.0 + 4 * frame, 100.0, 310.0 + 4 * frame, 110.0, 0.8, 1.0],
        ])
        tracker.update(dets, frame_idx=frame)
    assert len({t.id for t in tracker.tracks}) == 2


def test_tracker_handles_empty_detection_lists() -> None:
    """Empty input must be a no-op — both on a fresh tracker and repeated."""
    tracker = SORTTracker(SORTTrackerConfig(min_hits=1))
    assert tracker.update(np.zeros((0, 6)), frame_idx=0) == []
    assert tracker.update(np.zeros((0, 6)), frame_idx=1) == []
    # And an empty frame between detections must not crash either.
    det = np.array([[0.0, 0.0, 10.0, 10.0, 0.9, 0.0]])
    tracker.update(det, frame_idx=2)
    tracker.update(np.zeros((0, 6)), frame_idx=3)
    assert len(tracker.tracks) == 1


def test_kalman_survives_zero_area_box() -> None:
    """A degenerate (zero-area) detection must not poison the filter state."""
    tracker = KalmanBoxTracker(np.array([50.0, 50.0, 50.0, 50.0]), 0.9, 0)
    for f in range(5):
        box = tracker.predict()
        assert np.all(np.isfinite(box)), f"non-finite prediction at frame {f}"
    tracker.update(np.array([50.0, 50.0, 50.0, 50.0]), 0.9, 0, frame_idx=5)
    assert np.all(np.isfinite(tracker.box))


def test_kalman_stays_finite_over_long_coast() -> None:
    """60 frames of pure prediction (no updates) must stay finite — guards
    against covariance collapse/divergence in the linear CV model."""
    tracker = KalmanBoxTracker(np.array([100.0, 100.0, 110.0, 110.0]), 0.8, 0)
    for f in range(60):
        box = tracker.predict()
        assert np.all(np.isfinite(box)), f"diverged at frame {f}: {box}"


def test_evaluator_perfect_predictions_score_one() -> None:
    rng = np.random.default_rng(0)
    ev = DetectionEvaluator(num_classes=2)
    for img in range(3):
        boxes = np.array([[50.0, 50.0, 60.0, 60.0], [200.0, 200.0, 208.0, 208.0]])
        labels = np.array([0, 1])
        scores = np.array([0.95, 0.9])
        ev.update(img, boxes, scores, labels, boxes, labels)
    res = ev.results()
    assert res["mAP_all"] > 0.99, res
    assert res["mAP50"] > 0.99, res


def test_evaluator_small_split_and_fp_taxonomy() -> None:
    # GT boxes ~10 px -> area 100 < 32^2, so everything lands in 'small'.
    ev = DetectionEvaluator(num_classes=1)
    rng = np.random.default_rng(1)
    for img in range(3):
        box = np.array([[100.0, 100.0, 110.0, 110.0]])
        ev.update(img, box + rng.normal(0, 0.2, box.shape), np.array([0.9]), np.array([0]), box, np.array([0]))
        # Far-away false alarm in a corner (no GT re-added here).
        ev.update(img, np.array([[600.0, 600.0, 608.0, 608.0]]), np.array([0.6]), np.array([0]),
                  np.zeros((0, 4)), np.zeros(0, dtype=np.int64))
    res = ev.results()
    assert res["mAP_small"] > 0.5
    fp = ev.fp_analysis(conf_threshold=0.5)
    assert fp.true_positives == 3 and fp.background_fp == 3 and fp.missed_gt == 0


def test_domain_gap_report_keys() -> None:
    from evaluation.metrics import domain_gap_report

    src = {"mAP_all": 0.8, "mAP_small": 0.6}
    tgt = {"mAP_all": 0.6, "mAP_small": 0.3}
    gap = domain_gap_report(src, tgt)
    assert abs(gap["gap_mAP_all"] - 0.2) < 1e-9
    assert abs(gap["gap_rel_mAP_small"] - 0.5) < 1e-9


def test_synthetic_generator_deterministic() -> None:
    g = SyntheticAerialGenerator(SyntheticConfig(seed=42, size=128))
    img1, lbl1 = g.generate_one(3)
    img2, lbl2 = g.generate_one(3)
    assert np.array_equal(img1, img2)
    assert lbl1 == lbl2
    img3, _ = g.generate_one(4)  # different index → different scene
    assert not np.array_equal(img1, img3)


def test_clip_has_occlusion_gap() -> None:
    frames = generate_clip(num_frames=20, size=256, seed=5, num_objects=2)
    assert len(frames) == 20
    counts = [len(lbl) for _img, lbl in frames]
    # The sweeping occluder must hide at least one (object, frame) pair.
    assert max(counts) == 2 and sum(c == 2 for c in counts) < 20


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
