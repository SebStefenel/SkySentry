"""Multi-object tracking: Kalman state estimation + Hungarian association.

Design notes for tiny aerial targets
------------------------------------
* **State** follows SORT (Bewley et al., ICIP 2016): an 8-D state
  ``u = (cx, cy, s, r, vcx, vcy, vs, 0)`` with center position, box *area*
  ``s`` and aspect ratio ``r`` (kept constant in the motion model), driven by a
  constant-velocity model. Area/aspect parameterization keeps ``w, h`` positive
  and decouples scale from motion — important when a 10-px drone accelerates
  but its apparent size barely changes.
* **Association metric** is pluggable: IoU (default of classic trackers) or
  **NWD**, which tolerates the larger jitter-to-size ratio of tiny boxes — a
  3-px frame-to-frame jump destroys IoU on a 6-px bird but barely moves NWD.
* **Occlusion handling**: unmatched tracks are coasted on their Kalman
  prediction for up to ``max_age`` frames (momentary occlusion by a building,
  tree or cloud bank); track identity persists and trajectories stay connected.
* Tracks are confirmed after ``min_hits`` consecutive detections (birth
  debounce against false positives) and reported with full trajectory history
  for downstream spatiotemporal analysis.

Dependencies: ``filterpy`` (Kalman filter), ``scipy`` (Hungarian assignment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

try:  # torch not strictly required for tracking; used for NWD association
    import torch
except ImportError:  # pragma: no cover
    torch = None

from losses.nwd_loss import nwd_similarity  # numpy-friendly: accepts any tensor-like

__all__ = ["TrackState", "Track", "KalmanBoxTracker", "SORTTracker"]


class TrackState(Enum):
    TENTATIVE = 0   # detected fewer than min_hits times
    CONFIRMED = 1   # stable track, reported to the caller
    LOST = 2        # coasting through occlusion (age <= max_age)
    DELETED = 3     # removed


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``(N, 4)`` / ``(M, 4)`` xyxy arrays."""
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)[:, None]  # (N,1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)[None, :]  # (1,M)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.clip(area_a + area_b - inter, 1e-7, None)


def _association_matrix(tracks_xyxy: np.ndarray, dets_xyxy: np.ndarray, metric: str, nwd_const: float) -> np.ndarray:
    """Higher = better match, shaped (n_tracks, n_dets)."""
    if metric == "iou":
        return _iou_matrix(tracks_xyxy, dets_xyxy)
    if metric == "nwd":
        if torch is None:
            raise ImportError("torch is required for NWD association")
        t = torch.as_tensor(tracks_xyxy, dtype=torch.float32)
        d = torch.as_tensor(dets_xyxy, dtype=torch.float32)
        return nwd_similarity(t, d, constant=nwd_const).numpy()
    raise ValueError(f"unknown association metric {metric!r}")


@dataclass
class Track:
    """Public snapshot of one tracked object at the current frame."""

    track_id: int
    state: TrackState
    box_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    hits: int
    age: int
    time_since_update: int
    trajectory: list[tuple[int, float, float]] = field(default_factory=list)
    """(frame_idx, cx, cy) history — the spatiotemporal trajectory."""


class KalmanBoxTracker:
    """Kalman filter around one detected object.

    State: ``[x, y, vx, vy, w, h]`` — center position with constant-velocity
    motion and directly-tracked size (no area/aspect parameterization: for
    tiny boxes the area/aspect singularity hurts more than the extra
    flexibility helps, and w/h can be clamped positive on output).
    """

    _count = 0

    def __init__(self, box_xyxy: np.ndarray, score: float, class_id: int) -> None:
        box_xyxy = np.asarray(box_xyxy, dtype=np.float64).reshape(4)  # accept (4,) or (1, 4)
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        # x' = x + vx; y' = y + vy; velocities and size persist.
        self.kf.F = np.eye(6)
        self.kf.F[0, 2] = 1.0   # x += vx
        self.kf.F[1, 3] = 1.0   # y += vy
        self.kf.H = np.zeros((4, 6))
        self.kf.H[0, 0] = 1.0   # measure x
        self.kf.H[1, 1] = 1.0   # measure y
        self.kf.H[2, 4] = 1.0   # measure w
        self.kf.H[3, 5] = 1.0   # measure h
        self.kf.R[2:, 2:] *= 10.0            # size measured noisier than position
        self.kf.P[2:4, 2:4] *= 1000.0        # high initial velocity uncertainty
        self.kf.P *= 10.0
        self.kf.Q[2:4, 2:4] *= 0.01          # velocity noise
        z = self._bbox_to_z(box_xyxy)
        self.kf.x[0, 0], self.kf.x[1, 0] = z[0], z[1]   # center
        self.kf.x[4, 0], self.kf.x[5, 0] = z[2], z[3]   # size; velocities stay 0

        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        self.score = score
        self.class_id = class_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.trajectory: list[tuple[int, float, float]] = []
        cx, cy = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
        self.trajectory.append((0, cx, cy))

    @staticmethod
    def _bbox_to_z(box_xyxy: np.ndarray) -> np.ndarray:
        """xyxy box → measurement (cx, cy, w, h)."""
        w = max(box_xyxy[2] - box_xyxy[0], 1e-3)
        h = max(box_xyxy[3] - box_xyxy[1], 1e-3)
        cx = box_xyxy[0] + w / 2.0
        cy = box_xyxy[1] + h / 2.0
        return np.array([cx, cy, w, h])

    @staticmethod
    def _x_to_bbox(x: np.ndarray) -> tuple[float, float, float, float]:
        """State mean → xyxy box (guards degenerate sizes)."""
        cx, cy = float(x[0, 0]), float(x[1, 0])
        w = max(float(x[4, 0]), 1e-3)
        h = max(float(x[5, 0]), 1e-3)
        return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0

    def predict(self) -> tuple[float, float, float, float]:
        """Advance the motion model one frame; returns the predicted xyxy box."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._x_to_bbox(self.kf.x)

    def update(self, box_xyxy: np.ndarray, score: float, class_id: int, frame_idx: int) -> None:
        """Correct the state with a new detection and extend the trajectory."""
        self.kf.update(self._bbox_to_z(box_xyxy))
        self.time_since_update = 0
        self.hits += 1
        self.score = score
        self.class_id = class_id
        cx, cy = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
        self.trajectory.append((frame_idx, cx, cy))

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self._x_to_bbox(self.kf.x)

    def snapshot(self, frame_idx: int) -> Track:
        state = (
            TrackState.CONFIRMED if self.hits >= 2 and self.time_since_update == 0
            else TrackState.LOST if self.hits >= 2
            else TrackState.TENTATIVE
        )
        return Track(
            track_id=self.id,
            state=state,
            box_xyxy=self.box,
            score=self.score,
            class_id=self.class_id,
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
            trajectory=list(self.trajectory),
        )


class SORTTrackerConfig:
    """Knobs of the SORT-style tracker."""

    def __init__(
        self,
        assoc_metric: Literal["iou", "nwd"] = "nwd",
        assoc_threshold: float = 0.3,   # min similarity to accept a match
        nwd_constant: float = 12.8,
        max_age: int = 15,              # frames a LOST track is coasted
        min_hits: int = 3,              # detections before a track is confirmed
    ) -> None:
        self.assoc_metric = assoc_metric
        self.assoc_threshold = assoc_threshold
        self.nwd_constant = nwd_constant
        self.max_age = max_age
        self.min_hits = min_hits


class SORTTracker:
    """Frame-by-frame multi-object tracker with occlusion coasting.

    Usage::

        tracker = SORTTracker(cfg)
        for frame_dets in video:            # (N, 6) array: x1,y1,x2,y2,score,cls
            tracks = tracker.update(frame_dets, frame_idx)
            for t in tracks:                # confirmed tracks carry `trajectory`
                ...
    """

    def __init__(self, config: SORTTrackerConfig | None = None) -> None:
        self.cfg = config or SORTTrackerConfig()
        self.tracks: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(
        self,
        detections: np.ndarray | "torch.Tensor",
        frame_idx: int | None = None,
    ) -> list[Track]:
        """Feed one frame's detections; returns the current track snapshots.

        Args:
            detections: ``(N, 6)`` rows of ``(x1, y1, x2, y2, score, class_id)``.
                Empty ``(0, 6)`` input is fine (occluded frame).
            frame_idx: optional external frame counter; defaults to internal.
        """
        if torch is not None and isinstance(detections, torch.Tensor):
            detections = detections.detach().cpu().numpy()
        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 6)
        if frame_idx is None:
            frame_idx = self.frame_count
        self.frame_count = max(self.frame_count, frame_idx + 1)

        # 1. Predict: coast every existing track forward.
        predicted = np.array([t.predict() for t in self.tracks]) if self.tracks else np.zeros((0, 4))

        # 2. Associate (Hungarian on negative similarity = cost).
        matched_t, matched_d = [], []
        if len(self.tracks) and len(detections):
            sim = _association_matrix(
                predicted, detections[:, :4], self.cfg.assoc_metric, self.cfg.nwd_constant
            )
            cost = -sim
            cost[sim < self.cfg.assoc_threshold] = 1e6   # forbidden pairs
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if sim[r, c] >= self.cfg.assoc_threshold:
                    matched_t.append(r)
                    matched_d.append(c)

        # 3. Update matched tracks with their detections.
        for t_idx, d_idx in zip(matched_t, matched_d):
            self.tracks[t_idx].update(
                detections[d_idx, :4], float(detections[d_idx, 4]), int(detections[d_idx, 5]), frame_idx
            )

        # 4. Birth new tracks from unmatched detections.
        unmatched_d = [c for c in range(len(detections)) if c not in matched_d]
        for c in unmatched_d:
            self.tracks.append(
                KalmanBoxTracker(detections[c, :4], float(detections[c, 4]), int(detections[c, 5]))
            )

        # 5. Reap dead tracks (occluded for too long).
        alive: list[KalmanBoxTracker] = []
        for t in self.tracks:
            keep = t.time_since_update <= self.cfg.max_age
            if keep:
                alive.append(t)
        self.tracks = alive

        return [t.snapshot(frame_idx) for t in self.tracks]

    def confirmed_tracks(self) -> list[Track]:
        """Confirmed tracks with recent updates, ready for consumption."""
        return [
            t.snapshot(self.frame_count - 1)
            for t in self.tracks
            if t.hits >= self.cfg.min_hits and t.time_since_update <= 1
        ]
