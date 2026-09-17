"""Track-quality filtering: reject tracks with erratic motion patterns.

Real detections include hallucinations — reflections, sensor speckle,
swaying vegetation — whose tracks jitter or accelerate in ways real drones
and birds (near-constant-velocity over a few frames) do not. This module
watches each track's center history over a rolling window and flags tracks
whose *smoothed acceleration* or *direction-flip rate* exceeds thresholds.

It is a policy layer on top of the tracker: it never edits tracks, only
classifies them (``keep`` / ``reject``), so precision before/after filtering
is measurable and thresholds stay configurable in configs/tracking.yaml.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

__all__ = ["ErraticMotionFilter", "TrackMotionStats"]


@dataclass
class TrackMotionStats:
    track_id: int
    n_points: int
    median_speed: float          # px/frame over the window
    median_accel: float          # |Δvelocity| px/frame² over the window
    direction_flips: float       # flips per frame over the window
    rejected: bool
    reason: str = ""


class ErraticMotionFilter:
    """Rolling-window motion filter for tracked objects.

    Args:
        window: number of recent points per track considered.
        min_points: points required before a verdict is issued.
        accel_thresh: median |Δvelocity| (px/frame²) above which a track is
            erratic. Real 8–32 px targets move smoothly; detector noise does not.
        flip_thresh: fraction of direction flips (per frame) above which a
            track is erratic (random-walk false positives flip constantly).
        min_speed_for_flips: below this speed (px/frame) flips are noise, not
            evidence — skip flip counting.
    """

    def __init__(
        self,
        window: int = 7,
        min_points: int = 4,
        accel_thresh: float = 6.0,
        flip_thresh: float = 0.5,
        min_speed_for_flips: float = 1.0,
    ) -> None:
        self.window = window
        self.min_points = min_points
        self.accel_thresh = accel_thresh
        self.flip_thresh = flip_thresh
        self.min_speed_for_flips = min_speed_for_flips
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self._verdicts: dict[int, bool] = {}

    def update(self, track_id: int, cx: float, cy: float) -> bool:
        """Feed one frame's track center; returns True if the track is OK."""
        self._history[track_id].append((float(cx), float(cy)))
        stats = self.stats_for(track_id)
        ok = not (stats and stats.rejected)
        if stats:
            self._verdicts[track_id] = ok
        return ok

    def stats_for(self, track_id: int) -> TrackMotionStats | None:
        """Compute motion statistics (and verdict) for one track's window."""
        pts = list(self._history.get(track_id, ()))
        if len(pts) < self.min_points:
            return None
        pts_arr = np.asarray(pts)
        deltas = np.diff(pts_arr, axis=0)                       # (K-1, 2)
        speeds = np.linalg.norm(deltas, axis=1)
        accels = np.linalg.norm(np.diff(deltas, axis=0), axis=1) if len(deltas) >= 2 else np.zeros(1)
        med_speed = float(np.median(speeds))
        med_accel = float(np.median(accels))

        flips = 0.0
        turns = 0
        if med_speed >= self.min_speed_for_flips and len(deltas) >= 3:
            for a, b in zip(deltas[:-1], deltas[1:]):
                if float(a @ b) < 0:                             # velocity reversal
                    flips += 1.0
                turns += 1
        flip_rate = flips / turns if turns else 0.0

        rejected, reason = False, ""
        if med_accel > self.accel_thresh:
            rejected, reason = True, f"accel {med_accel:.1f} > {self.accel_thresh}"
        elif flip_rate > self.flip_thresh:
            rejected, reason = True, f"flip rate {flip_rate:.2f} > {self.flip_thresh}"
        return TrackMotionStats(track_id, len(pts), med_speed, med_accel, flip_rate, rejected, reason)

    def active_ids(self) -> list[int]:
        return sorted(self._history.keys())

    def summary(self) -> dict[str, int]:
        """Counts of kept/rejected tracks ever seen (for precision reports)."""
        kept = sum(1 for ok in self._verdicts.values() if ok)
        rejected = sum(1 for ok in self._verdicts.values() if not ok)
        return {"kept": kept, "rejected": rejected}
