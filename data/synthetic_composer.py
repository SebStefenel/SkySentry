"""Deterministic synthetic aerial-scene composer (the "sim" of sim-to-real).

Renders small-target scenes — drones, birds and background clutter — as
NumPy/OpenCV images with YOLO-format labels, fully controlled by a seed:

* **Backgrounds**: sky/terrain color gradients + Gaussian noise fields +
  rectangle "buildings"/"roads" (clutter), lightly blurred.
* **Objects**: tiny bright/dark ellipses with a soft blur (4-24 px), drawn
  from the class palette; optionally rotated motion streaks for drones.
* **Clips** (:func:`generate_clip`): constant-velocity objects + random
  rectangular occluders crossing frames, to exercise the Kalman tracker
  through momentary occlusions.

Determinism: every image is generated from ``np.random.default_rng(seed + i)``
so any subset is reproducible across machines (same NumPy version).

Labels use YOLO format: ``class_id cx cy w h`` normalized to [0, 1].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

__all__ = ["SyntheticConfig", "SyntheticAerialGenerator", "generate_clip"]

# class_id → (name, base BGR color, brightness mode)
# 0 = drone (bright, rigid, slight streak), 1 = bird (dark, elongated), 2 = clutter
CLASSES = {0: ("drone", (230, 235, 240), "bright"), 1: ("bird", (40, 45, 60), "dark"), 2: ("clutter", (150, 160, 170), "dark")}


@dataclass
class SyntheticConfig:
    """Generation parameters. Defaults target VisDrone-like statistics."""

    size: int = 640                       # square image side
    min_objects: int = 1
    max_objects: int = 8
    tiny_size_px: tuple[float, float] = (4.0, 24.0)   # min/max object half-extent
    p_class: tuple[float, float, float] = (0.45, 0.45, 0.10)  # drone/bird/clutter
    clutter_rects: tuple[int, int] = (2, 10)          # background rectangles
    motion_blur_prob: float = 0.35
    noise_sigma: float = 6.0
    seed: int = 0


class SyntheticAerialGenerator:
    """Seeded renderer of single-frame tiny-target scenes."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.cfg = config or SyntheticConfig()

    # ------------------------------------------------------------------ #
    # Background                                                          #
    # ------------------------------------------------------------------ #

    def _background(self, rng: np.random.Generator) -> np.ndarray:
        s = self.cfg.size
        base = rng.integers(60, 160, size=3, dtype=np.uint8)
        img = np.full((s, s, 3), base, dtype=np.float32)
        # Smooth vertical gradient (sky → ground).
        grad = rng.uniform(-0.4, 0.4)
        ramp = np.linspace(1.0 - grad, 1.0 + grad, s, dtype=np.float32)[:, None, None]
        img *= ramp
        # Low-frequency noise field (terrain texture).
        noise = rng.normal(0, 1, (s // 8 + 1, s // 8 + 1, 3)).astype(np.float32)
        noise = cv2.resize(noise, (s, s), interpolation=cv2.INTER_CUBIC)
        img += noise * 10.0
        # Clutter rectangles: buildings / roads / vehicles.
        n_rect = int(rng.integers(self.cfg.clutter_rects[0], self.cfg.clutter_rects[1] + 1))
        for _ in range(n_rect):
            x1, y1 = rng.integers(0, s - 2, size=2)
            w, h = int(rng.integers(8, s // 3)), int(rng.integers(6, s // 4))
            color = rng.integers(30, 200, size=3, dtype=np.uint8).astype(np.float32)
            img[y1 : y1 + h, x1 : x1 + w] = color
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img += rng.normal(0, self.cfg.noise_sigma, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Objects                                                             #
    # ------------------------------------------------------------------ #

    def _draw_object(self, img: np.ndarray, cx: float, cy: float, r: float, cls_id: int, rng: np.random.Generator) -> bool:
        """Draw one ellipse target; returns False if it left the canvas."""
        s = self.cfg.size
        if not (r + 1 <= cx < s - r - 1 and r + 1 <= cy < s - r - 1):
            return False
        name, color, mode = CLASSES[cls_id]
        aspect = rng.uniform(1.2, 2.2) if name == "bird" else rng.uniform(0.85, 1.2)
        ax = int(round(max(2, r * aspect)))           # semi-major
        ay = int(round(max(2, r / aspect)))           # semi-minor
        angle = rng.uniform(0, 180)
        col = np.array(color, dtype=np.float32) + rng.normal(0, 10, 3)
        thickness = -1
        cv2.ellipse(img, (int(cx), int(cy)), (ax, ay), angle, 0, 360, col.astype(np.uint8).tolist(), thickness)
        if mode == "bright":
            cv2.ellipse(img, (int(cx), int(cy)), (max(1, ax // 2), max(1, ay // 2)), angle, 0, 360, (255, 255, 255), -1)
        if rng.random() < self.cfg.motion_blur_prob:
            k = max(3, int(r))
            kernel = np.zeros((1, k), dtype=np.float32)
            kernel[0, :] = 1.0 / k
            patch = img[max(0, int(cy) - 4) : int(cy) + 5, max(0, int(cx) - ax - k) : int(cx) + ax + k]
            if patch.size:
                img[max(0, int(cy) - 4) : int(cy) + 5, max(0, int(cx) - ax - k) : int(cx) + ax + k] = cv2.filter2D(patch, -1, kernel)
        return True

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_one(self, index: int) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
        """Render scene ``index``; returns (image BGR, labels).

        Labels are YOLO-normalized ``(class_id, cx, cy, w, h)`` tuples.
        """
        rng = np.random.default_rng(self.cfg.seed + index)
        img = self._background(rng)
        s = self.cfg.size
        n_obj = int(rng.integers(self.cfg.min_objects, self.cfg.max_objects + 1))
        labels: list[tuple[int, float, float, float, float]] = []
        attempts = 0
        while len(labels) < n_obj and attempts < n_obj * 10:
            attempts += 1
            r = float(rng.uniform(*self.cfg.tiny_size_px))
            cx, cy = rng.uniform(r + 1, s - r - 1, size=2)
            cls_id = int(rng.choice(3, p=np.asarray(self.cfg.p_class) / np.sum(self.cfg.p_class)))
            if self._draw_object(img, cx, cy, r, cls_id, rng):
                labels.append((cls_id, cx / s, cy / s, 2 * r / s, 2 * r / s))
        return img, labels

    def write_dataset(self, out_dir: str | Path, num_images: int, split: str = "train") -> Path:
        """Render ``num_images`` scenes to ``out_dir/split`` with YOLO labels."""
        out = Path(out_dir) / split
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        iterator = range(num_images)
        if tqdm is not None:
            iterator = tqdm(iterator, desc=f"synthetic/{split}")
        for i in iterator:
            img, labels = self.generate_one(i)
            cv2.imwrite(str(out / "images" / f"{i:06d}.png"), img)
            with open(out / "labels" / f"{i:06d}.txt", "w", encoding="utf-8") as f:
                for cls_id, cx, cy, w, h in labels:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        return out


def generate_clip(
    num_frames: int,
    size: int = 640,
    seed: int = 123,
    num_objects: int = 3,
) -> list[tuple[np.ndarray, list[tuple[int, float, float, float, float]]]]:
    """Render a short constant-velocity clip with random occluders.

    Each object i starts at a seeded random position with a seeded velocity
    (px/frame) and keeps it — the exact motion model the Kalman filter assumes.
    From frame 40% onward, one rectangle occluder sweeps across, blocking
    objects for a few frames at a time (occlusion coasting test).

    Returns: list of (frame BGR, YOLO-normalized labels) — occluded objects
    are simply absent from that frame's labels.
    """
    base = np.random.default_rng(seed)
    gen = SyntheticAerialGenerator(SyntheticConfig(size=size, motion_blur_prob=0.2, seed=seed * 1000))
    starts = base.uniform(0.1, 0.9, size=(num_objects, 2)) * size
    vel = base.uniform(-6.0, 6.0, size=(num_objects, 2))
    radii = base.uniform(4.0, 12.0, size=(num_objects,))
    classes = base.choice(2, size=num_objects, p=[0.5, 0.5])  # drone / bird
    occ_x_start = base.uniform(0.2, 0.6) * size
    occ_width = base.uniform(40, 90)

    frames = []
    for t in range(num_frames):
        rng = np.random.default_rng(seed * 1000 + t)
        img = gen._background(rng)
        occ_active = t >= int(num_frames * 0.4)
        occ_x = occ_x_start + t * 8.0 if occ_active else -1e9
        labels = []
        for i in range(num_objects):
            cx = float(starts[i, 0] + vel[i, 0] * t)
            cy = float(starts[i, 1] + vel[i, 1] * t)
            r = float(radii[i])
            # Occluder band: vertical sweeping rectangle.
            if occ_active and occ_x - 4 <= cx <= occ_x + occ_width + 4:
                continue
            if gen._draw_object(img, cx, cy, r, int(classes[i]), rng):
                labels.append((int(classes[i]), cx / size, cy / size, 2 * r / size, 2 * r / size))
        if occ_active:
            x0 = int(max(0, occ_x))
            x1 = int(min(size, occ_x + occ_width))
            img[:, x0:x1] = np.clip(img[:, x0:x1].astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
        frames.append((img, labels))
    return frames
