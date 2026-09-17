"""Deterministic synthetic aerial-scene composer (the "sim" of sim-to-real).

Renders small-target scenes — drones, birds and background clutter — as
NumPy/OpenCV images with YOLO-format labels, fully controlled by a seed.
**No external assets**: drones are procedural quad-rotor sprites drawn with
OpenCV primitives (fuselage, arms, semi-transparent rotor discs), birds are
elongated bodies with wing strokes.

Per-frame effects: low-frequency terrain noise, clutter rectangles, global
lighting jitter (gain/bias/gamma) and localized sun glare.

Clips (:func:`generate_clip`) render constant-velocity objects with **ground-
truth track IDs** attached to every label, plus a sweeping occluder, so the
Kalman tracker can be scored against known identities.

Determinism: every image is generated from ``np.random.default_rng(seed + i)``
so any subset is reproducible across machines (same NumPy version).

Single-frame labels use YOLO format: ``class_id cx cy w h`` normalized to
[0, 1]. Clip labels prepend the track id: ``track_id class_id cx cy w h``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

__all__ = ["SyntheticConfig", "SyntheticAerialGenerator", "generate_clip"]

# class_id → (name, brightness mode). Colors are sampled per instance.
# 0 = drone (procedural quad-rotor sprite), 1 = bird (elongated, wings), 2 = clutter.
CLASSES = {0: "drone", 1: "bird", 2: "clutter"}


@dataclass
class SyntheticConfig:
    """Generation parameters. Defaults target VisDrone-like statistics."""

    size: int = 640                       # square image side
    min_objects: int = 1
    max_objects: int = 8
    object_size_px: tuple[float, float] = (8.0, 32.0)  # bbox max side, pixels
    p_class: tuple[float, float, float] = (0.45, 0.45, 0.10)  # drone/bird/clutter
    clutter_rects: tuple[int, int] = (2, 10)           # background rectangles
    motion_blur_prob: float = 0.35
    noise_sigma: float = 6.0
    lighting_jitter: bool = True         # gain/bias/gamma per image
    glare_prob: float = 0.4              # sun-glare blob probability
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
    # Procedural sprites                                                  #
    # ------------------------------------------------------------------ #

    def _sprite_canvas(self, size_px: float) -> tuple[int, float]:
        """Canvas side and center for a sprite of nominal bbox side ``size_px``."""
        side = int(np.ceil(size_px * 2.6)) + 2   # headroom for rotation
        return side, side / 2.0

    @staticmethod
    def _fit_to_size(color: np.ndarray, alpha: np.ndarray, size_px: float) -> tuple[np.ndarray, np.ndarray]:
        """Rescale a sprite so its rendered alpha extent equals ``size_px``.

        Drawn parts (rotor discs, wings) overshoot the nominal radius; this
        guarantees the 8–32 px spec is met exactly and labels match pixels.
        """
        ys, xs = np.nonzero(alpha > 0.12)
        if len(xs) == 0:
            return color, alpha
        extent = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
        scale = size_px / extent
        if abs(scale - 1.0) < 0.02:
            return color, alpha
        side = max(2, int(round(alpha.shape[0] * scale)))
        return cv2.resize(color, (side, side)), cv2.resize(alpha, (side, side))

    def _drone_sprite(self, rng: np.random.Generator, size_px: float) -> tuple[np.ndarray, np.ndarray]:
        """Quad-rotor drone: dark fuselage, 4 arms, semi-transparent rotor discs.

        Returns (bgr float32 canvas, alpha float32 canvas in [0, 1]).
        """
        s, c = self._sprite_canvas(size_px)
        r = size_px / 2.0
        alpha = np.zeros((s, s), np.float32)
        color = np.zeros((s, s, 3), np.float32)

        def put(mask: np.ndarray, part_alpha: float, rgb) -> None:
            m = mask > 0
            alpha[m] = np.maximum(alpha[m], part_alpha)
            color[m] = np.asarray(rgb, dtype=np.float32)

        body = rng.integers(25, 70, 3)                       # dark fuselage
        arm_len, arm_w = 1.5 * r, max(1.0, 0.14 * r)
        rotor_r = 0.62 * r
        body_angle = rng.uniform(0, 180)
        for ang in (45.0, 135.0, 225.0, 315.0):
            a = np.deg2rad(ang + body_angle)
            tip = (c + np.cos(a) * arm_len, c + np.sin(a) * arm_len)
            arm = np.zeros_like(alpha)
            cv2.line(arm, (int(c), int(c)), (int(tip[0]), int(tip[1])), 1.0, int(arm_w), lineType=cv2.LINE_AA)
            put(arm, 1.0, body * 0.85)
            disc = np.zeros_like(alpha)
            cv2.circle(disc, (int(tip[0]), int(tip[1])), max(1, int(rotor_r)), 1.0, -1, lineType=cv2.LINE_AA)
            put(disc, 0.5, rng.integers(150, 210, 3))        # blurred rotor disc
            hub = np.zeros_like(alpha)
            cv2.circle(hub, (int(tip[0]), int(tip[1])), max(1, int(0.16 * r)), 1.0, -1, lineType=cv2.LINE_AA)
            put(hub, 1.0, (230, 230, 230))
        fuselage = np.zeros_like(alpha)
        cv2.ellipse(fuselage, (int(c), int(c)), (max(1, int(0.85 * r)), max(1, int(0.45 * r))),
                    body_angle, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
        put(fuselage, 1.0, body)
        tail = np.zeros_like(alpha)
        cv2.circle(tail, (int(c - np.cos(np.deg2rad(body_angle)) * 0.8 * r),
                          int(c - np.sin(np.deg2rad(body_angle)) * 0.8 * r)),
                   max(1, int(0.18 * r)), 1.0, -1, lineType=cv2.LINE_AA)
        put(tail, 1.0, (200, 200, 210))

        # Rotate the whole sprite.
        m = cv2.getRotationMatrix2D((c, c), rng.uniform(0, 180), 1.0)
        alpha = cv2.warpAffine(alpha, m, (s, s), flags=cv2.INTER_LINEAR, borderValue=0.0)
        color = cv2.warpAffine(color, m, (s, s), flags=cv2.INTER_LINEAR, borderValue=0.0)
        return self._fit_to_size(color, alpha, size_px)

    def _bird_sprite(self, rng: np.random.Generator, size_px: float) -> tuple[np.ndarray, np.ndarray]:
        """Bird: elongated dark body + two swept wing strokes."""
        s, c = self._sprite_canvas(size_px)
        r = size_px / 2.0
        alpha = np.zeros((s, s), np.float32)
        color = np.zeros((s, s, 3), np.float32)

        def put(mask: np.ndarray, part_alpha: float, rgb) -> None:
            m = mask > 0
            alpha[m] = np.maximum(alpha[m], part_alpha)
            color[m] = np.asarray(rgb, dtype=np.float32)

        body_col = rng.integers(30, 75, 3)
        angle = rng.uniform(0, 180)
        body = np.zeros_like(alpha)
        cv2.ellipse(body, (int(c), int(c)), (max(1, int(1.0 * r)), max(1, int(0.32 * r))),
                    angle, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
        put(body, 1.0, body_col)
        wing_dir = rng.choice([-1.0, 1.0])
        for sweep in (-1, 1):
            wing = np.zeros_like(alpha)
            cv2.ellipse(wing, (int(c), int(c)),
                        (max(1, int(0.95 * r)), max(1, int(0.55 * r))),
                        angle + sweep * 55 * wing_dir, 200 if sweep > 0 else 20,
                        340 if sweep > 0 else 160, 1.0, max(1, int(0.16 * r)), lineType=cv2.LINE_AA)
            put(wing, 0.9, body_col * 0.9)

        m = cv2.getRotationMatrix2D((c, c), 0.0, 1.0)  # angle already baked in
        return self._fit_to_size(color, alpha, size_px)

    def _clutter_sprite(self, rng: np.random.Generator, size_px: float) -> tuple[np.ndarray, np.ndarray]:
        """Background clutter: dull irregular polygon blob (rooftop/vehicle-ish)."""
        s, c = self._sprite_canvas(size_px)
        r = size_px / 2.0
        alpha = np.zeros((s, s), np.float32)
        color = np.zeros((s, s, 3), np.float32)
        n = int(rng.integers(5, 8))
        pts = [
            [int(c + np.cos(2 * np.pi * k / n + rng.uniform(-0.2, 0.2)) * r * rng.uniform(0.5, 1.0)),
             int(c + np.sin(2 * np.pi * k / n + rng.uniform(-0.2, 0.2)) * r * rng.uniform(0.5, 1.0))]
            for k in range(n)
        ]
        cv2.fillPoly(alpha, np.array([pts], dtype=np.int32), 1.0)
        color[alpha > 0] = rng.integers(90, 170, 3).astype(np.float32)
        return self._fit_to_size(color, alpha, size_px)

    def _composite_sprite(
        self,
        img: np.ndarray,
        color: np.ndarray,
        alpha: np.ndarray,
        cx: float,
        cy: float,
    ) -> tuple[float, float, float, float] | None:
        """Alpha-composite a sprite centered at (cx, cy); returns its YOLO label.

        The label bbox is the *rendered* alpha extent, so labels always match
        the pixels exactly. Returns None if the sprite would leave the canvas.
        """
        s_img = img.shape[0]
        h, w = alpha.shape
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        x1, y1 = x0 + w, y0 + h
        if x0 < 0 or y0 < 0 or x1 > s_img or y1 > s_img:
            return None
        region = img[y0:y1, x0:x1].astype(np.float32)
        a = alpha[..., None]
        img[y0:y1, x0:x1] = np.clip(region * (1 - a) + color * a, 0, 255).astype(np.uint8)
        ys, xs = np.nonzero(alpha > 0.12)
        bw = float(xs.max() - xs.min() + 1)
        bh = float(ys.max() - ys.min() + 1)
        bcx = x0 + (xs.max() + xs.min() + 1) / 2.0
        bcy = y0 + (ys.max() + ys.min() + 1) / 2.0
        return bcx / s_img, bcy / s_img, bw / s_img, bh / s_img

    def _apply_motion_blur(self, img: np.ndarray, cx: float, cy: float, r: float, k: int) -> None:
        """Horizontal streak over the sprite's neighborhood (fast movers)."""
        kernel = np.zeros((1, k), dtype=np.float32)
        kernel[0, :] = 1.0 / k
        x0 = max(0, int(cx) - int(2 * r) - k)
        x1 = int(cx) + int(2 * r) + k
        y0, y1 = max(0, int(cy) - 6), int(cy) + 7
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = cv2.filter2D(img[y0:y1, x0:x1], -1, kernel)

    # ------------------------------------------------------------------ #
    # Global photometric effects                                          #
    # ------------------------------------------------------------------ #

    def _lighting(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Per-image gain/bias/gamma jitter (randomized lighting)."""
        if not self.cfg.lighting_jitter:
            return img
        gain = rng.uniform(0.8, 1.2)
        bias = rng.uniform(-18.0, 18.0)
        gamma = rng.uniform(0.8, 1.25)
        out = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        lut = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0).astype(np.uint8)
        return cv2.LUT(out, lut)

    def _glare(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Localized sun-glare blob: additive radial falloff from a random sun position."""
        s = self.cfg.size
        cx, cy = rng.uniform(0, s), rng.uniform(0, s * 0.5)
        rad = rng.uniform(0.2, 0.45) * s
        strength = rng.uniform(35.0, 90.0)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad
        falloff = np.clip(1.0 - d, 0.0, 1.0) ** 2
        out = img.astype(np.float32) + strength * falloff[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_one(self, index: int) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
        """Render scene ``index``; returns (image BGR, YOLO-normalized labels)."""
        rng = np.random.default_rng(self.cfg.seed + index)
        img = self._background(rng)
        s = self.cfg.size
        n_obj = int(rng.integers(self.cfg.min_objects, self.cfg.max_objects + 1))
        labels: list[tuple[int, float, float, float, float]] = []
        attempts = 0
        while len(labels) < n_obj and attempts < n_obj * 10:
            attempts += 1
            size_px = float(rng.uniform(*self.cfg.object_size_px))
            cx, cy = rng.uniform(size_px, s - size_px, size=2)
            cls_id = int(rng.choice(3, p=np.asarray(self.cfg.p_class) / np.sum(self.cfg.p_class)))
            if cls_id == 0:
                color, alpha = self._drone_sprite(rng, size_px)
            elif cls_id == 1:
                color, alpha = self._bird_sprite(rng, size_px)
            else:  # clutter: dull irregular blob
                color, alpha = self._clutter_sprite(rng, size_px)
            label = self._composite_sprite(img, color.astype(np.float32), alpha, cx, cy)
            if label is None:
                continue
            if rng.random() < self.cfg.motion_blur_prob:
                self._apply_motion_blur(img, cx, cy, size_px / 2, max(3, int(size_px / 3)))
            labels.append((cls_id, *label))
        img = self._lighting(img, rng)
        if rng.random() < self.cfg.glare_prob:
            img = self._glare(img, rng)
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
) -> list[tuple[np.ndarray, list[tuple[int, int, float, float, float, float]]]]:
    """Render a constant-velocity clip with ground-truth track IDs.

    Each object i keeps a stable ``track_id == i`` across frames, moves with a
    seeded constant velocity (the exact model the Kalman filter assumes), and
    disappears behind a sweeping occluder from 40% of the clip onward — the
    labels simply omit occluded objects on occluded frames, so a MOT scorer
    sees genuine identity gaps.

    Returns: list of (frame BGR, labels) where each label row is
    ``(track_id, class_id, cx, cy, w, h)`` normalized to [0, 1].
    """
    base = np.random.default_rng(seed)
    gen = SyntheticAerialGenerator(SyntheticConfig(
        size=size, motion_blur_prob=0.2, lighting_jitter=False, glare_prob=0.0,
        seed=seed * 1000, object_size_px=(8.0, 24.0),
    ))
    starts = base.uniform(0.12, 0.88, size=(num_objects, 2)) * size
    vel = base.uniform(-3.0, 3.0, size=(num_objects, 2))  # px/frame; slow aerial drift
    sizes = base.uniform(*gen.cfg.object_size_px, size=(num_objects,))
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
            size_px = float(sizes[i])
            if occ_active and occ_x - 4 <= cx <= occ_x + occ_width + 4:
                continue  # occluded: object present in the scene, absent from labels
            if int(classes[i]) == 0:
                color, alpha = gen._drone_sprite(rng, size_px)
            else:
                color, alpha = gen._bird_sprite(rng, size_px)
            label = gen._composite_sprite(img, color, alpha, cx, cy)
            if label is not None:
                labels.append((i, int(classes[i]), *label))
        if occ_active:
            x0 = int(max(0, occ_x))
            x1 = int(min(size, occ_x + occ_width))
            img[:, x0:x1] = np.clip(img[:, x0:x1].astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
        frames.append((img, labels))
    return frames
