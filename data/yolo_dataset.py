"""YOLO-format dataset with letterboxing and sim-to-real augmentations.

* Letterbox resize (preserve aspect, pad to square) — boxes transformed with
  the same affine, so aspect-distortion never masquerades as domain gap.
* ``build_augmentations`` returns the Phase-3 (d) domain-adaptation pipeline:
  flips, brightness/contrast (gamma-ish), HSV shifts, motion blur, Gaussian
  noise, JPEG compression and fog/haze — mild by default because aggressive
  warps destroy 8–32 px targets.
* ``class_map`` lets real datasets collapse into a class-agnostic "object"
  class (VisDrone has no drone/bird categories; see DATASET_CARD.md).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:  # pragma: no cover
    A = None

__all__ = ["YoloDetDataset", "letterbox", "build_augmentations"]


def letterbox(img: np.ndarray, size: int, boxes_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize preserving aspect and pad to ``size x size``; transform boxes.

    Returns (letterboxed image, boxes in letterboxed pixel coords, clamped).
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized

    if len(boxes_xyxy):
        out = boxes_xyxy.astype(np.float64).copy()
        out[:, [0, 2]] = out[:, [0, 2]] * scale + left
        out[:, [1, 3]] = out[:, [1, 3]] * scale + top
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, size)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, size)
    else:
        out = boxes_xyxy
    return canvas, out


def build_augmentations(p: float = 0.5) -> "A.Compose":
    """Domain-adaptation augmentations (Phase-3 variant d).

    Targets the synthetic→real gap: sensor noise, compression artifacts,
    lighting/gamma shifts, haze, motion blur, mirrored scenes. Geometry is
    limited to flips — 1-px warps matter at 8–32 px target sizes.
    """
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=25, p=0.5),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
            A.ImageCompression(quality_range=(40, 85), p=0.4),
            A.RandomFog(fog_coef_range=(0.15, 0.45), p=0.2),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.0),
        p=1.0,
    )


class YoloDetDataset(Dataset):
    """Images + YOLO labels for detection training/eval.

    Args:
        images_dir / labels_dir: SkySentry split layout
            (``<split>/{images,labels}`` — pass the split dir's children).
        img_size: square letterbox size.
        augment: an albumentations compose (training variant d only).
        class_map: optional remap of label class ids (e.g. VisDrone's 10
            classes → single 0 for class-agnostic transfer eval). Applied
            before augmentations so boxes keep their labels.
        max_images: cap for smoke runs (deterministic: first N sorted).
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        img_size: int = 640,
        augment: "A.Compose | None" = None,
        class_map: dict[int, int] | None = None,
        max_images: int | None = None,
    ) -> None:
        self.images = sorted(Path(images_dir).iterdir())
        self.images = [p for p in self.images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if max_images is not None:
            self.images = self.images[:max_images]
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        self.class_map = class_map or {}

    def __len__(self) -> int:
        return len(self.images)

    def _load_labels(self, stem: str, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """Parse one YOLO label file → (boxes xyxy pixels, class ids)."""
        path = self.labels_dir / f"{stem}.txt"
        if not path.exists():
            return np.zeros((0, 4)), np.zeros(0, dtype=np.int64)
        boxes, classes = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            if x2 - x1 < 0.5 or y2 - y1 < 0.5:
                continue
            boxes.append([x1, y1, x2, y2])
            classes.append(self.class_map.get(cls_id, cls_id))
        return np.asarray(boxes, dtype=np.float64).reshape(-1, 4), np.asarray(classes, dtype=np.int64)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        boxes, classes = self._load_labels(img_path.stem, *img.shape[:2])

        if self.augment is not None and len(boxes):
            augmented = self.augment(image=img, bboxes=boxes, class_labels=classes.tolist())
            img, boxes, classes = augmented["image"], np.asarray(augmented["bboxes"]).reshape(-1, 4), np.asarray(augmented["class_labels"], dtype=np.int64)

        img, boxes = letterbox(img, self.img_size, boxes)
        targets = np.concatenate([classes[:, None].astype(np.float32), boxes.astype(np.float32)], axis=1) \
            if len(boxes) else np.zeros((0, 5), dtype=np.float32)
        tensor = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        return tensor, {"boxes": targets}
