"""Visual QC: save a grid of dataset samples with GT boxes drawn.

New file (rather than extending the downloader) on purpose: this is a pure
rendering/QC utility reused across phases (dataset QC now, prediction QC in
Phase 3, track QC in Phase 4) — keeping it separate keeps the downloader a
pure acquisition/conversion CLI.

Reads any SkySentry dataset layout (``<root>/<split>/{images,labels}``), YOLO
labels with 5 columns (class cx cy w h) or 6 (track_id class cx cy w h), and
writes a matplotlib grid PNG.

Usage::

    python -m data.visualize_samples --root data/raw/synthetic --split train \
        --out results/sample_grid.png --num 16 --classes drone bird clutter
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed per-class colors (BGR for OpenCV, normalized RGB for matplotlib).
COLORS = [(0.20, 0.75, 0.30), (0.95, 0.60, 0.10), (0.30, 0.45, 0.95), (0.80, 0.20, 0.60)]


def draw_label(img: np.ndarray, row: list[float], color_bgr: tuple[int, int, int], tag: str) -> np.ndarray:
    """Draw one YOLO-normalized label row (class-first or trackid-first) on img."""
    cls_id, cx, cy, w, h = int(row[0]), row[1], row[2], row[3], row[4]
    s = img.shape[1], img.shape[0]
    x1, y1 = int((cx - w / 2) * s[0]), int((cy - h / 2) * s[1])
    x2, y2 = int((cx + w / 2) * s[0]), int((cy + h / 2) * s[1])
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 1)
    cv2.putText(img, tag, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color_bgr, 1)
    return img


def load_image_with_boxes(img_path: Path, label_path: Path, class_names: list[str]) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    if not label_path.exists():
        return img
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        row = [float(p) for p in parts[:5]]
        cls_id = int(row[0]) if len(parts) == 5 else int(row[1])
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        tag = f"{name}#{int(row[0])}" if len(parts) >= 6 else name
        color = tuple(int(c * 255) for c in COLORS[cls_id % len(COLORS)])
        # draw_label expects (cls, cx, cy, w, h): strip track id if present.
        draw_label(img, row[-5:], color, tag)
    return img


def make_grid(root: Path, split: str, out_path: Path, num: int = 16, class_names: list[str] | None = None) -> Path:
    """Render the first ``num`` images of a split with boxes into a grid PNG."""
    split_dir = root / split
    images = sorted((split_dir / "images").iterdir())[:num]
    if not images:
        raise FileNotFoundError(f"no images under {split_dir / 'images'}")
    class_names = class_names or [c.strip() for c in (root / "classes.txt").read_text().splitlines() if c.strip()]

    cols = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), dpi=110)
    axes = np.atleast_1d(axes).ravel()
    for ax, img_path in zip(axes, images):
        label_path = split_dir / "labels" / (img_path.stem + ".txt")
        img = load_image_with_boxes(img_path, label_path, class_names)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(img_path.stem, fontsize=7)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a QC grid of samples with GT boxes")
    parser.add_argument("--root", default="data/raw/synthetic")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="results/sample_grid.png")
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument("--classes", nargs="*", default=None, help="class names (default: classes.txt)")
    args = parser.parse_args()

    out = make_grid(Path(args.root), args.split, Path(args.out), args.num, args.classes)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
