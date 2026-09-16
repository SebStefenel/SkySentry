"""Dataset preparation CLI: reproducible sample splits for Phase-1 testing.

Sources
-------
* ``synthetic`` (default) — renders a seeded sample dataset with
  :mod:`data.synthetic_composer`. Always available, byte-reproducible, and
  sufficient for pipeline verification and tracking smoke tests.
* ``visdrone`` — VisDrone2019-DET (aerial drone scenes, 10 classes). The
  official distribution requires registration at
  https://github.com/VisDrone/VisDrone-Dataset; this script downloads from a
  *user-provided* mirror URL (``--url``), verifies it (SHA-256 via
  ``--sha256``), unpacks it and converts the annotation CSVs to YOLO format.
* ``drone-vs-bird`` — Drone-vs-Bird (https://wosodeteam.wixsite.com/wosodeteam/drone-bird)
  similarly requires manual registration; pass ``--url`` to a local/remote zip.

Output layout (YOLO convention)::

    data/raw/<dataset>/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt        # class_id cx cy w h  (normalized)
        manifest.json                   # provenance: source, seed, counts, hashes

Usage::

    python -m data.dataset_downloader --dataset synthetic --out data/raw --num-train 64 --num-val 16
    python -m data.dataset_downloader --dataset visdrone --out data/raw --url <zip-url> [--sha256 <hex>]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

__all__ = ["main", "sha256_of", "convert_visdrone_split"]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# VisDrone class ids (1-based, 0 = ignored regions) → SkySentry class map.
# Project focus: drone(0)... we collapse VisDrone's {pedestrian, people,
# bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor} into
# 'clutter'(2) and 'drone' is absent in DET (it is the *camera* platform) —
# DET is used as a *real-domain tiny-object* benchmark, so keep vehicles as
# clutter and pedestrians as 'bird'-scale small objects is wrong; we keep the
# standard 10 classes and let configs map them per experiment.
VISDRONE_CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
                    "tricycle", "awning-tricycle", "bus", "motor"]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a file through SHA-256 (archive integrity check)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` with a progress hook."""
    print(f"downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def hook(blocks: int, bs: int, total: int) -> None:
        done = min(blocks * bs, total)
        if total > 0:
            sys.stdout.write(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()
    return dest


def convert_visdrone_split(ann_dir: Path, out_label_dir: Path, class_map: dict[int, int] | None = None) -> int:
    """Convert VisDrone annotation txt files to YOLO label txt files.

    VisDrone DET format (comma-separated, one object per line)::

        <frame_idx>,<target_id>,<x>,<y>,<w>,<h>,<score>,<category>,<truncation>,<occlusion>

    Region categories 0 (ignored) and 11 (others) are dropped; ``score`` 0
    (unlabeled/ambiguous) is dropped. Image sizes are parsed from the paired
    ``.jpg`` in ``images/`` if present; VisDrone ships sizes in a meta file,
    but for robustness we read dimensions with OpenCV.

    Returns the number of converted label files.
    """
    import cv2  # local import: only needed for this conversion path

    out_label_dir.mkdir(parents=True, exist_ok=True)
    class_map = class_map or {i + 1: i for i in range(10)}  # 1..10 → 0..9
    count = 0
    for ann_file in sorted(ann_dir.glob("*.txt")):
        img_path = ann_file.with_suffix(".jpg")
        if not img_path.exists():
            for ext in (".png", ".jpeg"):
                cand = ann_file.with_suffix(ext)
                if cand.exists():
                    img_path = cand
                    break
        h_img, w_img = cv2.imread(str(img_path)).shape[:2] if img_path.exists() else (0, 0)
        lines_out = []
        with open(ann_file, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 8:
                    continue
                x, y, w, h = (int(row[2]), int(row[3]), int(row[4]), int(row[5]))
                score, category = int(row[6]), int(row[7])
                if score == 0 or category in (0, 11) or category not in class_map:
                    continue
                if w_img <= 0 or h_img <= 0 or w <= 0 or h <= 0:
                    continue
                cls_id = class_map[category]
                cx, cy = (x + w / 2) / w_img, (y + h / 2) / h_img
                lines_out.append(f"{cls_id} {cx:.6f} {cy:.6f} {w / w_img:.6f} {h / h_img:.6f}")
        with open(out_label_dir / ann_file.name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out) + ("\n" if lines_out else ""))
        count += 1
    return count


def _prepare_visdrone(zip_path: Path, out_dir: Path) -> dict:
    """Unpack a VisDrone DET archive and convert both standard splits."""
    extract_dir = out_dir / "visdrone_extracted"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    # The zips usually contain VisDrone2019-DET-{train,val}/annotations + images.
    manifest = {"source": "visdrone2019-det", "splits": {}}
    for split in ("train", "val", "test-dev"):
        root = next(extract_dir.rglob(f"*{split}"), None)
        if root is None or not root.is_dir():
            continue
        img_src = next(root.rglob("images"), None)
        ann_src = next(root.rglob("annotations"), None)
        if img_src is None or ann_src is None:
            continue
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        n = convert_visdrone_split(ann_src, out_dir / "labels" / split)
        for img in img_src.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy(img, out_dir / "images" / split / img.name)
        manifest["splits"][split] = {"images": len(list((out_dir / 'images' / split).iterdir())), "labels": n}
    return manifest


def _prepare_synthetic(out_dir: Path, num_train: int, num_val: int, seed: int) -> dict:
    """Render the seeded synthetic sample (default reproducible split)."""
    from data.synthetic_composer import SyntheticAerialGenerator, SyntheticConfig

    cfg = SyntheticConfig(seed=seed)
    gen = SyntheticAerialGenerator(cfg)
    dataset_dir = out_dir / "synthetic"
    gen.write_dataset(dataset_dir, num_train, split="train")
    # Val uses a different seed offset so it never overlaps train.
    gen_val = SyntheticAerialGenerator(SyntheticConfig(seed=seed + 100_000))
    gen_val.write_dataset(dataset_dir, num_val, split="val")
    with open(dataset_dir / "classes.txt", "w", encoding="utf-8") as f:
        f.write("drone\nbird\nclutter\n")
    return {
        "source": "synthetic",
        "seed": seed,
        "num_train": num_train,
        "num_val": num_val,
        "classes": ["drone", "bird", "clutter"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SkySentry dataset preparation")
    parser.add_argument("--dataset", choices=["synthetic", "visdrone", "drone-vs-bird"], default="synthetic")
    parser.add_argument("--out", default="data/raw", help="output root (default: data/raw)")
    parser.add_argument("--num-train", type=int, default=64, help="synthetic train images")
    parser.add_argument("--num-val", type=int, default=16, help="synthetic val images")
    parser.add_argument("--seed", type=int, default=0, help="synthetic generator seed")
    parser.add_argument("--url", default=None, help="archive URL for visdrone / drone-vs-bird")
    parser.add_argument("--sha256", default=None, help="expected archive SHA-256 (integrity check)")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.dataset == "synthetic":
        manifest = _prepare_synthetic(out_dir, args.num_train, args.num_val, args.seed)
    else:
        if not args.url:
            raise SystemExit(
                f"{args.dataset} requires manual registration; pass --url to a downloaded/mirrored "
                f"archive (see module docstring for the official source)."
            )
        archive = out_dir / f"{args.dataset}.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            print(f"reusing cached archive {archive}")
        else:
            _download(args.url, archive)
        digest = sha256_of(archive)
        if args.sha256 and digest != args.sha256:
            raise SystemExit(f"SHA-256 mismatch: got {digest}, expected {args.sha256}")
        if args.dataset == "visdrone":
            manifest = _prepare_visdrone(archive, out_dir)
            manifest["sha256"] = digest
        else:
            raise SystemExit("drone-vs-bird: unpack manually and point configs/data.yaml at it "
                             "(video clips + MOT-style annotations; conversion lands in Phase 2).")

    manifest_path = out_dir / f"{args.dataset}" / "manifest.json" if args.dataset == "synthetic" else out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
