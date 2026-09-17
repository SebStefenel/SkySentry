"""Dataset preparation CLI: reproducible sample splits for testing.

Sources
-------
* ``synthetic`` (default) — renders a seeded sample dataset with
  :mod:`data.synthetic_composer`. Always available, byte-reproducible.
* ``visdrone`` — VisDrone2019-DET. The official distribution requires
  registration (https://github.com/VisDrone/VisDrone-Dataset); this script
  accepts either a user-provided archive (``--url`` or an already-downloaded
  ``--zip`` path) and verifies it via SHA-256. A known scriptable mirror is
  the YOLOv5 v1.0 GitHub release asset — used for the Phase-2 real subset.

Splits & leakage
----------------
Real data is split **by sequence, never by frame**: images whose filename
prefix (``uav0000013_...`` → sequence ``uav0000013``) falls in the same group
always land in the same split, so val/test never share a recording. The
assignment is seeded and stored in the manifest.

Outputs (YOLO convention)::

    data/raw/<dataset>/
        images/{split}/*.jpg
        labels/{split}/*.txt          # class_id cx cy w h (normalized)
        classes.txt
        manifest.json                 # provenance: source, seed, counts, hashes
        DATASET_CARD.md               # source, license, class map, stats, histogram
        size_histogram.png            # GT box-size distribution (real splits)

Usage::

    python -m data.dataset_downloader --dataset synthetic --out data/raw --num-train 64 --num-val 16
    python -m data.dataset_downloader --dataset visdrone --out data/raw --zip path/to/VisDrone2019-DET-val.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

__all__ = ["main", "sha256_of", "convert_visdrone_split", "sequence_of", "split_sequences"]

# VisDrone DET category ids are 1-based; 0 = ignored regions, 11 = others.
# Identity mapping 1..10 → 0..9 keeps the original names.
VISDRONE_CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
                    "tricycle", "awning-tricycle", "bus", "motor"]
VISDRONE_MIRROR = "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip"


@dataclass
class SizeRecord:
    """One GT instance's absolute pixel size, for the dataset card histogram."""

    width_px: float
    height_px: float
    class_id: int
    sequence: str


@dataclass
class SplitStats:
    images: int = 0
    instances: int = 0
    sequences: list[str] = field(default_factory=list)
    sizes: list[SizeRecord] = field(default_factory=list)


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


def sequence_of(stem: str) -> str:
    """Sequence key of a VisDrone frame: ``uav0000013_00000_d_0000001`` → ``uav0000013``."""
    return stem.split("_")[0]


def split_sequences(sequences: list[str], seed: int = 0, val_frac: float = 0.7) -> tuple[list[str], list[str]]:
    """Deterministically partition sequences into (val, test).

    Whole sequences go to one side only, so val/test never share a recording
    (no frame-level leakage). Deterministic for a given seed.
    """
    seqs = sorted(set(sequences))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(seqs))
    n_val = max(1, int(round(len(seqs) * val_frac)))
    val = [seqs[i] for i in perm[:n_val]]
    test = [seqs[i] for i in perm[n_val:]]
    return val, test


def convert_visdrone_split(
    ann_dir: Path,
    out_label_dir: Path,
    class_map: dict[int, int] | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[int, list[SizeRecord]]:
    """Convert VisDrone annotation txt files to YOLO label txt files.

    VisDrone DET format (comma-separated, one object per line)::

        <frame_idx>,<target_id>,<x>,<y>,<w>,<h>,<score>,<category>,<truncation>,<occlusion>

    Region categories 0 (ignored) and 11 (others) are dropped; ``score`` 0
    (ambiguous) is dropped. Image sizes come from the paired image file; pass
    ``image_size=(w, h)`` to skip disk reads (all frames equal, or tests).

    Returns (number of label files written, per-instance size records).
    """
    out_label_dir.mkdir(parents=True, exist_ok=True)
    class_map = class_map or {i + 1: i for i in range(10)}  # 1..10 → 0..9
    count = 0
    records: list[SizeRecord] = []
    for ann_file in sorted(ann_dir.glob("*.txt")):
        stem = ann_file.stem
        w_img = h_img = 0
        if image_size is not None:
            w_img, h_img = image_size
        else:
            import cv2  # local import: only needed when sizes are not provided

            img_path = ann_file.with_suffix(".jpg")
            for ext in (".png", ".jpeg"):
                if not img_path.exists():
                    img_path = ann_file.with_suffix(ext)
            img = cv2.imread(str(img_path))
            if img is not None:
                h_img, w_img = img.shape[:2]
        lines_out = []
        for row in csv_rows(ann_file):
            parsed = parse_visdrone_row(row)
            if parsed is None:
                continue
            x, y, w, h, score, category = parsed
            if score == 0 or category in (0, 11) or category not in class_map:
                continue
            if w_img <= 0 or h_img <= 0 or w <= 0 or h <= 0:
                continue
            cls_id = class_map[category]
            cx, cy = (x + w / 2) / w_img, (y + h / 2) / h_img
            lines_out.append(f"{cls_id} {cx:.6f} {cy:.6f} {w / w_img:.6f} {h / h_img:.6f}")
            records.append(SizeRecord(float(w), float(h), cls_id, sequence_of(stem)))
        with open(out_label_dir / ann_file.name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out) + ("\n" if lines_out else ""))
        count += 1
    return count, records


def csv_rows(ann_file: Path):
    """Iterate comma-separated rows of an annotation file."""
    with open(ann_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line.split(",")


def parse_visdrone_row(row: list[str]) -> tuple[int, int, int, int, int, int] | None:
    """Parse one annotation row, handling both DET layouts.

    * val GT (8 fields): ``x, y, w, h, score, category, truncation, occlusion``
    * train GT (10 fields): ``frame, target_id, x, y, w, h, score, category, ...``

    Returns ``(x, y, w, h, score, category)`` or None for malformed rows.
    """
    try:
        vals = [int(v) for v in row]
    except ValueError:
        return None
    if len(vals) >= 10:
        x, y, w, h, score, category = vals[2:8]
    elif len(vals) == 8:
        x, y, w, h, score, category = vals[0:6]
    else:
        return None
    return x, y, w, h, score, category


def _plot_size_histogram(records: list[SizeRecord], out_png: Path) -> None:
    """Box-size histogram (longer side, log-ish bins) for the dataset card."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("[warn] matplotlib missing — skipping size histogram")
        return
    sides = np.array([max(r.width_px, r.height_px) for r in records])
    fig, ax = plt.subplots(figsize=(5, 3), dpi=130)
    ax.hist(sides, bins=np.arange(0, 260, 10), color="#4878a8", edgecolor="white")
    ax.axvline(32, color="crimson", ls="--", lw=1, label="COCO small (<32 px)")
    ax.axvline(96, color="darkorange", ls="--", lw=1, label="COCO large (>=96 px)")
    ax.set_xlabel("longer GT side (px)")
    ax.set_ylabel("instances")
    ax.set_title("GT box sizes")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _write_dataset_card(out_dir: Path, stats: dict[str, SplitStats], sha: str, seed: int) -> None:
    tiny = {
        split: (sum(1 for r in s.sizes if max(r.width_px, r.height_px) < 32), len(s.sizes))
        for split, s in stats.items()
    }
    lines = [
        "# VisDrone2019-DET (subset) — dataset card",
        "",
        f"- Retrieved: {date.today().isoformat()}",
        f"- Archive SHA-256: `{sha}`",
        f"- Split seed: {seed} (sequences partitioned with numpy default_rng)",
        f"- Mirror used: {VISDRONE_MIRROR}",
        "- Official source (registration): https://github.com/VisDrone/VisDrone-Dataset",
        "- License: made available by the VisDrone organizers for academic research;",
        "  non-commercial use. Verify the terms in the official repo before any redistribution.",
        "",
        "## Class mapping (identity, DET 1-based → YOLO 0-based)",
        "",
        "| YOLO id | VisDrone category |",
        "|---|---|",
    ]
    lines += [f"| {i} | {name} |" for i, name in enumerate(VISDRONE_CLASSES)]
    lines += [
        "",
        "Note: DET has no drone/bird categories (the UAV is the camera platform).",
        "For this project VisDrone serves as the *real-domain tiny-object* benchmark",
        "(vehicles/pedestrians); drone-vs-bird class-space alignment comes from the",
        "Drone-vs-Bird dataset (manual download — see README roadmap).",
        "",
        "## Splits (by sequence — no frame leakage)",
        "",
        "| split | images | instances | sequences | % instances < 32 px |",
        "|---|---|---|---|---|",
    ]
    for split, s in stats.items():
        n_tiny, n_all = tiny[split]
        pct = f"{100.0 * n_tiny / n_all:.1f}" if n_all else "—"
        lines.append(f"| {split} | {s.images} | {s.instances} | {len(s.sequences)} | {pct} |")
    lines += [
        "",
        "Sequences: " + "; ".join(f"{sp}: {', '.join(sorted(s.sequences))}" for sp, s in stats.items()),
        "",
        "See `size_histogram.png` for the GT size distribution.",
        "",
    ]
    (out_dir / "DATASET_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def _prepare_visdrone(zip_path: Path, out_dir: Path, seed: int = 0) -> dict:
    """Unpack a VisDrone DET archive, convert to YOLO, split by sequence.

    Produces ``images/real-val``, ``images/real-test`` (+labels), a dataset
    card, a size histogram and a manifest with full provenance.
    """
    import cv2

    extract_dir = out_dir / "visdrone_extracted"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

    # Gather all archive images into one pool and cache their sizes.
    pool = out_dir / "visdrone"
    if pool.exists():
        shutil.rmtree(pool)
    (pool / "images").mkdir(parents=True)
    for split_name in ("train", "val", "test-dev"):
        root = next(extract_dir.rglob(f"*{split_name}"), None)
        img_src = next(root.rglob("images"), None) if root else None
        if img_src is None:
            continue
        for img_file in sorted(img_src.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy(img_file, pool / "images" / img_file.name)

    all_img = sorted((pool / "images").iterdir())
    sizes = {}
    for img_file in all_img:
        im = cv2.imread(str(img_file))
        if im is not None:
            sizes[img_file.stem] = (im.shape[1], im.shape[0])

    # Sequence-partition the pool into real-val / real-test.
    seqs = sorted({sequence_of(p.stem) for p in all_img})
    val_seqs, test_seqs = split_sequences(seqs, seed=seed)
    stats: dict[str, SplitStats] = {"real-val": SplitStats(), "real-test": SplitStats()}
    seq_to_split = {s: "real-val" for s in val_seqs}
    seq_to_split.update({s: "real-test" for s in test_seqs})

    # Convert each annotation file once, routed by its image's sequence.
    # Layout matches the synthetic dataset: <root>/<split>/{images,labels}.
    for split_name in ("train", "val", "test-dev"):
        root = next(extract_dir.rglob(f"*{split_name}"), None)
        ann_src = next(root.rglob("annotations"), None) if root else None
        if ann_src is None:
            continue
        for ann_file in sorted(ann_src.glob("*.txt")):
            split = seq_to_split.get(sequence_of(ann_file.stem))
            if split is None:
                continue
            dest = pool / split / "labels"
            dest.mkdir(parents=True, exist_ok=True)
            _, recs = convert_visdrone_split_one(ann_file, dest, image_size=sizes.get(ann_file.stem))
            stats[split].sizes += recs

    # Move images into per-split dirs and finalize stats.
    for img_file in all_img:
        split = seq_to_split.get(sequence_of(img_file.stem))
        if split is None:
            continue
        dest = pool / split / "images"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_file), dest / img_file.name)
        stats[split].images += 1
        stats[split].sequences.append(sequence_of(img_file.stem))
    for s in stats.values():
        s.instances = len(s.sizes)
        s.sequences = sorted(set(s.sequences))

    sha = sha256_of(zip_path)
    records = stats["real-val"].sizes + stats["real-test"].sizes
    _plot_size_histogram(records, out_dir / "size_histogram.png")
    _write_dataset_card(pool, stats, sha, seed)
    (pool / "classes.txt").write_text("\n".join(VISDRONE_CLASSES) + "\n", encoding="utf-8")

    manifest = {
        "source": "visdrone2019-det",
        "mirror": VISDRONE_MIRROR,
        "sha256": sha,
        "split_seed": seed,
        "leakage_control": "split by sequence (filename prefix), not by frame",
        "classes": VISDRONE_CLASSES,
        "splits": {
            sp: {"images": s.images, "instances": s.instances, "sequences": sorted(s.sequences)}
            for sp, s in stats.items()
        },
        "card": str((pool / "DATASET_CARD.md").relative_to(out_dir)),
    }
    (out_dir / "visdrone" / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "visdrone" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def convert_visdrone_split_one(
    ann_file: Path,
    out_label_dir: Path,
    class_map: dict[int, int] | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[int, list[SizeRecord]]:
    """Convert a single VisDrone annotation file to one YOLO label file."""
    out_label_dir.mkdir(parents=True, exist_ok=True)
    class_map = class_map or {i + 1: i for i in range(10)}
    stem = ann_file.stem
    w_img, h_img = image_size if image_size else (0, 0)
    records: list[SizeRecord] = []
    lines_out = []
    for row in csv_rows(ann_file):
        parsed = parse_visdrone_row(row)
        if parsed is None:
            continue
        x, y, w, h, score, category = parsed
        if score == 0 or category in (0, 11) or category not in class_map:
            continue
        if w_img <= 0 or h_img <= 0 or w <= 0 or h <= 0:
            continue
        cls_id = class_map[category]
        cx, cy = (x + w / 2) / w_img, (y + h / 2) / h_img
        lines_out.append(f"{cls_id} {cx:.6f} {cy:.6f} {w / w_img:.6f} {h / h_img:.6f}")
        records.append(SizeRecord(float(w), float(h), cls_id, sequence_of(stem)))
    (out_label_dir / ann_file.name).write_text(
        "\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8"
    )
    return 1, records


def _prepare_synthetic(out_dir: Path, num_train: int, num_val: int, seed: int) -> dict:
    """Render the seeded synthetic sample (default reproducible split)."""
    from data.synthetic_composer import SyntheticAerialGenerator, SyntheticConfig

    dataset_dir = out_dir / "synthetic"
    gen = SyntheticAerialGenerator(SyntheticConfig(seed=seed))
    gen.write_dataset(dataset_dir, num_train, split="train")
    # Val uses a different seed offset so it never overlaps train.
    gen_val = SyntheticAerialGenerator(SyntheticConfig(seed=seed + 100_000))
    gen_val.write_dataset(dataset_dir, num_val, split="val")
    (dataset_dir / "classes.txt").write_text("drone\nbird\nclutter\n", encoding="utf-8")

    def count_split(split: str) -> dict:
        labels = sorted((dataset_dir / split / "labels").glob("*.txt"))
        instances = sum(len(l.read_text(encoding="utf-8").splitlines()) for l in labels)
        return {"images": len(labels), "instances": instances}

    return {
        "source": "synthetic",
        "seed": seed,
        "classes": ["drone", "bird", "clutter"],
        "splits": {"train": count_split("train"), "val": count_split("val")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SkySentry dataset preparation")
    parser.add_argument("--dataset", choices=["synthetic", "visdrone"], default="synthetic")
    parser.add_argument("--out", default="data/raw", help="output root (default: data/raw)")
    parser.add_argument("--num-train", type=int, default=64, help="synthetic train images")
    parser.add_argument("--num-val", type=int, default=16, help="synthetic val images")
    parser.add_argument("--seed", type=int, default=0, help="generator / split seed")
    parser.add_argument("--url", default=None, help="archive URL for visdrone")
    parser.add_argument("--zip", default=None, help="path to an already-downloaded archive")
    parser.add_argument("--sha256", default=None, help="expected archive SHA-256 (integrity check)")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.dataset == "synthetic":
        manifest = _prepare_synthetic(out_dir, args.num_train, args.num_val, args.seed)
        manifest_path = out_dir / "synthetic" / "manifest.json"
    else:
        archive = Path(args.zip) if args.zip else out_dir / "visdrone.zip"
        if not archive.exists():
            if not args.url:
                raise SystemExit(
                    "visdrone needs an archive: pass --zip <path> or --url <direct link>.\n"
                    f"A scriptable mirror is {VISDRONE_MIRROR}"
                )
            _download(args.url, archive)
        digest = sha256_of(archive)
        if args.sha256 and digest != args.sha256:
            raise SystemExit(f"SHA-256 mismatch: got {digest}, expected {args.sha256}")
        manifest = _prepare_visdrone(archive, out_dir, seed=args.seed)
        manifest_path = out_dir / "visdrone" / "manifest.json"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
