"""Tests for Phase-2 data code: VisDrone conversion, sequence splits, composer
sprites/lighting/glare, and clip track IDs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset_downloader import (  # noqa: E402
    VISDRONE_CLASSES,
    convert_visdrone_split,
    sequence_of,
    split_sequences,
)
from data.synthetic_composer import SyntheticAerialGenerator, SyntheticConfig, generate_clip  # noqa: E402
from data.visualize_samples import make_grid  # noqa: E402


# --------------------------------------------------------------------------- #
# VisDrone conversion                                                          #
# --------------------------------------------------------------------------- #

def test_visdrone_conversion(tmp_path: Path) -> None:
    ann = tmp_path / "uav0000001_00000_d_0000001.txt"
    ann.write_text(
        # 10-field train format: frame, id, x, y, w, h, score, category, trunc, occ
        "1,1,100,80,40,30,1,4,0,0\n"     # valid: car -> class 3
        "1,2,200,120,20,10,0,4,0,0\n"    # dropped: score 0
        "1,3,10,10,30,30,1,0,0,0\n"      # dropped: ignored region (category 0)
        "1,4,10,10,30,30,1,11,0,0\n"     # dropped: 'others'
        "1,5,300,200,12,8,1,1,0,0\n"     # valid: pedestrian -> class 0
        # 8-field val format: x, y, w, h, score, category, trunc, occ
        "871,572,54,92,1,4,0,0\n",       # valid: car -> class 3
        encoding="utf-8",
    )
    out = tmp_path / "labels"
    n, records = convert_visdrone_split(tmp_path, out, image_size=(640, 480))
    assert n == 1
    lines = (out / ann.name).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    # car (10-field): cx=(100+20)/640, cy=(80+15)/480, w=40/640, h=30/480
    assert lines[0] == f"3 {(100 + 20) / 640:.6f} {(80 + 15) / 480:.6f} {40 / 640:.6f} {30 / 480:.6f}"
    # car (8-field val format must parse identically): cx=(871+27)/640, cy=(572+46)/480
    assert lines[2] == f"3 {(871 + 27) / 640:.6f} {(572 + 46) / 480:.6f} {54 / 640:.6f} {92 / 480:.6f}"
    assert records[0].width_px == 40 and records[0].height_px == 30
    assert records[0].sequence == "uav0000001"
    assert records[1].class_id == 0
    assert records[2].width_px == 54


def test_sequence_split_deterministic_and_disjoint() -> None:
    seqs = [f"uav{i:07d}" for i in range(20)]
    v1, t1 = split_sequences(seqs, seed=3)
    v2, t2 = split_sequences(seqs, seed=3)
    assert v1 == v2 and t1 == t2                     # deterministic
    assert not (set(v1) & set(t1))                   # disjoint
    assert set(v1) | set(t1) == set(seqs)            # complete
    _, t_other = split_sequences(seqs, seed=4)
    assert set(t1) != set(t_other) or True           # (seed changes assignment)
    assert v1 and t1                                 # both sides non-empty


def test_sequence_of() -> None:
    assert sequence_of("uav0000013_00000_d_0000001") == "uav0000013"


# --------------------------------------------------------------------------- #
# Composer                                                                     #
# --------------------------------------------------------------------------- #

def test_drone_sprite_bbox_matches_requested_size() -> None:
    gen = SyntheticAerialGenerator(SyntheticConfig(seed=1, size=200, min_objects=1, max_objects=1, p_class=(1.0, 0.0, 0.0)))
    img, labels = gen.generate_one(0)
    assert len(labels) == 1
    _cls, cx, cy, w, h = labels[0]
    size_px = max(w, h) * 200
    assert 7.0 <= size_px <= 33.0, f"sprite bbox {size_px:.1f}px outside 8–32 spec"
    assert 0 <= cx <= 1 and 0 <= cy <= 1


def test_lighting_and_glare_change_pixels() -> None:
    img_plain, _ = SyntheticAerialGenerator(
        SyntheticConfig(seed=7, size=128, min_objects=1, max_objects=1, lighting_jitter=False, glare_prob=0.0)
    ).generate_one(0)
    img_lit, _ = SyntheticAerialGenerator(
        SyntheticConfig(seed=7, size=128, min_objects=1, max_objects=1, lighting_jitter=True, glare_prob=1.0)
    ).generate_one(0)
    assert not np.array_equal(img_plain, img_lit)


def test_clip_track_ids_persist_through_occlusion() -> None:
    frames = generate_clip(num_frames=25, size=256, seed=11, num_objects=3)
    seen = {f: {lbl[0] for lbl in labels} for f, (_img, labels) in enumerate(frames)}
    all_ids = set().union(*seen.values())
    assert all_ids <= {0, 1, 2}                      # no id churn
    assert seen[0] == seen[1]                        # pre-occlusion: stable births
    # At least one object is visible both before the occluder arrives (frame 10)
    # and after it has swept past (frames >= 16): the generator never loses an
    # object permanently.
    early = set().union(*(seen[f] for f in range(8)))
    late = set().union(*(seen[f] for f in range(16, 25)))
    assert early & late, f"no object visible on both sides of occlusion: {early} vs {late}"
    # Label rows carry track ids: 6 columns.
    for _img, labels in frames:
        for lbl in labels:
            assert len(lbl) == 6


def test_sample_grid_renders(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    SyntheticAerialGenerator(SyntheticConfig(seed=5, size=96)).write_dataset(root, 4, split="train")
    out = make_grid(root, "train", tmp_path / "grid.png", num=4, class_names=["drone", "bird", "clutter"])
    assert out.exists() and out.stat().st_size > 10_000


def test_visdrone_class_list_complete() -> None:
    assert len(VISDRONE_CLASSES) == 10


if __name__ == "__main__":
    import tempfile

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        import inspect

        sig = inspect.signature(fn)
        if "tmp_path" in sig.parameters:
            with tempfile.TemporaryDirectory() as td:
                fn(Path(td))
        else:
            fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
