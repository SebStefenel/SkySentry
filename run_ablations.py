"""Run the Phase-3 ablation grid end to end.

For each experiment config: train, then evaluate the best checkpoint on
**synthetic-val** (in-domain, 3 classes) and **real-test** (VisDrone,
class-agnostic transfer protocol — every category collapses to one "object"
class on BOTH prediction and GT side, because VisDrone has no drone/bird
classes; see data/raw/visdrone/DATASET_CARD.md).

Outputs (all measured, nothing hand-written):
    results/ablations.csv            one row per (experiment, dataset)
    results/gap_<exp>.json           sim-to-real gap report per experiment
    results/logs/<exp>/log.csv       per-epoch training curves

Usage:
    python run_ablations.py --smoke                     # pipeline proof (~minutes, CPU)
    python run_ablations.py                             # full runs (see STOP estimate first)
    python run_ablations.py --exps exp_b_p2,exp_c_p2_nwd
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.yolo_dataset import YoloDetDataset
from evaluation.metrics import domain_gap_report
from infer import postprocess
from models.detector_p2 import DetectorConfig, TinyDetector
from train import ROOT, evaluate, load_experiment, make_collate, train

EXPERIMENTS = {
    "exp_a_baseline": "configs/experiments/exp_a_baseline.yaml",
    "exp_b_p2": "configs/experiments/exp_b_p2.yaml",
    "exp_c_p2_nwd": "configs/experiments/exp_c_p2_nwd.yaml",
    "exp_d_p2_nwd_aug": "configs/experiments/exp_d_p2_nwd_aug.yaml",
}

REAL_TEST_ROOT = Path("data/raw/visdrone")
VISDRONE_CLASSES = 10  # all collapsed to one agnostic class at eval time


def eval_real(model: TinyDetector, exp: dict, smoke: bool, device: torch.device) -> dict[str, float]:
    """Class-agnostic evaluation on the VisDrone real-test split."""
    size = exp["train"]["img_size"]
    ds = YoloDetDataset(
        REAL_TEST_ROOT / "real-test" / "images", REAL_TEST_ROOT / "real-test" / "labels",
        img_size=size, class_map={i: 0 for i in range(VISDRONE_CLASSES)},
        max_images=8 if smoke else None,
    )
    loader = DataLoader(ds, batch_size=exp["train"]["batch_size"], shuffle=False,
                        collate_fn=make_collate(), num_workers=0)
    return evaluate(model, loader, num_classes=1, class_agnostic=True, device=device)


def run(smoke: bool, exps: list[str], overrides: dict) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    csv_path = ROOT / "results" / "ablations.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "dataset", "protocol", "mAP_all", "mAP50", "mAP75",
              "mAP_small", "eval_images", "train_epochs", "train_seconds", "params", "notes"]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()

        for name in exps:
            exp = load_experiment(ROOT / EXPERIMENTS[name])
            exp = json.loads(json.dumps(exp))
            if smoke:
                exp["train"].update(img_size=320, batch_size=4, early_stop_patience=6)
                exp["train"]["max_train"] = 32
                exp["train"]["max_val"] = 8
                exp["train"]["epochs"] = min(exp["train"]["epochs"], 6)
            exp["train"].update(overrides)

            print(f"\n== [{name}] training ({'SMOKE' if smoke else 'full'}) ==")
            t0 = time.perf_counter()
            summary = train(exp, smoke=smoke)
            train_seconds = time.perf_counter() - t0

            model_cfg = DetectorConfig(
                num_classes=exp["model"]["num_classes"], width=exp["model"].get("width", 0.5),
                depth=exp["model"].get("depth", 0.33), strides=tuple(exp["model"]["strides"]),
            )
            model = TinyDetector(model_cfg)
            ckpt_path = ROOT / "checkpoints" / name / "best.pt"
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.to(device)

            # In-domain eval on synthetic-val (3-class protocol).
            val_ds = YoloDetDataset(
                Path(exp["data"]["root"]) / "val" / "images", Path(exp["data"]["root"]) / "val" / "labels",
                img_size=exp["train"]["img_size"], max_images=8 if smoke else None,
            )
            val_ld = DataLoader(val_ds, batch_size=exp["train"]["batch_size"], shuffle=False,
                                collate_fn=make_collate(), num_workers=0)
            syn = evaluate(model, val_ld, model_cfg.num_classes, False, device)
            # Transfer eval on real-test (class-agnostic protocol).
            real = eval_real(model, exp, smoke, device)

            n_params = sum(p.numel() for p in model.parameters())
            notes = ("class-agnostic transfer eval (VisDrone→'object')" if REAL_TEST_ROOT.exists() else
                     "real-test split missing — synthetic only")
            for ds_name, res in (("synthetic-val", syn), ("real-test", real)):
                row = {
                    "experiment": name, "dataset": ds_name,
                    "protocol": "3-class" if ds_name == "synthetic-val" else "class-agnostic",
                    "mAP_all": round(res["mAP_all"], 5), "mAP50": round(res["mAP50"], 5),
                    "mAP75": round(res["mAP75"], 5), "mAP_small": round(res["mAP_small"], 5),
                    "eval_images": (8 if smoke else "all"), "train_epochs": summary["epochs_run"],
                    "train_seconds": round(train_seconds, 1), "params": n_params, "notes": notes,
                }
                writer.writerow(row)
                rows.append(row)
                print(f"  {ds_name}: mAP {row['mAP_all']}  mAP50 {row['mAP50']}  mAP_small {row['mAP_small']}")

            gap = domain_gap_report(syn, real)
            gap_path = ROOT / "results" / f"gap_{name}.json"
            gap_path.write_text(json.dumps(gap, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="SkySentry Phase-3 ablations")
    parser.add_argument("--smoke", action="store_true", help="tiny config to prove the pipeline")
    parser.add_argument("--exps", default=",".join(EXPERIMENTS), help="comma list of experiment names")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--size", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.size:
        overrides["img_size"] = args.size
    rows = run(args.smoke, [e for e in args.exps.split(",") if e], overrides)
    print("\n== ablations.csv ==")
    for r in rows:
        print(f"{r['experiment']:20s} {r['dataset']:14s} mAP={r['mAP_all']} mAP50={r['mAP50']} mAP_small={r['mAP_small']}")


if __name__ == "__main__":
    main()
