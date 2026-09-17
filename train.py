"""Training loop for SkySentry experiments.

Reads one experiment YAML (see configs/experiments/*.yaml), trains the P2
detector with the configured loss/strides/augmentations, evaluates on the
validation split every epoch, and supports checkpointing, resume, early
stopping and CSV logging. Every number this module reports lands in
``results/logs/<exp>/log.csv`` — nothing is reported that wasn't measured.

Usage::

    python train.py --config configs/experiments/exp_c_p2_nwd.yaml
    python train.py --config ... --resume checkpoints/exp_c/last.pt
    # smoke override (beats YAML values):
    python train.py --config ... --smoke --epochs 2 --size 320 --max-train 16 --max-val 8
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.yolo_dataset import YoloDetDataset, build_augmentations
from evaluation.metrics import DetectionEvaluator
from infer import postprocess
from losses.detection_loss import DetectionLossConfig, TinyDetectionLoss
from models.detector_p2 import DetectorConfig, TinyDetector

ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch (CPU determinism best-effort)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_experiment(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_collate():
    """Variable-length targets: images stacked, targets passed as a list."""
    def collate(batch):
        images = torch.stack([b[0] for b in batch])
        targets = [torch.from_numpy(b[1]["boxes"]) for b in batch]
        return images, targets
    return collate


def build_loaders(exp: dict, smoke: bool) -> tuple[DataLoader, DataLoader]:
    data = exp["data"]
    size = exp["train"]["img_size"]
    aug = build_augmentations() if data.get("augment", False) else None
    train_ds = YoloDetDataset(
        Path(data["root"]) / "train" / "images", Path(data["root"]) / "train" / "labels",
        img_size=size, augment=aug,
        max_images=exp["train"].get("max_train") if smoke else None,
    )
    val_ds = YoloDetDataset(
        Path(data["root"]) / "val" / "images", Path(data["root"]) / "val" / "labels",
        img_size=size,
        max_images=exp["train"].get("max_val") if smoke else None,
    )
    train_ld = DataLoader(train_ds, batch_size=exp["train"]["batch_size"], shuffle=True,
                          collate_fn=make_collate(), num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=exp["train"]["batch_size"], shuffle=False,
                        collate_fn=make_collate(), num_workers=0)
    return train_ld, val_ld


def evaluate(model: TinyDetector, loader: DataLoader, num_classes: int,
             class_agnostic: bool, device: torch.device, conf: float = 0.0001) -> dict[str, float]:
    # conf floor is deliberately tiny (standard for AP): the PR curve integrates
    # over the full score range, so early-epoch models with low objectness
    # still get measured instead of silently scoring 0.0.
    """Full inference + COCO-style metrics on one split."""
    ev = DetectionEvaluator(num_classes=num_classes if not class_agnostic else 1)
    image_idx = 0
    for images, targets in loader:
        dets = postprocess(model, images.to(device), conf_threshold=conf, nms_iou=0.5,
                           class_agnostic=class_agnostic)
        for det, gt in zip(dets, targets):
            ev.update(image_idx, det["boxes"].numpy(), det["scores"].numpy(), det["labels"].numpy(),
                      gt[:, 1:5].numpy(), gt[:, 0].long().numpy())
            image_idx += 1
    return ev.results()


def train(exp: dict, smoke: bool = False, resume_from: str | None = None) -> dict:
    """Train one experiment; returns the best-val summary dict."""
    set_seed(exp["train"].get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = DetectorConfig(
        num_classes=exp["model"]["num_classes"],
        width=exp["model"].get("width", 0.5),
        depth=exp["model"].get("depth", 0.33),
        strides=tuple(exp["model"]["strides"]),
    )
    model = TinyDetector(model_cfg).to(device)
    loss_cfg = DetectionLossConfig(
        num_classes=model_cfg.num_classes,
        strides=model_cfg.strides,
        scale_ranges=((0.0, 64.0), (64.0, 128.0), (128.0, float("inf"))),
        nwd_constant=exp["loss"].get("nwd_constant", 12.8),
        nwd_alpha=exp["loss"].get("nwd_alpha", 0.5),
        weight_box=exp["loss"].get("weight_box", 5.0),
    )
    criterion = TinyDetectionLoss(loss_cfg)

    t_cfg = exp["train"]
    epochs = t_cfg["epochs"]   # the caller (CLI overrides / run_ablations) owns sizing
    opt = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg.get("weight_decay", 5e-4))
    start_epoch, best_map50 = 0, -1.0

    out_dir = ROOT / "checkpoints" / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "results" / "logs" / exp["name"] / "log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch, best_map50 = ckpt["epoch"] + 1, ckpt["best_map50"]
        print(f"  resumed from {resume_from} (epoch {start_epoch})")

    train_ld, val_ld = build_loaders(exp, smoke)

    log_fields = ["epoch", "lr", "loss", "loss_box", "loss_cls", "loss_obj",
                  "val_mAP_all", "val_mAP50", "val_mAP_small", "seconds"]
    new_log = not log_path.exists() or start_epoch > 0
    log_file = open(log_path, "a" if (log_path.exists() and start_epoch > 0) else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if new_log or log_path.stat().st_size == 0:
        writer.writeheader()

    patience = t_cfg.get("early_stop_patience", 20)
    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()
        model.train()
        sums = {"loss": 0.0, "loss_box": 0.0, "loss_cls": 0.0, "loss_obj": 0.0}
        n_steps = 0
        for images, targets in train_ld:
            loss, stats = criterion(model(images.to(device)), targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            for k in sums:
                sums[k] += float(stats[k])
            n_steps += 1
        train_seconds = time.perf_counter() - t0

        results = evaluate(model, val_ld, model_cfg.num_classes, False, device)
        row = {
            "epoch": epoch, "lr": opt.param_groups[0]["lr"],
            "loss": sums["loss"] / max(n_steps, 1), "loss_box": sums["loss_box"] / max(n_steps, 1),
            "loss_cls": sums["loss_cls"] / max(n_steps, 1), "loss_obj": sums["loss_obj"] / max(n_steps, 1),
            "val_mAP_all": results["mAP_all"], "val_mAP50": results["mAP50"],
            "val_mAP_small": results["mAP_small"], "seconds": round(train_seconds, 1),
        }
        writer.writerow(row)
        log_file.flush()

        ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                "epoch": epoch, "best_map50": best_map50, "exp": exp["name"]}
        torch.save(ckpt, out_dir / "last.pt")
        if results["mAP50"] > best_map50:
            best_map50 = results["mAP50"]
            torch.save(ckpt, out_dir / "best.pt")
        print(f"  epoch {epoch}: loss {row['loss']:.4f}  val mAP50 {row['val_mAP50']:.4f}  "
              f"({train_seconds:.0f}s)")

        if epoch - _best_epoch(log_path, best_map50) >= patience and epoch > start_epoch:
            print(f"  early stop at epoch {epoch} (no val mAP50 gain for {patience})")
            break

    log_file.close()
    return {"best_val_mAP50": best_map50, "epochs_run": epochs}


def _best_epoch(log_path: Path, best_map50: float) -> int:
    """Last epoch that set the current best (rough early-stop bookkeeping)."""
    try:
        rows = list(csv.DictReader(open(log_path, encoding="utf-8")))
        best_rows = [int(r["epoch"]) for r in rows if abs(float(r["val_mAP50"]) - best_map50) < 1e-9]
        return best_rows[-1] if best_rows else 0
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="SkySentry training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny subset, ≤2 epochs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    args = parser.parse_args()

    exp = load_experiment(args.config)
    if args.smoke or any(v is not None for v in (args.epochs, args.size, args.max_train, args.max_val)):
        exp = json.loads(json.dumps(exp))  # deep copy before overriding
        if args.smoke:
            exp["train"].update(img_size=320, batch_size=4, early_stop_patience=2)
        if args.epochs:
            exp["train"]["epochs"] = args.epochs
        if args.size:
            exp["train"]["img_size"] = args.size
        if args.max_train:
            exp["train"]["max_train"] = args.max_train
        if args.max_val:
            exp["train"]["max_val"] = args.max_val

    print(f"== training {exp['name']} ==")
    summary = train(exp, smoke=args.smoke, resume_from=args.resume)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
