# Resume bullets (only real numbers; full-training values are TBD)

Ground rule: every number below comes from a file in `results/` produced by a
command recorded in the README. Bullets marked **[TBD]** await the full
training runs — fill from `results/ablations.csv` when they land. Do NOT
publish detection accuracy numbers before then.

## Ready to use (measured, sources cited)

- Built an end-to-end aerial tiny-object tracking pipeline (custom stride-4
  PyTorch detector, Wasserstein-distance box loss, Kalman multi-object
  tracker); tracks survive artificial 15-frame detection dropouts with
  **IDF1 0.964, 0 identity switches, 100% identity preservation** across an
  8-clip occlusion benchmark scored with motmetrics
  (`results/tracking/occlusion_summary.csv`).
- Implemented a rolling-window motion filter that raises track precision from
  **0.44 to 0.53** under injected detector hallucinations
  (`results/tracking/fp_precision.csv`); thresholds are config-driven.
- Achieved a **757 FPS** full detect→track loop on CPU in the tracking demo
  (`results/demo/gt_sim_report.json`); tracker association alone runs at
  **0.8–2.5 ms/frame** depending on metric (IoU vs NWD).
- Authored a Normalized Wasserstein Distance (NWD) regression loss for 8–32 px
  bounding boxes with closed-form Gaussian 2-Wasserstein distance and a
  math-property unit suite (gradient signal on disjoint boxes where IoU is
  exactly zero) (`tests/test_nwd_loss.py`, 12 tests).

## [TBD] after full training runs (fill from results/ablations.csv)

- Trained a 4-variant ablation (P2 head on/off × NWD vs CIoU × domain
  augmentation) on N synthetic aerial images: adding the stride-4 P2 head
  improved small-object AP @[.5:.95] from **TBD** to **TBD**; NWD blending
  added **TBD**; domain augmentation closed the synthetic→real gap by
  **TBD** points of mAP50 (`results/ablations.csv`).
