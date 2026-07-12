# exp01 — baseline

Baseline config exactly as shipped in `configs/train_config.yaml`:
yolov8s, imgsz 960, 150 epochs, mosaic + copy_paste augmentation.

**Goal:** establish a reference mAP@0.50 before tuning anything.

**Result:** _(fill in after running)_
- mAP@0.50:
- Train time:
- Notes: watch for small-object recall specifically (Section 5.2 sim-to-real gap) —
  if recall on far-range/low-PoT drones is weak, exp02/exp03 target that directly.
