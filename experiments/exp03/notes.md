# exp03 — stronger augmentation + yolov8m backbone

**Change vs exp01:** `yolov8s -> yolov8m`, heavier copy_paste/scale/mixup,
added `flipud: 0.1` for aerial-angle diversity, `mixup: 0.15`.

**Goal:** reduce the sim-to-real generalization gap (Section 5.2) by making
the model rely less on sim-specific texture cues, at the cost of a bigger/
slower model.

**Result:** _(fill in after running)_
- mAP@0.50:
- Train time:
- Notes: if this beats exp01/exp02 on the real_reference set specifically
  (not just synthetic val), that's strong evidence augmentation strength
  matters more than resolution for this challenge — worth ensembling
  exp02+exp03 (see ensemble/) rather than picking just one.
