# exp02 — larger input resolution (1280 vs 960)

**Change vs exp01:** `imgsz: 960 -> 1280`, `batch: 16 -> 8` (memory tradeoff).

**Goal:** test whether small-object recall (far-range drones, low
pixels-on-target) improves enough to justify ~2x slower training/inference.

**Result:** _(fill in after running)_
- mAP@0.50:
- Train time:
- Notes: compare per-modality mAP via `scripts/evaluate_modality.py` — if
  the gain is concentrated in one modality (EO vs IR) that's a signal to
  generate more low-PoT sim data for the other rather than just scaling imgsz further.
