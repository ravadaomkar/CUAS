# Data Generation Strategy

## Goal
Maximize `(mAP@0.50_EO + mAP@0.50_IR) / 2` on an **unseen real-world** eval set,
using only Vibe Sim–generated synthetic training data.

## Core principle
The bottleneck in this challenge is not model architecture — it's the sim-to-real
gap (explicitly called out in Sections 1 and 5.2). Every design choice below is aimed
at shrinking that gap.

## 1. Pixels-on-target distribution matching (highest priority)
- Real drone imagery has a heavy-tailed apparent-size distribution: most detections
  are small (far drones), few are large (close drones).
- Default sim sampling (uniform in a fixed range) will over-represent large drones
  relative to the real IEEE distribution → model won't specialize enough on small
  objects, which is where mAP is lost.
- Action: load `ieee_pixels_on_target.npy`, fit the shape, feed it back into Vibe Sim
  as a **custom** pixels-on-target distribution (`analyze_pot_distribution.py` does
  this).
- For IR (Phase 2), no equivalent .npy exists yet — approximate using the Duality IR
  reference set once released, by running the same analysis script on ground-truth
  boxes extracted from its labels.

## 2. Background & environment diversity
- Use **both** rural and urban scenarios, not just one — real eval images will span
  environments the model has never trained on otherwise.
- Sweep altitude angle (sensor pitch) across a wide range (e.g. 5°–60°) so the model
  sees sky, treeline, rooftop, and horizon backgrounds, not just one framing.
- Use random map location sampling (small batches first, per the doc's own
  recommendation of 10-20 images to sanity-check) to avoid overfitting to a single
  background twin.
- In the urban scenario, sweep weather conditions once available — fog/overcast/rain
  variation is cheap synthetic diversity that maps to real seasonal/weather variance
  in the eval set.

## 3. Sensor realism
- Vary f-stop (depth-of-field blur) rather than fixing it — real cameras aren't
  always in perfect focus at all ranges.
- Add film grain at realistic intensity — synthetic images that are "too clean"
  are a classic sim-to-real failure mode for classifiers trained on them.

## 4. Drone placement diversity
- Don't cluster the drone in the image center. Sweep vertical/horizontal placement
  ranges across full-frame coverage over multiple generation runs (e.g., upper-30%,
  center-band, lower-30%) so the detector isn't spatially biased.
- Vary drone rotation (X/Y/Z) via USD edits to cover front, side, top-down, and
  banking-turn silhouettes — real drones are rarely captured face-on.

## 5. Clutter handling (false-positive control)
- Enable birds and manned aircraft in a meaningful fraction (not 100%) of generation
  runs. Include them in training images as background context, but:
  - **Do not** train the detector to predict their class — keep the loss/head
    scoped to class 0 (drone) at train time, OR train a 3-class head and simply
    ignore classes 1/2 at inference/eval time (both are valid; the merge script
    supports both — see `prepare_dataset.py --keep_clutter_classes`).
  - Rationale: real eval imagery will contain birds; if the model never sees birds
    during training it is far more likely to false-positive on them.

## 6. IR-specific considerations (Phase 2)
- Thermal signature depends heavily on: viewing angle (motor/battery hotspots
  visible face-on, less so edge-on), range (signal-to-background ratio drops with
  distance), and background temperature (drone-vs-sky contrast is higher on cold
  days/altitudes, lower against warm rooftops or the sun-heated ground).
- Deliberately sample across all three axes rather than a single "typical" IR
  condition, and generate at multiple times-of-day/background-temperature settings
  if the scenario exposes that control.
- Watch for fixed-pattern noise settings in the IR pipeline — real IR sensors have
  this; if the sim omits it, add synthetic fixed-pattern noise as an augmentation
  step in `prepare_dataset.py` to close that specific gap.

## 7. Dataset sizing
- Don't over-invest in one giant single-distribution run. Prefer several smaller
  batches (100–300 images) that each vary one axis (angle, weather, placement,
  clutter), verified via `dataset_analysis.ipynb` between runs, over one large
  monolithic run — this makes it easy to isolate which axis is actually helping
  validation mAP and drop what isn't.

## 8. Validation discipline
- Reference sets (IEEE EO, Duality IR) are for **analysis and local validation
  only** — never train on them (disqualifying per Rule 2). Keep them in
  `data/real_reference/` and only ever pass that path to `evaluate.py`, never to
  `train_yolo.py`.
