# Submission Guide

The official submission format is "to be announced separately" per Section 6
of the problem statement. This doc captures our team's working assumptions,
the three most likely shapes, and how to swap between them the moment the
official spec drops.

## TL;DR

```bash
# Default — YOLO .txt per image, bundled as a zip
python scripts/submit_predictions.py \
    --weights runs/detect/cuas_train/weights/best.pt \
    --test_dir /path/to/test_set \
    --format yolo --out submission/

# COCO-style results.json
python scripts/submit_predictions.py --format coco --out submission_coco/

# Flat CSV
python scripts/submit_predictions.py --format csv --out submission_csv/
```

## Format A — YOLO per-image .txt (current default)

* One `.txt` file per test image, named `<image_stem>.txt`.
* Each non-empty line: `class x_center y_center width height confidence`
  with values in normalized image coordinates (0–1).
* `class` is always `0` (drone). Birds and manned aircraft are not scored.
* Multiple lines per file are allowed when multiple drones appear.
* Bundle: a single zip of all `.txt` files (no images), named
  `submission.zip`.

This matches the output of `scripts/evaluate.py` exactly, so if the official
spec is YOLO, we are already done.

## Format B — COCO results.json

* Single `results.json` file.
* Each row: `{"image_id": int, "category_id": 1, "bbox": [x, y, w, h], "score": float}`
  with absolute pixel coordinates.
* `image_id` is a 1-indexed integer per test image. The mapping is recorded
  in `submission/manifest.json` for audit.

## Format C — Flat CSV

* Single `predictions.csv` with columns
  `image_id, confidence, x_min, y_min, x_max, y_max`
  (absolute pixel coords, integers).
* No `category_id` column (we only score drones).

## Switching formats when the official spec drops

Two options:

1. Re-run with `--format <yolo|coco|csv>` and ship the new artifact.
2. Drop a JSON spec file at the location and re-run with `--spec <path>`:

```json
{
  "format": "coco",
  "conf": 0.10
}
```

The spec-driven path lets a non-engineer change the submission shape without
touching the script.

## Recommended confidence threshold

* Default is `0.15`. Lower it to `0.05` for higher recall (more detections,
  more false positives), raise it to `0.30` for higher precision.
* Run `python scripts/evaluate.py` against the IEEE EO reference at three
  thresholds (0.05 / 0.15 / 0.30) and pick the one that maximizes mAP@0.50.
  That's almost always the right value to ship.

## File integrity checklist

Before submitting, verify:

- [ ] Number of `.txt` (or rows in `results.json` / `predictions.csv`)
      matches the number of test images that contained drone detections
      (anything that *should* have detections; an empty image = empty file is fine).
- [ ] `submission/manifest.json` is included in the zip.
- [ ] Confidence column is present and not all zero.
- [ ] No negative or >1 normalized values in YOLO format.
- [ ] Bounding box coordinates make physical sense (no boxes bigger than
      the image, no negative coordinates).

## What we will NOT do

* Train on the IEEE or Duality reference sets (Rule 2 — disqualifying).
* Use any non-Vibe-Sim synthetic or real data source (Rule 1).
* Submit model weights that saw the reference sets during training.

The submission pipeline above operates on a single trained `.pt` file and
the unseen test set; nothing upstream of `train_yolo.py` is touched during
submission generation.
