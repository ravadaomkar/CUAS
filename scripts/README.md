# Scripts

All scripts in this folder are CLI-only Python (no notebook required).
Each one is a thin wrapper around a single stage of the pipeline.

## Pipeline at a glance

```
Vibe Sim exports  →  analyze_pot_distribution  →  prepare_dataset  →  validate_dataset
                                                                       ↓
                                                              train_yolo
                                                                       ↓
                              ┌──────────────────────────────────┐   ↓
                              ↓                                  ↓   ↓
                       evaluate (overall)              evaluate_modality (EO/IR split)
                              ↓                                  ↓
                       visualize_predictions               submit_predictions
```

## Scripts

| Script | Purpose | When to run |
|---|---|---|
| `analyze_pot_distribution.py` | Compute pixels-on-target statistics from `ieee_pixels_on_target.npy` or a YOLO labels folder. | After you download the IEEE reference .npy; before generating large Vibe Sim batches. |
| `prepare_dataset.py` | Merge Vibe Sim scenario outputs (rural / urban / ir, possibly nested in batch subfolders) into a single YOLO train/val split. | After Vibe Sim exports land in `data/<scenario>/`. |
| `validate_dataset.py` | Sanity check the merged dataset — orphan files, corrupt images, bad label values, class balance. | **Always** after `prepare_dataset.py`, before `train_yolo.py`. Catches mistakes that would silently waste GPU hours. |
| `train_yolo.py` | Train a YOLOv8 model on the merged dataset using hyperparameters from `configs/train_config.yaml`. | After `validate_dataset.py` passes. |
| `evaluate.py` | Compute mAP@0.50 on a single labeled set (e.g. the local real-world reference). Also writes per-image predictions in YOLO format for later submission. | After training. |
| `evaluate_modality.py` | Run inference on EO and IR halves separately and report mAP@0.50 per modality (plus the combined final score). | After Phase 2 IR data is added; use to find out whether the model is weak on IR specifically. |
| `visualize_predictions.py` | Draw predicted (and optionally ground-truth) boxes on a folder of images. | Whenever you need to debug failure cases by eye. |
| `submit_predictions.py` | Run inference on the unseen test set and package the submission artifact (YOLO zip / COCO JSON / CSV). | When the official submission window opens. |
| `run_pipeline.sh` | Chains merge → train → evaluate in one command. | Convenience wrapper — not for production use. |

## Common invocation patterns

### First-time setup, full pipeline

```bash
# 1. (One time) Pinpoint the real pixel-size distribution
python scripts/analyze_pot_distribution.py \
    --ieee_npy /path/to/ieee_pixels_on_target.npy \
    --out strategy/pot_report.png

# 2. (After Vibe Sim exports) Merge and split
python scripts/prepare_dataset.py --scenarios rural urban --out data/merged --val_split 0.15

# 3. (Before training) Validate
python scripts/validate_dataset.py --data configs/data.yaml

# 4. Train
python scripts/train_yolo.py --data configs/data.yaml --cfg configs/train_config.yaml

# 5. Evaluate against the local reference (validation only — never train on it)
python scripts/evaluate.py \
    --weights runs/detect/cuas_train/weights/best.pt \
    --data_dir data/real_reference \
    --out predictions/

# 6. Visualize failures
python scripts/visualize_predictions.py \
    --weights runs/detect/cuas_train/weights/best.pt \
    --images_dir data/real_reference/images \
    --labels_dir data/real_reference/labels \
    --out viz/
```

### After Phase 2 IR release

```bash
python scripts/prepare_dataset.py --scenarios rural urban ir --out data/merged
python scripts/validate_dataset.py --data configs/data.yaml
python scripts/train_yolo.py --data configs/data.yaml --cfg configs/train_config.yaml
python scripts/evaluate_modality.py \
    --weights runs/detect/cuas_train/weights/best.pt \
    --eo_dir data/real_reference/eo --ir_dir data/real_reference/ir
```

### Final submission

```bash
python scripts/submit_predictions.py \
    --weights runs/detect/cuas_train/weights/best.pt \
    --test_dir /path/to/unseen_test_set \
    --format yolo --out submission/
```

## Dependencies

All scripts depend only on `ultralytics`, `numpy`, `Pillow`, `pyyaml`, and
`matplotlib`. `tqdm` is used where present but not required. See
`requirements.txt` at the repo root for the pinned list.
