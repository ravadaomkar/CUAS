# Duality Drone Detection Challenge — Complete Project Pipeline

## What this is (read this first)

This repo is a **complete, ready-to-run pipeline** for the challenge: data strategy,
Vibe Sim prompt scripts, dataset preparation, training, and evaluation code.

**What it does NOT include:** actual generated images. Vibe Sim is a platform your
team must be logged into — I have no access to it, so no one (including an AI) can
"complete" this competition without a human on your team running the sim and
downloading the resulting images/labels. What I've built removes every other piece
of manual work: once images land in `data/<scenario>/images` + `labels`, everything
downstream (training, augmentation, distribution matching, evaluation) is automated.

## Recommended strategy (the "best version" of this project)

Given the rules and scoring, the highest-leverage strategy is:

1. **Model**: YOLOv8 (small/medium) — best tradeoff of small-object accuracy, training
   speed, and ease of exporting a submission. Ultralytics' `yolov8s.pt`/`yolov8m.pt`
   COCO-pretrained backbones are allowed under Rule 3 (pretraining didn't see the
   eval set).
2. **Phase 1 (EO)**: generate data from *both* rural and urban scenarios, matched to
   the real IEEE pixels-on-target distribution (`ieee_pixels_on_target.npy`) using
   the custom-distribution feature — this directly targets the sim-to-real gap called
   out in Section 5.2, rather than guessing at a Gaussian/uniform spread.
3. **Clutter strategy**: enable birds + manned aircraft as background clutter classes
   (not trained as positives, but present in scenes) so the model learns *not* to
   fire on them — this is what Section 5's "strategic decision" is pointing at, and
   skipping it is the single most common way teams lose points to false positives.
4. **Weather/background diversity**: use `Hackathon_UrbanCUAS_ChangingWeather` to
   sweep weather conditions and random map locations — this is cheap variance that
   directly improves generalization to the unseen real eval set.
5. **Phase 2 (IR)**: same distribution-matching approach but for thermal signature —
   vary viewing angle, range, and background temperature deliberately (see
   `vibesim_prompts/phase2_ir.md`).
6. **Augmentation**: mosaic + copy-paste (allowed under Rule 4) on top of sim
   diversity, not as a replacement for it.
7. **Validation discipline**: never train on the IEEE/Duality reference sets (Rule 2 —
   disqualifying if violated). Use them only inside `scripts/evaluate.py` as a local
   proxy for real-world performance.

## Directory layout

```
cuas_project/
├── strategy/DATA_STRATEGY.md        # Full data-generation strategy & rationale
├── vibesim_prompts/                 # Copy-paste prompt sequences for the Vibe Sim agent
│   ├── phase1_rural.md
│   ├── phase1_urban.md
│   └── phase2_ir.md
├── notebooks/
│   └── dataset_analysis.ipynb       # Post-generation QA notebook (PoT, placement, balance)
├── scripts/
│   ├── analyze_pot_distribution.py  # Load ieee_pixels_on_target.npy, plan sim sampling
│   ├── prepare_dataset.py           # Merge scenario outputs into YOLO train/val split
│   ├── validate_dataset.py          # Sanity check the merged dataset (run before training!)
│   ├── train_yolo.py                # Train YOLOv8; --save_checkpoint logs to experiments/
│   ├── tune.py                      # Hyperparameter search (grid or ultralytics genetic tuner)
│   ├── evaluate.py                  # mAP@0.50 vs a labeled set; --plots writes to results/
│   ├── evaluate_modality.py         # Per-modality (EO/IR) mAP@0.50 + combined score
│   ├── inference_tta.py             # Test-time augmentation (multi-scale + flip) + WBF fusion
│   ├── error_analysis.py            # Bucket failures into analysis/{false_positive,...}
│   ├── visualize_predictions.py     # Draw predicted vs GT boxes on a folder of images
│   ├── submit_predictions.py        # Package the final submission (YOLO/COCO/CSV)
│   ├── run_pipeline.sh              # One-command end-to-end runner
│   └── README.md                    # Detailed per-script usage guide
├── ensemble/
│   ├── weighted_boxes_fusion.py     # Dependency-light WBF (single-class), used by TTA + ensembling
│   └── ensemble.py                  # Fuse predictions from 2+ trained checkpoints
├── utils/
│   ├── helper.py                    # Path/glob helpers, experiment-dir bookkeeping, CSV logging
│   ├── metrics.py                   # Shared IoU/AP/confusion-matrix math (backs evaluate*.py, tests/)
│   ├── visualization.py             # PR-curve / confusion-matrix / box-drawing plot helpers
│   └── logger.py                    # Shared logging setup -> logs/<script>.log + stdout
├── tests/
│   ├── test_metrics.py              # Unit tests for utils/metrics.py (no GPU/data needed)
│   └── test_weighted_boxes_fusion.py
├── experiments/                     # One folder per training run: config + metrics.csv + notes
│   ├── exp01/ ... exp03/            # Seeded templates: baseline, higher-res, heavier-aug + yolov8m
│   └── tuning_results.csv           # Appended to by scripts/tune.py
├── checkpoints/                     # best.pt / last.pt land here (git-ignored; see checkpoints/README.md)
├── logs/                            # Per-script log files, written by utils/logger.py
├── results/                         # Generated by evaluate.py --plots: PR curve, confusion matrix, metrics.csv
│   └── predictions/                 # Per-image YOLO-format prediction .txt files
├── analysis/                        # Generated by error_analysis.py: bucketed failure-case images
│   ├── false_positive/
│   ├── false_negative/
│   └── difficult_cases/
├── .github/workflows/train.yml      # CI: config/CLI smoke tests + unit tests on every push;
│                                     # optional full-training job on a self-hosted GPU runner
├── configs/
│   ├── data.yaml                    # YOLO dataset config (classes, paths)
│   └── train_config.yaml            # Hyperparameters
├── docs/
│   └── SUBMISSION.md                # Submission format guide (3 supported shapes)
├── requirements.txt                 # Pinned Python dependencies
├── LICENSE                          # MIT
├── .gitignore
└── data/                            # Drop Vibe Sim outputs here (see below)
    ├── rural/{images,labels}
    ├── urban/{images,labels}
    ├── ir/{images,labels}
    └── real_reference/{images,labels}   # IEEE + Duality IR sets, VALIDATION ONLY
```

## How to actually run this, end to end

### Step 0 — Setup
```bash
pip install -r requirements.txt
```

### Step 1 — Get access & pull starting datasets
Follow Section 4.1/9 of the problem statement: log into Vibe Sim, download the
rural + urban starter datasets, and place them under `data/rural/` and `data/urban/`
in YOLO format (an `images/` and `labels/` folder each, matched by filename).

### Step 2 — Plan your synthetic sampling against the real distribution
```bash
python scripts/analyze_pot_distribution.py \
    --ieee_npy /path/to/ieee_pixels_on_target.npy \
    --out strategy/pot_report.png
```
This tells you the mean/std/shape to feed back into the Vibe Sim "custom distribution"
prompt (see `vibesim_prompts/phase1_rural.md`), so your generated data's size
distribution actually matches reality instead of guessing.

### Step 3 — Generate data in Vibe Sim
Open the Vibe Sim chat pane and run the prompt sequences in
`vibesim_prompts/phase1_rural.md` and `phase1_urban.md` (then `phase2_ir.md` in
Week 4). Export each run's images+labels into the matching `data/<scenario>/`
folder.

### Step 3.5 — QA the generated batch in the notebook
```bash
jupyter lab notebooks/dataset_analysis.ipynb
```
Edit the `DATASETS` list to point at your batch, run all cells, and look at
the pixels-on-target histogram + placement heatmap + go/no-go verdict before
merging. The notebook will tell you whether the batch is ready to train on
or whether to re-run the Vibe Sim prompt with a different axis.

### Step 4 — Merge into a train/val split
```bash
python scripts/prepare_dataset.py --scenarios rural urban --out data/merged --val_split 0.15
```

### Step 4.5 — Validate before training
```bash
python scripts/validate_dataset.py --data configs/data.yaml
```
This catches the failures that have historically cost this project the most
GPU-hours: orphan files, corrupt images, out-of-range labels, zero-drone
splits. Don't skip it.

### Step 5 — Train
```bash
python scripts/train_yolo.py --data configs/data.yaml --cfg configs/train_config.yaml \
    --save_checkpoint --exp_notes "baseline yolov8s @960"
```
`--save_checkpoint` copies `best.pt`/`last.pt` into `checkpoints/` and logs the
run's hyperparameters + mAP into `experiments/<expNN>/metrics.csv` automatically.

### Step 5.5 — (Optional) Hyperparameter tuning before a full run
```bash
# cheap ranking search over a couple of params, 1-epoch trials
python scripts/tune.py --mode grid --grid lr0=0.001,0.01,0.05 imgsz=640,960 --epochs 1

# or hand the whole augmentation/optimizer space to ultralytics' own tuner
python scripts/tune.py --mode ultra --iterations 30 --epochs 10
```
Results land in `experiments/tuning_results.csv`. See `experiments/exp01`–`exp03`
for three seeded configs (baseline / higher-res / heavier-augmentation +
yolov8m) worth running before inventing your own sweep.

### Step 6 — Evaluate locally against real reference images (validation only!)
```bash
python scripts/evaluate.py --weights checkpoints/best.pt \
    --data_dir data/real_reference --out results/predictions --plots results/
```
`--plots` writes `results/pr_curve.png`, `results/confusion_matrix.png`, and
appends a row to `results/metrics.csv`.

### Step 6.5 — Per-modality breakdown (after Phase 2)
```bash
python scripts/evaluate_modality.py \
    --weights checkpoints/best.pt \
    --eo_dir data/real_reference/eo --ir_dir data/real_reference/ir
```

### Step 6.6 — Visualize failure cases
```bash
python scripts/visualize_predictions.py \
    --weights checkpoints/best.pt \
    --images_dir data/real_reference/eo/images \
    --labels_dir data/real_reference/eo/labels \
    --out viz/eo --num 30
```

### Step 6.7 — Bucket failures for systematic error analysis
```bash
python scripts/error_analysis.py --weights checkpoints/best.pt \
    --data_dir data/real_reference --conf 0.15 --num_per_bucket 25
```
Annotated images land in `analysis/false_positive/`, `analysis/false_negative/`,
and `analysis/difficult_cases/` — a faster way to spot systematic failure
modes (e.g. birds triggering false positives, low-PoT range shots being missed)
than reading prediction files by hand.

### Step 6.8 — (Optional) squeeze out extra mAP without retraining
```bash
# test-time augmentation: multi-scale + flip views, fused with weighted box fusion
python scripts/inference_tta.py --weights checkpoints/best.pt \
    --data_dir data/real_reference --out results/predictions_tta --scales 0.83,1.0,1.17 --flip

# or ensemble two+ trained checkpoints (e.g. exp02 + exp03) instead
python ensemble/ensemble.py --weights checkpoints/exp02_best.pt checkpoints/exp03_best.pt \
    --data_dir data/real_reference --out results/predictions_ensemble
```

### Step 7 — Submit (when the test set drops)
```bash
python scripts/submit_predictions.py \
    --weights checkpoints/best.pt \
    --test_dir /path/to/unseen_test_set \
    --format yolo --out submission/
```
See `docs/SUBMISSION.md` for how to switch to COCO or CSV format the moment
the official spec drops.

### Step 8 — Repeat with Phase 2 IR data once released (Week 4), same commands,
just add `data/ir` into the merge step.

`scripts/run_pipeline.sh` chains the merge → train → evaluate steps for
convenience once data exists.

## Testing & CI
`tests/` has fast, GPU-free unit tests for the shared math in `utils/metrics.py`
and `ensemble/weighted_boxes_fusion.py` — run them locally with:
```bash
pytest tests/ -v
```
`.github/workflows/train.yml` runs these tests plus a `--help` smoke test on
every script on each push/PR (catching broken imports/argparse typos before
they reach a GPU box), and exposes an optional `workflow_dispatch` job that
runs full training on a self-hosted GPU runner if you wire one up.

## Submission format
The official format will be "announced separately" per Section 6/8.5. We ship
support for the three most likely shapes (YOLO per-image .txt, COCO
`results.json`, flat CSV) — see `docs/SUBMISSION.md` for details and
`scripts/submit_predictions.py` for the implementation. The YOLO default
matches what `scripts/evaluate.py` already emits, so no rework is needed if
the official spec turns out to be YOLO.
