#!/usr/bin/env bash
# End-to-end runner: merge -> train -> evaluate.
# Run from the project root, after Vibe Sim data is placed under data/<scenario>/.
set -euo pipefail

SCENARIOS=${1:-"rural urban"}   # e.g. run_pipeline.sh "rural urban ir"

echo "== Step 1: Merging scenario data =="
python scripts/prepare_dataset.py --scenarios $SCENARIOS --out data/merged --val_split 0.15

echo "== Step 2: Training =="
python scripts/train_yolo.py --data configs/data.yaml --cfg configs/train_config.yaml

echo "== Step 3: Evaluating against local real-world reference set =="
BEST_WEIGHTS=$(find runs/detect -name "best.pt" | tail -n 1)
python scripts/evaluate.py --weights "$BEST_WEIGHTS" --data_dir data/real_reference --out predictions/

echo "Pipeline complete. Weights: $BEST_WEIGHTS"
