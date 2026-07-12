#!/usr/bin/env python3
"""
Train a YOLOv8 model on the merged CUAS dataset.

Usage:
    python train_yolo.py --data ../configs/data.yaml --cfg ../configs/train_config.yaml

    # also copy best/last.pt into checkpoints/ and log the run to
    # experiments/<expNN>/metrics.csv:
    python train_yolo.py --save_checkpoint --exp_notes "baseline yolov8s @960"

Requires: pip install ultralytics --break-system-packages
"""
import argparse
import os
import shutil
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper import append_metrics_row, next_experiment_dir
from utils.logger import get_logger

log = get_logger("train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="configs/data.yaml")
    ap.add_argument("--cfg", type=str, default="configs/train_config.yaml")
    ap.add_argument("--save_checkpoint", action="store_true",
                     help="Copy best.pt/last.pt into checkpoints/ after training")
    ap.add_argument("--exp_notes", type=str, default="",
                     help="Free-text note stored alongside this run's hyperparameters/mAP")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed. Run:\n"
            "  pip install ultralytics --break-system-packages"
        )

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.pop("model", "yolov8s.pt")
    model = YOLO(model_name)

    log.info(f"Training {model_name} on {args.data} with config: {cfg}")
    start = time.time()
    results = model.train(data=args.data, **cfg)
    elapsed = time.time() - start

    run_dir = getattr(results, "save_dir", None)
    best_map50 = None
    try:
        best_map50 = float(results.results_dict.get("metrics/mAP50(B)", None))
    except Exception:
        pass

    log.info(f"Training finished in {elapsed / 60:.1f} min. Run dir: {run_dir}")

    if args.save_checkpoint and run_dir:
        os.makedirs("checkpoints", exist_ok=True)
        for fname in ("best.pt", "last.pt"):
            src = os.path.join(run_dir, "weights", fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join("checkpoints", fname))
                log.info(f"Copied {src} -> checkpoints/{fname}")

    # Log this run into experiments/ so hyperparameters + mAP are tracked
    # over time instead of only living in runs/ (which is git-ignored).
    exp_dir = next_experiment_dir("experiments")
    append_metrics_row(os.path.join(exp_dir, "metrics.csv"), {
        "model": model_name,
        "epochs": cfg.get("epochs"),
        "imgsz": cfg.get("imgsz"),
        "batch": cfg.get("batch"),
        "mAP50": best_map50,
        "train_minutes": round(elapsed / 60, 1),
        "run_dir": run_dir,
        "notes": args.exp_notes,
    })
    log.info(f"Run recorded in {exp_dir}/metrics.csv")


if __name__ == "__main__":
    main()
