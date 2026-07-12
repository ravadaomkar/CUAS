#!/usr/bin/env python3
"""
Hyperparameter tuning for the YOLOv8 training run.

Two modes:
  * 'grid'   — exhaustive search over a small explicit grid (cheap, predictable
               cost, good for 2-3 params like lr0 x imgsz).
  * 'ultra'  — delegate to ultralytics' own built-in genetic-algorithm tuner
               (model.tune), which searches the full augmentation/optimizer
               space but costs much more compute.

Every trial's config + resulting mAP@0.50 is appended to
experiments/tuning_results.csv (via utils.helper.append_metrics_row) so you
can compare trials without re-parsing ultralytics' own run folders.

Usage:
    # cheap 2-param grid search, 1 epoch each just to rank configs
    python scripts/tune.py --mode grid --data configs/data.yaml \
        --base_cfg configs/train_config.yaml --epochs 1 \
        --grid lr0=0.001,0.01,0.05 imgsz=640,960

    # full ultralytics genetic-algorithm tuner (expensive — use on a subset
    # of full training epochs, e.g. 30 iterations x 10 epochs each)
    python scripts/tune.py --mode ultra --data configs/data.yaml \
        --base_cfg configs/train_config.yaml --iterations 30 --epochs 10

Requires: pip install ultralytics --break-system-packages
"""
import argparse
import itertools
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper import append_metrics_row
from utils.logger import get_logger

log = get_logger("tune")

RESULTS_CSV = os.path.join("experiments", "tuning_results.csv")


def parse_grid(grid_args):
    """['lr0=0.001,0.01', 'imgsz=640,960'] -> {'lr0': [0.001, 0.01], 'imgsz': [640, 960]}"""
    grid = {}
    for item in grid_args:
        key, values = item.split("=")
        parsed = []
        for v in values.split(","):
            try:
                parsed.append(int(v))
            except ValueError:
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(v)
        grid[key] = parsed
    return grid


def run_grid_search(args):
    from ultralytics import YOLO

    with open(args.base_cfg) as f:
        base_cfg = yaml.safe_load(f)
    model_name = base_cfg.pop("model", "yolov8s.pt")

    grid = parse_grid(args.grid)
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    log.info(f"Grid search over {keys}: {len(combos)} combinations")

    best = {"mAP50": -1, "combo": None}
    for i, combo in enumerate(combos):
        trial_cfg = dict(base_cfg)
        trial_cfg.update(dict(zip(keys, combo)))
        trial_cfg["epochs"] = args.epochs  # override to keep trials cheap
        trial_cfg["name"] = f"tune_grid_{i:03d}"

        log.info(f"[{i + 1}/{len(combos)}] trying {dict(zip(keys, combo))}")
        model = YOLO(model_name)
        results = model.train(data=args.data, **trial_cfg)
        try:
            map50 = float(results.results_dict.get("metrics/mAP50(B)", 0.0))
        except Exception:
            map50 = 0.0

        row = {"trial": i, "model": model_name, "mAP50": map50, **dict(zip(keys, combo))}
        append_metrics_row(RESULTS_CSV, row)
        log.info(f"  -> mAP50={map50:.4f}")

        if map50 > best["mAP50"]:
            best = {"mAP50": map50, "combo": dict(zip(keys, combo))}

    log.info(f"Best combo: {best['combo']} (mAP50={best['mAP50']:.4f})")
    log.info(f"Full results in {RESULTS_CSV}")


def run_ultra_tune(args):
    from ultralytics import YOLO

    with open(args.base_cfg) as f:
        base_cfg = yaml.safe_load(f)
    model_name = base_cfg.pop("model", "yolov8s.pt")

    log.info(f"Running ultralytics genetic-algorithm tuner: "
             f"{args.iterations} iterations x {args.epochs} epochs each")
    model = YOLO(model_name)
    # model.tune searches lr0, momentum, weight_decay, augmentation params, etc.
    # and writes its own runs/detect/tune/ folder with best_hyperparameters.yaml.
    model.tune(
        data=args.data,
        epochs=args.epochs,
        iterations=args.iterations,
        imgsz=base_cfg.get("imgsz", 960),
        optimizer=base_cfg.get("optimizer", "auto"),
        plots=True,
        save=True,
        val=True,
    )
    log.info("Tuning complete — see runs/detect/tune/best_hyperparameters.yaml")
    log.info("Copy the winning values into configs/train_config.yaml before a full training run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grid", "ultra"], default="grid")
    ap.add_argument("--data", type=str, default="configs/data.yaml")
    ap.add_argument("--base_cfg", type=str, default="configs/train_config.yaml")
    ap.add_argument("--epochs", type=int, default=1,
                     help="Epochs per trial — keep small, this is for ranking configs, not final training")
    ap.add_argument("--grid", nargs="+", default=["lr0=0.001,0.01,0.05"],
                     help="grid mode only, e.g. --grid lr0=0.001,0.01 imgsz=640,960")
    ap.add_argument("--iterations", type=int, default=20,
                     help="ultra mode only — number of genetic-algorithm iterations")
    args = ap.parse_args()

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        raise SystemExit(
            "ultralytics not installed. Run:\n"
            "  pip install ultralytics --break-system-packages"
        )

    if args.mode == "grid":
        run_grid_search(args)
    else:
        run_ultra_tune(args)


if __name__ == "__main__":
    main()
