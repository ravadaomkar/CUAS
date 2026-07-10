#!/usr/bin/env python3
"""
Train a YOLOv8 model on the merged CUAS dataset.

Usage:
    python train_yolo.py --data ../configs/data.yaml --cfg ../configs/train_config.yaml

Requires: pip install ultralytics --break-system-packages
"""
import argparse

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="configs/data.yaml")
    ap.add_argument("--cfg", type=str, default="configs/train_config.yaml")
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

    print(f"Training {model_name} on {args.data} with config:\n{cfg}\n")
    model.train(data=args.data, **cfg)


if __name__ == "__main__":
    main()
