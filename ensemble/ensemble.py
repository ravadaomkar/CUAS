#!/usr/bin/env python3
"""
Ensemble multiple trained models (e.g. exp01's yolov8s vs exp03's yolov8m,
or an EO-specialist + IR-specialist) by running each on the same image set
and fusing their predictions with weighted box fusion.

This is the natural next step after experiments/ shows you several
checkpoints with different strengths — e.g. exp02 (imgsz=1280) might have
better far-range recall while exp03 (heavier augmentation) generalizes
better to real-world texture; ensembling both is often better than picking
either individually.

Usage:
    python ensemble/ensemble.py \
        --weights checkpoints/exp02_best.pt checkpoints/exp03_best.pt \
        --model_weights 1.0 1.2 \
        --data_dir data/real_reference \
        --out results/predictions_ensemble

Then score it the normal way:
    python scripts/evaluate.py --weights checkpoints/exp03_best.pt \
        --data_dir data/real_reference --out results/predictions_ensemble --plots results/
    (--weights here is only used by evaluate.py to know the imgsz/labels map;
     the actual predictions read from --out are the fused ones, not re-run.)
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)  # so `weighted_boxes_fusion` resolves even though this
                                 # file is itself named ensemble.py inside package ensemble/
from utils.helper import list_images
from utils.logger import get_logger
from weighted_boxes_fusion import weighted_boxes_fusion

log = get_logger("ensemble")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, nargs="+", required=True,
                     help="Paths to two or more trained .pt checkpoints")
    ap.add_argument("--model_weights", type=float, nargs="+", default=None,
                     help="Optional per-model vote weight, same length as --weights "
                          "(default: equal weighting)")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/predictions_ensemble")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--wbf_iou_thresh", type=float, default=0.55)
    args = ap.parse_args()

    if args.model_weights and len(args.model_weights) != len(args.weights):
        raise SystemExit("--model_weights must match --weights in length")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed. Run:\n"
            "  pip install ultralytics --break-system-packages"
        )

    os.makedirs(args.out, exist_ok=True)
    models = [YOLO(w) for w in args.weights]
    log.info(f"Ensembling {len(models)} models: {args.weights}")

    images_dir = os.path.join(args.data_dir, "images")
    img_paths = list_images(images_dir)
    if not img_paths:
        raise SystemExit(f"No images found in {images_dir}")

    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        all_boxes, all_scores = [], []
        w = h = None

        for model in models:
            results = model.predict(img_path, conf=args.conf, verbose=False)[0]
            w, h = results.orig_shape[1], results.orig_shape[0]
            boxes, scores = [], []
            for box in results.boxes:
                if int(box.cls.item()) != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
                scores.append(float(box.conf.item()))
            all_boxes.append(boxes)
            all_scores.append(scores)

        fused_boxes, fused_scores = weighted_boxes_fusion(
            all_boxes, all_scores, iou_thr=args.wbf_iou_thresh, model_weights=args.model_weights
        )

        pred_lines = []
        for (x1, y1, x2, y2), conf in zip(fused_boxes, fused_scores):
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            pred_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")

        with open(os.path.join(args.out, stem + ".txt"), "w") as f:
            f.write("\n".join(pred_lines))

    log.info(f"Ensembled predictions written to {args.out}/")


if __name__ == "__main__":
    main()
