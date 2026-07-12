#!/usr/bin/env python3
"""
Evaluate a trained model's mAP@0.50 against a labeled dataset (e.g. the local
real-world reference set for validation, or later the EO/IR halves separately).

IMPORTANT: only ever point this at data/real_reference or held-out synthetic
val data — never train on the reference sets (Rule 2 of the challenge).

Usage:
    python evaluate.py --weights runs/detect/cuas_train/weights/best.pt \
        --data_dir data/real_reference --out predictions/

This will:
  1. Run inference over data_dir/images
  2. Compute mAP@0.50 against data_dir/labels (YOLO format, class 0 = drone)
  3. Write per-image predictions in YOLO format to --out for later submission
"""
import argparse
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper import append_metrics_row
from utils.logger import get_logger
from utils.metrics import compute_ap, confusion_counts, match_predictions, yolo_to_xyxy

log = get_logger("evaluate")


def compute_map50(all_preds, all_gts):
    """Thin wrapper kept for backward compatibility with any external
    callers; the real implementation now lives in utils/metrics.py so
    tune.py and inference_tta.py can reuse the same AP math."""
    n_gt = sum(len(v) for v in all_gts.values())
    if n_gt == 0:
        return 0.0
    tp, fp, _, _, _ = match_predictions(all_preds, all_gts, iou_thresh=0.5)
    ap, _, _ = compute_ap(tp, fp, n_gt)
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True,
                     help="Folder with images/ and labels/ subfolders")
    ap.add_argument("--out", type=str, default="predictions")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--plots", type=str, default=None,
                     help="If set, write confusion_matrix.png, pr_curve.png, and "
                          "metrics.csv into this directory (e.g. results/)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed. Run:\n"
            "  pip install ultralytics --break-system-packages"
        )

    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)

    images_dir = os.path.join(args.data_dir, "images")
    labels_dir = os.path.join(args.data_dir, "labels")
    img_paths = sorted(
        p for p in glob(os.path.join(images_dir, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not img_paths:
        raise SystemExit(f"No images found in {images_dir}")

    all_preds = []
    all_gts = {}

    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        with Image.open(img_path) as im:
            w, h = im.size

        # Ground truth (class 0 only)
        gt_boxes = []
        label_path = os.path.join(labels_dir, stem + ".txt")
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5 or int(parts[0]) != 0:
                        continue
                    box = list(map(float, parts[1:5]))
                    gt_boxes.append(yolo_to_xyxy(box, w, h))
        all_gts[stem] = gt_boxes

        # Predictions
        results = model.predict(img_path, conf=args.conf, verbose=False)[0]
        pred_lines = []
        for box in results.boxes:
            cls = int(box.cls.item())
            if cls != 0:
                continue
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            all_preds.append((stem, [x1, y1, x2, y2], conf))

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            pred_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")

        with open(os.path.join(args.out, stem + ".txt"), "w") as f:
            f.write("\n".join(pred_lines))

    n_gt = sum(len(v) for v in all_gts.values())
    tp, fp, fp_ids, fn_ids, sorted_preds = match_predictions(all_preds, all_gts, iou_thresh=0.5)
    map50, recall_curve, precision_curve = compute_ap(tp, fp, n_gt)
    counts = confusion_counts(tp, fp, n_gt)

    log.info(f"Evaluated {len(img_paths)} images")
    log.info(f"mAP@0.50 (drone class) = {map50:.4f}")
    log.info(f"TP={counts['tp']} FP={counts['fp']} FN={counts['fn']} "
             f"P={counts['precision']:.3f} R={counts['recall']:.3f} F1={counts['f1']:.3f}")
    log.info(f"Per-image predictions written to {args.out}/")

    if args.plots:
        from utils.visualization import plot_confusion_matrix, plot_pr_curve
        os.makedirs(args.plots, exist_ok=True)
        plot_pr_curve(recall_curve, precision_curve, map50,
                      os.path.join(args.plots, "pr_curve.png"))
        plot_confusion_matrix(counts, os.path.join(args.plots, "confusion_matrix.png"))
        append_metrics_row(os.path.join(args.plots, "metrics.csv"), {
            "weights": args.weights,
            "data_dir": args.data_dir,
            "conf": args.conf,
            "mAP50": round(map50, 4),
            **{k: v for k, v in counts.items()},
        })
        log.info(f"Plots + metrics.csv written to {args.plots}/")


if __name__ == "__main__":
    main()
