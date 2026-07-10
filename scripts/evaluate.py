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
from glob import glob

import numpy as np


def yolo_to_xyxy(box, w, h):
    xc, yc, bw, bh = box
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return [x1, y1, x2, y2]


def iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_map50(all_preds, all_gts):
    """
    all_preds: list of (image_id, [x1,y1,x2,y2], conf)
    all_gts:   dict image_id -> list of [x1,y1,x2,y2]
    Simple single-class AP@0.5 via 11-point-free precision/recall integration.
    """
    all_preds = sorted(all_preds, key=lambda x: -x[2])
    n_gt = sum(len(v) for v in all_gts.values())
    if n_gt == 0:
        return 0.0

    matched = {k: [False] * len(v) for k, v in all_gts.items()}
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for i, (img_id, pred_box, _) in enumerate(all_preds):
        gts = all_gts.get(img_id, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gts):
            if matched[img_id][j]:
                continue
            cur_iou = iou(pred_box, gt_box)
            if cur_iou > best_iou:
                best_iou, best_j = cur_iou, j
        if best_iou >= 0.5 and best_j >= 0:
            tp[i] = 1
            matched[img_id][best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # Append sentinel points and compute AP via area under PR curve (all-point interpolation)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return float(ap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True,
                     help="Folder with images/ and labels/ subfolders")
    ap.add_argument("--out", type=str, default="predictions")
    ap.add_argument("--conf", type=float, default=0.15)
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

    map50 = compute_map50(all_preds, all_gts)
    print(f"\nEvaluated {len(img_paths)} images")
    print(f"mAP@0.50 (drone class) = {map50:.4f}")
    print(f"Per-image predictions written to {args.out}/")


if __name__ == "__main__":
    main()
