#!/usr/bin/env python3
"""
Run inference on a multi-modal dataset and report mAP@0.50 per modality.

The challenge's final score is `(mAP@0.50_EO + mAP@0.50_IR) / 2`, so a single
overall number hides modality-specific problems (e.g. a great EO model that
fails on IR will report ~50% final score). This script splits the evaluation
by folder name convention:

  <root>/
    eo/  images, labels
    ir/  images, labels

or any other explicit --eo_dir / --ir_dir paths.

Each half is evaluated independently and printed separately. The combined
score is also computed (for comparison against the official final score
formula).

Usage:
    python scripts/evaluate_modality.py --weights runs/detect/cuas_train/weights/best.pt \
        --root data/real_reference

    python scripts/evaluate_modality.py --weights best.pt \
        --eo_dir data/real_reference/eo --ir_dir data/real_reference/ir
"""
import argparse
import os
import sys
from glob import glob

# Reuse the mAP logic from evaluate.py to keep one source of truth.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import compute_map50, yolo_to_xyxy  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg")


def evaluate_split(weights, images_dir, labels_dir, conf=0.15, modality=""):
    from PIL import Image
    from ultralytics import YOLO

    if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
        return None, None

    model = YOLO(weights)
    img_paths = sorted(
        p for p in glob(os.path.join(images_dir, "*")) if p.lower().endswith(IMG_EXTS)
    )
    if not img_paths:
        return None, None

    all_preds, all_gts = [], {}
    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        with Image.open(img_path) as im:
            w, h = im.size

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

        results = model.predict(img_path, conf=conf, verbose=False)[0]
        for box in results.boxes:
            cls = int(box.cls.item())
            if cls != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            all_preds.append((stem, [x1, y1, x2, y2], float(box.conf.item())))

    map50 = compute_map50(all_preds, all_gts)
    n_imgs = len(img_paths)
    n_drones = sum(len(v) for v in all_gts.values())
    return {
        "n_images": n_imgs,
        "n_drones": n_drones,
        "mAP50": map50,
    }, (all_preds, all_gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", default=None,
                     help="Root dir containing eo/ and ir/ subdirs (alternative to --eo_dir/--ir_dir)")
    ap.add_argument("--eo_dir", default=None, help="Override EO images dir (with labels/ sibling)")
    ap.add_argument("--ir_dir", default=None, help="Override IR images dir (with labels/ sibling)")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--out", default=None,
                     help="Optional dir to dump per-image predictions for submission")
    args = ap.parse_args()

    if args.root:
        eo_img = args.eo_dir or os.path.join(args.root, "eo", "images")
        eo_lbl = args.eo_dir.replace(os.sep + "images", os.sep + "labels") if args.eo_dir else os.path.join(args.root, "eo", "labels")
        ir_img = args.ir_dir or os.path.join(args.root, "ir", "images")
        ir_lbl = args.ir_dir.replace(os.sep + "images", os.sep + "labels") if args.ir_dir else os.path.join(args.root, "ir", "labels")
    else:
        if not (args.eo_dir and args.ir_dir):
            ap.error("Provide --root or both --eo_dir and --ir_dir")
        eo_img, eo_lbl = args.eo_dir, args.eo_dir.replace(os.sep + "images", os.sep + "labels")
        ir_img, ir_lbl = args.ir_dir, args.ir_dir.replace(os.sep + "images", os.sep + "labels")

    print(f"Weights: {args.weights}  conf={args.conf}\n")

    eo_result, _ = evaluate_split(args.weights, eo_img, eo_lbl, args.conf, "EO")
    print(f"EO  images_dir={eo_img}  labels_dir={eo_lbl}")
    if eo_result is None:
        print("  (no images found — skipping)")
    else:
        print(f"  n_images={eo_result['n_images']}  n_drones={eo_result['n_drones']}  "
              f"mAP@0.50={eo_result['mAP50']:.4f}")
    print()
    ir_result, _ = evaluate_split(args.weights, ir_img, ir_lbl, args.conf, "IR")
    print(f"IR  images_dir={ir_img}  labels_dir={ir_lbl}")
    if ir_result is None:
        print("  (no images found — skipping)")
    else:
        print(f"  n_images={ir_result['n_images']}  n_drones={ir_result['n_drones']}  "
              f"mAP@0.50={ir_result['mAP50']:.4f}")

    print()
    if eo_result and ir_result:
        combined = (eo_result["mAP50"] + ir_result["mAP50"]) / 2
        print(f"Combined final score  = (EO {eo_result['mAP50']:.4f} + IR {ir_result['mAP50']:.4f}) / 2 = {combined:.4f}")
    else:
        print("Combined final score  = N/A (need both EO and IR splits)")


if __name__ == "__main__":
    main()
