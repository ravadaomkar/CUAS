#!/usr/bin/env python3
"""
Bucket a model's failures on a labeled set into analysis/{false_positive,
false_negative,difficult_cases}/ as annotated images, so failure modes are
a quick visual scan instead of manually cross-referencing prediction and
label files.

Buckets:
  * false_positive   — a predicted box with no matching GT box at IoU >= 0.5
  * false_negative     — a GT box never matched by any prediction at IoU >= 0.5
  * difficult_cases     — matched, but only in the 0.5-0.7 IoU band (borderline
                            localization — worth a look even though technically "correct")

Usage:
    python scripts/error_analysis.py --weights checkpoints/best.pt \
        --data_dir data/real_reference --conf 0.15 --num_per_bucket 25
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper import list_images, read_yolo_labels
from utils.logger import get_logger
from utils.metrics import iou, yolo_to_xyxy
from utils.visualization import draw_box

log = get_logger("error_analysis")

GT_COLOR = (31, 119, 180)     # blue
PRED_COLOR = (44, 160, 44)     # green
OUT_DIRS = {
    "false_positive": "analysis/false_positive",
    "false_negative": "analysis/false_negative",
    "difficult_cases": "analysis/difficult_cases",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--num_per_bucket", type=int, default=25,
                     help="Cap on saved images per bucket, to keep this from dumping thousands of files")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed. Run:\n"
            "  pip install ultralytics --break-system-packages"
        )
    from PIL import Image, ImageDraw

    for d in OUT_DIRS.values():
        os.makedirs(d, exist_ok=True)

    model = YOLO(args.weights)
    images_dir = os.path.join(args.data_dir, "images")
    labels_dir = os.path.join(args.data_dir, "labels")
    img_paths = list_images(images_dir)
    if not img_paths:
        raise SystemExit(f"No images found in {images_dir}")

    saved = {k: 0 for k in OUT_DIRS}

    for img_path in img_paths:
        if all(saved[k] >= args.num_per_bucket for k in saved):
            break

        stem = os.path.splitext(os.path.basename(img_path))[0]
        with Image.open(img_path) as im:
            w, h = im.size
            img_rgb = im.convert("RGB")

        gt_boxes = [
            yolo_to_xyxy(box, w, h)
            for cls, box in read_yolo_labels(os.path.join(labels_dir, stem + ".txt"), class_filter=0)
        ]

        results = model.predict(img_path, conf=args.conf, verbose=False)[0]
        pred_boxes = []
        for box in results.boxes:
            if int(box.cls.item()) != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred_boxes.append(([x1, y1, x2, y2], float(box.conf.item())))

        gt_matched = [False] * len(gt_boxes)
        fp_boxes, difficult_pairs = [], []

        for pred_box, conf in pred_boxes:
            best_iou, best_j = 0.0, -1
            for j, gt_box in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                cur = iou(pred_box, gt_box)
                if cur > best_iou:
                    best_iou, best_j = cur, j
            if best_iou >= 0.5:
                gt_matched[best_j] = True
                if best_iou < 0.7:
                    difficult_pairs.append((pred_box, gt_boxes[best_j], best_iou))
            else:
                fp_boxes.append((pred_box, conf))

        fn_boxes = [gt_boxes[j] for j, matched in enumerate(gt_matched) if not matched]

        def save(bucket, annotate_fn):
            if saved[bucket] >= args.num_per_bucket:
                return
            canvas = img_rgb.copy()
            draw = ImageDraw.Draw(canvas)
            annotate_fn(draw)
            canvas.save(os.path.join(OUT_DIRS[bucket], f"{stem}.jpg"), quality=90)
            saved[bucket] += 1

        if fp_boxes:
            def _draw_fp(draw, boxes=fp_boxes):
                for box, conf in boxes:
                    draw_box(draw, box, PRED_COLOR, label=f"FP {conf:.2f}")
            save("false_positive", _draw_fp)

        if fn_boxes:
            def _draw_fn(draw, boxes=fn_boxes):
                for box in boxes:
                    draw_box(draw, box, GT_COLOR, label="FN (missed)")
            save("false_negative", _draw_fn)

        if difficult_pairs:
            def _draw_diff(draw, pairs=difficult_pairs):
                for pred_box, gt_box, iou_val in pairs:
                    draw_box(draw, gt_box, GT_COLOR, label="GT")
                    draw_box(draw, pred_box, PRED_COLOR, label=f"pred IoU={iou_val:.2f}")
            save("difficult_cases", _draw_diff)

    for bucket, count in saved.items():
        log.info(f"{bucket}: {count} images -> {OUT_DIRS[bucket]}/")


if __name__ == "__main__":
    main()
