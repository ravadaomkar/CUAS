#!/usr/bin/env python3
"""
Visual debugging tool: draw predicted vs ground-truth boxes on a folder of
images and save an annotated copy side-by-side.

Useful for:
  * Spot-checking which image types the model is missing
  * Sharing "failure cases" with the team without dumping whole images
  * Quickly comparing EO vs IR model behavior on a small set

Usage:
    python scripts/visualize_predictions.py --weights best.pt \
        --images_dir data/real_reference/eo/images \
        --labels_dir data/real_reference/eo/labels \
        --out viz/eo --num 20
"""
import argparse
import os
import random
from glob import glob

import numpy as np

CLASS_COLORS = {0: (31, 119, 180), 1: (214, 39, 40), 2: (148, 103, 189)}
CLASS_NAMES = {0: "drone (gt)", 1: "bird (gt)", 2: "aircraft (gt)"}
PRED_COLOR = (44, 160, 44)  # green for predictions


def yolo_to_xyxy(box, w, h):
    xc, yc, bw, bh = box
    return [
        (xc - bw / 2) * w,
        (yc - bh / 2) * h,
        (xc + bw / 2) * w,
        (yc + bh / 2) * h,
    ]


def draw_box(arr, box, color, label=None, lw=2):
    from PIL import Image, ImageDraw, ImageFont
    if isinstance(arr, np.ndarray):
        img = Image.fromarray(arr)
    else:
        img = arr
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
    if label:
        # White background + colored text
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        tb = draw.textbbox((x1, max(y1 - 12, 0)), label, font=font) if font else (x1, y1 - 10, x1 + 60, y1)
        draw.rectangle(tb, fill=color)
        draw.text((tb[0] + 1, tb[1] + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_dir", default=None,
                     help="Optional. If provided, ground truth is drawn in red/blue/purple. "
                          "If absent, only predictions are shown.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    os.makedirs(args.out, exist_ok=True)
    img_paths = sorted(
        p for p in glob(os.path.join(args.images_dir, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not img_paths:
        raise SystemExit(f"No images in {args.images_dir}")
    random.Random(args.seed).shuffle(img_paths)
    img_paths = img_paths[: args.num]

    model = YOLO(args.weights)
    n_with_pred = 0
    n_with_gt = 0
    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        with Image.open(img_path) as im:
            iw, ih = im.size
            arr = np.asarray(im.convert("RGB")).copy()

        # GT
        if args.labels_dir:
            lbl = os.path.join(args.labels_dir, stem + ".txt")
            if os.path.exists(lbl):
                n_with_gt += 1
                with open(lbl) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls = int(parts[0])
                        box = list(map(float, parts[1:5]))
                        x1, y1, x2, y2 = yolo_to_xyxy(box, iw, ih)
                        color = CLASS_COLORS.get(cls, (200, 200, 0))
                        name = CLASS_NAMES.get(cls, f"cls{cls}")
                        arr = draw_box(arr, [x1, y1, x2, y2], color, label=name)

        # Pred
        results = model.predict(img_path, conf=args.conf, verbose=False)[0]
        n_this = 0
        for box in results.boxes:
            cls = int(box.cls.item())
            if cls != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf.item())
            arr = draw_box(arr, [x1, y1, x2, y2], PRED_COLOR, label=f"drone {conf:.2f}")
            n_this += 1
        if n_this:
            n_with_pred += 1

        Image.fromarray(arr).save(os.path.join(args.out, stem + ".jpg"), quality=90)

    print(f"Wrote {len(img_paths)} annotated images to {args.out}/")
    print(f"  with at least one prediction: {n_with_pred}")
    if args.labels_dir:
        print(f"  with ground truth:            {n_with_gt}")


if __name__ == "__main__":
    main()
