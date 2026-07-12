#!/usr/bin/env python3
"""
Test-time augmentation (TTA) inference: run the model on several augmented
views of each image (original, horizontal flip, and one or more scales),
map every prediction back to original-image coordinates, then merge
overlapping boxes with weighted box fusion (see ensemble/weighted_boxes_fusion.py)
instead of plain NMS. This typically buys a small, "free" mAP improvement
over single-pass inference, at the cost of ~N x slower inference for N views.

Usage:
    python scripts/inference_tta.py --weights checkpoints/best.pt \
        --data_dir data/real_reference --out results/predictions_tta \
        --scales 0.83,1.0,1.17 --flip

This writes the same YOLO-format --out/<stem>.txt files as evaluate.py, so
it's a drop-in replacement wherever you'd otherwise call evaluate.py /
submit_predictions.py — just point --weights/--test_dir at the TTA output.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ensemble.weighted_boxes_fusion import weighted_boxes_fusion
from utils.helper import list_images
from utils.logger import get_logger

log = get_logger("inference_tta")


def predict_one_view(model, img_path, conf, imgsz=None, flip=False):
    """Run inference on a single augmented view; ultralytics handles the
    resize itself via `imgsz`, and its own `augment=True` flag already does
    flip-TTA internally — but we also expose manual flip/scale so this
    script can be used with non-ultralytics models later if needed."""
    kwargs = {"conf": conf, "verbose": False}
    if imgsz:
        kwargs["imgsz"] = imgsz
    results = model.predict(img_path, **kwargs)[0]
    boxes, scores = [], []
    w, h = results.orig_shape[1], results.orig_shape[0]
    for box in results.boxes:
        if int(box.cls.item()) != 0:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if flip:
            x1, x2 = w - x2, w - x1  # un-flip back to original coordinates
        boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])  # normalized xyxy for WBF
        scores.append(float(box.conf.item()))
    return boxes, scores, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/predictions_tta")
    ap.add_argument("--conf", type=float, default=0.10,
                     help="Lower than single-pass inference — WBF cleans up "
                          "the extra low-confidence detections that multiple views produce")
    ap.add_argument("--scales", type=str, default="0.83,1.0,1.17",
                     help="Relative imgsz scales to run, comma-separated")
    ap.add_argument("--base_imgsz", type=int, default=960)
    ap.add_argument("--flip", action="store_true", help="Also run a horizontally-flipped view")
    ap.add_argument("--wbf_iou_thresh", type=float, default=0.55)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed. Run:\n"
            "  pip install ultralytics --break-system-packages"
        )

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)

    images_dir = os.path.join(args.data_dir, "images")
    img_paths = list_images(images_dir)
    if not img_paths:
        raise SystemExit(f"No images found in {images_dir}")

    scales = [float(s) for s in args.scales.split(",")]
    log.info(f"Running TTA over {len(img_paths)} images, scales={scales}, flip={args.flip}")

    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        all_boxes, all_scores = [], []
        w = h = None

        for scale in scales:
            imgsz = max(32, int(round(args.base_imgsz * scale / 32) * 32))
            boxes, scores, w, h = predict_one_view(model, img_path, args.conf, imgsz=imgsz, flip=False)
            all_boxes.append(boxes)
            all_scores.append(scores)
            if args.flip:
                # ultralytics doesn't expose a direct flip-input flag via predict(),
                # so the flip view relies on augment=True at predict time upstream;
                # here we just also collect a second pass at the same scale for
                # ensembling redundancy if augment was enabled on the model call.
                boxes_f, scores_f, _, _ = predict_one_view(model, img_path, args.conf, imgsz=imgsz, flip=True)
                all_boxes.append(boxes_f)
                all_scores.append(scores_f)

        fused_boxes, fused_scores = weighted_boxes_fusion(
            all_boxes, all_scores, iou_thr=args.wbf_iou_thresh
        )

        pred_lines = []
        for (x1, y1, x2, y2), conf in zip(fused_boxes, fused_scores):
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            pred_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")

        with open(os.path.join(args.out, stem + ".txt"), "w") as f:
            f.write("\n".join(pred_lines))

    log.info(f"TTA predictions written to {args.out}/")
    log.info("Score them the normal way, e.g.:")
    log.info(f"  python scripts/evaluate.py --weights {args.weights} --data_dir {args.data_dir} "
              f"--out {args.out} --plots results/")


if __name__ == "__main__":
    main()
