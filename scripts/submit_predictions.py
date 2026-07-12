#!/usr/bin/env python3
"""
Generate the final submission artifact for the challenge.

The problem statement (Section 6) says "Submission format details will be
announced separately" — this script supports the two most likely formats and
makes it trivial to switch between them. As soon as the official spec drops,
edit SUBMISSION_FORMAT and re-run.

Currently supported formats:
  * 'yolo'   — one .txt per image, same format as evaluate.py emits.
               Conventionally bundled as a zip of all .txt files (no images).
  * 'coco'   — single results.json in COCO submission format
               (image_id, category_id, bbox[x,y,w,h], score).
  * 'csv'    — one CSV with columns image_id,confidence,x_min,y_min,x_max,y_max
               (always absolute pixel coords; the official form if they want a flat table).

Usage:
    python scripts/submit_predictions.py \
        --weights runs/detect/cuas_train/weights/best.pt \
        --test_dir /path/to/unseen_test_set \
        --format yolo --out submission/

    python scripts/submit_predictions.py --weights best.pt \
        --test_dir /path/to/test --format coco --out submission/
"""
import argparse
import json
import os
import sys
import zipfile
from glob import glob

IMG_EXTS = (".png", ".jpg", ".jpeg")


def detect_format_from_spec(spec_path):
    """If a JSON spec is provided, read SUBMISSION_FORMAT and overrides from it."""
    if not spec_path or not os.path.isfile(spec_path):
        return None, None
    with open(spec_path) as f:
        spec = json.load(f)
    return spec.get("format"), spec


def run_inference(weights, test_dir, conf):
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(weights)
    img_paths = sorted(
        p for p in glob(os.path.join(test_dir, "*")) if p.lower().endswith(IMG_EXTS)
    )
    if not img_paths:
        raise SystemExit(f"No images found in {test_dir}")

    records = []  # list of dicts: image_id, image_path, w, h, x1, y1, x2, y2, conf
    for img_path in img_paths:
        with Image.open(img_path) as im:
            w, h = im.size
        results = model.predict(img_path, conf=conf, verbose=False)[0]
        for box in results.boxes:
            cls = int(box.cls.item())
            if cls != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            records.append({
                "image_id": os.path.splitext(os.path.basename(img_path))[0],
                "image_path": img_path,
                "w": w, "h": h,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": float(box.conf.item()),
            })
    return records, img_paths


def write_yolo(records, out_dir, image_paths, conf, zip_bundle=True):
    """One .txt per image, lines: class xc yc w h conf (normalized)."""
    os.makedirs(out_dir, exist_ok=True)
    by_image = {}
    for r in records:
        by_image.setdefault(r["image_id"], []).append(r)
    written = 0
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, stem + ".txt")
        with open(out_path, "w") as f:
            for r in by_image.get(stem, []):
                xc = (r["x1"] + r["x2"]) / 2 / r["w"]
                yc = (r["y1"] + r["y2"]) / 2 / r["h"]
                bw = (r["x2"] - r["x1"]) / r["w"]
                bh = (r["y2"] - r["y1"]) / r["h"]
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {r['conf']:.4f}\n")
        written += 1
    print(f"  wrote {written} prediction files to {out_dir}/")
    if zip_bundle:
        zip_path = out_dir.rstrip("/") + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for img_path in image_paths:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                p = os.path.join(out_dir, stem + ".txt")
                if os.path.isfile(p):
                    zf.write(p, arcname=stem + ".txt")
        print(f"  bundled as {zip_path}")
        return zip_path
    return out_dir


def write_coco(records, out_dir, image_paths):
    """COCO submission format: results.json with image_id, category_id, bbox, score."""
    os.makedirs(out_dir, exist_ok=True)
    # COCO requires a contiguous integer image_id; build a mapping.
    img_id_map = {
        os.path.splitext(os.path.basename(p))[0]: i + 1
        for i, p in enumerate(image_paths)
    }
    results = []
    for r in records:
        x, y, w, h = r["x1"], r["y1"], r["x2"] - r["x1"], r["y2"] - r["y1"]
        results.append({
            "image_id": img_id_map[r["image_id"]],
            "category_id": 1,  # drone
            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            "score": round(r["conf"], 4),
        })
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  wrote {len(results)} detections to {out_path}")
    return out_path


def write_csv(records, out_dir, image_paths):
    """Flat CSV — always absolute pixel coords."""
    import csv
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "confidence", "x_min", "y_min", "x_max", "y_max"])
        for r in records:
            w.writerow([
                r["image_id"],
                f"{r['conf']:.4f}",
                int(round(r["x1"])), int(round(r["y1"])),
                int(round(r["x2"])), int(round(r["y2"])),
            ])
    print(f"  wrote {len(records)} rows to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test_dir", required=True,
                     help="Folder of unseen test images (no labels).")
    ap.add_argument("--format", choices=["yolo", "coco", "csv"], default="yolo")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--out", default="submission")
    ap.add_argument("--spec", default=None,
                     help="Optional JSON file with {format, conf, ...} overrides. "
                          "Lets the team swap submission shape the moment the official spec is released.")
    args = ap.parse_args()

    fmt, spec = detect_format_from_spec(args.spec)
    if fmt:
        args.format = fmt
    if spec and "conf" in spec:
        args.conf = float(spec["conf"])
    print(f"Format: {args.format}  conf={args.conf}  test_dir={args.test_dir}\n")

    records, image_paths = run_inference(args.weights, args.test_dir, args.conf)
    print(f"Inferred on {len(image_paths)} images, {len(records)} drone detections.\n")

    os.makedirs(args.out, exist_ok=True)
    if args.format == "yolo":
        artifact = write_yolo(records, args.out, image_paths, args.conf)
    elif args.format == "coco":
        artifact = write_coco(records, args.out, image_paths)
    else:
        artifact = write_csv(records, args.out, image_paths)

    # Always also dump a small manifest so the team can audit
    manifest = {
        "weights": os.path.abspath(args.weights),
        "test_dir": os.path.abspath(args.test_dir),
        "n_images": len(image_paths),
        "n_detections": len(records),
        "format": args.format,
        "conf_threshold": args.conf,
        "artifact": os.path.abspath(str(artifact)),
    }
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
