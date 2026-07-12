#!/usr/bin/env python3
"""
Merge Vibe Sim scenario outputs (rural / urban / ir, each possibly split into
multiple batch subfolders) into a single YOLO-format train/val dataset.

Expected input layout (per scenario), e.g.:
    data/rural/batchA/images/*.png
    data/rural/batchA/labels/*.txt
    data/rural/batchB/images/*.png
    data/rural/batchB/labels/*.txt
    data/urban/.../images, labels
    data/ir/.../images, labels

Also supports a flat layout:
    data/rural/images/*.png
    data/rural/labels/*.txt

Output:
    data/merged/images/train, data/merged/images/val
    data/merged/labels/train, data/merged/labels/val

Usage:
    python prepare_dataset.py --scenarios rural urban --out data/merged --val_split 0.15
    python prepare_dataset.py --scenarios rural urban ir --out data/merged --val_split 0.15 \
        --drop_clutter_classes
"""
import argparse
import os
import random
import shutil
from glob import glob


IMG_EXTS = (".png", ".jpg", ".jpeg")


def find_image_label_pairs(scenario_dir: str):
    """Recursively find images/labels folders anywhere under scenario_dir and
    return matched (image_path, label_path) pairs."""
    pairs = []
    for images_dir in glob(os.path.join(scenario_dir, "**", "images"), recursive=True):
        labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
        if not os.path.isdir(labels_dir):
            continue
        for img_path in glob(os.path.join(images_dir, "*")):
            if not img_path.lower().endswith(IMG_EXTS):
                continue
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            if os.path.exists(label_path):
                pairs.append((img_path, label_path))
    return pairs


def filter_clutter(label_path: str, out_label_path: str):
    """Write a copy of the label file keeping only class 0 (drone) rows."""
    with open(label_path) as fin, open(out_label_path, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if not parts:
                continue
            if int(parts[0]) == 0:
                fout.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--scenarios", nargs="+", default=["rural", "urban"],
                     help="Subfolders under data_root to merge, e.g. rural urban ir")
    ap.add_argument("--out", type=str, default="data/merged")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop_clutter_classes", action="store_true",
                     help="If set, strip bird(1)/aircraft(2) labels so only class 0 remains "
                          "(use if you want a single-class detector). Default: keep all classes.")
    args = ap.parse_args()

    random.seed(args.seed)

    all_pairs = []
    for scenario in args.scenarios:
        scenario_dir = os.path.join(args.data_root, scenario)
        if not os.path.isdir(scenario_dir):
            print(f"WARNING: {scenario_dir} does not exist, skipping.")
            continue
        pairs = find_image_label_pairs(scenario_dir)
        print(f"{scenario}: found {len(pairs)} image/label pairs")
        all_pairs.extend(pairs)

    if not all_pairs:
        raise SystemExit(
            "No image/label pairs found. Populate data/<scenario>/images and "
            "data/<scenario>/labels with Vibe Sim exports first."
        )

    random.shuffle(all_pairs)
    n_val = int(len(all_pairs) * args.val_split)
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    for split, pairs in (("train", train_pairs), ("val", val_pairs)):
        img_out = os.path.join(args.out, "images", split)
        lbl_out = os.path.join(args.out, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_path, label_path in pairs:
            base = os.path.basename(img_path)
            stem = os.path.splitext(base)[0]
            dst_img = os.path.join(img_out, base)
            dst_lbl = os.path.join(lbl_out, stem + ".txt")

            # avoid filename collisions across scenarios
            if os.path.exists(dst_img):
                dst_img = os.path.join(img_out, f"{stem}_{random.randint(0, 999999)}{os.path.splitext(base)[1]}")
                dst_lbl = os.path.join(lbl_out, os.path.splitext(os.path.basename(dst_img))[0] + ".txt")

            shutil.copy2(img_path, dst_img)
            if args.drop_clutter_classes:
                filter_clutter(label_path, dst_lbl)
            else:
                shutil.copy2(label_path, dst_lbl)

    print(f"\nDone. train={len(train_pairs)} val={len(val_pairs)}")
    print(f"Output at: {args.out}")


if __name__ == "__main__":
    main()
