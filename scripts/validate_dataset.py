#!/usr/bin/env python3
"""
Pre-train sanity check for the merged YOLO dataset.

Catches the failure modes that have historically cost this project the most
GPU-hours:
  * Empty / unbalanced train vs val split
  * Orphaned images (no matching label) or vice versa
  * Corrupt images
  * Labels with out-of-range or non-numeric values
  * Labels with class IDs not in the data.yaml names list
  * Boxes that are degenerate (w<=0 or h<=0 or outside [0,1])
  * Suspiciously few drone boxes (model won't learn)
  * Train/val distribution skew (val mAP will be misleading)

Usage:
    python scripts/validate_dataset.py --data configs/data.yaml
    python scripts/validate_dataset.py --images_dir data/merged/images/train \
                                       --labels_dir data/merged/labels/train \
                                       --class_names drone bird manned_aircraft
"""
import argparse
import os
import sys
from glob import glob

import yaml


def load_data_yaml(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.get("path", ".")
    train_rel = cfg["train"]
    val_rel = cfg.get("val")
    # Resolve relative to the yaml file's directory, not cfg['path'] (which
    # ultralytics treats as the dataset root).
    yaml_dir = os.path.dirname(os.path.abspath(path))
    train = os.path.normpath(os.path.join(yaml_dir, train_rel))
    val = os.path.normpath(os.path.join(yaml_dir, val_rel)) if val_rel else None
    return {
        "names": cfg.get("names", {}),
        "nc": len(cfg.get("names", {})),
        "train": train,
        "val": val,
    }


IMG_EXTS = (".png", ".jpg", ".jpeg")


def find_pairs(images_dir, labels_dir):
    pairs, orphan_imgs, orphan_lbls = [], [], []
    img_files = sorted(
        p for p in glob(os.path.join(images_dir, "*"))
        if p.lower().endswith(IMG_EXTS)
    )
    lbl_files = sorted(glob(os.path.join(labels_dir, "*.txt")))

    img_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in img_files}
    lbl_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in lbl_files}

    for stem, img in img_stems.items():
        if stem in lbl_stems:
            pairs.append((img, lbl_stems[stem]))
        else:
            orphan_imgs.append(img)
    for stem, lbl in lbl_stems.items():
        if stem not in img_stems:
            orphan_lbls.append(lbl)
    return pairs, orphan_imgs, orphan_lbls


def validate_label_file(path, allowed_classes):
    """Yield (level, message) tuples. level in {'ok','warn','error'}."""
    from PIL import Image
    issues = []
    n_rows = 0
    n_drone = 0
    n_clutter = 0
    n_bad_box = 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            n_rows += 1
            if len(parts) < 5:
                issues.append(("error", f"{path}:{ln} <5 fields"))
                continue
            try:
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
            except ValueError:
                issues.append(("error", f"{path}:{ln} non-numeric"))
                continue
            if allowed_classes is not None and cls not in allowed_classes:
                issues.append(("error", f"{path}:{ln} unknown class {cls}"))
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                issues.append(("warn", f"{path}:{ln} center outside [0,1]"))
            if bw <= 0 or bh <= 0 or bw > 1.0 or bh > 1.0:
                issues.append(("error", f"{path}:{ln} bad box w={bw} h={bh}"))
                n_bad_box += 1
            if cls == 0:
                n_drone += 1
            else:
                n_clutter += 1
    return n_rows, n_drone, n_clutter, n_bad_box, issues


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="YOLO data.yaml (uses train/val paths from it)")
    src.add_argument("--images_dir", help="Raw images dir (use with --labels_dir)")
    ap.add_argument("--labels_dir")
    ap.add_argument("--class_names", nargs="+", default=None,
                     help="Optional list of class names (matched to integer IDs in order)")
    args = ap.parse_args()

    if args.data:
        cfg = load_data_yaml(args.data)
        print(f"Loaded {args.data}: {cfg['nc']} classes, names={cfg['names']}")
        splits = []
        for split_name in ("train", "val"):
            img_dir = cfg[split_name]
            if img_dir is None:
                continue
            lbl_dir = img_dir.replace(os.sep + "images", os.sep + "labels")
            if not os.path.isdir(img_dir):
                print(f"  WARNING: split '{split_name}' dir {img_dir} does not exist")
                continue
            splits.append((split_name, img_dir, lbl_dir))
        allowed = set(int(k) for k in cfg["names"].keys()) if cfg["names"] else None
    else:
        if not args.labels_dir:
            ap.error("--labels_dir is required with --images_dir")
        allowed = set(range(len(args.class_names))) if args.class_names else None
        splits = [("custom", args.images_dir, args.labels_dir)]

    if not splits:
        print("No splits to validate. Populate data/merged/{train,val} first.")
        sys.exit(1)

    overall_ok = True
    for split_name, img_dir, lbl_dir in splits:
        print(f"\n=== Split: {split_name} ===")
        print(f"  images: {img_dir}")
        print(f"  labels: {lbl_dir}")
        pairs, orphan_imgs, orphan_lbls = find_pairs(img_dir, lbl_dir)
        print(f"  paired: {len(pairs)}   orphan_images: {len(orphan_imgs)}   orphan_labels: {len(orphan_lbls)}")
        if orphan_imgs:
            print(f"  WARNING: {len(orphan_imgs)} images have no matching label (will be skipped by ultralytics)")
        if orphan_lbls:
            print(f"  WARNING: {len(orphan_lbls)} labels have no matching image (will be skipped by ultralytics)")
            overall_ok = False

        # Corrupt image check
        from PIL import Image
        corrupt = 0
        for img, _ in pairs[:min(200, len(pairs))]:  # sample
            try:
                with Image.open(img) as im:
                    im.size
            except Exception:
                corrupt += 1
        if corrupt:
            print(f"  WARNING: {corrupt} of first {min(200, len(pairs))} images failed to open")
            overall_ok = False

        # Label content check
        n_rows = n_drone = n_clutter = n_bad = 0
        all_issues = []
        for _, lbl in pairs:
            r, d, c, b, issues = validate_label_file(lbl, allowed)
            n_rows += r
            n_drone += d
            n_clutter += c
            n_bad += b
            all_issues.extend(issues)
        n_err = sum(1 for lvl, _ in all_issues if lvl == "error")
        n_warn = sum(1 for lvl, _ in all_issues if lvl == "warn")
        print(f"  total label rows: {n_rows}  (drone={n_drone}, clutter={n_clutter})")
        print(f"  bad boxes: {n_bad}  errors: {n_err}  warnings: {n_warn}")
        if n_err:
            for lvl, msg in all_issues:
                if lvl == "error":
                    print(f"    [{lvl}] {msg}")
                    overall_ok = False
        if n_warn:
            for lvl, msg in all_issues[:5]:
                if lvl == "warn":
                    print(f"    [{lvl}] {msg}")
            if n_warn > 5:
                print(f"    ... and {n_warn - 5} more warnings")
        if n_drone == 0:
            print(f"  ERROR: split '{split_name}' has zero drone labels — model will not learn")
            overall_ok = False
        elif n_drone < 50:
            print(f"  WARNING: split '{split_name}' has only {n_drone} drone labels — consider generating more data")

    # Train/val distribution skew (only meaningful when both splits exist)
    if len(splits) == 2:
        n_imgs = [len(find_pairs(s[1], s[2])[0]) for s in splits]
        if min(n_imgs) > 0:
            ratio = max(n_imgs) / min(n_imgs)
            if ratio > 5:
                print(f"\nWARNING: train/val size ratio is {ratio:.1f}x — val mAP will be noisy on the smaller split")

    print()
    if overall_ok:
        print("OK — dataset is ready for training.")
        sys.exit(0)
    else:
        print("FAIL — fix the errors above before training.")
        sys.exit(2)


if __name__ == "__main__":
    main()
