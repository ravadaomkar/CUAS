#!/usr/bin/env python3
"""
Analyze pixels-on-target (bounding box area) distributions.

Two modes:
1. --ieee_npy: load the provided ieee_pixels_on_target.npy and report stats
   (mean, std, percentiles) to feed back into Vibe Sim's custom distribution.
2. --labels_dir: compute a pixels-on-target distribution from a folder of
   YOLO-format label .txt files + matching images (used for the IR reference
   set in Phase 2, which has no pre-supplied .npy). Saves a .npy you can then
   load into Vibe Sim as a custom distribution target.

Usage:
    python analyze_pot_distribution.py --ieee_npy path/to/ieee_pixels_on_target.npy \
        --out strategy/pot_report.png

    python analyze_pot_distribution.py --labels_dir data/real_reference/labels \
        --images_dir data/real_reference/images \
        --save_npy strategy/duality_ir_pixels_on_target.npy \
        --out strategy/ir_pot_report.png
"""
import argparse
import glob
import os

import numpy as np


def load_yolo_areas(labels_dir: str, images_dir: str) -> np.ndarray:
    """Compute bbox pixel areas from YOLO-format labels (normalized xywh)
    matched against image dimensions. Requires Pillow for image size lookup."""
    from PIL import Image

    areas = []
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No .txt label files found in {labels_dir}")

    for lf in label_files:
        stem = os.path.splitext(os.path.basename(lf))[0]
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(images_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if cls != 0:  # drone class only
                    continue
                _, _, _, bw, bh = map(float, parts[:5])
                areas.append((bw * w) * (bh * h))

    return np.array(areas, dtype=np.float64)


def report(areas: np.ndarray, out_path: str, title: str):
    if areas.size == 0:
        print("No areas found — check input paths.")
        return

    print(f"\n=== {title} ===")
    print(f"n = {areas.size}")
    print(f"mean area (px^2)   = {areas.mean():.1f}")
    print(f"std area (px^2)    = {areas.std():.1f}")
    print(f"median area (px^2) = {np.median(areas):.1f}")
    for p in (5, 25, 50, 75, 95):
        print(f"p{p:<3} = {np.percentile(areas, p):.1f}")

    side_equiv = np.sqrt(areas)
    print("\n(equivalent square side length, px, for intuition)")
    print(f"mean side = {side_equiv.mean():.1f}, std side = {side_equiv.std():.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(areas, bins=40)
        ax[0].set_title(f"{title}: bbox area (px^2)")
        ax[0].set_xlabel("area")
        ax[1].hist(side_equiv, bins=40)
        ax[1].set_title(f"{title}: equivalent side length (px)")
        ax[1].set_xlabel("side length")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved plot to {out_path}")
    except ImportError:
        print("matplotlib not installed — skipping plot.")

    print(
        "\nFeed these numbers into the Vibe Sim agent, e.g.:\n"
        f'  "Make sure the drones are uniformly distributed between '
        f"{np.percentile(areas, 5):.0f} and {np.percentile(areas, 95):.0f} square pixels\"\n"
        "or prefer loading the .npy directly as a Custom distribution if your "
        "Vibe Sim build supports it (recommended — preserves the real shape, "
        "not just a uniform envelope)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ieee_npy", type=str, default=None,
                     help="Path to provided ieee_pixels_on_target.npy")
    ap.add_argument("--labels_dir", type=str, default=None,
                     help="YOLO labels dir to compute a distribution from scratch (e.g. IR reference set)")
    ap.add_argument("--images_dir", type=str, default=None,
                     help="Matching images dir (required with --labels_dir)")
    ap.add_argument("--save_npy", type=str, default=None,
                     help="Where to save computed areas as .npy (use with --labels_dir)")
    ap.add_argument("--out", type=str, default="pot_report.png")
    args = ap.parse_args()

    if args.ieee_npy:
        areas = np.load(args.ieee_npy)
        report(areas, args.out, "IEEE EO reference")

    if args.labels_dir:
        if not args.images_dir:
            raise ValueError("--images_dir is required with --labels_dir")
        areas = load_yolo_areas(args.labels_dir, args.images_dir)
        report(areas, args.out, "Computed from labels")
        if args.save_npy:
            np.save(args.save_npy, areas)
            print(f"Saved computed distribution to {args.save_npy}")

    if not args.ieee_npy and not args.labels_dir:
        ap.error("Provide --ieee_npy and/or --labels_dir")


if __name__ == "__main__":
    main()
