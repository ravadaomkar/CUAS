"""
Small shared utilities that don't belong in metrics.py, visualization.py,
or logger.py — path handling, run bookkeeping, and generic file I/O used
across scripts/, ensemble/, and analysis/.
"""
import csv
import glob
import os
from datetime import datetime

IMG_EXTS = (".png", ".jpg", ".jpeg")


def list_images(images_dir):
    """Sorted list of image paths in a directory, restricted to IMG_EXTS."""
    return sorted(
        p for p in glob.glob(os.path.join(images_dir, "*"))
        if p.lower().endswith(IMG_EXTS)
    )


def read_yolo_labels(label_path, class_filter=None):
    """Parse a YOLO-format .txt label file into a list of (cls, [xc,yc,w,h])."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if class_filter is not None and cls != class_filter:
                continue
            boxes.append((cls, list(map(float, parts[1:5]))))
    return boxes


def next_experiment_dir(experiments_root="experiments"):
    """Return the next free experiments/expNN directory, e.g. experiments/exp04."""
    existing = sorted(glob.glob(os.path.join(experiments_root, "exp*")))
    nums = []
    for path in existing:
        name = os.path.basename(path)
        digits = "".join(ch for ch in name if ch.isdigit())
        if digits:
            nums.append(int(digits))
    next_num = (max(nums) + 1) if nums else 1
    new_dir = os.path.join(experiments_root, f"exp{next_num:02d}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def append_metrics_row(csv_path, row: dict):
    """Append a row (dict) to a metrics CSV, writing the header if the file
    is new. Used by tune.py and evaluate*.py so every run's numbers land in
    one place instead of only stdout."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
