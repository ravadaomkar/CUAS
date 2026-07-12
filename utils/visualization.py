"""
Shared plotting helpers for results/. Kept dependency-light (matplotlib
only) since this runs after every evaluation, not just in notebooks.
"""
import os

import numpy as np


def plot_pr_curve(recall, precision, ap, out_path, title="Precision-Recall (drone, IoU=0.5)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#1f77b4", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{title}\nAP@0.50 = {ap:.4f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(counts, out_path, class_name="drone"):
    """counts: dict with tp/fp/fn from utils.metrics.confusion_counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # 2x2 grid: rows = actual (object, background), cols = predicted (object, background)
    matrix = np.array([
        [counts["tp"], counts["fn"]],
        [counts["fp"], 0],  # true negatives aren't well-defined for detection
    ])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"pred: {class_name}", "pred: none"])
    ax.set_yticklabels([f"actual: {class_name}", "actual: none"])
    for i in range(2):
        for j in range(2):
            if i == 1 and j == 1:
                continue  # TN not applicable, leave blank
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                     fontsize=14, fontweight="bold")
    ax.set_title(f"Confusion counts (P={counts['precision']:.3f}, R={counts['recall']:.3f}, F1={counts['f1']:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_box(draw, box, color, label=None, lw=2):
    """Draw a single box + optional label on a PIL ImageDraw context.
    Shared by visualize_predictions.py and the analysis/ bucketing script."""
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
    if label:
        try:
            draw.text((x1 + 2, max(y1 - 12, 0)), label, fill=color)
        except Exception:
            pass
