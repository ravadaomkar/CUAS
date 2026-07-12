"""
Shared detection-metrics helpers, factored out of scripts/evaluate.py so
tune.py, inference_tta.py, and analysis scripts all score predictions the
same way instead of re-implementing IoU/AP math.

All functions operate on the plain-Python box format used everywhere else
in this repo: [x1, y1, x2, y2] in absolute pixel coordinates.
"""
from collections import defaultdict

import numpy as np


def yolo_to_xyxy(box, w, h):
    xc, yc, bw, bh = box
    return [
        (xc - bw / 2) * w,
        (yc - bh / 2) * h,
        (xc + bw / 2) * w,
        (yc + bh / 2) * h,
    ]


def iou(box1, box2):
    xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
    xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_predictions(all_preds, all_gts, iou_thresh=0.5):
    """
    Greedy-match predictions to ground truth at a single IoU threshold.

    all_preds: list of (image_id, box_xyxy, conf)
    all_gts:   dict image_id -> list of box_xyxy

    Returns per-prediction tp/fp arrays (sorted by descending conf) plus,
    for each image, the indices of GT boxes that were never matched
    (i.e. false negatives) — this is what analysis/ scripts need to bucket
    failure cases.
    """
    all_preds = sorted(all_preds, key=lambda x: -x[2])
    matched = {k: [False] * len(v) for k, v in all_gts.items()}
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    fp_image_ids = []

    for i, (img_id, pred_box, _) in enumerate(all_preds):
        gts = all_gts.get(img_id, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gts):
            if matched.get(img_id, [])[j] if img_id in matched else False:
                pass
            if img_id in matched and matched[img_id][j]:
                continue
            cur_iou = iou(pred_box, gt_box)
            if cur_iou > best_iou:
                best_iou, best_j = cur_iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp[i] = 1
            matched[img_id][best_j] = True
        else:
            fp[i] = 1
            fp_image_ids.append(img_id)

    fn_image_ids = defaultdict(list)
    for img_id, flags in matched.items():
        for j, was_matched in enumerate(flags):
            if not was_matched:
                fn_image_ids[img_id].append(j)

    return tp, fp, fp_image_ids, fn_image_ids, all_preds


def compute_ap(tp, fp, n_gt):
    """All-point-interpolated AP from cumulative tp/fp arrays (already
    sorted by descending confidence)."""
    if n_gt == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:])), recall, precision


def compute_map50(all_preds, all_gts, iou_thresh=0.5):
    """Convenience wrapper matching the signature evaluate.py already uses."""
    n_gt = sum(len(v) for v in all_gts.values())
    tp, fp, _, _, _ = match_predictions(all_preds, all_gts, iou_thresh)
    ap, _, _ = compute_ap(tp, fp, n_gt)
    return ap


def confusion_counts(tp, fp, n_gt):
    """Single-class confusion summary: TP / FP / FN counts at the operating
    point implied by the full prediction set (i.e. after applying whatever
    --conf threshold was used at inference time — this is NOT swept)."""
    n_tp = int(tp.sum())
    n_fp = int(fp.sum())
    n_fn = int(n_gt - n_tp)
    precision = n_tp / max(n_tp + n_fp, 1)
    recall = n_tp / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "tp": n_tp, "fp": n_fp, "fn": n_fn,
        "precision": precision, "recall": recall, "f1": f1,
    }
