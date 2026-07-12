"""
Weighted Boxes Fusion (WBF), single-class version, dependency-light
(numpy only — no `ensemble-boxes` package required, since this challenge
is single-class (drone) and full multi-class WBF is more machinery than
we need).

Reference: Solovyev et al., "Weighted boxes fusion: Ensembling boxes for
object detection models" (2021) — this is a from-scratch reimplementation
of the core algorithm, not a copy of their code.

Used by:
  * scripts/inference_tta.py — fuse predictions across augmented views
  * ensemble/ensemble.py       — fuse predictions across different trained models

All boxes are normalized [x1, y1, x2, y2] in [0, 1], matching how
inference_tta.py already produces them (so results are resolution-independent
before being converted back to pixels/YOLO format by the caller).
"""
import numpy as np


def _iou(box1, box2):
    xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
    xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def weighted_boxes_fusion(boxes_list, scores_list, iou_thr=0.55, skip_box_thr=0.0, model_weights=None):
    """
    boxes_list:  list of lists of [x1,y1,x2,y2] (one inner list per model/view)
    scores_list: list of lists of confidence scores, same shape as boxes_list
    model_weights: optional per-model/view weight (e.g. weight a stronger
                   model's votes more heavily); defaults to equal weighting.

    Returns (fused_boxes, fused_scores) — flat lists, one entry per cluster.
    """
    n_sources = len(boxes_list)
    if model_weights is None:
        model_weights = [1.0] * n_sources

    # Flatten with a source id + weight tag so clustering can pool everything.
    flat = []
    for src_idx, (boxes, scores) in enumerate(zip(boxes_list, scores_list)):
        w = model_weights[src_idx]
        for box, score in zip(boxes, scores):
            if score < skip_box_thr:
                continue
            flat.append({"box": np.array(box, dtype=float), "score": score * w, "raw_score": score})

    if not flat:
        return [], []

    flat.sort(key=lambda d: -d["score"])

    clusters = []  # each cluster: list of entries
    for entry in flat:
        placed = False
        for cluster in clusters:
            # compare against the current fused box of the cluster
            if _iou(entry["box"], cluster["fused_box"]) >= iou_thr:
                cluster["members"].append(entry)
                cluster["fused_box"] = _fuse_cluster(cluster["members"])
                cluster["fused_score"] = _fuse_score(cluster["members"], n_sources)
                placed = True
                break
        if not placed:
            clusters.append({
                "members": [entry],
                "fused_box": entry["box"],
                "fused_score": _fuse_score([entry], n_sources),
            })

    fused_boxes = [c["fused_box"].tolist() for c in clusters]
    fused_scores = [c["fused_score"] for c in clusters]

    # Re-sort by fused score, descending, for a stable/expected output order.
    order = np.argsort(-np.array(fused_scores))
    fused_boxes = [fused_boxes[i] for i in order]
    fused_scores = [fused_scores[i] for i in order]
    return fused_boxes, fused_scores


def _fuse_cluster(members):
    """Confidence-weighted average of box coordinates in a cluster."""
    weights = np.array([m["raw_score"] for m in members])
    boxes = np.array([m["box"] for m in members])
    return np.average(boxes, axis=0, weights=weights)


def _fuse_score(members, n_sources):
    """
    WBF's score rule: average confidence across the members that voted for
    this cluster, then attenuate by how many of the N sources actually
    agreed (a box only one out of five views produced is less trustworthy
    than one all five agreed on).
    """
    avg_conf = float(np.mean([m["raw_score"] for m in members]))
    agreement = min(len(members), n_sources) / n_sources
    return avg_conf * agreement
