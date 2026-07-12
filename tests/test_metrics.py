"""
Pure-Python/numpy unit tests for utils/metrics.py — no GPU, no dataset,
no ultralytics import required. Run with `pytest tests/`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compute_ap, confusion_counts, iou, match_predictions, yolo_to_xyxy


def test_iou_identical_boxes():
    box = [0, 0, 10, 10]
    assert iou(box, box) == 1.0


def test_iou_no_overlap():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_partial_overlap():
    val = iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert 0.0 < val < 1.0


def test_yolo_to_xyxy_center_box():
    # Full-image box: xc=0.5, yc=0.5, w=1.0, h=1.0 on a 100x100 image -> [0,0,100,100]
    box = yolo_to_xyxy([0.5, 0.5, 1.0, 1.0], 100, 100)
    assert box == [0.0, 0.0, 100.0, 100.0]


def test_perfect_predictions_give_ap_1():
    gts = {"img1": [[0, 0, 10, 10]], "img2": [[5, 5, 15, 15]]}
    preds = [("img1", [0, 0, 10, 10], 0.99), ("img2", [5, 5, 15, 15], 0.95)]
    tp, fp, _, _, _ = match_predictions(preds, gts, iou_thresh=0.5)
    ap, _, _ = compute_ap(tp, fp, n_gt=2)
    assert ap == 1.0


def test_no_predictions_gives_ap_0():
    gts = {"img1": [[0, 0, 10, 10]]}
    ap = compute_ap(*match_predictions([], gts)[:2], n_gt=1)[0]
    assert ap == 0.0


def test_confusion_counts_basic():
    gts = {"img1": [[0, 0, 10, 10]], "img2": [[0, 0, 10, 10]]}
    # One correct match (img1), one false positive with no matching GT (img2 mismatched box),
    # img2's real GT is never matched -> also a false negative.
    preds = [("img1", [0, 0, 10, 10], 0.9), ("img2", [50, 50, 60, 60], 0.8)]
    tp, fp, _, _, _ = match_predictions(preds, gts, iou_thresh=0.5)
    counts = confusion_counts(tp, fp, n_gt=2)
    assert counts["tp"] == 1
    assert counts["fp"] == 1
    assert counts["fn"] == 1
