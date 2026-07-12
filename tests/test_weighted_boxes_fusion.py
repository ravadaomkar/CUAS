"""
Unit tests for ensemble/weighted_boxes_fusion.py. Pure numpy, no GPU/model
required — run with `pytest tests/`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble.weighted_boxes_fusion import weighted_boxes_fusion


def test_wbf_merges_overlapping_boxes_from_two_sources():
    # Two "models" each detect roughly the same box -> should fuse to one.
    boxes_list = [
        [[0.10, 0.10, 0.30, 0.30]],
        [[0.12, 0.11, 0.31, 0.29]],
    ]
    scores_list = [[0.9], [0.8]]
    fused_boxes, fused_scores = weighted_boxes_fusion(boxes_list, scores_list, iou_thr=0.5)
    assert len(fused_boxes) == 1
    assert 0.0 < fused_scores[0] <= 1.0


def test_wbf_keeps_non_overlapping_boxes_separate():
    boxes_list = [
        [[0.0, 0.0, 0.1, 0.1]],
        [[0.8, 0.8, 0.9, 0.9]],
    ]
    scores_list = [[0.9], [0.9]]
    fused_boxes, fused_scores = weighted_boxes_fusion(boxes_list, scores_list, iou_thr=0.5)
    assert len(fused_boxes) == 2


def test_wbf_empty_input_returns_empty():
    fused_boxes, fused_scores = weighted_boxes_fusion([[], []], [[], []])
    assert fused_boxes == []
    assert fused_scores == []


def test_wbf_single_source_passthrough_shape():
    boxes_list = [[[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]]]
    scores_list = [[0.9, 0.7]]
    fused_boxes, fused_scores = weighted_boxes_fusion(boxes_list, scores_list)
    assert len(fused_boxes) == 2
    # Higher-confidence box should be first after re-sorting.
    assert fused_scores[0] >= fused_scores[1]
