"""KITTI-protocol-inspired 2D detection evaluation.

Not a byte-exact reproduction of KITTI's official C++ devkit (that also
does 40-point interpolated AP and neutral-class don't-care handling we
don't replicate here) — this applies the same difficulty tiers and
per-class IoU thresholds so results are comparable in spirit, and says so
rather than claiming exactness.

Difficulty tiers (by ground-truth box height / occlusion / truncation):
  easy:     height >= 40px, occlusion == 0, truncation <= 0.15
  moderate: height >= 25px, occlusion <= 1, truncation <= 0.30
  hard:     height >= 25px, occlusion <= 2, truncation <= 0.50

IoU thresholds: 0.7 for Car/Van/Truck, 0.5 for Pedestrian/Person_sitting/Cyclist/Tram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IOU_THRESHOLDS = {
    "Car": 0.7,
    "Van": 0.7,
    "Truck": 0.7,
    "Pedestrian": 0.5,
    "Person_sitting": 0.5,
    "Cyclist": 0.5,
    "Tram": 0.5,
    "Misc": 0.5,
}

DIFFICULTY_CRITERIA = {
    "easy": {"min_height": 40, "max_occlusion": 0, "max_truncation": 0.15},
    "moderate": {"min_height": 25, "max_occlusion": 1, "max_truncation": 0.30},
    "hard": {"min_height": 25, "max_occlusion": 2, "max_truncation": 0.50},
}


@dataclass
class GroundTruthBox:
    cls: str
    bbox: tuple[float, float, float, float]
    occluded: int
    truncated: float


@dataclass
class PredBox:
    cls: str
    bbox: tuple[float, float, float, float]
    confidence: float
    frame: int


def box_height(bbox: tuple[float, float, float, float]) -> float:
    return bbox[3] - bbox[1]


def matches_difficulty(gt: GroundTruthBox, difficulty: str) -> bool:
    c = DIFFICULTY_CRITERIA[difficulty]
    return (
        box_height(gt.bbox) >= c["min_height"]
        and gt.occluded <= c["max_occlusion"]
        and gt.truncated <= c["max_truncation"]
    )


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def average_precision(preds: list[PredBox], gts_by_frame: list[list[GroundTruthBox]], cls: str, difficulty: str) -> float:
    """Continuous (non-interpolated) precision/recall AP for one class + difficulty tier.

    Difficulty tiers are nested (easy GT boxes also satisfy moderate/hard
    criteria), so a same-class GT box that's real but doesn't meet *this*
    tier's criteria (too small/occluded/truncated) is a genuine object, just
    not one this tier scores on — a prediction matching it is neither a hit
    nor a miss and must be excluded from the curve entirely. Without this,
    every correct detection of a harder instance scores as a false positive
    at the easier tiers, which inverts the expected easy >= moderate >= hard
    ordering.
    """
    iou_thresh = IOU_THRESHOLDS[cls]

    valid_gt = []
    ignore_boxes_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    for frame_idx, gts in enumerate(gts_by_frame):
        for gt in gts:
            if gt.cls != cls:
                continue
            if matches_difficulty(gt, difficulty):
                valid_gt.append({"frame": frame_idx, "box": gt.bbox, "matched": False})
            else:
                ignore_boxes_by_frame.setdefault(frame_idx, []).append(gt.bbox)

    n_gt = len(valid_gt)
    if n_gt == 0:
        return float("nan")

    class_preds = sorted(
        [p for p in preds if p.cls == cls], key=lambda p: p.confidence, reverse=True
    )

    valid_gt_by_frame: dict[int, list[dict]] = {}
    for g in valid_gt:
        valid_gt_by_frame.setdefault(g["frame"], []).append(g)

    tp = np.zeros(len(class_preds))
    fp = np.zeros(len(class_preds))
    ignored = np.zeros(len(class_preds), dtype=bool)

    for i, pred in enumerate(class_preds):
        frame_gts = valid_gt_by_frame.get(pred.frame, [])
        best_iou, best_gt = 0.0, None
        for g in frame_gts:
            if g["matched"]:
                continue
            score = iou(pred.bbox, g["box"])
            if score > best_iou:
                best_iou, best_gt = score, g
        if best_iou >= iou_thresh and best_gt is not None:
            tp[i] = 1
            best_gt["matched"] = True
            continue

        ignore_boxes = ignore_boxes_by_frame.get(pred.frame, [])
        best_ignore_iou = max((iou(pred.bbox, b) for b in ignore_boxes), default=0.0)
        if best_ignore_iou >= iou_thresh:
            ignored[i] = True
        else:
            fp[i] = 1

    keep = ~ignored
    tp_cum = np.cumsum(tp[keep])
    fp_cum = np.cumsum(fp[keep])
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # Standard AP via the area under the precision-recall curve (precision
    # made monotonically non-increasing from the right, as in VOC/KITTI-style AP).
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision, [0.0]])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
