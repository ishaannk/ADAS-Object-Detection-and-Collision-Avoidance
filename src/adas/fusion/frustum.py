"""Associate projected LIDAR points to 2D detections by frustum membership.

The original prototype (docs/PLAN.md, DTC-03) picked the globally nearest
point in the whole frame for every detection, with no requirement that the
point actually fall inside that detection's box — a pedestrian could inherit
a truck's distance. Here, a point only counts for a detection if its
projected pixel falls inside that detection's box, and depth is summarized
robustly (median) rather than by a single nearest point, so a handful of
LIDAR returns that land on the background just past a box edge don't skew
the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusedDetection:
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    cls: str
    confidence: float
    distance_m: float | None  # None if no LIDAR points fell inside the box
    num_points: int


def fuse_detections_with_points(
    detections: list[dict],
    pixels: np.ndarray,
    depth: np.ndarray,
) -> list[FusedDetection]:
    """
    detections: list of {"bbox": (xmin, ymin, xmax, ymax), "cls": str, "confidence": float}
    pixels: (M, 2) projected [u, v] pixel coordinates
    depth: (M,) camera-frame depth in meters, aligned with `pixels`
    """
    fused = []
    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        inside = (
            (pixels[:, 0] >= xmin)
            & (pixels[:, 0] <= xmax)
            & (pixels[:, 1] >= ymin)
            & (pixels[:, 1] <= ymax)
        )
        box_depths = depth[inside]

        distance = float(np.median(box_depths)) if box_depths.size > 0 else None

        fused.append(
            FusedDetection(
                bbox=(xmin, ymin, xmax, ymax),
                cls=det["cls"],
                confidence=det["confidence"],
                distance_m=distance,
                num_points=int(box_depths.size),
            )
        )
    return fused
