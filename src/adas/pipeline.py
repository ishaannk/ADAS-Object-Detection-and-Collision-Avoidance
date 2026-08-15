"""End-to-end per-frame pipeline: detect -> project LIDAR -> fuse.

This is the concrete fix for DTC-01/02/03 (docs/PLAN.md): real calibration,
real velodyne points, and box-constrained association, replacing the
original prototype's ground-truth-as-sensor-data shortcut and global
nearest-neighbor match.
"""

from __future__ import annotations

from PIL import Image

from adas.data.calibration import Calibration
from adas.detection.detector import Detector
from adas.fusion.frustum import FusedDetection, fuse_detections_with_points
from adas.fusion.projection import load_velodyne_bin, project_velodyne_to_image


def run_frame(
    detector: Detector,
    image_path: str,
    velodyne_path: str,
    calib: Calibration,
) -> list[FusedDetection]:
    with Image.open(image_path) as img:
        width, height = img.size
        detections = detector.detect(img)

    points = load_velodyne_bin(velodyne_path)
    pixels, depth = project_velodyne_to_image(points[:, :3], calib, width, height)

    return fuse_detections_with_points(detections, pixels, depth)
