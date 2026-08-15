"""Project raw velodyne LIDAR points into the left color camera's image plane.

This replaces the original prototype's approach of dividing raw 3D
coordinates by z with no calibration at all (see docs/PLAN.md, DTC-02).
"""

from __future__ import annotations

import numpy as np

from adas.data.calibration import Calibration


def load_velodyne_bin(path: str) -> np.ndarray:
    """Load a KITTI .bin point cloud: (N, 4) of [x, y, z, reflectance]."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def project_velodyne_to_image(
    points_xyz: np.ndarray,
    calib: Calibration,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project velodyne-frame points into pixel coordinates.

    Returns (pixels, depth) where `pixels` is (M, 2) float32 [u, v] and
    `depth` is (M,) float32 camera-frame Z in meters, for the M points that
    land in front of the camera and inside the image bounds. M <= N.
    """
    n = points_xyz.shape[0]
    homogeneous = np.hstack([points_xyz[:, :3], np.ones((n, 1), dtype=np.float64)])

    cam_coords = homogeneous @ calib.velo_to_image.T  # (N, 3): [u*d, v*d, d]
    depth = cam_coords[:, 2]

    in_front = depth > 0.1  # drop points behind or right at the camera
    safe_depth = np.where(in_front, depth, 1.0)
    pixels = cam_coords[:, :2] / safe_depth[:, None]

    in_bounds = (
        in_front
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < image_width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < image_height)
    )

    return pixels[in_bounds].astype(np.float32), depth[in_bounds].astype(np.float32)
