"""Parsing for KITTI per-frame calibration files.

KITTI calib/*.txt files hold, per frame: four 3x4 camera projection matrices
(P0-P3), the 3x3 rectifying rotation R0_rect, and the 3x4 rigid transforms
Tr_velo_to_cam and Tr_imu_to_velo. Projecting a LIDAR point into the left
color image (cam 2) requires composing all three: P2 @ R0_rect @ Tr_velo_to_cam.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Calibration:
    P2: np.ndarray  # (3, 4) projection matrix for the left color camera
    R0_rect: np.ndarray  # (4, 4) homogeneous rectifying rotation
    Tr_velo_to_cam: np.ndarray  # (4, 4) homogeneous velodyne -> camera transform

    @property
    def velo_to_image(self) -> np.ndarray:
        """(3, 4) matrix mapping homogeneous velodyne points straight to pixel coords."""
        return self.P2 @ self.R0_rect @ self.Tr_velo_to_cam


def _to_homogeneous_4x4(flat_3x4: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :4] = flat_3x4.reshape(3, 4)
    return mat


def parse_calib_file(path: str | Path) -> Calibration:
    values: dict[str, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, raw = line.split(":", 1)
            values[key.strip()] = np.array([float(v) for v in raw.split()], dtype=np.float64)

    p2 = values["P2"].reshape(3, 4)

    r0_flat = values["R0_rect"].reshape(3, 3)
    r0_rect = np.eye(4, dtype=np.float64)
    r0_rect[:3, :3] = r0_flat

    tr_velo_to_cam = _to_homogeneous_4x4(values["Tr_velo_to_cam"])

    return Calibration(P2=p2, R0_rect=r0_rect, Tr_velo_to_cam=tr_velo_to_cam)
