"""Minimal KITTI Object Detection Benchmark reader (training split only).

We deliberately do not depend on tensorflow-datasets (docs/PLAN.md flags the
original `tfds.load('kitti', ...)` path as fragile) — this reads the raw
KITTI files directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adas.data.calibration import Calibration, parse_calib_file

# KITTI's own object taxonomy. "DontCare" regions are excluded — they mark
# areas evaluators should ignore, not objects to detect or train on.
CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
IGNORED_CLASSES = {"DontCare"}


@dataclass
class KittiObject:
    cls: str
    truncated: float
    occluded: int
    alpha: float
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax (pixels)
    dimensions_hwl: tuple[float, float, float]
    location_xyz: tuple[float, float, float]
    rotation_y: float


@dataclass
class KittiFrame:
    frame_id: str
    image_path: Path
    label_path: Path
    calib_path: Path
    velodyne_path: Path

    def load_calib(self) -> Calibration:
        return parse_calib_file(self.calib_path)

    def load_objects(self) -> list[KittiObject]:
        objects = []
        with open(self.label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = parts[0]
                if cls in IGNORED_CLASSES:
                    continue
                objects.append(
                    KittiObject(
                        cls=cls,
                        truncated=float(parts[1]),
                        occluded=int(parts[2]),
                        alpha=float(parts[3]),
                        bbox=(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
                        dimensions_hwl=(float(parts[8]), float(parts[9]), float(parts[10])),
                        location_xyz=(float(parts[11]), float(parts[12]), float(parts[13])),
                        rotation_y=float(parts[14]),
                    )
                )
        return objects


class KittiObjectDataset:
    """Indexes the training split of the KITTI Object Detection Benchmark."""

    def __init__(self, root: str | Path):
        self.root = Path(root) / "training"
        image_dir = self.root / "image_2"
        self.frame_ids = sorted(p.stem for p in image_dir.glob("*.png"))

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> KittiFrame:
        fid = self.frame_ids[idx]
        return KittiFrame(
            frame_id=fid,
            image_path=self.root / "image_2" / f"{fid}.png",
            label_path=self.root / "label_2" / f"{fid}.txt",
            calib_path=self.root / "calib" / f"{fid}.txt",
            velodyne_path=self.root / "velodyne" / f"{fid}.bin",
        )
