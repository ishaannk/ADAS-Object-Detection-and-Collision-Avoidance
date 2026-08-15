#!/usr/bin/env python3
"""Visual sanity check for the corrected fusion pipeline: draws detections
with calibrated, box-constrained LIDAR distances on a handful of KITTI
training frames. Saves to runs/fusion_demo/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from adas.data.kitti import KittiObjectDataset  # noqa: E402
from adas.detection.detector import Detector  # noqa: E402
from adas.pipeline import run_frame  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", default="data/raw/object")
    parser.add_argument("--weights", default="yolo11m.pt")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--out-dir", default="runs/fusion_demo")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = KittiObjectDataset(args.kitti_root)
    detector = Detector(args.weights, device=args.device)

    for i in range(min(args.num_frames, len(dataset))):
        frame = dataset[i]
        calib = frame.load_calib()
        fused = run_frame(detector, str(frame.image_path), str(frame.velodyne_path), calib)

        img = Image.open(frame.image_path)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(img)
        for det in fused:
            xmin, ymin, xmax, ymax = det.bbox
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            label = f"{det.cls} {det.distance_m:.1f}m ({det.num_points}pt)" if det.distance_m is not None else f"{det.cls} no LIDAR"
            ax.text(xmin, ymin, label, color="white", fontsize=9, bbox=dict(facecolor="red", alpha=0.6))
        ax.axis("off")
        ax.set_title(f"frame {frame.frame_id}")
        fig.savefig(out_dir / f"{frame.frame_id}.png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"frame {frame.frame_id}: {len(fused)} detections")

    print(f"saved to {out_dir}")
