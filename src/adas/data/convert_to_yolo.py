"""Convert KITTI labels to YOLO format and produce a deterministic train/val split.

Not the literature-standard Chen et al. 3712/3769 split (we don't have that
id list on hand) — a fixed-seed 85/15 split over sorted frame ids, chosen so
runs are reproducible. See docs/PLAN.md.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from adas.data.kitti import CLASSES, KittiObjectDataset

VAL_FRACTION = 0.15
SPLIT_SEED = 42


def split_ids(frame_ids: list[str]) -> tuple[set[str], set[str]]:
    """Deterministic 85/15 train/val split, fixed seed — shared by dataset
    conversion and evaluation so both agree on which frames are held out."""
    rng = random.Random(SPLIT_SEED)
    shuffled = frame_ids[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_FRACTION)
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])
    return train_ids, val_ids


def convert(kitti_root: str, out_root: str) -> None:
    dataset = KittiObjectDataset(kitti_root)
    out_root_path = Path(out_root)
    images_out = out_root_path / "images"
    labels_out = out_root_path / "labels"

    frame_ids = list(dataset.frame_ids)
    _, val_ids = split_ids(frame_ids)
    n_val = len(val_ids)

    for split in ("train", "val"):
        (images_out / split).mkdir(parents=True, exist_ok=True)
        (labels_out / split).mkdir(parents=True, exist_ok=True)

    class_index = {name: i for i, name in enumerate(CLASSES)}

    for i in range(len(dataset)):
        frame = dataset[i]
        split = "val" if frame.frame_id in val_ids else "train"

        with Image.open(frame.image_path) as img:
            width, height = img.size

        lines = []
        for obj in frame.load_objects():
            xmin, ymin, xmax, ymax = obj.bbox
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            box_w = (xmax - xmin) / width
            box_h = (ymax - ymin) / height
            cls_id = class_index[obj.cls]
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        label_dst = labels_out / split / f"{frame.frame_id}.txt"
        label_dst.write_text("\n".join(lines) + ("\n" if lines else ""))

        image_dst = images_out / split / f"{frame.frame_id}.png"
        if not image_dst.exists():
            image_dst.symlink_to(frame.image_path.resolve())

    print(f"train: {len(frame_ids) - n_val} frames, val: {n_val} frames -> {out_root_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", default="data/raw/object")
    parser.add_argument("--out-root", default="data/processed/kitti_yolo")
    args = parser.parse_args()
    convert(args.kitti_root, args.out_root)
