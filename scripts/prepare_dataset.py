#!/usr/bin/env python3
"""CLI entry point: KITTI raw labels -> YOLO-format dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas.data.convert_to_yolo import convert  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", default="data/raw/object")
    parser.add_argument("--out-root", default="data/processed/kitti_yolo")
    args = parser.parse_args()
    convert(args.kitti_root, args.out_root)
