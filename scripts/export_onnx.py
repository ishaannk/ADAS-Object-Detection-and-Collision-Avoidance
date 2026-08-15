#!/usr/bin/env python3
"""Export the fine-tuned detector to ONNX — a hardware-agnostic stand-in for
an embedded/edge deployment path (TensorRT, ONNX Runtime on-device), used to
benchmark latency without physical automotive-grade hardware (P5)."""

import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/kitti_finetune/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    model = YOLO(args.weights)
    path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    print(f"exported to {path}")
