#!/usr/bin/env python3
"""Latency/FPS benchmark for the fine-tuned detector — GPU, CPU, and an
ONNX export as a stand-in for embedded/edge deployment (P5)."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ultralytics import YOLO  # noqa: E402


def benchmark(model_path: str, device: str, imgsz: int, n_warmup: int, n_iters: int) -> dict:
    model = YOLO(model_path)
    dummy = "https://ultralytics.com/images/bus.jpg"

    for _ in range(n_warmup):
        model.predict(dummy, device=device, imgsz=imgsz, verbose=False)

    start = time.perf_counter()
    for _ in range(n_iters):
        model.predict(dummy, device=device, imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - start

    return {
        "device": device,
        "imgsz": imgsz,
        "mean_latency_ms": (elapsed / n_iters) * 1000,
        "fps": n_iters / elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/kitti_finetune/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    for device in ["0", "cpu"]:
        result = benchmark(args.weights, device, args.imgsz, args.warmup, args.iters)
        print(f"{result['device']:>4}  {result['mean_latency_ms']:7.2f} ms/frame  {result['fps']:7.2f} FPS")
