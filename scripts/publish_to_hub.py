#!/usr/bin/env python3
"""Publish the fine-tuned checkpoint to Hugging Face Hub with a model card.

Not run automatically — needs a target repo id and an authenticated HF
token (`huggingface-cli login` or HF_TOKEN env var) supplied by whoever owns
the Hub account this should land in.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, ModelCard, ModelCardData

MODEL_CARD_TEMPLATE = """
This is a YOLO11 detector fine-tuned on the KITTI Object Detection Benchmark
(training split, 7,481 frames) for an ADAS perception pipeline. See the
full project: https://github.com/ishaannk/ADAS-Object-Detection-and-Collision-Avoidance

## Classes
Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc — KITTI's
own taxonomy, not remapped to COCO classes.

## Training data
KITTI Object Detection Benchmark, training split only. Deterministic 85/15
train/val split (seed 42) over sorted frame ids — not the literature Chen et
al. 3712/3769 split.

## Metrics
See `metrics.json` in this repo for per-class mAP50 / mAP50-95 and
KITTI-protocol-style easy/moderate/hard AP.

## Intended use
Research and portfolio demonstration of a calibrated camera-LIDAR fusion +
collision-risk pipeline. **Not validated for deployment in a vehicle.**

## License
Base model (Ultralytics YOLO11) is AGPL-3.0. KITTI's terms restrict this
dataset to non-commercial research use — these weights are not licensed for
commercial/production use as-is.
"""


def publish(weights_path: str, onnx_path: str | None, repo_id: str, metrics_path: str | None) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    card = ModelCard.from_template(
        ModelCardData(license="agpl-3.0", tags=["object-detection", "yolo11", "kitti", "adas"]),
        model_summary=MODEL_CARD_TEMPLATE,
    )
    card.push_to_hub(repo_id)

    api.upload_file(path_or_fileobj=weights_path, path_in_repo="best.pt", repo_id=repo_id)
    if onnx_path and Path(onnx_path).exists():
        api.upload_file(path_or_fileobj=onnx_path, path_in_repo="best.onnx", repo_id=repo_id)
    if metrics_path and Path(metrics_path).exists():
        api.upload_file(path_or_fileobj=metrics_path, path_in_repo="metrics.json", repo_id=repo_id)

    print(f"Published to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/kitti_finetune/weights/best.pt")
    parser.add_argument("--onnx", default="runs/detect/kitti_finetune/weights/best.onnx")
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/adas-kitti-yolo11m")
    parser.add_argument("--metrics", default="runs/detect/kitti_finetune/kitti_eval_metrics.json")
    args = parser.parse_args()
    publish(args.weights, args.onnx, args.repo_id, args.metrics)
