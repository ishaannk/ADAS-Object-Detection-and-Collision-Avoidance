#!/usr/bin/env python3
"""Run the fine-tuned detector over the KITTI val split and score it with
the KITTI-protocol-inspired eval (easy/moderate/hard, per-class IoU
thresholds) in src/adas/eval/kitti_eval.py — not the generic YOLO/COCO-style
mAP ultralytics reports during training.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas.data.convert_to_yolo import split_ids  # noqa: E402
from adas.data.kitti import CLASSES, KittiObjectDataset  # noqa: E402
from adas.detection.detector import Detector  # noqa: E402
from adas.eval.kitti_eval import DIFFICULTY_CRITERIA, GroundTruthBox, PredBox, average_precision  # noqa: E402


def evaluate(kitti_root: str, weights: str, device: str, conf: float) -> dict:
    dataset = KittiObjectDataset(kitti_root)
    _, val_ids = split_ids(list(dataset.frame_ids))
    val_indices = [i for i in range(len(dataset)) if dataset.frame_ids[i] in val_ids]

    detector = Detector(weights, device=device, conf=conf)

    gts_by_frame: list[list[GroundTruthBox]] = []
    all_preds: list[PredBox] = []

    for frame_idx, i in enumerate(val_indices):
        frame = dataset[i]
        gts_by_frame.append(
            [
                GroundTruthBox(cls=obj.cls, bbox=obj.bbox, occluded=obj.occluded, truncated=obj.truncated)
                for obj in frame.load_objects()
            ]
        )
        detections = detector.detect(str(frame.image_path))
        all_preds.extend(
            PredBox(cls=det["cls"], bbox=det["bbox"], confidence=det["confidence"], frame=frame_idx)
            for det in detections
        )

    results: dict = {"num_val_frames": len(val_indices), "per_class": {}}
    for cls in CLASSES:
        results["per_class"][cls] = {
            difficulty: average_precision(all_preds, gts_by_frame, cls, difficulty)
            for difficulty in DIFFICULTY_CRITERIA
        }

    for difficulty in DIFFICULTY_CRITERIA:
        aps = [v[difficulty] for v in results["per_class"].values() if v[difficulty] == v[difficulty]]  # drop NaN
        results.setdefault("mean_ap", {})[difficulty] = sum(aps) / len(aps) if aps else float("nan")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", default="data/raw/object")
    parser.add_argument("--weights", default="runs/detect/kitti_finetune/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default="runs/detect/kitti_finetune/kitti_eval_metrics.json")
    args = parser.parse_args()

    results = evaluate(args.kitti_root, args.weights, args.device, args.conf)

    print(f"{'class':<16}{'easy':>8}{'moderate':>10}{'hard':>8}")
    for cls, aps in results["per_class"].items():
        print(f"{cls:<16}{aps['easy']:>8.3f}{aps['moderate']:>10.3f}{aps['hard']:>8.3f}")
    print(f"{'mean':<16}{results['mean_ap']['easy']:>8.3f}{results['mean_ap']['moderate']:>10.3f}{results['mean_ap']['hard']:>8.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"saved to {args.out}")
