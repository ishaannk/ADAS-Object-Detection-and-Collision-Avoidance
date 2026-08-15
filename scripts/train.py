#!/usr/bin/env python3
"""Fine-tune a YOLO detector on the KITTI taxonomy.

Runs on both available GPUs via ultralytics' built-in DDP support
(device="0,1"). Logs and checkpoints land under runs/detect/<name>/.
"""

import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11m.pt", help="base checkpoint to fine-tune from")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--name", default="kitti_finetune")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        name=args.name,
        exist_ok=True,
        plots=True,
    )
