"""Thin wrapper around a pinned `ultralytics` install.

Replaces the original prototype's `torch.hub.load('ultralytics/yolov5', ...)`
(docs/PLAN.md, DTC-06), which re-clones GitHub at runtime instead of using an
installed, version-pinned package.
"""

from __future__ import annotations

from ultralytics import YOLO


class Detector:
    def __init__(self, weights_path: str, device: str | int = 0, conf: float = 0.25, imgsz: int = 960):
        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image) -> list[dict]:
        """Run detection on a single image (numpy array or path). Returns a
        list of {"bbox": (xmin, ymin, xmax, ymax), "cls": str, "confidence": float}."""
        results = self.model.predict(image, device=self.device, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        names = results.names
        out = []
        for box in results.boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            out.append(
                {
                    "bbox": (xmin, ymin, xmax, ymax),
                    "cls": names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                }
            )
        return out
