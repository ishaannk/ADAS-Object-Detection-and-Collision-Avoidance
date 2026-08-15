"""Multi-object tracking, reusing ultralytics' built-in tracker rather than
hand-rolling one — ByteTrack/BoT-SORT are proven, and `model.track()` already
ships with the `ultralytics` dependency we install for detection.
"""

from __future__ import annotations

from ultralytics import YOLO


class Tracker:
    def __init__(self, weights_path: str, device: str | int = 0, conf: float = 0.25, tracker: str = "bytetrack.yaml"):
        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.tracker = tracker

    def track(self, image, persist: bool = True) -> list[dict]:
        """Run detection + tracking on one frame of a stream. `persist=True`
        keeps track state across calls for the same video/stream.
        Returns [{"track_id": int, "bbox": (xmin,ymin,xmax,ymax), "cls": str, "confidence": float}]."""
        results = self.model.track(
            image, device=self.device, conf=self.conf, tracker=self.tracker, persist=persist, verbose=False
        )[0]
        names = results.names
        out = []
        if results.boxes.id is None:
            return out
        for box in results.boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            out.append(
                {
                    "track_id": int(box.id[0]),
                    "bbox": (xmin, ymin, xmax, ymax),
                    "cls": names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                }
            )
        return out
