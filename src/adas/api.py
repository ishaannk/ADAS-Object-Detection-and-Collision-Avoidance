"""FastAPI inference service — the "Docker for the runtime" half of the
deployment story (the model itself lives on Hugging Face Hub, versioned
independently). Accepts an image and, optionally, matching KITTI calib +
velodyne files for the full calibrated fusion pipeline; falls back to
detection-only when no LIDAR data is available, which is the common case
for an arbitrary uploaded frame.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image

from adas.data.calibration import parse_calib_file
from adas.detection.detector import Detector
from adas.fusion.frustum import fuse_detections_with_points
from adas.fusion.projection import load_velodyne_bin, project_velodyne_to_image

MODEL_REPO = os.environ.get("ADAS_MODEL_REPO", "mokshhere/adas-kitti-yolo11m")
MODEL_FILE = os.environ.get("ADAS_MODEL_FILE", "best.onnx")

_detector: Detector | None = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        _detector = Detector(weights_path, device="cpu")
    return _detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_detector()  # load once at boot rather than on the first request
    yield


app = FastAPI(title="ADAS Perception API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_repo": MODEL_REPO, "model_file": MODEL_FILE}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    calib: UploadFile | None = File(None),
    velodyne: UploadFile | None = File(None),
):
    """Run detection, and — if calib + velodyne are both provided — calibrated
    LIDAR fusion, returning per-object distance alongside the box."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="`image` must be an image file")
    if bool(calib) != bool(velodyne):
        raise HTTPException(status_code=422, detail="calib and velodyne must be provided together, or not at all")

    detector = get_detector()

    with tempfile.TemporaryDirectory() as tmp:
        image_path = os.path.join(tmp, "frame.png")
        with open(image_path, "wb") as f:
            f.write(await image.read())

        with Image.open(image_path) as img:
            width, height = img.size
            detections = detector.detect(img)

        if calib is None:
            return JSONResponse({"detections": detections, "fused": False})

        calib_path = os.path.join(tmp, "calib.txt")
        velodyne_path = os.path.join(tmp, "velodyne.bin")
        with open(calib_path, "wb") as f:
            f.write(await calib.read())
        with open(velodyne_path, "wb") as f:
            f.write(await velodyne.read())

        parsed_calib = parse_calib_file(calib_path)
        points = load_velodyne_bin(velodyne_path)
        pixels, depth = project_velodyne_to_image(points[:, :3], parsed_calib, width, height)
        fused = fuse_detections_with_points(detections, pixels, depth)

        return JSONResponse({"detections": [asdict(f) for f in fused], "fused": True})
