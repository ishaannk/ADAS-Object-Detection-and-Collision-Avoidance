import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from adas import api

client = TestClient(api.app)


def _png_bytes(width=64, height=48) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_rejects_non_image_upload():
    resp = client.post("/analyze", files={"image": ("frame.txt", b"not an image", "text/plain")})
    assert resp.status_code == 422


def test_analyze_requires_calib_and_velodyne_together():
    resp = client.post(
        "/analyze",
        files={
            "image": ("frame.png", _png_bytes(), "image/png"),
            "calib": ("calib.txt", b"P2: 1 0 0 0 0 1 0 0 0 0 1 0", "text/plain"),
        },
    )
    assert resp.status_code == 422


class _StubDetector:
    def detect(self, image):
        return [{"bbox": (5.0, 5.0, 20.0, 20.0), "cls": "Car", "confidence": 0.9}]


def test_analyze_detection_only_uses_detector(monkeypatch):
    monkeypatch.setattr(api, "get_detector", lambda: _StubDetector())

    resp = client.post("/analyze", files={"image": ("frame.png", _png_bytes(), "image/png")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["fused"] is False
    assert body["detections"][0]["cls"] == "Car"


CALIB_TEXT = """P0: 1 0 0 0 0 1 0 0 0 0 1 0
P1: 1 0 0 0 0 1 0 0 0 0 1 0
P2: 500 0 320 0 0 500 240 0 0 0 1 0
P3: 1 0 0 0 0 1 0 0 0 0 1 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0
Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0
"""


def test_analyze_fuses_when_calib_and_velodyne_present(monkeypatch):
    monkeypatch.setattr(api, "get_detector", lambda: _StubDetector())

    velodyne_points = np.array([[10.0, 0.0, 0.0, 0.5]], dtype=np.float32)  # projects near principal point

    resp = client.post(
        "/analyze",
        files={
            "image": ("frame.png", _png_bytes(640, 480), "image/png"),
            "calib": ("calib.txt", CALIB_TEXT.encode(), "text/plain"),
            "velodyne": ("velodyne.bin", velodyne_points.tobytes(), "application/octet-stream"),
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["fused"] is True
