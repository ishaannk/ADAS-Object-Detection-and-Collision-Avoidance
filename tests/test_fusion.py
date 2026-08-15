import numpy as np

from adas.data.calibration import Calibration
from adas.fusion.frustum import fuse_detections_with_points
from adas.fusion.projection import project_velodyne_to_image


def simple_calib() -> Calibration:
    # fx=fy=500, principal point (320, 240), no rotation/translation.
    p2 = np.array([[500, 0, 320, 0], [0, 500, 240, 0], [0, 0, 1, 0]], dtype=np.float64)
    r0 = np.eye(4)
    tr = np.eye(4)
    return Calibration(P2=p2, R0_rect=r0, Tr_velo_to_cam=tr)


def test_project_velodyne_to_image_drops_behind_camera_points():
    calib = simple_calib()
    points = np.array(
        [
            [0.0, 0.0, 10.0],  # straight ahead -> principal point
            [0.0, 0.0, -5.0],  # behind the camera -> dropped
            [1000.0, 1000.0, 5.0],  # projects way outside image bounds -> dropped
        ]
    )
    pixels, depth = project_velodyne_to_image(points, calib, image_width=640, image_height=480)

    assert pixels.shape[0] == 1
    np.testing.assert_allclose(pixels[0], [320.0, 240.0], atol=1e-4)
    np.testing.assert_allclose(depth[0], 10.0, atol=1e-4)


def test_fusion_only_uses_points_inside_the_box_not_globally_nearest():
    """Regression test for DTC-03: a point closer in depth but outside every
    box must not be attached to a detection it doesn't geometrically belong to."""
    detections = [
        {"bbox": (100, 100, 200, 200), "cls": "Pedestrian", "confidence": 0.9},
        {"bbox": (400, 100, 500, 200), "cls": "Car", "confidence": 0.9},
    ]
    pixels = np.array(
        [
            [150, 150],  # inside pedestrian box
            [160, 160],  # inside pedestrian box
            [450, 150],  # inside car box
            [10, 10],  # inside neither box, but globally the closest point
        ]
    )
    depth = np.array([8.0, 8.5, 20.0, 1.0])

    fused = fuse_detections_with_points(detections, pixels, depth)

    pedestrian = next(f for f in fused if f.cls == "Pedestrian")
    car = next(f for f in fused if f.cls == "Car")

    assert pedestrian.num_points == 2
    assert pedestrian.distance_m == 8.25  # median of 8.0, 8.5 — not the globally-nearest 1.0
    assert car.num_points == 1
    assert car.distance_m == 20.0


def test_fusion_handles_no_points_in_box():
    detections = [{"bbox": (0, 0, 10, 10), "cls": "Car", "confidence": 0.5}]
    pixels = np.array([[500, 500]])
    depth = np.array([5.0])

    fused = fuse_detections_with_points(detections, pixels, depth)

    assert fused[0].distance_m is None
    assert fused[0].num_points == 0
