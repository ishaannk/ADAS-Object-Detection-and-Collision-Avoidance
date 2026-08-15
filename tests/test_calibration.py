import numpy as np

from adas.data.calibration import parse_calib_file

CALIB_TEXT = """P0: 1 0 0 0 0 1 0 0 0 0 1 0
P1: 1 0 0 0 0 1 0 0 0 0 1 0
P2: 500 0 320 0 0 500 240 0 0 0 1 0
P3: 1 0 0 0 0 1 0 0 0 0 1 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0
Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0
"""


def test_parse_calib_file(tmp_path):
    calib_path = tmp_path / "000000.txt"
    calib_path.write_text(CALIB_TEXT)

    calib = parse_calib_file(calib_path)

    assert calib.P2.shape == (3, 4)
    assert calib.R0_rect.shape == (4, 4)
    assert calib.Tr_velo_to_cam.shape == (4, 4)
    np.testing.assert_allclose(calib.R0_rect[:3, :3], np.eye(3))


def test_velo_to_image_composes_all_three_transforms(tmp_path):
    calib_path = tmp_path / "000000.txt"
    calib_path.write_text(CALIB_TEXT)
    calib = parse_calib_file(calib_path)

    # A point straight ahead of the velodyne sensor (+x) should land at the
    # principal point once rotated into the camera frame and projected.
    point = np.array([10.0, 0.0, 0.0, 1.0])
    projected = calib.velo_to_image @ point
    u, v, d = projected
    assert d > 0
    np.testing.assert_allclose([u / d, v / d], [320.0, 240.0], atol=1e-6)
