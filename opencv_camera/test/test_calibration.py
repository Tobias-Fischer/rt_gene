from pathlib import Path

from opencv_camera.calibration import default_camera_info, load_camera_info


def test_default_camera_info_has_expected_shape():
    info = default_camera_info(640, 480, "camera", "camera_optical_frame")
    assert info.width == 640
    assert info.height == 480
    assert len(info.k) == 9
    assert len(info.p) == 12
    assert info.header.frame_id == "camera_optical_frame"


def test_load_camera_info_from_yaml():
    config = Path(__file__).resolve().parents[1] / "config" / "default_calibration.yaml"
    info = load_camera_info(config, 1, 1, "camera", "camera_optical_frame")
    assert info.width == 640
    assert info.height == 480
    assert info.k[0] == 1130.394061179079
    assert info.header.frame_id == "camera_optical_frame"
