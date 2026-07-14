import numpy as np

from rt_gene.estimate_gaze_pytorch import open_cv_bgr_to_rgb


def test_gaze_preprocessing_converts_opencv_bgr_to_training_rgb():
    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    rgb = open_cv_bgr_to_rgb(bgr)

    assert rgb.tolist() == [[[30, 20, 10], [60, 50, 40]]]


def test_gaze_preprocessing_rejects_non_color_images():
    gray = np.zeros((2, 2), dtype=np.uint8)

    try:
        open_cv_bgr_to_rgb(gray)
    except ValueError as exc:
        assert "3-channel" in str(exc)
    else:
        raise AssertionError("Expected grayscale images to be rejected")
