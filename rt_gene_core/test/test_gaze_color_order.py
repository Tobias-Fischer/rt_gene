import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from rt_gene.estimate_gaze_pytorch import make_gaze_image_transform, open_cv_bgr_to_rgb


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


def test_gaze_preprocessing_matches_ros1_pytorch_training_transform():
    bgr = np.zeros((36, 60, 3), dtype=np.uint8)
    bgr[:, :, 0] = 10
    bgr[:, :, 1] = 20
    bgr[:, :, 2] = 30
    rgb = open_cv_bgr_to_rgb(bgr)

    ros2_inference = make_gaze_image_transform()(rgb)
    ros1_training = transforms.Compose([
        transforms.Resize((36, 60), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(Image.fromarray(rgb, "RGB"))

    assert torch.allclose(ros2_inference, ros1_training)
