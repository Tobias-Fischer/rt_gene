from pathlib import Path

import pytest

from rt_gene.download_tools import download_gaze_pytorch_models
from rt_gene.single_image_demo import default_camera_matrix, main


def test_default_camera_matrix_uses_image_shape():
    matrix = default_camera_matrix(width=640, height=480)

    assert matrix.tolist() == [[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]]


def test_single_image_demo_reports_missing_image(capsys, tmp_path):
    assert main([str(tmp_path / "missing.jpg")]) == 1

    assert "Could not read image" in capsys.readouterr().err


def test_download_gaze_pytorch_models_accepts_subset(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rt_gene.download_tools.request_if_not_exist",
        lambda target, *_: calls.append(Path(target).name),
    )

    download_gaze_pytorch_models(["gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model"])

    assert calls == ["gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model"]


def test_download_gaze_pytorch_models_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown gaze model"):
        download_gaze_pytorch_models(["unknown.model"])
