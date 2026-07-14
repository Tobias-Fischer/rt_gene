from pathlib import Path

import pytest

from rt_bene.single_image_demo import main
from rt_gene.download_tools import download_blink_pytorch_models


def test_blink_single_image_demo_reports_missing_face_image(capsys, tmp_path):
    assert main([str(tmp_path / "missing.jpg")]) == 1

    assert "Could not read image" in capsys.readouterr().err


def test_blink_single_image_demo_requires_eye_pair(capsys, tmp_path):
    assert main(["--left-eye", str(tmp_path / "left.jpg")]) == 1

    assert "--left-eye and --right-eye must be provided together" in capsys.readouterr().err


def test_download_blink_pytorch_models_accepts_subset(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rt_gene.download_tools.request_if_not_exist",
        lambda target, *_: calls.append(Path(target).name),
    )

    download_blink_pytorch_models(["blink_model_pytorch_vgg16_allsubjects1.model"])

    assert calls == ["blink_model_pytorch_vgg16_allsubjects1.model"]


def test_download_blink_pytorch_models_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown blink model"):
        download_blink_pytorch_models(["unknown.model"])
