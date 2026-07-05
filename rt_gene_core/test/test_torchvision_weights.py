from pathlib import Path


def test_torchvision_models_use_weights_api():
    repo = Path(__file__).resolve().parents[1]
    files = [
        repo / "src" / "rt_gene" / "gaze_estimation_models_pytorch.py",
        repo / "src" / "rt_bene" / "blink_estimation_models_pytorch.py",
    ]

    offenders = [str(path.relative_to(repo)) for path in files if "pretrained=True" in path.read_text()]

    assert offenders == []
