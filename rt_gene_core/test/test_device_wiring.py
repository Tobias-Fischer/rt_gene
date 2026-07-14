import numpy as np

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase


def test_landmark_detector_gets_resolved_device(monkeypatch):
    seen = {}

    class FakeSFDDetector:
        def __init__(self, device, path_to_detector=None):
            seen["device"] = device

    monkeypatch.setattr("rt_gene.extract_landmarks_method_base.download_external_landmark_models", lambda: None)
    monkeypatch.setattr("rt_gene.extract_landmarks_method_base.SFDDetector", FakeSFDDetector)
    monkeypatch.setattr(LandmarkMethodBase, "load_face_landmark_model", lambda self, path=None: object())
    monkeypatch.setattr(LandmarkMethodBase, "get_full_model_points", lambda self, path=None: np.zeros((68, 3)))
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False, raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    LandmarkMethodBase(device_id_facedetection="auto")

    assert str(seen["device"]) == "cpu"
