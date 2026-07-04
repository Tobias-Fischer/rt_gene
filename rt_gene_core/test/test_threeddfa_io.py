import importlib
import pickle
import sys

import numpy as np

from rt_gene.ThreeDDFA.io import _load
from rt_gene.ThreeDDFA.inference import get_suffix
from rt_gene_core import paths


def test_load_accepts_pathlike_filename(tmp_path):
    path = tmp_path / "array.npy"
    expected = np.array([1, 2, 3])
    np.save(path, expected)

    np.testing.assert_array_equal(_load(path), expected)


def test_inference_suffix_accepts_pathlike_filename(tmp_path):
    assert get_suffix(tmp_path / "image.jpg") == ".jpg"


def test_params_import_accepts_pathlike_model_paths(monkeypatch, tmp_path):
    model_dir = tmp_path / "ThreeDDFA"
    model_dir.mkdir()
    np.save(model_dir / "keypoints_sim.npy", np.array([0, 1, 2]))
    np.save(model_dir / "w_shp_sim.npy", np.ones((6, 2)))
    np.save(model_dir / "w_exp_sim.npy", np.ones((6, 2)))
    np.save(model_dir / "u_shp.npy", np.ones((6, 1)))
    np.save(model_dir / "u_exp.npy", np.ones((6, 1)))
    with (model_dir / "param_whitening.pkl").open("wb") as stream:
        pickle.dump({"param_mean": np.zeros(1), "param_std": np.ones(1)}, stream)

    monkeypatch.setattr(paths, "model_path", lambda *parts, writable=False: tmp_path.joinpath(*parts))
    sys.modules.pop("rt_gene.ThreeDDFA.params", None)
    threeddfa_package = importlib.import_module("rt_gene.ThreeDDFA")
    monkeypatch.delattr(threeddfa_package, "params", raising=False)
    try:
        params = importlib.import_module("rt_gene.ThreeDDFA.params")
        assert params.keypoints.tolist() == [0, 1, 2]
    finally:
        sys.modules.pop("rt_gene.ThreeDDFA.params", None)
