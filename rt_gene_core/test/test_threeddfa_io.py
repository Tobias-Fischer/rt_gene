import numpy as np

from rt_gene.ThreeDDFA.io import _load
from rt_gene.ThreeDDFA.inference import get_suffix


def test_load_accepts_pathlike_filename(tmp_path):
    path = tmp_path / "array.npy"
    expected = np.array([1, 2, 3])
    np.save(path, expected)

    np.testing.assert_array_equal(_load(path), expected)


def test_inference_suffix_accepts_pathlike_filename(tmp_path):
    assert get_suffix(tmp_path / "image.jpg") == ".jpg"
