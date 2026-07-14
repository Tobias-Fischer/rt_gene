from pathlib import Path
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def assert_editable_source(module, source_root):
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(source_root):
        raise RuntimeError(
            f"{module.__name__} imported from {module_path}, not editable source {source_root}. "
            "Run `pixi install` to refresh local editable packages."
        )


def main():
    import rt_gene.ThreeDDFA.io as threeddfa_io
    import rt_gene.ThreeDDFA.inference as inference
    import rt_gene_ros.model_paths as ros_model_paths

    assert_editable_source(threeddfa_io, REPO_ROOT / "rt_gene_core" / "src")
    assert_editable_source(ros_model_paths, REPO_ROOT / "rt_gene_ros")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "array.npy"
        expected = np.array([1, 2, 3])
        np.save(path, expected)
        np.testing.assert_array_equal(threeddfa_io._load(path), expected)
        assert inference.get_suffix(path) == ".npy"


if __name__ == "__main__":
    main()
