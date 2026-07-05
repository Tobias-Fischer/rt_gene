from pathlib import Path

from setuptools import find_packages, setup


BUNDLED_MODEL_FILES = [
    Path("model_nets/face_model_68.txt"),
    Path("model_nets/SFD/README.md"),
    Path("model_nets/ThreeDDFA/keypoints_sim.npy"),
    Path("model_nets/ThreeDDFA/param_whitening.pkl"),
    Path("model_nets/ThreeDDFA/param_whitening_py2.pkl"),
    Path("model_nets/ThreeDDFA/u_exp.npy"),
    Path("model_nets/ThreeDDFA/u_shp.npy"),
]


DEMO_IMAGE_FILES = [
    Path("demo_images/gaze_center.jpg"),
]


def model_data_files():
    return [
        (str(Path("share/rt_gene_core") / path.parent), [str(path)])
        for path in [*BUNDLED_MODEL_FILES, *DEMO_IMAGE_FILES]
    ]


setup(
    name="rt_gene_core",
    version="5.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/rt_gene_core"]),
        ("share/rt_gene_core", ["package.xml"]),
        *model_data_files(),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
)
