from glob import glob

from setuptools import find_packages, setup


setup(
    name="opencv_camera",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/opencv_camera"]),
        ("share/opencv_camera", ["package.xml"]),
        ("share/opencv_camera/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "camera_node = opencv_camera.camera_node:main",
        ],
    },
)
