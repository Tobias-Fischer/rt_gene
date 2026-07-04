from glob import glob

from setuptools import find_packages, setup


setup(
    name="rt_gene_ros",
    version="5.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/rt_gene_ros"]),
        ("share/rt_gene_ros", ["package.xml"]),
        ("share/rt_gene_ros/launch", glob("launch/*.launch.py")),
        ("share/rt_gene_ros/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "extract_landmarks = rt_gene_ros.landmark_node:main",
            "estimate_gaze = rt_gene_ros.gaze_node:main",
            "estimate_blink = rt_gene_ros.blink_node:main",
        ],
    },
)
