from pathlib import Path


def test_active_packages_do_not_reference_tensorflow():
    repo = Path(__file__).resolve().parents[2]
    active = ["rt_gene_core/src", "rt_gene_ros", "opencv_camera", "rt_gene_interfaces"]
    offenders = []
    for root in active:
        for path in (repo / root).rglob("*"):
            if any(part.endswith(".egg-info") or part == "__pycache__" for part in path.parts):
                continue
            if path.is_file() and path.suffix in {".py", ".xml", ".txt", ".toml", ".md", ".launch.py", ".msg"}:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
                if "tensorflow" in text or "tf.keras" in text:
                    offenders.append(path.relative_to(repo))
    assert offenders == []


def test_core_package_has_no_ros_imports():
    repo = Path(__file__).resolve().parents[2]
    forbidden = [
        "rclpy",
        "rospy",
        "roslib",
        "rospkg",
        "sensor_msgs",
        "geometry_msgs",
        "std_msgs",
        "cv_bridge",
        "tf2_ros",
        "rt_gene_interfaces",
    ]
    offenders = []
    for path in (repo / "rt_gene_core" / "src").rglob("*.py"):
        if any(part.endswith(".egg-info") or part == "__pycache__" for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in forbidden:
            if f"import {name}" in text or f"from {name}" in text:
                offenders.append(path.relative_to(repo))
                break
    assert offenders == []
