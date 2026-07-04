import re
from pathlib import Path


DISTRO_SPECIFIC = re.compile(
    r"ros-(humble|jazzy|kilted|rolling|iron)-|"
    r"robostack-(humble|jazzy|kilted|rolling|iron)|"
    r"/opt/ros/(humble|jazzy|kilted|rolling|iron)|"
    r"ROS_DISTRO[^\\n]*(humble|jazzy|kilted|rolling|iron)",
    re.IGNORECASE,
)


def test_ros_source_does_not_hardcode_distro_names():
    repo = Path(__file__).resolve().parents[2]
    allowed_names = {"pixi.toml", "pixi.lock", "README.md"}
    skipped_parts = {".git", ".pixi", "__pycache__"}
    suffixes = {".py", ".xml", ".txt", ".toml", ".md", ".msg", ".cpp", ".hpp", ".cmake"}
    offenders = []

    for path in repo.rglob("*"):
        rel = path.relative_to(repo)
        if not path.is_file() or any(part in skipped_parts for part in rel.parts):
            continue
        if any(part.endswith(".egg-info") for part in rel.parts):
            continue
        if path == Path(__file__).resolve() or path.name in allowed_names or path.name == "files.txt":
            continue
        if path.name != "CMakeLists.txt" and path.suffix not in suffixes:
            continue
        if DISTRO_SPECIFIC.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(rel))

    assert offenders == []
