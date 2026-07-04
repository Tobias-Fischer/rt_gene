from pathlib import Path
import os


PACKAGE_NAME = "rt_gene_core"


def _share_dir():
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory(PACKAGE_NAME))
    except Exception:
        return None


def _source_dir():
    return Path(__file__).resolve().parents[2]


def _cache_dir():
    return Path(os.environ.get("RT_GENE_MODEL_DIR", Path.home() / ".cache" / "rt_gene" / "model_nets"))


def model_path(*parts, writable=False):
    rel = Path(*parts)
    if rel.is_absolute():
        return rel

    candidates = []
    source = _source_dir() / "model_nets" / rel
    share = _share_dir()
    if share is not None:
        candidates.append(share / "model_nets" / rel)
    candidates.append(source)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if writable:
        target = _cache_dir() / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    return candidates[0]
