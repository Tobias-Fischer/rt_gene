from pathlib import Path

from rt_gene_core.paths import model_path


def resolve_model_files(files, writable=False):
    resolved = []
    for item in files:
        path = Path(item)
        resolved.append(str(path if path.is_absolute() else model_path(item, writable=writable)))
    return resolved
