from rt_gene_core import paths


def test_model_path_reads_existing_cache_file(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    source = tmp_path / "source"
    cached = cache / "SFD" / "detector.pth"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"model")

    monkeypatch.setenv("RT_GENE_MODEL_DIR", str(cache))
    monkeypatch.setattr(paths, "_share_dir", lambda: None)
    monkeypatch.setattr(paths, "_source_dir", lambda: source)

    assert paths.model_path("SFD/detector.pth") == cached
