import hashlib

import pytest

from rt_gene.download_tools import request_if_not_exist


class FakeResponse:
    def __init__(self, body, headers=None, status_error=None):
        self.body = body
        self.headers = headers or {}
        self.status_error = status_error
        self.closed = False

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def iter_content(self, chunksize):
        yield self.body

    def close(self):
        self.closed = True


def test_download_without_content_length(monkeypatch, tmp_path):
    body = b"model"
    response = FakeResponse(body)
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response)

    target = tmp_path / "model.bin"
    request_if_not_exist(target, "https://example.invalid/model", hashlib.md5(body).hexdigest())

    assert target.read_bytes() == body
    assert not target.with_name("model.bin.part").exists()
    assert response.closed


def test_download_removes_partial_on_checksum_error(monkeypatch, tmp_path):
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeResponse(b"wrong"))

    target = tmp_path / "model.bin"
    with pytest.raises(RuntimeError, match="MD5 checksum mismatch"):
        request_if_not_exist(target, "https://example.invalid/model", "not-the-md5")

    assert not target.exists()
    assert not target.with_name("model.bin.part").exists()
