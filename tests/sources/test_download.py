import io
from pathlib import Path

import pytest

from puzzle_recsys.sources import download as dl


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _fake_urlopen(payload: bytes):
    def _open(request, timeout):
        return _FakeResponse(payload)

    return _open


def test_downloads_valid_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl.urllib.request, "urlopen", _fake_urlopen(b"PK\x03\x04data"))
    dest = tmp_path / "a.zip"

    result = dl.download_file("https://x/a.zip", dest)

    assert result == dest
    assert dest.read_bytes() == b"PK\x03\x04data"
    assert not dest.with_suffix(".zip.part").exists()


def test_rejects_html_disguised_as_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl.urllib.request, "urlopen", _fake_urlopen(b"<html>error</html>"))
    dest = tmp_path / "a.zip"

    with pytest.raises(dl.InvalidDownloadError):
        dl.download_file("https://x/a.zip", dest)

    assert not dest.exists()
    assert not dest.with_suffix(".zip.part").exists()


def test_skips_existing_valid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "a.zip"
    dest.write_bytes(b"PK\x03\x04old")

    def _boom(*args, **kwargs):
        raise AssertionError("no debe descargar")

    monkeypatch.setattr(dl.urllib.request, "urlopen", _boom)

    assert dl.download_file("https://x/a.zip", dest) == dest
    assert dest.read_bytes() == b"PK\x03\x04old"


def test_download_source_uses_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl.urllib.request, "urlopen", _fake_urlopen(b"PK\x03\x04x"))
    sources = {"denue": {"url": "https://x/d.zip", "file": "d.zip"}}

    path = dl.download_source("denue", raw_dir=tmp_path, sources=sources)

    assert path == tmp_path / "d.zip"
    assert path.exists()
