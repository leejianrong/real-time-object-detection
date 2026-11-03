import hashlib

import pytest

from rtod import hub


def test_fetch_downloads_once_and_verifies(tmp_path, monkeypatch):
    src = tmp_path / "remote"
    src.mkdir()
    data = b"weights"
    (src / "m.onnx").write_bytes(data)
    monkeypatch.setitem(hub.MODELS, "t", ("m.onnx", hashlib.sha256(data).hexdigest()))
    got = hub.fetch_model("t", tmp_path / "cache", base=f"file://{src}")
    assert got.read_bytes() == data
    (src / "m.onnx").unlink()  # a second call must not need the network
    assert hub.fetch_model("t", tmp_path / "cache", base=f"file://{src}") == got


def test_corrupt_download_is_rejected_and_removed(tmp_path, monkeypatch):
    src = tmp_path / "remote"
    src.mkdir()
    (src / "m.onnx").write_bytes(b"tampered")
    monkeypatch.setitem(hub.MODELS, "t", ("m.onnx", hashlib.sha256(b"real").hexdigest()))
    with pytest.raises(ValueError, match="sha256"):
        hub.fetch_model("t", tmp_path / "cache", base=f"file://{src}")
    assert not (tmp_path / "cache" / "m.onnx").exists()
