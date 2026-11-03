import warnings

import pytest

ort = pytest.importorskip("onnxruntime")

from rtod import detect  # noqa: E402


def test_cuda_request_without_a_gpu_provider_warns_and_uses_the_cpu(monkeypatch):
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.warns(UserWarning, match="CUDAExecutionProvider is not available"):
        assert detect._providers("cuda") == ["CPUExecutionProvider"]


def test_cuda_provider_is_used_when_present(monkeypatch):
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert detect._providers("cuda")[0] == "CUDAExecutionProvider"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="device"):
        detect._providers("tpu")
