import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from rtod.bench import bench, load_result, write_result  # noqa: E402
from rtod.models import build_yolox_tiny, export_onnx  # noqa: E402


def test_quick_bench_writes_valid_labelled_result(tmp_path):
    onnx = export_onnx(build_yolox_tiny(), tmp_path / "m.onnx")
    res = bench(onnx, "yolox-tiny/random", warmup=2, iters=5, quick=True)
    back = load_result(write_result(res, tmp_path))
    assert back.quick and back.iters == 5
    assert 0 < back.model_ms_p50 <= back.e2e_ms_p50 and back.hardware["cpu"]


def test_resource_measurement_runs_in_a_fresh_process(tmp_path):
    from rtod.resources import measure_resources

    onnx = export_onnx(build_yolox_tiny(), tmp_path / "m.onnx")
    r = measure_resources(onnx, threads=1, iters=5)
    assert r["peak_rss_mb"] > 5  # an 20 MB model plus activations must add something
    assert 0 < r["cores_used"] < 1.6 and r["cpu_ms_per_inference"] > 0


def test_gpu_bench_refuses_to_time_the_cpu_when_cuda_is_missing(tmp_path):
    """A silent CPU fallback would publish CPU numbers under a GPU label."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("this machine has CUDA")
    onnx = export_onnx(build_yolox_tiny(), tmp_path / "m.onnx")
    with (
        pytest.warns(UserWarning),
        pytest.raises(ValueError, match="fell back to CPUExecutionProvider"),
    ):
        bench(onnx, "yolox-tiny/gpu", warmup=1, iters=2, quick=True, device="cuda")
