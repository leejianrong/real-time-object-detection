import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxconverter_common")

from rtod.detect import OrtDetector  # noqa: E402
from rtod.fp16 import to_fp16  # noqa: E402
from rtod.models import build_yolox_tiny, export_onnx  # noqa: E402


def test_fp16_conversion_halves_the_weights_and_keeps_float32_io(tmp_path):
    src = export_onnx(build_yolox_tiny(), tmp_path / "m.onnx")
    dst = to_fp16(src, tmp_path / "m16.onnx")
    model = onnx.load(str(dst))
    onnx.checker.check_model(model)
    assert dst.stat().st_size < 0.6 * src.stat().st_size
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def test_detector_reports_the_providers_it_actually_uses(tmp_path):
    path = export_onnx(build_yolox_tiny(), tmp_path / "m.onnx")
    has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        det = OrtDetector(path, device="cuda")
    assert det.providers[0] == ("CUDAExecutionProvider" if has_cuda else "CPUExecutionProvider")
    assert has_cuda or any("CUDAExecutionProvider" in str(w.message) for w in caught)
    assert np.isfinite(det.run_raw(np.zeros((1, 3, 416, 416), np.float32))).all()


@pytest.mark.gpu
def test_gpu_boxes_match_cpu_boxes_on_20_images(tmp_path):
    """On a CUDA machine, FP32 on the GPU finds the CPU's boxes (each matched at IoU >= 0.9)."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("needs onnxruntime-gpu and a CUDA device")
    import cv2

    from rtod import coco
    from rtod.models import load_official_weights

    path = export_onnx(load_official_weights(), tmp_path / "fp32.onnx")
    cpu, gpu = OrtDetector(path), OrtDetector(path, device="cuda")
    assert gpu.providers[0] == "CUDAExecutionProvider"
    ids = coco.read_ids("val200")[:20]
    for img_path in coco.fetch_images(ids, "val"):
        frame = cv2.imread(str(img_path))
        for box in cpu.detect(frame).boxes:
            assert max(_iou(box, other) for other in gpu.detect(frame).boxes) >= 0.9


def _iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
