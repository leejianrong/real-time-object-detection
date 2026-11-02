import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from rtod.detect import INPUT_SIZE, OrtDetector  # noqa: E402
from rtod.models import build_yolox_tiny, export_onnx  # noqa: E402
from rtod.quant import decode_tail, quantize_ptq  # noqa: E402


@pytest.fixture(scope="module")
def fp32(tmp_path_factory):
    return export_onnx(build_yolox_tiny(), tmp_path_factory.mktemp("m") / "fp32.onnx")


def test_decode_tail_excludes_sigmoids_but_no_convs(fp32):
    import onnx

    tail = set(decode_tail(fp32))
    ops = {n.name: n.op_type for n in onnx.load(str(fp32)).graph.node}
    assert tail and "Conv" not in {ops[n] for n in tail}


def test_ptq_is_smaller_and_scores_survive(fp32, tmp_path):
    rng = np.random.default_rng(0)
    imgs = []
    for i in range(4):
        p = tmp_path / f"{i}.jpg"
        cv2.imwrite(str(p), rng.integers(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
        imgs.append(p)
    ptq = quantize_ptq(fp32, tmp_path / "ptq.onnx", imgs)
    assert ptq.stat().st_size < 0.35 * fp32.stat().st_size
    blob = np.full((1, 3, INPUT_SIZE, INPUT_SIZE), 114, dtype=np.float32)
    # regression: a shared INT8 output scale once rounded every objectness/class score to 0
    assert OrtDetector(ptq).run_raw(blob)[:, 4:].max() > 0


def test_conv_only_quantizes_fewer_tensors_than_full_coverage(fp32, tmp_path):
    import onnx

    rng = np.random.default_rng(1)
    imgs = []
    for i in range(3):
        p = tmp_path / f"c{i}.jpg"
        cv2.imwrite(str(p), rng.integers(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
        imgs.append(p)

    def n_quant(path):
        return sum(n.op_type == "QuantizeLinear" for n in onnx.load(str(path)).graph.node)

    full = quantize_ptq(fp32, tmp_path / "full.onnx", imgs)
    conv = quantize_ptq(fp32, tmp_path / "conv.onnx", imgs, op_types=["Conv"])
    assert 0 < n_quant(conv) < n_quant(full)
