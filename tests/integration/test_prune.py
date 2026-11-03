import numpy as np
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("torch")
pytest.importorskip("torch_pruning")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from rtod.models import INPUT_SIZE, build_yolox_tiny, export_onnx  # noqa: E402
from rtod.prune import macs_params, prune_to_macs  # noqa: E402


def test_pruned_onnx_runs_and_its_size_matches_the_torch_model(tmp_path):
    pruned = prune_to_macs(build_yolox_tiny(), 0.35)
    pruned.head.decode_in_inference = True
    path = export_onnx(pruned, tmp_path / "pruned.onnx")
    onnx_params = sum(int(np.prod(i.dims)) for i in onnx.load(str(path)).graph.initializer)
    assert abs(onnx_params - macs_params(pruned)[1]) / macs_params(pruned)[1] < 0.02
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(
        None, {sess.get_inputs()[0].name: np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), np.float32)}
    )
    assert out[0].shape == (1, 3549, 85)
