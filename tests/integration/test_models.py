import numpy as np
import pytest

pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from rtod.models import INPUT_SIZE, build_yolox_tiny, export_onnx  # noqa: E402


def test_onnx_matches_pytorch_on_20_seeded_inputs(tmp_path):
    torch.manual_seed(0)
    model = build_yolox_tiny()
    path = export_onnx(model, tmp_path / "m.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.random((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32) * 255
        expected = model(torch.from_numpy(x)).detach().numpy()
        (got,) = sess.run(None, {"images": x})
        assert got.shape == (1, 3549, 85)
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.slow
def test_official_weights_load_strictly():
    from rtod.models import load_official_weights

    model = load_official_weights()
    assert sum(p.numel() for p in model.parameters()) == pytest.approx(5.06e6, rel=0.01)
