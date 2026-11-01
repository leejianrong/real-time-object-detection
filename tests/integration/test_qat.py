import numpy as np
import pytest

pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from rtod.models import INPUT_SIZE, build_yolox_tiny, export_onnx  # noqa: E402
from rtod.qat import QConv, calibrate, prepare_qat, set_fake_quant  # noqa: E402


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(0)
    return [torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE, generator=g) * 255 for _ in range(4)]


def test_prepare_replaces_every_conv_and_folding_is_exact(data):
    torch.manual_seed(0)
    model = build_yolox_tiny()
    for bn in (m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)):
        bn.running_mean.normal_(0, 0.1)  # make folding non-trivial for a random init
        bn.running_var.uniform_(0.5, 1.5)
    model.head.decode_in_inference = True
    ref = model(data[0]).detach()
    q = prepare_qat(model)
    assert not any(type(m) is torch.nn.Conv2d for m in q.modules())
    assert sum(isinstance(m, QConv) for m in q.modules()) == 83
    set_fake_quant(q, False)
    torch.testing.assert_close(q.eval()(data[0]).detach(), ref, rtol=1e-3, atol=1e-2)


def test_qdq_export_matches_torch_fake_quant_in_ort(data, tmp_path):
    torch.manual_seed(0)
    q = prepare_qat(build_yolox_tiny())
    q.head.decode_in_inference = True
    calibrate(q, data)
    with torch.no_grad():
        expected = q(data[0]).numpy()
    path = export_onnx(q, tmp_path / "qdq.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (got,) = sess.run(None, {"images": data[0].numpy()})
    assert np.corrcoef(got[0, :, 4], expected[0, :, 4])[0, 1] > 0.99
