import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]  # downloads weights, annotations, images
pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from rtod import coco  # noqa: E402
from rtod.models import export_onnx, load_official_weights  # noqa: E402


def test_val200_eval_is_deterministic_and_plausible(tmp_path):
    onnx = export_onnx(load_official_weights(), tmp_path / "fp32.onnx")
    ids = coco.read_ids("val200")
    a, b = coco.evaluate(onnx, ids), coco.evaluate(onnx, ids)
    assert a == b
    assert 0.30 < a["map50_95"] < 0.45
