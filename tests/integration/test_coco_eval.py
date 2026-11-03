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


def test_per_class_ap_agrees_with_the_overall_map(tmp_path):
    onnx = export_onnx(load_official_weights(), tmp_path / "fp32.onnx")
    r = coco.evaluate(onnx, coco.read_ids("val200"))
    valid = [v for v in r["per_class_ap"].values() if v == v]
    assert len(r["per_class_ap"]) == 80
    assert abs(sum(valid) / len(valid) - r["map50_95"]) < 1e-6
