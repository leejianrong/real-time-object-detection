import asyncio
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("indah")

from rtod.dashboard import Dashboard  # noqa: E402
from rtod.models import build_yolox_tiny, export_onnx  # noqa: E402

CLIP = Path(__file__).parents[2] / "assets/demo/traffic_ljubljana.mp4"


@pytest.fixture(scope="module")
def onnx(tmp_path_factory):
    return export_onnx(build_yolox_tiny(), tmp_path_factory.mktemp("m") / "m.onnx")


def test_both_lanes_publish_statistics_for_the_same_clip(onnx):
    async def run():
        dash = Dashboard(CLIP, {"FP32": onnx, "INT8 PTQ": onnx}, show_fps=30)
        dash.start()
        await asyncio.sleep(4)
        dash.task.cancel()
        return dash

    dash = asyncio.run(run())
    for lane in (dash.left, dash.right):
        assert len(lane.bars.value) > 0 and float(lane.p50.value) > 0
    assert dash.frame.value.startswith("data:image/jpeg")
    assert dash.session().snapshot()["type"]  # the page renders to a component tree


def test_choosing_a_variant_swaps_the_right_hand_model(onnx):
    dash = Dashboard(CLIP, {"FP32": onnx, "INT8 PTQ": onnx, "INT8 QAT": onnx})
    assert dash.right.title.value == "INT8 PTQ"
    dash.choose("INT8 QAT")
    assert dash.right.title.value == "INT8 QAT" and not dash.right.latencies


def test_needs_fp32_and_another_variant(onnx):
    with pytest.raises(ValueError, match="FP32"):
        Dashboard(CLIP, {"INT8 PTQ": onnx})
