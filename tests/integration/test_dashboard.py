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


async def _get(app, path):
    """One GET through the ASGI interface: (status, headers) without needing a test client."""
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": path, "raw_path": path.encode(),
             "query_string": b"", "headers": [], "server": ("t", 80), "scheme": "http",
             "root_path": "", "client": ("c", 1), "http_version": "1.1"}  # fmt: skip
    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], {k.decode(): v.decode() for k, v in start["headers"]}


def test_the_root_url_redirects_to_the_page(onnx):
    """Regression: only /ui was served, so Colab's inline view of the root showed a 404."""
    from rtod.dashboard import build_app

    app = build_app(Dashboard(CLIP, {"FP32": onnx, "INT8 PTQ": onnx}))
    status, headers = asyncio.run(_get(app, "/"))
    assert status in (302, 307) and headers["location"] == "ui/"
    assert asyncio.run(_get(app, "/ui/"))[0] == 200
