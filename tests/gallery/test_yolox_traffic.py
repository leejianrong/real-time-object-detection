"""The yolox_traffic gallery example: slot, fallback, replay honesty."""

import asyncio
import importlib.util
import json
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("indah")
pytest.importorskip("onnxruntime")
pytest.importorskip("PIL")

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "gallery/examples/yolox_traffic.py"


def load():
    spec = importlib.util.spec_from_file_location("yolox_traffic_example", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = load()


class FakeCtx:
    """What a session factory gets: close hooks and background tasks that end with the session."""

    def __init__(self):
        self.closers, self.tasks = [], []

    def on_close(self, fn):
        self.closers.append(fn)

    def spawn(self, coro):
        self.tasks.append(asyncio.get_running_loop().create_task(coro))

    def close(self):
        for t in self.tasks:
            t.cancel()
        for fn in self.closers:
            fn()


class StubModel:
    """Answers with the recorded boxes after a fixed delay, so no model file is needed."""

    def __init__(self, ms=5.0):
        self.ms, self.clip, self.confs = ms, ex.Clip(), []

    def detect(self, bgr, conf=0.3):
        self.confs.append(conf)
        time.sleep(self.ms / 1e3)
        return self.clip.meta["frames_detections"][13]  # a frame with a dozen boxes


@pytest.fixture(autouse=True)
def fresh_slot(monkeypatch):
    monkeypatch.setattr(ex, "SLOT", ex.Slot())
    monkeypatch.setattr(ex, "_model", None)


async def until(condition, seconds=4.0):
    """Wait for `condition()`: the clip's first moments have no objects, so boxes take a beat."""
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert condition()


def player_of(ctx):
    """The Player a session built: reach it through its first spawned task."""
    return ctx.tasks[1].get_coro().cr_frame.f_locals["self"]


# ---- unit: slot, fallback predicate, slow detector, statistics ---------------------------------
def test_slot_grants_one_holder_and_queues_nobody():
    slot, a, b = ex.Slot(), object(), object()
    assert slot.try_acquire(a) and not slot.try_acquire(b)
    slot.release(b)  # not the holder: nothing happens
    assert slot.held
    slot.release(a)
    assert not slot.held and slot.try_acquire(b)


def test_fallback_reason_names_the_cause():
    assert ex.fallback_reason(slot_free=True, model_ok=True) is None
    assert ex.fallback_reason(slot_free=False, model_ok=True) == "server busy"
    assert ex.fallback_reason(slot_free=True, model_ok=False) == "model unavailable"
    assert ex.fallback_reason(slot_free=False, model_ok=False) == "model unavailable"


def test_too_slow_needs_enough_samples_and_a_full_window():
    fast = [i * 0.1 for i in range(40)]  # 10 per second
    assert not ex.too_slow(fast, now=4.0)
    slow = [i * 0.5 for i in range(10)]  # 2 per second
    assert ex.too_slow(slow, now=5.0)
    assert not ex.too_slow(slow[:3], now=1.5)  # not enough evidence yet


def test_percentile():
    assert ex.percentile([], 0.5) == 0.0
    assert ex.percentile([3, 1, 2], 0.5) == 2


def test_assets_are_downloaded_when_only_the_py_file_was_copied(monkeypatch, tmp_path):
    """A Colab or Space copies just yolox_traffic.py, not the folder beside it."""
    real = ex.ASSETS
    monkeypatch.setattr(ex, "ASSETS", tmp_path / "missing")
    monkeypatch.setattr(ex.tempfile, "gettempdir", lambda: str(tmp_path))
    fetched = []

    def fake_retrieve(url, dest):
        fetched.append(url.rsplit("/", 1)[1])
        Path(dest).write_bytes((real / url.rsplit("/", 1)[1]).read_bytes())

    monkeypatch.setattr(ex.urllib.request, "urlretrieve", fake_retrieve)
    clip = ex.Clip()
    assert len(clip) == 180 and sorted(fetched) == ["frames.bin", "replay.json"]
    ex.Clip()  # a second clip reuses the cache: no more downloads
    assert len(fetched) == 2


def test_the_page_credits_the_clip_and_the_model():
    """CC BY needs visible attribution, not just a file beside the example."""
    import json as _json

    text = _json.dumps(ex.Session(ex.page(ex.Player(ex.Clip()))).snapshot())
    assert "Sounds of Changes" in text and "CC BY 3.0" in text and "Megvii" in text


# ---- the recorded data -------------------------------------------------------------------------
def test_replay_file_is_small_versioned_and_covers_every_frame():
    path = ROOT / "gallery/examples/yolox_traffic/replay.json"
    assert path.stat().st_size < 3_000_000
    meta = json.loads(path.read_text())
    assert meta["schema_version"] == 1 and meta["recorded_on"]
    assert len(meta["frames"]) == len(meta["frames_detections"]) == len(meta["latency_ms"]) == 180


# ---- sessions: replay by default, live on request ----------------------------------------------
def test_a_new_session_opens_on_the_replay_and_draws_boxes_without_the_model(monkeypatch):
    monkeypatch.setattr(
        ex, "load_model", lambda: pytest.fail("the model must not load until asked")
    )

    async def run():
        ctx = FakeCtx()
        ex.build_session(ctx)
        p = player_of(ctx)
        await until(lambda: p.boxes.value)
        assert not p.live and p.badge.value.startswith("Replay: recorded on")
        assert not ex.SLOT.held
        ctx.close()

    asyncio.run(run())


def test_two_visitors_ask_for_live_one_gets_it_and_the_slot_frees_on_close(monkeypatch):
    monkeypatch.setattr(ex, "load_model", lambda: StubModel())

    async def run():
        a, b = FakeCtx(), FakeCtx()
        ex.build_session(a)
        ex.build_session(b)
        pa, pb = player_of(a), player_of(b)
        await pa.go_live()
        await pb.go_live()
        await until(lambda: pa.boxes.value and pb.boxes.value)
        assert pa.live and pa.badge.value == ""  # the control says it; no footnote
        assert not pb.live and "server busy" in pb.badge.value
        a.close()  # abrupt disconnect
        assert not ex.SLOT.held
        await pb.go_live()
        assert pb.live
        b.close()

    asyncio.run(run())


def test_the_recorded_live_control_takes_and_gives_back_the_model(monkeypatch):
    monkeypatch.setattr(ex, "load_model", lambda: StubModel())

    async def run():
        c = FakeCtx()
        ex.build_session(c)
        p = player_of(c)
        p.mode.set(ex.LIVE)
        await p.sync_mode()
        assert p.live and ex.SLOT.held and p.mode.value == ex.LIVE
        p.mode.set(ex.RECORDED)
        await p.sync_mode()
        assert not p.live and not ex.SLOT.held and p.badge.value.startswith("Replay: recorded on")
        c.close()

    asyncio.run(run())


def test_a_refused_live_request_puts_the_control_back_on_recorded(monkeypatch):
    monkeypatch.setattr(ex, "load_model", lambda: None)

    async def run():
        c = FakeCtx()
        ex.build_session(c)
        p = player_of(c)
        p.mode.set(ex.LIVE)
        await p.sync_mode()
        assert p.mode.value == ex.RECORDED and "model unavailable" in p.badge.value
        c.close()

    asyncio.run(run())


def test_the_threshold_slider_reaches_the_model_only_in_live(monkeypatch):
    stub = StubModel()
    monkeypatch.setattr(ex, "load_model", lambda: stub)

    async def run():
        c = FakeCtx()
        ex.build_session(c)
        p = player_of(c)
        p.conf.set(0.1)  # a drag while recorded: read-only, so it snaps back
        await p.sync_mode()
        assert p.conf.value == ex.CONF
        p.mode.set(ex.LIVE)
        await p.sync_mode()
        p.conf.set(0.1)
        await p.sync_mode()
        assert p.conf.value == 0.1  # live: the slider is free
        await until(lambda: stub.confs and stub.confs[-1] == 0.1)
        p.mode.set(ex.RECORDED)
        await p.sync_mode()
        assert p.conf.value == ex.CONF  # back to the recording's own threshold
        c.close()

    asyncio.run(run())


def test_a_lower_threshold_finds_more_boxes_than_the_recording_has():
    import numpy as np

    raw = np.zeros((3, 85), dtype=np.float32)
    raw[:, 2:4] = 20.0
    raw[:, 0] = [50, 150, 250]
    raw[:, 1] = 50
    raw[:, 4] = [0.9, 0.5, 0.2]
    raw[:, 5] = 1.0  # class 0 at full confidence, so the score is the objectness
    few = ex.decode(raw, 1.0, (416, 416, 3), conf=ex.CONF)
    many = ex.decode(raw, 1.0, (416, 416, 3), conf=0.1)
    assert len(few) == 2 and len(many) == 3


def test_the_page_has_the_short_title_and_a_subtitle_that_matches_the_shipped_model():
    text = json.dumps(ex.Session(ex.page(ex.Player(ex.Clip()))).snapshot())
    assert "Real-Time Object Detection with YOLOX-Tiny (INT8)" in text
    assert "first time seeing this clip" in text
    assert "ptq" in ex.MODEL_FILE and "quantization-aware" not in text  # say what actually runs
    assert "Recorded: boxes and timings" not in text


def test_a_model_that_cannot_load_says_why(monkeypatch, caplog):
    monkeypatch.delenv("YOLOX_TRAFFIC_MODEL", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)  # import fails
    with caplog.at_level("WARNING"):
        assert ex.load_model() is None
    assert "huggingface_hub" in caplog.text


def test_missing_model_stays_in_replay_with_the_reason(monkeypatch):
    monkeypatch.setattr(ex, "load_model", lambda: None)

    async def run():
        ctx = FakeCtx()
        ex.build_session(ctx)
        p = player_of(ctx)
        await p.go_live()
        await until(lambda: p.boxes.value)
        assert not p.live and "model unavailable" in p.badge.value
        assert float(p.p50.value) > 0
        ctx.close()

    asyncio.run(run())


def test_a_server_that_is_too_slow_falls_back_to_replay_and_frees_the_slot(monkeypatch):
    monkeypatch.setattr(ex, "load_model", lambda: StubModel(ms=500))  # 2 fps

    async def run():
        ctx = FakeCtx()
        ex.build_session(ctx)
        p = player_of(ctx)
        await p.go_live()
        assert p.live
        deadline = time.monotonic() + 12
        while p.live and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        assert not p.live and "slow" in p.badge.value and not ex.SLOT.held
        ctx.close()

    asyncio.run(run())


def test_a_live_run_ends_after_its_time_cap_and_frees_the_slot(monkeypatch):
    monkeypatch.setattr(ex, "load_model", lambda: StubModel())
    monkeypatch.setattr(ex, "LIVE_MAX_S", 0.5)

    async def run():
        ctx = FakeCtx()
        ex.build_session(ctx)
        p = player_of(ctx)
        await p.go_live()
        assert p.live
        await until(lambda: not p.live, seconds=4)
        assert "live run ended" in p.badge.value and not ex.SLOT.held
        ctx.close()

    asyncio.run(run())


# ---- review follow-ups: labels, a failing model, shared state, a pinned download ---------------
def test_the_class_names_are_the_80_coco_classes():
    assert len(ex.COCO) == 80
    assert ex.COCO[0] == "person" and ex.COCO[10] == "fire hydrant" and ex.COCO[11] == "stop sign"
    assert ex.COCO[-1] == "toothbrush" and len(set(ex.COCO)) == 80


class BrokenModel:
    def detect(self, bgr, conf=0.3):
        raise RuntimeError("bad frame")


def test_a_model_error_frees_the_slot_and_falls_back_instead_of_killing_the_loop(monkeypatch):
    monkeypatch.setattr(ex, "load_model", lambda: BrokenModel())

    async def run():
        ctx = FakeCtx()
        ex.build_session(ctx)
        p = player_of(ctx)
        await p.go_live()
        assert p.live
        await until(lambda: not p.live, seconds=4)
        assert "model error" in p.badge.value and not ex.SLOT.held
        await asyncio.sleep(0.3)
        assert not ctx.tasks[1].done()  # the run loop is still alive, now replaying
        ctx.close()

    asyncio.run(run())


def test_a_run_of_very_slow_frames_falls_back_even_below_one_fps(monkeypatch):
    monkeypatch.setattr(ex, "load_model", lambda: StubModel(ms=300))
    monkeypatch.setattr(ex, "SLOW_FRAME_MS", 100.0)  # stands in for "over a second a frame"

    async def run():
        ctx = FakeCtx()
        ex.build_session(ctx)
        p = player_of(ctx)
        await p.go_live()
        await until(lambda: not p.live, seconds=4)
        assert "slow" in p.badge.value and not ex.SLOT.held
        ctx.close()

    asyncio.run(run())


def test_sessions_share_one_clip():
    a, b = FakeCtx(), FakeCtx()

    async def run():
        ex.build_session(a)
        ex.build_session(b)
        assert player_of(a).clip is player_of(b).clip is ex.get_clip()
        a.close()
        b.close()

    asyncio.run(run())


def test_a_frame_is_encoded_once_not_per_visitor():
    clip = ex.get_clip()
    assert clip.uri(3) is clip.uri(3)


def test_two_clicks_load_the_model_once(monkeypatch, tmp_path):
    import threading

    made = []

    class Counting:
        def __init__(self, path):
            time.sleep(0.2)
            made.append(path)

    f = tmp_path / "m.onnx"
    f.write_bytes(b"x")
    monkeypatch.setenv("YOLOX_TRAFFIC_MODEL", str(f))
    monkeypatch.setattr(ex, "Model", Counting)
    monkeypatch.setattr(ex, "_model", None)
    threads = [threading.Thread(target=ex.load_model) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(made) == 1


def test_the_download_is_pinned_to_a_commit_and_checked_against_the_recording(
    monkeypatch, tmp_path, caplog
):
    import sys
    import types

    seen = {}
    wrong = tmp_path / "other.onnx"
    wrong.write_bytes(b"not the model the replay was recorded with")

    def fake_download(repo, filename, **kw):
        seen.update(repo=repo, filename=filename, **kw)
        return str(wrong)

    monkeypatch.delenv("YOLOX_TRAFFIC_MODEL", raising=False)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download)
    )
    monkeypatch.setattr(ex, "_model", None)
    with caplog.at_level("WARNING"):
        assert ex.load_model() is None  # a different file than the recording used: not served
    assert seen["revision"] == ex.MODEL_REVISION and len(ex.MODEL_REVISION) == 40
    assert "checksum" in caplog.text


def test_the_page_credit_links_the_source_and_the_licence():
    text = json.dumps(ex.Session(ex.page(ex.Player(ex.Clip()))).snapshot())
    assert "https://commons.wikimedia.org/wiki/File:Road_traffic_in_Ljubljana.webm" in text
    assert "https://creativecommons.org/licenses/by/3.0/" in text


# ---- honesty: replay equals what the real model says on the same frames ------------------------
def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    w, h = min(ax2, bx2) - max(a[0], b[0]), min(ay2, by2) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


@pytest.mark.slow
def test_replay_boxes_match_the_live_model_on_the_same_frames():
    from rtod.hub import fetch_model

    model = ex.Model(str(fetch_model("ptq-int8")))
    clip = ex.Clip()
    for i in range(0, len(clip), 12):  # every 12th frame keeps this to a few seconds
        live, recorded = model.detect(clip.array_bgr(i)), clip.meta["frames_detections"][i]
        assert len(live) == len(recorded)
        for box in recorded:
            assert max(iou(box, other) for other in live if other[4] == box[4]) >= 0.9


def test_the_class_names_match_the_rtod_list():
    from rtod.coco_names import COCO_NAMES

    assert tuple(ex.COCO) == tuple(COCO_NAMES)


# ---- mounting: the gallery serves each demo under /<slug>, with a strict CSP -------------
async def _get(app, path):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "method": "GET", "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": [(b"host", b"gallery.test")],
        "server": ("gallery.test", 80), "scheme": "http", "root_path": "",
        "client": ("c", 1), "http_version": "1.1",
    }  # fmt: skip
    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], {k.decode().lower(): v.decode() for k, v in start["headers"]}


def test_the_demo_serves_under_a_sub_path_mount_and_under_a_strict_csp():
    from starlette.applications import Starlette
    from starlette.routing import Mount

    strict = ex.create_app(session_factory=ex.build_session, csp="strict", spa=True)
    gallery = Starlette(routes=[Mount("/yolox-traffic", app=strict)])
    status, headers = asyncio.run(_get(gallery, "/yolox-traffic/"))
    assert status == 200
    assert "content-security-policy" in headers
    assert (
        "img-src" in headers["content-security-policy"]
        and "data:" in headers["content-security-policy"]
    )
