"""Real-time object detection on a traffic clip: a quantized YOLOX-Tiny, live when the server is
free.

A 15-second road-traffic clip loops while an INT8 YOLOX-Tiny (5.6 MB, Apache-2.0) runs on the
frame that was current when it started, so a slow server shows boxes that trail the video, which
is what the latency bars and the frames-per-second readout are there to show.

The page opens on a **replay**: detections recorded earlier from the same model on the same
frames, through the same UI. "Run live" runs the model on this server for up to 30 seconds. It is
a button, not the default, because the gallery box is a shared CPU (``shared-cpu-1x``) that bursts
at about 38 fps for roughly a minute and then throttles to about 2 fps; capping each run keeps the
demo from spending the credit that the other demos need. Inference is also bounded to **one
visitor at a time** (a global slot), and a run that turns out too slow drops back to the replay. A
badge says which one you are looking at and why. There are no uploads and no camera.

The model downloads once from the Hugging Face Hub (``jianlee/yolox-tiny-quantized``) and is
cached; if it is unavailable the demo still plays, in replay. ``onnxruntime``, Pillow and
``huggingface_hub`` are this demo's own dependencies (the ``indah[demos]`` extra) and are imported
lazily.

Clip: "Road traffic in Ljubljana" by Sounds of Changes, CC BY 3.0 (first 15 s, downscaled; see
``yolox_traffic/ATTRIBUTION.txt``). Model and code: https://github.com/leejianrong/real-time-
object-detection

Run it with:  python examples/yolox_traffic.py"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

import indah
from indah import (
    Badge,
    BarStrip,
    Button,
    Card,
    Column,
    ImageOverlay,
    Row,
    Session,
    Signal,
    Stat,
    Text,
    create_app,
)

log = logging.getLogger("yolox_traffic")
ASSETS = Path(__file__).with_name("yolox_traffic")
# A Colab or Space copies only this file: missing assets are fetched.
ASSET_URL = "https://raw.githubusercontent.com/leejianrong/indah/main/examples/yolox_traffic"
FPS_FLOOR = 8.0  # live below this for a few seconds falls back to replay
LIVE_MAX_S = 30.0  # one live run lasts at most this long (the shared CPU's burst credit is finite)
BUDGET_MS = 1000 / 30  # a frame takes under this to keep up with 30 fps video
MODEL_REPO, MODEL_FILE = "jianlee/yolox-tiny-quantized", "yolox_tiny_ptq_int8_calib32.onnx"
COLORS = {"car": "#3fb6c4", "truck": "#e9a23b", "bus": "#c77dd6", "person": "#7fd36b"}

COCO = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire,"
    "hydrant,stop sign,parking,"
    "meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports,"
    "ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis,"
    "racket,bottle,wine,"
    "glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot,"
    "dog,pizza,donut,cake,chair,couch,potted plant,bed,dining,"
    "table,toilet,tv,laptop,mouse,remote,keyboard,cell,"
    "phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy,"
    "bear,hair drier,toothbrush"
).split(",")


# ---- the clip: JPEG frames bundled in one file, so no video decoder is needed ------------------
def asset_folder() -> Path:
    """The bundled assets, or a downloaded copy when only this file was copied somewhere."""
    if (ASSETS / "replay.json").exists():
        return ASSETS
    cache = Path(tempfile.gettempdir()) / "yolox_traffic_assets"
    cache.mkdir(exist_ok=True)
    for name in ("replay.json", "frames.bin"):
        if not (cache / name).exists():
            urllib.request.urlretrieve(f"{ASSET_URL}/{name}", cache / name)  # noqa: S310 (fixed https URL)
    return cache


class Clip:
    def __init__(self, folder: Path | None = None) -> None:
        folder = folder or asset_folder()
        meta = json.loads((folder / "replay.json").read_text())
        blob = (folder / "frames.bin").read_bytes()
        self.meta, self.fps = meta, meta["fps"]
        self.frames = [blob[o : o + n] for o, n in meta["frames"]]

    def __len__(self) -> int:
        return len(self.frames)

    def uri(self, i: int) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.frames[i]).decode("ascii")

    def array_bgr(self, i: int) -> Any:
        import numpy as np
        from PIL import Image

        rgb = np.asarray(Image.open(io.BytesIO(self.frames[i])).convert("RGB"))
        return rgb[:, :, ::-1]  # the model was trained on BGR


# ---- the model: letterbox, run, decode, per-class NMS (numpy only, no OpenCV) ------------------
def letterbox(bgr: Any, size: int = 416) -> tuple[Any, float]:
    import numpy as np
    from PIL import Image

    ratio = min(size / bgr.shape[0], size / bgr.shape[1])
    h, w = round(bgr.shape[0] * ratio), round(bgr.shape[1] * ratio)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:h, :w] = np.asarray(
        Image.fromarray(np.ascontiguousarray(bgr)).resize((w, h), Image.BILINEAR)
    )
    return canvas.transpose(2, 0, 1)[None].astype(np.float32), ratio


def nms(boxes: Any, scores: Any, iou_thr: float) -> list[int]:
    import numpy as np

    order, keep = scores.argsort()[::-1], []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        wh = np.clip(
            np.minimum(boxes[i, 2:], boxes[rest, 2:]) - np.maximum(boxes[i, :2], boxes[rest, :2]),
            0,
            None,
        )
        inter = wh[:, 0] * wh[:, 1]
        order = rest[inter / (area[i] + area[rest] - inter + 1e-9) <= iou_thr]
    return keep


def decode(
    raw: Any, ratio: float, shape: tuple[int, ...], conf: float = 0.3, iou: float = 0.45
) -> list[list[float]]:
    """Rows [cx, cy, w, h, obj, 80 class scores] to [x, y, w, h, class, score], as fractions."""
    import numpy as np

    cls = raw[:, 5:].argmax(1)
    scores = raw[:, 4] * raw[np.arange(len(raw)), 5 + cls]
    m = scores >= conf
    raw, cls, scores = raw[m], cls[m], scores[m]
    boxes = np.concatenate([raw[:, :2] - raw[:, 2:4] / 2, raw[:, :2] + raw[:, 2:4] / 2], 1) / ratio
    boxes = np.clip(boxes, 0, np.array([shape[1], shape[0]] * 2, dtype=np.float32))
    keep: list[int] = []
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        keep += [int(idx[k]) for k in nms(boxes[idx], scores[idx], iou)]
    keep.sort(key=lambda i: -scores[i])
    h, w = shape[:2]
    rows = []
    for i in keep:
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        rows.append([round(x1 / w, 4), round(y1 / h, 4), round((x2 - x1) / w, 4),
                     round((y2 - y1) / h, 4), int(cls[i]), round(float(scores[i]), 3)])  # fmt: skip
    return rows


class Model:
    def __init__(self, path: str) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    def detect(self, bgr: Any) -> list[list[float]]:
        blob, ratio = letterbox(bgr)
        (out,) = self.session.run(None, {"images": blob})
        return decode(out[0], ratio, bgr.shape)


_model: Model | None = None


def load_model() -> Model | None:
    """The shared model, loaded once; None if it cannot be (then everyone gets the replay)."""
    global _model
    if _model is None:
        try:
            path = os.environ.get("YOLOX_TRAFFIC_MODEL")
            if not path:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(MODEL_REPO, MODEL_FILE)
            _model = Model(path)
        except Exception:
            log.warning("model unavailable, serving the replay only", exc_info=True)
            return None
    return _model


# ---- bounded work: one live holder at a time ---------------------------------------------------
class Slot:
    """Grants the model to one session at a time; everyone else is told "busy", never queued.

    Only the holder can release it, so a session that already gave the slot up (too slow) cannot
    free it again when its tab closes, out from under whoever holds it now.
    """

    def __init__(self) -> None:
        self.owner: object | None = None

    @property
    def held(self) -> bool:
        return self.owner is not None

    def try_acquire(self, owner: object) -> bool:
        if self.owner is not None:
            return False
        self.owner = owner
        return True

    def release(self, owner: object) -> None:
        if self.owner is owner:
            self.owner = None


SLOT = Slot()


def fallback_reason(*, slot_free: bool, model_ok: bool) -> str | None:
    """Why a session cannot run live, or None if it can."""
    if not model_ok:
        return "model unavailable"
    if not slot_free:
        return "server busy"
    return None


def too_slow(
    finished: list[float], now: float, window: float = 3.0, floor: float = FPS_FLOOR
) -> bool:
    """True when inference finished under ``floor`` times a second over the last ``window`` s."""
    recent = [t for t in finished if now - t <= window]
    return len(finished) >= 8 and now - finished[0] >= window and len(recent) / window < floor


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


# ---- one session: the same UI whether the boxes come from the model or from the recording ------
class Player:
    def __init__(self, clip: Clip) -> None:
        self.clip = clip
        self.frame, self.boxes, self.bars = Signal(clip.uri(0)), Signal([]), Signal([])
        self.fps, self.p50, self.p95, self.objects = (
            Signal("0"),
            Signal("0"),
            Signal("0"),
            Signal("0"),
        )
        self.badge = Signal("starting")
        self.latencies: deque[float] = deque(maxlen=60)
        self.finished: list[float] = []
        self.t0 = time.monotonic()
        self.live, self.live_until = False, 0.0
        self.model: Model | None = None
        self.recorded = clip.meta["frames_detections"]
        self.recorded_ms = clip.meta["latency_ms"]

    def index(self) -> int:
        return int((time.monotonic() - self.t0) * self.clip.fps) % len(self.clip)

    def set_mode(self, live: bool, reason: str | None = None) -> None:
        self.live = live
        hw = self.clip.meta["recorded_on"]
        if live:
            self.badge.set(f"Live: running the model on this server (up to {LIVE_MAX_S:.0f} s)")
        elif reason:
            self.badge.set(f"Replay ({reason}): recorded on {hw}")
        else:
            self.badge.set(f"Replay: recorded on {hw}")
        self.latencies.clear()
        self.finished = []

    async def go_live(self) -> None:
        """The "Run live" button: take the slot and run the model here, or say why not."""
        if self.live:
            return
        model = await asyncio.to_thread(load_model)  # the first call may download the model
        reason = fallback_reason(slot_free=not SLOT.held, model_ok=model is not None)
        if reason is None and SLOT.try_acquire(self):
            self.model, self.live_until = model, time.monotonic() + LIVE_MAX_S
            self.set_mode(True)
        else:
            self.set_mode(False, f"can't run live: {reason or 'server busy'}")

    async def display(self) -> None:
        last = -1
        while True:
            i = self.index()
            if i != last:
                self.frame.set(self.clip.uri(i))
                last = i
            await asyncio.sleep(1 / self.clip.fps / 2)

    def to_boxes(self, rows: list[list[float]]) -> list[dict[str, Any]]:
        return [
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": COCO[c],
                "score": s,
                "color": COLORS.get(COCO[c], "#d0d7de"),
            }
            for x, y, w, h, c, s in rows
        ]

    async def infer_once(self, i: int) -> tuple[list[list[float]], float]:
        if self.live and self.model is not None:
            bgr = self.clip.array_bgr(i)
            t0 = time.perf_counter()
            rows = await asyncio.to_thread(self.model.detect, bgr)
            return rows, (time.perf_counter() - t0) * 1e3
        ms = self.recorded_ms[i]
        await asyncio.sleep(ms / 1e3)
        return self.recorded[i], ms

    async def run(self) -> None:
        while True:
            i = self.index()
            rows, ms = await self.infer_once(i)
            now = time.monotonic()
            self.latencies.append(ms)
            self.finished = [t for t in self.finished if now - t <= 10] + [now]
            self.boxes.set(self.to_boxes(rows))
            self.bars.set([[min(x, BUDGET_MS), max(x - BUDGET_MS, 0.0)] for x in self.latencies])
            self.objects.set(str(len(rows)))
            self.p50.set(f"{percentile(list(self.latencies), 0.5):.0f}")
            self.p95.set(f"{percentile(list(self.latencies), 0.95):.0f}")
            self.fps.set(str(sum(1 for t in self.finished if now - t <= 1.0)))
            if self.live and too_slow(self.finished, now):
                SLOT.release(self)
                self.set_mode(False, f"slow: under {FPS_FLOOR:.0f} fps")
            elif self.live and now > self.live_until:
                SLOT.release(self)
                self.set_mode(False, f"live run ended after {LIVE_MAX_S:.0f} s")
            await asyncio.sleep(0)


def page(p: Player) -> Column:
    return Column(
        [
            Text("YOLOX-Tiny, INT8, on a traffic clip it was never trained on", variant="title"),
            Row(
                [Badge(p.badge, tone="info"), Button("Run live on this server", on_click=p.go_live)]
            ),
            Card(
                [
                    ImageOverlay(p.frame, boxes=p.boxes, label_mode="always"),
                    Row(
                        [
                            Stat(p.fps, label="inference fps"),
                            Stat(p.p50, label="p50 ms"),
                            Stat(p.p95, label="p95 ms"),
                            Stat(p.objects, label="objects"),
                        ]
                    ),  # fmt: skip
                    Text(
                        "Latency of each frame, last 60 (dark = over the 33 ms of a 30 fps frame)"
                    ),
                    BarStrip(
                        p.bars,
                        height=64,
                        tones=["ok", "warn"],
                        names=["within 33 ms", "over budget"],
                        label="Per-frame latency",
                    ),  # fmt: skip
                ]
            ),
            Text(
                "Boxes update when the model finishes a frame, so a slow server trails the video. "
                "Live runs last up to 30 s, one visitor at a time; replay is the same model."
            ),
            Text(
                "Clip: 'Road traffic in Ljubljana' by Sounds of Changes, CC BY 3.0 "
                "(first 15 s, downscaled). Model: YOLOX-Tiny (Megvii, Apache-2.0), INT8."
            ),
        ]
    )


def build_session(ctx: Any) -> Session:
    player = Player(Clip())
    player.set_mode(False)
    ctx.on_close(lambda: SLOT.release(player))  # a closed or abandoned tab frees the slot
    ctx.spawn(player.display())
    ctx.spawn(player.run())
    return Session(page(player))


app = create_app(session_factory=build_session, title="Live object detection", factory_timeout=2.0)

if __name__ == "__main__":
    indah.launch(app)
