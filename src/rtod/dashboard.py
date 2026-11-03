"""Live detection dashboard built with indah (needs the `demo` extra).

A traffic clip plays at a fixed rate while each model runs on the frame that was current when it
started. A slow model therefore shows boxes that lag the video, which is what the per-frame latency
strip and the fps readout are there to explain. One `Dashboard` is shared by every viewer.
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
from indah import (
    BarStrip,
    Card,
    Chart,
    Column,
    ImageOverlay,
    Row,
    Segmented,
    Session,
    Signal,
    Stat,
    Text,
)

from rtod.coco_names import COCO_NAMES
from rtod.detect import OrtDetector

BUDGET_MS = 1000 / 30  # a model keeps up with 30 fps video when a frame takes under this
SHOW = (480, 270)  # frames are sent downscaled: a Colab proxy carries about a megabyte a second
COLORS = {"car": "#3fb6c4", "truck": "#e9a23b", "bus": "#c77dd6", "person": "#7fd36b"}
FALLBACK_COLOR = "#d0d7de"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


def jpeg_uri(frame: Any, size: tuple[int, int] = SHOW, quality: int = 60) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.resize(frame, size), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode the frame")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def split_bar(ms: float) -> list[float]:
    """A latency as a stacked bar: the part inside the frame budget, then the part over it."""
    return [min(ms, BUDGET_MS), max(ms - BUDGET_MS, 0.0)]


class Lane:
    """One model on the stream: runs inference when idle and publishes boxes and statistics."""

    def __init__(self, model: Path, *, threads: int = 1, device: str = "cpu", conf: float = 0.3):
        self.threads, self.device, self.conf = threads, device, conf
        self.latencies: deque[float] = deque(maxlen=60)
        self.done_at: deque[float] = deque()
        self.busy = False
        self.title = Signal("")
        self.boxes: Signal[list[dict[str, Any]]] = Signal([])
        self.bars: Signal[list[list[float]]] = Signal([])
        self.fps, self.p50, self.p95 = Signal("0"), Signal("0"), Signal("0")
        self.objects = Signal("0")
        self.fps_chart = Chart(series=["inference fps"], height=120, y_range=(0, 35), legend="off")
        self.set_model(model)

    def set_model(self, model: Path) -> None:
        self.detector = OrtDetector(model, threads=self.threads, conf=self.conf, device=self.device)
        self.latencies.clear()
        self.done_at.clear()

    def infer(self, frame: Any) -> tuple[list[dict[str, Any]], float]:
        """Detect on one frame (runs in a worker thread); boxes are fractions of the frame."""
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        det = self.detector.detect(frame)
        ms = (time.perf_counter() - t0) * 1e3
        boxes = [
            {
                "x": float(x1 / w), "y": float(y1 / h),
                "w": float((x2 - x1) / w), "h": float((y2 - y1) / h),
                "label": COCO_NAMES[int(c)], "score": round(float(s), 2),
                "color": COLORS.get(COCO_NAMES[int(c)], FALLBACK_COLOR),
            }
            for (x1, y1, x2, y2), s, c in zip(det.boxes, det.scores, det.class_ids, strict=True)
        ]  # fmt: skip
        return boxes, ms

    def record(self, boxes: list[dict[str, Any]], ms: float, now: float) -> None:
        self.latencies.append(ms)
        self.done_at.append(now)
        while self.done_at and now - self.done_at[0] > 1.0:
            self.done_at.popleft()
        self.boxes.set(boxes)
        self.bars.set([split_bar(x) for x in self.latencies])
        self.objects.set(str(len(boxes)))
        self.p50.set(f"{percentile(list(self.latencies), 0.5):.0f}")
        self.p95.set(f"{percentile(list(self.latencies), 0.95):.0f}")
        self.fps.set(str(len(self.done_at)))

    async def run_once(self, frame: Any) -> None:
        self.busy = True
        try:
            boxes, ms = await asyncio.to_thread(self.infer, frame)
            self.record(boxes, ms, time.monotonic())
        finally:
            self.busy = False


class Dashboard:
    """FP32 on the left, a chosen INT8 variant on the right, the same clip and the same frames."""

    def __init__(self, clip: Path, models: dict[str, Path], *, threads: int = 1, show_fps: int = 15,
                 device: str = "cpu") -> None:  # fmt: skip
        if "FP32" not in models or len(models) < 2:
            raise ValueError("models needs 'FP32' and at least one other variant")
        self.clip, self.models, self.show_fps = Path(clip), models, show_fps
        self.others = [k for k in models if k != "FP32"]
        self.variant: Signal[str] = Signal(self.others[0])
        self.frame = Signal("")
        self.left = Lane(models["FP32"], threads=threads, device=device)
        self.right = Lane(models[self.variant.value], threads=threads, device=device)
        self.left.title.set("FP32")
        self.right.title.set(self.variant.value)
        self.task: asyncio.Task[None] | None = None

    def choose(self, name: str) -> None:
        """Swap the right-hand model (the Segmented control calls this through the signal)."""
        if name in self.models and name != self.right.title.value:
            self.right.set_model(self.models[name])
            self.right.title.set(name)

    async def step(self, frame: Any) -> None:
        """Publish one frame and start inference on every idle lane."""
        self.choose(self.variant.value)
        self.frame.set(jpeg_uri(frame))
        for lane in (self.left, self.right):
            if not lane.busy:
                asyncio.create_task(lane.run_once(frame.copy()))  # noqa: RUF006

    async def play(self) -> None:
        cap = cv2.VideoCapture(str(self.clip))
        period, tick = 1 / self.show_fps, 0
        while True:
            started = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            await self.step(frame)
            tick += 1
            if tick % self.show_fps == 0:  # once a second, a point on each fps chart
                for lane in (self.left, self.right):
                    lane.fps_chart.push(time.time() % 3600, int(lane.fps.value))
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    def start(self) -> None:
        if self.task is None:
            self.task = asyncio.get_running_loop().create_task(self.play())

    def lane_card(self, lane: Lane) -> Card:
        return Card(
            [
                Text(lane.title, variant="title"),
                ImageOverlay(self.frame, boxes=lane.boxes, label_mode="always"),
                Row(
                    [
                        Stat(lane.fps, label="inference fps"),
                        Stat(lane.p50, label="p50 ms"),
                        Stat(lane.p95, label="p95 ms"),
                        Stat(lane.objects, label="objects"),
                    ]
                ),  # fmt: skip
                Text("Latency of each frame, last 60 (dark part is over the 33 ms budget)"),
                BarStrip(
                    lane.bars,
                    height=64,
                    tones=["ok", "warn"],
                    names=["within 33 ms", "over budget"],
                    label="Per-frame latency",
                ),  # fmt: skip
                lane.fps_chart,
            ],
            variant="outlined",
        )

    def page(self) -> Column:
        return Column(
            [
                Text("YOLOX-Tiny on a traffic clip it was never trained on", variant="title"),
                Text(
                    "Both models see the same frames. Boxes update when a model finishes, so a "
                    "slower model's boxes trail the video."
                ),
                Segmented(self.variant, options=self.others, label="Compare FP32 with"),
                Row([self.lane_card(self.left), self.lane_card(self.right)]),
            ]
        )

    def session(self) -> Session:
        return Session(self.page())


def launch_dashboard(dashboard: Dashboard, **launch_kwargs: Any) -> Any:
    """Serve the dashboard (inline in a notebook) and start the video loop on the server loop."""
    import contextlib

    import indah
    from starlette.applications import Starlette
    from starlette.routing import Mount

    ui = indah.create_app(session_factory=lambda ctx: dashboard.session(), title="rtod", spa=True)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> Any:
        dashboard.start()
        try:
            yield
        finally:
            if dashboard.task:
                dashboard.task.cancel()

    app = Starlette(routes=[Mount("/ui", app=ui)], lifespan=lifespan)
    return indah.launch(app, **launch_kwargs)
