"""Measure the demo where it runs: raw inference rate over time, the live player, memory.

    python measure_on_box.py raw|player [seconds]
    fly ssh console -C "python measure_on_box.py player 90"      # on a Fly machine

`raw` times inference alone; `player` runs the demo's own live player (display, inference, UI
signals). Both print fps and latency in 5-second windows and the peak memory. Run `player` first on
a fresh machine: a shared CPU starts fast and throttles once its burst credit is spent, so each run
spends the next one's.
"""

import asyncio
import importlib.util
import resource
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def load():
    spec = importlib.util.spec_from_file_location("yolox_traffic", HERE / "yolox_traffic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def raw_loop(ex, seconds: float) -> None:
    model, clip = ex.load_model(), ex.Clip()
    assert model is not None, "model did not load"
    print("raw inference (decode + detect, one thread):", flush=True)
    t_end, window_start, i, lat = time.monotonic() + seconds, time.monotonic(), 0, []
    while time.monotonic() < t_end:
        t0 = time.perf_counter()
        model.detect(clip.array_bgr(i % len(clip)))
        lat.append((time.perf_counter() - t0) * 1e3)
        i += 1
        if time.monotonic() - window_start >= 5:
            print(f"  {len(lat) / 5:5.1f} fps  p50 {statistics.median(lat):5.1f} ms  p95 "
                  f"{sorted(lat)[int(len(lat) * .95)]:5.1f} ms", flush=True)  # fmt: skip
            window_start, lat = time.monotonic(), []


async def player_loop(ex, seconds: float) -> None:
    class Ctx:
        def __init__(self):
            self.tasks, self.closers = [], []

        def on_close(self, fn):
            self.closers.append(fn)

        def spawn(self, coro):
            self.tasks.append(asyncio.get_running_loop().create_task(coro))

    print("the demo's live player (display + inference + UI signals):", flush=True)
    ctx = Ctx()
    ex.build_session(ctx)
    player = ctx.tasks[1].get_coro().cr_frame.f_locals["self"]
    ex.LIVE_MAX_S = seconds + 10  # measure without the demo's own 30 s cap
    await player.go_live()
    for _ in range(int(seconds // 5)):
        await asyncio.sleep(5)
        mode = "live" if player.live else "replay"
        print(f"  mode={mode}  fps {player.fps.value}  p50 {player.p50.value} ms"
              f"  p95 {player.p95.value} ms  badge: {player.badge.value}", flush=True)  # fmt: skip
    for t in ctx.tasks:
        t.cancel()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "raw"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    ex = load()
    if mode == "raw":
        raw_loop(ex, seconds)
    else:
        asyncio.run(player_loop(ex, seconds))
    print(
        f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True
    )
