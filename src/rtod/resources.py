"""Peak memory and CPU use of one model, measured in a fresh process (Linux: reads /proc).

A fresh process per model keeps another model's allocations out of the number. Memory is reported
above the post-import baseline, so it is what loading and running this model adds, not what Python,
NumPy, OpenCV and ONNX Runtime cost.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def status_mb(field: str, text: str) -> float:
    """A `/proc/self/status` field such as VmRSS or VmHWM, in MB (it reports kB)."""
    for line in text.splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024
    raise KeyError(field)


def _rss(field: str) -> float:
    return status_mb(field, Path("/proc/self/status").read_text())


def _child(model: Path, threads: int, iters: int) -> dict[str, float]:
    import numpy as np

    from rtod.detect import OrtDetector, letterbox

    base = _rss("VmRSS")
    det = OrtDetector(model, threads=threads, conf=0.25)
    loaded = _rss("VmRSS")
    blob = letterbox(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8))[0]
    for _ in range(10):
        det.run_raw(blob)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for _ in range(iters):
        det.run_raw(blob)
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
    return {
        "baseline_rss_mb": base,
        "load_rss_mb": loaded - base,
        "peak_rss_mb": _rss("VmHWM") - base,
        "cpu_ms_per_inference": cpu / iters * 1e3,
        "cores_used": cpu / wall,
    }


def measure_resources(model: Path, threads: int = 1, iters: int = 100) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, "-m", "rtod.resources", str(model), str(threads), str(iters)],
        capture_output=True,
        text=True,
        check=True,
    )
    result: dict[str, Any] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


if __name__ == "__main__":
    print(json.dumps(_child(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))))
