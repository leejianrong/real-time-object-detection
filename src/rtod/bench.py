"""Latency benchmark and the results JSON (schema v1). Numbers always carry their hardware."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rtod.detect import F32, INPUT_SIZE, Image, OrtDetector, letterbox, postprocess

SCHEMA_VERSION = 1
MIN_ITERS = 200


@dataclass
class Result:
    variant: str
    runtime: str
    hardware: dict[str, Any]
    threads: int
    input_size: int
    warmup: int
    iters: int
    quick: bool
    model_ms_p50: float
    model_ms_p95: float
    e2e_ms_p50: float
    e2e_ms_p95: float
    fps: float
    size_bytes: int
    git_sha: str
    map50_95: float | None = None
    map50: float | None = None
    map50_95_se: float | None = None  # delete-d jackknife standard error
    eval_subset: str | None = None  # val, val_dev, val_test, val200
    peak_rss_mb: float | None = None  # memory the model adds above the post-import baseline
    cpu_ms_per_inference: float | None = None  # process CPU time, summed over threads
    cores_used: float | None = None  # CPU time / wall time while running
    schema_version: int = field(default=SCHEMA_VERSION)


def load_result(path: Path) -> Result:
    data = json.loads(path.read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {data.get('schema_version')!r}")
    return Result(**data)


def hardware_fingerprint() -> dict[str, Any]:
    import onnxruntime

    model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    import os

    return {
        "cpu": model,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": onnxruntime.__version__,
    }


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _percentiles(ms: list[float]) -> tuple[float, float]:
    return float(np.percentile(ms, 50)), float(np.percentile(ms, 95))


def bench(
    model: Path,
    variant: str,
    *,
    threads: int = 1,
    warmup: int = 20,
    iters: int = MIN_ITERS,
    quick: bool = False,
    frame: Image | None = None,
) -> Result:
    """Time model-only and end-to-end (letterbox + model + decode/NMS) latency on CPU.

    `frame` defaults to seeded noise, so decode/NMS cost is lower than on a real scene; pass a
    real frame when that matters. Refuses fewer than 200 timed iterations unless `quick`.
    """
    if iters < MIN_ITERS and not quick:
        raise ValueError(f"need at least {MIN_ITERS} timed iterations (got {iters}); use quick")
    if frame is None:
        frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    det = OrtDetector(model, threads=threads, conf=0.25)
    blob: F32 = letterbox(frame)[0]
    for _ in range(warmup):
        det.detect(frame)
    model_ms, e2e_ms = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        blob, ratio = letterbox(frame)
        t1 = time.perf_counter()
        raw = det.run_raw(blob)
        t2 = time.perf_counter()
        postprocess(raw, ratio, frame.shape)
        t3 = time.perf_counter()
        model_ms.append((t2 - t1) * 1e3)
        e2e_ms.append((t3 - t0) * 1e3)
    m50, m95 = _percentiles(model_ms)
    e50, e95 = _percentiles(e2e_ms)
    return Result(
        variant=variant,
        runtime="ort-cpu",
        hardware=hardware_fingerprint(),
        threads=threads,
        input_size=INPUT_SIZE,
        warmup=warmup,
        iters=iters,
        quick=quick,
        model_ms_p50=m50,
        model_ms_p95=m95,
        e2e_ms_p50=e50,
        e2e_ms_p95=e95,
        fps=1e3 / e50,
        size_bytes=model.stat().st_size,
        git_sha=_git_sha(),
    )


def write_result(res: Result, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{res.variant.replace('/', '_')}__{res.runtime}__t{res.threads}.json"
    path = out_dir / name
    path.write_text(json.dumps(asdict(res), indent=2) + "\n")
    return path
