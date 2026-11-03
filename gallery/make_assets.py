"""Build the demo's bundled assets from the clip and the published model.

    uv run python gallery/make_assets.py [--clip assets/demo/traffic_ljubljana.mp4]

Writes gallery/examples/yolox_traffic/{frames.bin, replay.json}. The recorded detections come from
the example's own `Model.detect` on the very JPEG frames the demo ships, so replay shows what live
would show for the same frame (a test checks that). Boxes are deterministic; latencies are the
median of three runs on this machine and are labelled with it in the file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

import cv2

HERE = Path(__file__).parent
OUT = HERE / "examples/yolox_traffic"
FPS, SECONDS, SIZE, QUALITY = 12, 15, (640, 360), 55


def load_example():
    spec = importlib.util.spec_from_file_location(
        "yolox_traffic", HERE / "examples/yolox_traffic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, default=HERE.parent / "assets/demo/traffic_ljubljana.mp4")
    ap.add_argument("--model", type=Path, required=True, help="the published INT8 ONNX file")
    args = ap.parse_args()
    ex = load_example()

    cap = cv2.VideoCapture(str(args.clip))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    blob, offsets = bytearray(), []
    for k in range(FPS * SECONDS):
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(k / FPS * src_fps))
        ok, frame = cap.read()
        assert ok, f"clip is shorter than {SECONDS} s"
        _, jpg = cv2.imencode(".jpg", cv2.resize(frame, SIZE), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        offsets.append([len(blob), len(jpg)])
        blob += jpg.tobytes()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames.bin").write_bytes(bytes(blob))

    meta = {"fps": FPS, "frames": offsets}
    clip = ex.Clip.__new__(ex.Clip)
    clip.meta, clip.fps = meta, FPS
    clip.frames = [bytes(blob[o : o + n]) for o, n in offsets]
    model = ex.Model(str(args.model))
    detections, latency = [], []
    for i in range(len(clip.frames)):
        bgr = clip.array_bgr(i)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            rows = model.detect(bgr)
            times.append((time.perf_counter() - t0) * 1e3)
        detections.append(rows)
        latency.append(round(statistics.median(times), 1))

    import onnxruntime

    meta.update(
        schema_version=1,
        size=list(SIZE),
        model={
            "repo": ex.MODEL_REPO,
            "file": ex.MODEL_FILE,
            "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        },  # fmt: skip
        recorded_on=f"{cpu_name()}, 1 thread, onnxruntime {onnxruntime.__version__}",
        frames_detections=detections,
        latency_ms=latency,
    )
    (OUT / "replay.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n")
    print(
        f"frames.bin {len(blob) / 1e6:.2f} MB, "
        f"replay.json {(OUT / 'replay.json').stat().st_size / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
