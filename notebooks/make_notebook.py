"""Build notebooks/rtod_showcase.ipynb from the cells below (one source of truth, diffable).

    python notebooks/make_notebook.py

The notebook only loads published artifacts and the repo's results JSON; nothing heavy runs in
it unless LIVE_TRAIN is switched on (see ADR-0004: heavy jobs run on RunPod).
"""

from __future__ import annotations

import json
from pathlib import Path

REF = "main"  # pinned to a release tag by the release step (REA-30)
OUT = Path(__file__).with_name("rtod_showcase.ipynb")

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Making YOLOX-Tiny small and fast: quantization, QAT and what we measured

[YOLOX-Tiny](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) detects 80 kinds of
objects at 416 px. This notebook shrinks it with **INT8 quantization** (post-training and
quantization-aware training), measures what that costs in accuracy and buys in speed, and runs the
variants on a real traffic clip.

Everything heavy (training on a GPU) was done ahead of time on RunPod. This notebook downloads
the published models and the benchmark results, so it runs on a **free CPU runtime in a few
minutes**. All code lives in [real-time-object-detection](https://github.com/leejianrong/real-time-object-detection).

The honest headline: quantization gives a 2.5x CPU speed-up and a 3.6x smaller file for under one
point of mAP, but **well-calibrated post-training quantization is as accurate as quantization-aware
training here**, and that is the more useful thing to know.
""")

md("## 1. Set up")
code(f"""
import importlib.util

if importlib.util.find_spec("rtod") is None:  # Colab: install from GitHub; locally it is already there
    %pip install -q "rtod[onnx] @ git+https://github.com/leejianrong/real-time-object-detection@{REF}" matplotlib
""")
code(f"""
import json, urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from rtod.hub import fetch_model

REF = "{REF}"
RAW = f"https://raw.githubusercontent.com/leejianrong/real-time-object-detection/{{REF}}"
LIVE_TRAIN = False  # True re-runs QAT here (needs a GPU runtime, about 5 minutes, see section 4)

MODELS = {{name: fetch_model(name) for name in ("fp32", "ptq-int8", "qat-int8")}}
{{k: round(v.stat().st_size / 1e6, 1) for k, v in MODELS.items()}}  # size in MB
""")

md("""
## 2. The model and one frame

`OrtDetector` runs an ONNX model with ONNX Runtime: letterbox the image to 416x416, run the network
(box decoding is inside the graph), then score-filter and non-maximum suppression in plain NumPy.
The clip is 40 seconds of road traffic in Ljubljana, [CC BY 3.0 by Sounds of Changes](https://commons.wikimedia.org/wiki/File:Road_traffic_in_Ljubljana.webm).
""")
code("""
from rtod.detect import OrtDetector

CLIP = Path("traffic.mp4")
if not CLIP.exists():
    urllib.request.urlretrieve(f"{RAW}/assets/demo/traffic_ljubljana.mp4", CLIP)

def read_frame(i):
    cap = cv2.VideoCapture(str(CLIP)); cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read(); cap.release()
    return frame

from rtod.coco_names import COCO_NAMES  # noqa: E402

def draw(frame, det):
    out = frame.copy()
    for (x1, y1, x2, y2), s, c in zip(det.boxes, det.scores, det.class_ids):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (60, 220, 90), 2)
        cv2.putText(out, f"{COCO_NAMES[int(c)]} {s:.2f}", (int(x1), max(14, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 90), 2)
    return out

frame = read_frame(350)
fp32 = OrtDetector(MODELS["fp32"], threads=1, conf=0.3)
plt.figure(figsize=(10, 5.6)); plt.axis("off")
plt.imshow(cv2.cvtColor(draw(frame, fp32.detect(frame)), cv2.COLOR_BGR2RGB));
""")

md("""
## 3. Post-training quantization (PTQ)

PTQ converts the trained FP32 weights and activations to 8-bit integers without any training.
It needs **calibration**: run some images through the network to learn each layer's value range.
Our recipe (`rtod quantize`, ONNX Runtime static quantization with per-channel weights) has two
details that mattered:

- **The decode tail stays in float.** The network's output mixes pixel-scale box coordinates with
  0-to-1 confidences in one tensor. One INT8 scale for both rounded every confidence to zero and gave
  an mAP of 0. Keeping the last stage in float fixes it.
- **Fewer calibration images were better.** Min/max calibration only ever widens a range as images
  are added, so accuracy *fell* from 32 images (about 32 mAP on a dev set) to 5,000 images (30.7).

Re-run it yourself (about a minute on CPU, downloads 32 COCO images):
`rtod quantize yolox_tiny_fp32.onnx out.onnx --calib 32`.
""")

md("""
## 4. Quantization-aware training (QAT)

QAT fine-tunes the network *with* quantization noise in the loop, so the weights can adapt to it.
`rtod.qat` is a small hand-written implementation (PyTorch's own tool cannot trace YOLOX's head):
BatchNorm is folded into each convolution, then every convolution fake-quantizes its input, its
per-channel INT8 weights and its output. The trained graph is exported as a QDQ ONNX model.

What we found, measured on held-out COCO images (section 5):

- QAT works: quantizing the trained network costs about 0.6 mAP, versus about 2 for naive PTQ.
- But **tuned PTQ is just as accurate** (32.75 vs 32.81, far inside the noise), and its model is
  smaller (5.6 MB vs 20.5 MB) and faster, because the ONNX Runtime PTQ graph quantizes more of the
  network than the trained QDQ graph does.
- Folding BatchNorm means the weights carry its scale, so the fine-tuning learning rate must be
  tiny: 5e-5 lost 2.4 mAP before any quantization, 1e-5 lost 0.3.
""")
code(
    """
if LIVE_TRAIN:
    # needs the torch extra and a GPU runtime; the published run took about 5 minutes on an RTX 4000 Ada
    %pip install -q "rtod[torch,coco] @ git+https://github.com/leejianrong/real-time-object-detection@"""
    + REF
    + """"
    !rtod qat --lr 2e-6 --steps 1250 --device cuda --no-ptq --out qat_out
    !rtod export-qdq qat_out/qat_model.pt qat_pathA.onnx
else:
    print("LIVE_TRAIN is off: using the published QAT model", MODELS["qat-int8"].name)
"""
)

md("""
## 5. Results on held-out images

Accuracy is COCO mAP50-95 on `val_test`, half of val2017 (2,476 images) that was never used to choose
anything. `±` is a standard error from random 80% subsets, so differences under about one point are
noise. Latency is one CPU thread of an Intel i7-13620H in ONNX Runtime, all models measured in one
session.
""")
code("""
FILES = {
    "FP32": "yolox-tiny_fp32",
    "PTQ INT8 (32 calibration images)": "yolox-tiny_ptq-int8-calib32",
    "PTQ INT8 (5000 calibration images)": "yolox-tiny_ptq-int8-calib5000",
    "QAT INT8 (trained graph)": "yolox-tiny_qat-int8-pathA",
    "QAT weights through the PTQ pipeline": "yolox-tiny_qat-int8-pathB",
}
def result(stem):
    with urllib.request.urlopen(f"{RAW}/results/final/{stem}__ort-cpu__t1.json") as r:
        return json.load(r)
RESULTS = {name: result(stem) for name, stem in FILES.items()}

print(f"{'model':38s} {'mAP50-95':>14s} {'latency':>9s} {'size':>8s}")
for name, r in RESULTS.items():
    print(f"{name:38s} {r['map50_95']*100:7.2f} ± {r['map50_95_se']*100:.2f} "
          f"{r['model_ms_p50']:7.1f} ms {r['size_bytes']/1e6:6.1f} MB")
""")
code("""
fig, ax = plt.subplots(figsize=(7.5, 4.5))
LABEL = {  # name: (short label, dx, dy, alignment); placed to keep nearby points readable
    "FP32": ("FP32", -10, 8, "right"),
    "PTQ INT8 (32 calibration images)": ("PTQ, 32 images", -10, 6, "right"),
    "QAT INT8 (trained graph)": ("QAT (trained graph)", 10, 8, "left"),
    "PTQ INT8 (5000 calibration images)": ("PTQ, 5000 images", -10, -14, "right"),
    "QAT weights through the PTQ pipeline": ("QAT weights via PTQ", 10, 4, "left"),
}
for name, r in RESULTS.items():
    ax.errorbar(r["model_ms_p50"], r["map50_95"] * 100, yerr=r["map50_95_se"] * 100, fmt="o", capsize=3)
    label, dx, dy, ha = LABEL[name]
    ax.annotate(label, (r["model_ms_p50"], r["map50_95"] * 100), textcoords="offset points",
                xytext=(dx, dy), ha=ha, fontsize=8)
ax.set_xlabel("CPU latency per image, ms (1 thread, i7-13620H; lower is faster)")
ax.set_ylabel("mAP50-95 on val_test")
ax.set_xlim(0, 150); ax.grid(alpha=0.3); ax.set_title("Accuracy against speed (up and left is better)");
""")

md("""
## 6. Speed on your runtime

The table above was measured on one laptop; numbers from different machines are not comparable.
Here is the same benchmark on whatever this notebook is running on (single thread, 100 timed runs each).
""")
code("""
from rtod.bench import bench

LIVE = {}
for name, key in (("FP32", "fp32"), ("PTQ INT8", "ptq-int8"), ("QAT INT8", "qat-int8")):
    r = bench(MODELS[key], f"nb/{key}", threads=1, warmup=10, iters=100, quick=True)
    LIVE[name] = r
    print(f"{name:10s} model p50 {r.model_ms_p50:6.1f} ms   p95 {r.model_ms_p95:6.1f} ms   "
          f"end-to-end {r.fps:5.1f} fps")
print("hardware:", r.hardware["cpu"])
""")

md("""
## 7. FP32 against INT8 on the traffic clip

The same frames through both models. INT8 should find nearly the same objects.
""")
code("""
int8 = OrtDetector(MODELS["ptq-int8"], threads=1, conf=0.3)
fig, axes = plt.subplots(2, 2, figsize=(12, 6.8))
for col, idx in enumerate((120, 600)):
    f = read_frame(idx)
    for row, (title, det) in enumerate((("FP32", fp32), ("INT8 PTQ", int8))):
        d = det.detect(f)
        axes[row, col].imshow(cv2.cvtColor(draw(f, d), cv2.COLOR_BGR2RGB)); axes[row, col].axis("off")
        axes[row, col].set_title(f"{title}: {len(d.boxes)} detections")
plt.tight_layout();
""")

md("""
## What to take away

1. **Calibration moved accuracy more than training did.** Tune the number of calibration images before reaching for QAT.
2. **INT8 is a real win on CPU:** about 2.5x faster and 3.6x smaller for under one point of mAP.
3. **Measure everything with a standard error and a named machine.** Our first PTQ baseline was
   handicapped by over-calibration, which made QAT look better than it is. The earlier numbers are kept in the repo's
   `results/history/` for the record.

Code, results and the full write-up: [github.com/leejianrong/real-time-object-detection](https://github.com/leejianrong/real-time-object-detection)
([RESULTS.md](https://github.com/leejianrong/real-time-object-detection/blob/main/docs/RESULTS.md)).
Weights: [jianlee/yolox-tiny-quantized](https://huggingface.co/jianlee/yolox-tiny-quantized) (Apache-2.0).
""")


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        cell = {
            "cell_type": kind,
            "id": f"rtod{len(cells):03d}",
            "metadata": {},
            "source": src.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(OUT)
