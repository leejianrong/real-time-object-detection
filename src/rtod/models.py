"""Build, load and export YOLOX-Tiny (needs the `torch` extra)."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Any

from torch.nn import BatchNorm2d

from rtod._yolox.yolo_head import YOLOXHead
from rtod._yolox.yolo_pafpn import YOLOPAFPN
from rtod._yolox.yolox import YOLOX

INPUT_SIZE = 416
WEIGHTS_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
)
WEIGHTS_SHA256 = "9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0"


def build_yolox_tiny(num_classes: int = 80) -> YOLOX:
    """YOLOX-Tiny (depth 0.33, width 0.375) with random weights, in eval mode."""
    model = YOLOX(YOLOPAFPN(0.33, 0.375, act="silu"), YOLOXHead(num_classes, 0.375, act="silu"))
    for m in model.modules():
        if isinstance(m, BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    return model.eval()


def fetch_official_weights(cache_dir: Path = Path(".cache")) -> Path:
    """Download the official COCO checkpoint once and verify its sha256."""
    path = cache_dir / "yolox_tiny.pth"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, path)  # noqa: S310 (fixed https URL)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"{path}: sha256 {digest} != pinned {WEIGHTS_SHA256}")
    return path


def load_official_weights(path: Path | None = None) -> YOLOX:
    import torch

    model = build_yolox_tiny()
    ckpt: dict[str, Any] = torch.load(path or fetch_official_weights(), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model


def export_onnx(model: YOLOX, out: Path, size: int = INPUT_SIZE) -> Path:
    """Export the raw head output, shape (1, N, 85), before decode and NMS (ADR-0002).

    Uses the head's decode_in_inference=True so grid offsets are in the graph; box
    coordinates are in input pixels. NMS stays outside in shared numpy code.
    """
    import torch

    model.head.decode_in_inference = True
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, size, size),),
        str(out),
        input_names=["images"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    return out
