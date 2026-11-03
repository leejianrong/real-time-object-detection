"""Download the published model files from the Hugging Face Hub, checking each sha256.

Stdlib only, so a Colab notebook can use it without torch.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

REPO = "jianlee/yolox-tiny-quantized"
BASE = f"https://huggingface.co/{REPO}/resolve/main"

# name -> (file in the repo, sha256). See docs/RESULTS.md for what each one is.
MODELS = {
    "fp32": (
        "yolox_tiny_fp32.onnx",
        "733a6b2b49c9705fb88642f6a6b7538e1c3c965b83d27dbd8cc0b5ac90d78ca3",
    ),
    "ptq-int8": (
        "yolox_tiny_ptq_int8_calib32.onnx",
        "0d0418035dd5e6ddf4cb108b253490f6fade8292a4fb217d3eaf570f30aa8efb",
    ),
    "qat-int8": (
        "yolox_tiny_qat_int8_pathA.onnx",
        "509e070329b407836cf3c0910e0768072efebb21a7e0810a719faeecf6a171af",
    ),
    "qat-checkpoint": (
        "yolox_tiny_qat_lr2e-6.pt",
        "2540d73581bcec2f9848fba689b82d278903ab0992335c9125bf3283ceb172c9",
    ),
}


def fetch_model(name: str, cache_dir: Path = Path(".cache"), base: str = BASE) -> Path:
    """Download `name` once into `cache_dir` and verify it against the pinned sha256."""
    filename, sha = MODELS[name]
    path = cache_dir / filename
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{base}/{filename}", path)  # noqa: S310 (fixed https URL)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        path.unlink()
        raise ValueError(f"{filename}: sha256 {digest} != pinned {sha}")
    return path
