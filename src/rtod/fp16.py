"""FP32 ONNX to FP16 (needs the `fp16` extra). Inputs and outputs stay float32."""

from __future__ import annotations

from pathlib import Path


def to_fp16(src: Path, dst: Path) -> Path:
    """Convert weights and ops to FP16 for a GPU; `keep_io_types` leaves the I/O contract unchanged.

    Ops without an FP16 kernel keep float32 with casts around them, so the model still loads.
    """
    import onnx
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(model, str(dst))
    return dst
