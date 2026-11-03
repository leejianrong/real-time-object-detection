"""FP32 ONNX to FP16 (needs the `fp16` extra). Inputs and outputs stay float32."""

from __future__ import annotations

from pathlib import Path


def to_fp16(src: Path, dst: Path) -> Path:
    """Convert weights and ops to FP16 for a GPU; `keep_io_types` leaves the I/O contract unchanged.

    Ops without an FP16 kernel keep float32 with casts around them, so the model still loads.
    """
    import onnx
    from onnxconverter_common import float16

    from rtod.quant import decode_tail

    # The decode tail (sigmoids, grid offsets, concat) stays float32: it mixes pixel-scale boxes with
    # 0-1 scores, and converting it left Add nodes with mixed float16/float32 inputs that ORT rejects.
    model = float16.convert_float_to_float16(
        onnx.load(str(src)), keep_io_types=True, node_block_list=decode_tail(src)
    )
    del model.graph.value_info[:]  # stale intermediate types; let ORT infer them
    onnx.save(model, str(dst))
    return dst
