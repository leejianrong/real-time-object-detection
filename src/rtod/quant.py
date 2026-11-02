"""Post-training INT8 quantization with ONNX Runtime static QDQ (ADR-0003)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from rtod.detect import F32, letterbox


class _Reader:
    def __init__(self, paths: Sequence[Path], input_name: str) -> None:
        self._it = iter(paths)
        self._name = input_name
        self._n, self._total, self._t0 = 0, len(paths), time.monotonic()

    def get_next(self) -> dict[str, F32] | None:
        path = next(self._it, None)
        if path is None:
            return None
        self._n += 1
        if self._n % 250 == 0 or self._n == self._total:
            rate = self._n / (time.monotonic() - self._t0)
            print(f"CALIB {self._n}/{self._total} rate={rate:.2f}/s", flush=True)
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"unreadable image {path}")
        return {self._name: letterbox(np.asarray(img, dtype=np.uint8))[0]}


def decode_tail(model: Path) -> list[str]:
    """Names of the nodes between the last Convs and the output (sigmoids, grid decode, concat).

    The output mixes pixel-scale boxes with 0-1 scores in one tensor, so INT8 there rounds every
    score to zero. These nodes stay float; the Convs feeding them are still quantized.
    """
    import onnx

    g = onnx.load(str(model)).graph
    producer = {o: n for n in g.node for o in n.output}
    seen: set[str] = set()
    todo = [o.name for o in g.output]
    while todo:
        n = producer.get(todo.pop())
        if n is None or n.op_type == "Conv" or n.name in seen:
            continue
        seen.add(n.name)
        todo += list(n.input)
    return sorted(seen)


def quantize_ptq(
    fp32: Path,
    out: Path,
    images: Sequence[Path],
    op_types: list[str] | None = None,
    method: str = "minmax",
) -> Path:
    """QDQ static INT8, calibrated on `images` (the first 5k of train10k, ADR-0006).

    `op_types=["Conv"]` quantizes only the convolutions, the same coverage as the trained fake-quant
    graph of `rtod.qat` (path A); the default quantizes everything ORT can (SiLU, Add, Concat, ...).
    `method` is the range calibrator: minmax (absolute extremes, so more images only widen the
    ranges), moving_avg (averaged extremes, outlier-robust, cheap), percentile or entropy (robust,
    but they keep every tensor in memory).
    """
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    quantize_static(
        str(fp32),
        str(out),
        _Reader(images, "images"),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=decode_tail(fp32),
        op_types_to_quantize=op_types,
        calibrate_method={
            "minmax": CalibrationMethod.MinMax,
            "moving_avg": CalibrationMethod.MinMax,
            "percentile": CalibrationMethod.Percentile,
            "entropy": CalibrationMethod.Entropy,
        }[method],
        # moving-average min/max: per-batch extremes are averaged, so one outlier image cannot
        # stretch a range (the estimator the QAT observers use); needs no extra memory
        extra_options={"CalibMovingAverage": True} if method == "moving_avg" else {},
    )
    return out
