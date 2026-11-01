"""Detector protocol, shared pre/post-processing and the ONNX Runtime CPU backend (ADR-0002).

Every runtime returns raw (N, 85) rows [cx, cy, w, h, objectness, 80 class scores] in input
pixels; decode and NMS live here in numpy so results are comparable across runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

INPUT_SIZE = 416
PAD_VALUE = 114

Image = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Detections:
    """Boxes are xyxy in original-image pixels."""

    boxes: F32
    scores: F32
    class_ids: npt.NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.scores)


class Detector(Protocol):
    def detect(self, frame_bgr: Image) -> Detections: ...


def letterbox(img: Image, size: int = INPUT_SIZE) -> tuple[F32, float]:
    """Resize keeping aspect ratio, pad bottom/right with 114. Returns (1,3,size,size) and ratio."""
    ratio = min(size / img.shape[0], size / img.shape[1])
    h, w = round(img.shape[0] * ratio), round(img.shape[1] * ratio)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:h, :w] = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return canvas.transpose(2, 0, 1)[None].astype(np.float32), ratio


def nms(boxes: F32, scores: F32, iou_thr: float) -> list[int]:
    """Greedy NMS on xyxy boxes. Returns kept indices, highest score first."""
    order = scores.argsort()[::-1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        wh = np.clip(
            np.minimum(boxes[i, 2:], boxes[rest, 2:]) - np.maximum(boxes[i, :2], boxes[rest, :2]),
            0,
            None,
        )
        inter = wh[:, 0] * wh[:, 1]
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def postprocess(
    raw: F32, ratio: float, frame_shape: tuple[int, ...], conf: float = 0.25, iou: float = 0.45
) -> Detections:
    """Decode (N,85) rows to class-aware NMS'd detections in original-image pixels."""
    cls = raw[:, 5:].argmax(1)
    scores = raw[:, 4] * raw[np.arange(len(raw)), 5 + cls]
    m = scores >= conf
    raw, cls, scores = raw[m], cls[m], scores[m]
    cxcy, wh = raw[:, :2], raw[:, 2:4]
    boxes = (np.concatenate([cxcy - wh / 2, cxcy + wh / 2], 1) / ratio).astype(np.float32)
    boxes = np.clip(boxes, 0, np.array([frame_shape[1], frame_shape[0]] * 2, dtype=np.float32))
    keep: list[int] = []
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        keep += [int(idx[k]) for k in nms(boxes[idx], scores[idx], iou)]
    keep.sort(key=lambda i: -scores[i])
    return Detections(boxes[keep], scores[keep].astype(np.float32), cls[keep].astype(np.int64))


class OrtDetector:
    """ONNX Runtime CPU backend. `threads` pins intra-op threads for benchmarking."""

    def __init__(
        self, model: Path, threads: int | None = None, conf: float = 0.25, iou: float = 0.45
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self._conf, self._iou = conf, iou

    def run_raw(self, blob: F32) -> F32:
        (out,) = self._sess.run(None, {"images": blob})
        return np.asarray(out[0], dtype=np.float32)

    def detect(self, frame_bgr: Image) -> Detections:
        blob, ratio = letterbox(frame_bgr)
        return postprocess(self.run_raw(blob), ratio, frame_bgr.shape, self._conf, self._iou)
