"""QAT fine-tuning of YOLOX-Tiny on a COCO subset (ADR-0003, ADR-0006).

Short recipe: official weights, BN folded, fake-quant on every conv, SGD with cosine decay, random
horizontal flip only (no mosaic: this is recovery, not training from scratch). The deployable model
is the folded float weights run through the PTQ pipeline (docs/spikes/qat-export.md, path B).
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rtod import coco
from rtod.detect import F32, Image, letterbox
from rtod.metrics import MetricsWriter
from rtod.qat import calibrate, prepare_qat, set_fake_quant

MAX_LABELS = 120


def load_boxes(ids: Sequence[int], split: str) -> dict[int, F32]:
    """image id -> (n, 5) array of [class_index, x, y, w, h] in image pixels; crowds skipped."""
    ann = json.loads(coco.annotation_file(split).read_text())
    cat_index = {c: i for i, c in enumerate(sorted(c["id"] for c in ann["categories"]))}
    wanted = set(ids)
    out: dict[int, list[list[float]]] = {i: [] for i in ids}
    for a in ann["annotations"]:
        if a["image_id"] in wanted and not a["iscrowd"] and a["bbox"][2] > 1 and a["bbox"][3] > 1:
            out[a["image_id"]].append([cat_index[a["category_id"]], *a["bbox"]])
    return {i: np.asarray(v, dtype=np.float32).reshape(-1, 5) for i, v in out.items()}


def prepare_sample(img: Image, boxes: F32, flip: bool) -> tuple[F32, F32]:
    """Letterbox + YOLOX targets: (MAX_LABELS, 5) rows of [cls, cx, cy, w, h] in input pixels."""
    boxes = boxes.copy()
    if flip:
        img = np.ascontiguousarray(img[:, ::-1])
        boxes[:, 1] = img.shape[1] - boxes[:, 1] - boxes[:, 3]
    blob, ratio = letterbox(img)
    t = np.zeros((MAX_LABELS, 5), dtype=np.float32)
    n = min(len(boxes), MAX_LABELS)
    t[:n, 0] = boxes[:n, 0]
    t[:n, 1:3] = (boxes[:n, 1:3] + boxes[:n, 3:5] / 2) * ratio
    t[:n, 3:5] = boxes[:n, 3:5] * ratio
    return blob[0], t


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA). Does not force deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Ema:
    """Exponential moving average of the parameters, started from the pretrained weights.

    Fixed decay, no warm-up ramp: YOLOX's `d * (1 - exp(-updates / 2000))` ramp is for training from
    scratch, and over a 1250-step fine-tune it only reaches ~0.46, which makes the average almost
    identical to the raw weights (the first sweep's `float_ema` config accidentally tested that).
    Buffers (fake-quant ranges) are copied from the live model, not averaged.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.model, self.decay = copy.deepcopy(model).eval(), decay

    @torch.no_grad()
    def update(self, live: torch.nn.Module) -> None:
        for e, m in zip(self.model.parameters(), live.parameters(), strict=True):
            e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
        for e, m in zip(self.model.buffers(), live.buffers(), strict=True):
            e.copy_(m)


class CocoSubset(Dataset[tuple[F32, F32]]):
    def __init__(self, ids: Sequence[int], split: str, train: bool = True) -> None:
        self.ids, self.train = list(ids), train
        self.paths = coco.fetch_images(self.ids, split)
        self.boxes = load_boxes(self.ids, split)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> tuple[F32, F32]:
        img = cv2.imread(str(self.paths[i]))
        flip = self.train and bool(torch.rand(()) < 0.5)
        return prepare_sample(np.asarray(img, dtype=np.uint8), self.boxes[self.ids[i]], flip)


def qat_finetune(
    ids: Sequence[int],
    split: str = "train",
    *,
    steps: int,
    batch: int = 16,
    lr: float = 2e-4,
    clip: float = 10.0,
    calib_batches: int = 8,
    device: str = "cpu",
    workers: int = 4,
    metrics_dir: Path | None = None,
    seed: int = 0,
    ema: bool = False,
    ema_decay: float = 0.999,
    fake_quant: bool = True,
) -> torch.nn.Module:
    """Fine-tune and return the model (the EMA weights if `ema`).

    `fake_quant=False` runs the identical recipe as a plain float fine-tune: the control that
    separates "the recipe hurts the model" from "quantization noise hurts the model".
    """
    from rtod.models import load_official_weights

    seed_everything(seed)
    model = prepare_qat(load_official_weights())
    ds = CocoSubset(ids, split)
    loader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model.head.decode_in_inference = False
    model.to(device)
    calibrate(model, (x.to(device) for x, _ in _take(loader, calib_batches)))
    for m in (
        model.modules()
    ):  # weights keep changing, so their ranges keep tracking; activations stay frozen
        if hasattr(m, "w_fq"):
            m.w_fq.enable_observer()

    if not fake_quant:
        set_fake_quant(model, False)
    averaged = Ema(model, ema_decay) if ema else None
    model.train()
    writer = MetricsWriter(metrics_dir) if metrics_dir else None
    t0 = time.monotonic()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    step = 0
    while step < steps:
        for x, y in loader:
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
            out: dict[str, Any] = model(x.to(device), y.to(device))
            opt.zero_grad()
            out["total_loss"].backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if averaged:
                averaged.update(model)
            step += 1
            loss = float(out["total_loss"])
            if not math.isfinite(loss):  # fail in seconds, not after the whole run has burned money
                raise FloatingPointError(f"non-finite loss {loss} at step {step}")
            rate = step / (time.monotonic() - t0)
            if writer:
                writer.log(
                    step,
                    loss=loss,
                    iou=float(out["iou_loss"]),
                    conf=float(out["conf_loss"]),
                    cls=float(out["cls_loss"]),
                    lr=opt.param_groups[0]["lr"],
                    gnorm=float(gnorm),
                    it_s=rate,
                )
            if step in (1, steps) or step % 10 == 0:
                mem = torch.cuda.max_memory_allocated() // 2**20 if device != "cpu" else 0
                print(
                    f"TRAIN step={step}/{steps} loss={loss:.3f} iou={float(out['iou_loss']):.3f} "
                    f"conf={float(out['conf_loss']):.3f} cls={float(out['cls_loss']):.3f} "
                    f"lr={opt.param_groups[0]['lr']:.2e} gnorm={float(gnorm):.2f} it_s={rate:.2f} "
                    f"eta_s={(steps - step) / rate:.0f} gpu_mem_mb={mem}",
                    flush=True,
                )
            if step >= steps:
                break
    if writer:
        writer.close()
    return (averaged.model if averaged else model).eval().cpu()


def _take(loader: DataLoader[tuple[F32, F32]], n: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        out.append(batch)
    return out


def export_folded(model: torch.nn.Module, out: Path) -> Path:
    """Export the fine-tuned, BN-folded float weights (fake-quant off) as ONNX."""
    from rtod.models import export_onnx

    set_fake_quant(model, False)
    return export_onnx(model, out)  # type: ignore[arg-type]
