"""COCO subsets and mAP evaluation (ADR-0006). Images are fetched by id and never committed.

Id lists live in `data_lists/`: val200 (CI-sized), train10k (calibration/QAT/prune fine-tune;
the first 5k are the calibration set). `make_lists` regenerates them from the annotation files.
"""

from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rtod.detect import Detector, OrtDetector

LISTS = Path(__file__).resolve().parents[2] / "data_lists"
CACHE = Path(".cache/coco")
ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}2017/{id:012d}.jpg"
EVAL_CONF, EVAL_IOU = 0.001, 0.65  # YOLOX's own eval settings, so numbers match its published mAP


def annotation_file(split: str = "val", cache: Path = CACHE) -> Path:
    path = cache / f"instances_{split}2017.json"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        zpath = cache / "annotations.zip"
        if not zpath.exists():
            urllib.request.urlretrieve(ANN_ZIP, zpath)  # noqa: S310 (fixed http URL)
        with zipfile.ZipFile(zpath) as z:
            path.write_bytes(z.read(f"annotations/instances_{split}2017.json"))
    return path


def read_ids(name: str) -> list[int]:
    return [int(x) for x in (LISTS / f"{name}.txt").read_text().split()]


def make_lists(seed: int = 0) -> None:
    """Regenerate the pinned lists. Only images with at least one annotation are eligible."""
    for name, split, n in (("val200", "val", 200), ("train10k", "train", 10_000)):
        ann = json.loads(annotation_file(split).read_text())
        ids = sorted({a["image_id"] for a in ann["annotations"]})
        picked = random.Random(seed).sample(ids, n)
        LISTS.mkdir(exist_ok=True)
        (LISTS / f"{name}.txt").write_text("\n".join(map(str, picked)) + "\n")


def fetch_images(ids: Sequence[int], split: str = "val", cache: Path = CACHE) -> list[Path]:
    (cache / split).mkdir(parents=True, exist_ok=True)

    def one(i: int) -> Path:
        p = cache / split / f"{i:012d}.jpg"
        if not p.exists():
            urllib.request.urlretrieve(IMG_URL.format(split=split, id=i), p)  # noqa: S310
        return p

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, ids))


def evaluate(model: Path, ids: Sequence[int] | None = None, split: str = "val") -> dict[str, Any]:
    """mAP50-95 and mAP50 of an ONNX model on `ids` (default: all of val2017)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(annotation_file(split)))
    ids = sorted(gt.getImgIds()) if ids is None else list(ids)
    cat_ids = sorted(gt.getCatIds())
    det: Detector = OrtDetector(model, conf=EVAL_CONF, iou=EVAL_IOU)
    results: list[dict[str, Any]] = []
    for i, path in zip(ids, fetch_images(ids, split), strict=True):
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"unreadable image {path}")
        d = det.detect(np.asarray(img, dtype=np.uint8))
        for (x1, y1, x2, y2), s, c in zip(d.boxes, d.scores, d.class_ids, strict=True):
            results.append(
                {
                    "image_id": i,
                    "category_id": cat_ids[c],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                }
            )
    if not results:
        return {"map50_95": 0.0, "map50": 0.0, "images": len(ids)}
    ev = COCOeval(gt, gt.loadRes(results), "bbox")
    ev.params.imgIds = ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"map50_95": float(ev.stats[0]), "map50": float(ev.stats[1]), "images": len(ids)}
