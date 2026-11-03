"""COCO subsets and mAP evaluation (ADR-0006). Images are fetched by id and never committed.

Id lists live in `src/rtod/id_lists/`: val200 (CI-sized), train10k (calibration/QAT/prune fine-tune;
the first 5k are the calibration set). `make_lists` regenerates them from the annotation files.
"""

from __future__ import annotations

import random
import sys
import time
import urllib.request
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rtod.detect import Detector, OrtDetector

LISTS = Path(__file__).resolve().parent / "id_lists"
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


def annotated_ids(split: str) -> list[int]:
    """Ids of images with at least one annotation, read by regex so the 450 MB train file is
    never parsed into Python objects."""
    import re

    pat = re.compile(rb'"image_id": ?(\d+)')
    found: set[int] = set()
    tail = b""
    with annotation_file(split).open("rb") as f:
        while chunk := f.read(1 << 26):
            data = tail + chunk
            found.update(int(m) for m in pat.findall(data))
            tail = data[-64:]
    return sorted(found)


def _write(name: str, ids: Sequence[int]) -> None:
    LISTS.mkdir(exist_ok=True)
    (LISTS / f"{name}.txt").write_text("\n".join(map(str, ids)) + "\n")


def make_lists(seed: int = 0) -> None:
    """Create any id list that does not exist yet. Existing lists are pinned and never rewritten.

    - val200: a 200-image quick check.
    - val_dev / val_test: a disjoint halves split of val2017 (tune on dev, report on test once).
    - train10k: calibration / fine-tune subset (the first 5k are the calibration set).
    - train40k: train10k plus 30k more, for the data-size ablation.
    """
    rng = random.Random(seed)
    if not (LISTS / "val200.txt").exists():
        _write("val200", rng.sample(annotated_ids("val"), 200))
    if not (LISTS / "val_dev.txt").exists():
        ids = annotated_ids("val")
        random.Random(seed + 1).shuffle(ids)
        half = len(ids) // 2
        _write("val_dev", sorted(ids[:half]))
        _write("val_test", sorted(ids[half:]))
    if not (LISTS / "train10k.txt").exists():
        _write("train10k", rng.sample(annotated_ids("train"), 10_000))
    if not (LISTS / "train40k.txt").exists():
        have = read_ids("train10k")
        rest = sorted(set(annotated_ids("train")) - set(have))
        _write("train40k", have + random.Random(seed + 2).sample(rest, 30_000))


def fetch_images(ids: Sequence[int], split: str = "val", cache: Path = CACHE) -> list[Path]:
    (cache / split).mkdir(parents=True, exist_ok=True)

    def one(i: int) -> Path:
        p = cache / split / f"{i:012d}.jpg"
        for attempt in range(3):
            if p.exists():
                break
            try:
                urllib.request.urlretrieve(IMG_URL.format(split=split, id=i), p)  # noqa: S310
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        return p

    t0, out = time.monotonic(), []
    with ThreadPoolExecutor(16) as ex:
        for n, p in enumerate(ex.map(one, ids), 1):
            out.append(p)
            if len(ids) >= 500 and (n % 500 == 0 or n == len(ids)):
                rate = n / (time.monotonic() - t0)
                print(f"DATA split={split} fetched={n}/{len(ids)} rate={rate:.1f}/s", flush=True)
    return out


def per_class_ap(precision: Any, names: Sequence[str]) -> dict[str, float]:
    """AP50-95 per class from COCOeval's precision array (T, R, K, A, M), all areas, 100 dets.

    Entries of -1 mark recall levels a class never reaches, as in pycocotools' own summary.
    """
    out: dict[str, float] = {}
    for k, name in enumerate(names):
        p = precision[:, :, k, 0, 2]
        out[name] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return out


def _score(gt: Any, results: list[dict[str, Any]], ids: Sequence[int], quiet: bool) -> Any:
    from contextlib import redirect_stdout
    from io import StringIO

    from pycocotools.cocoeval import COCOeval

    ev = COCOeval(gt, gt.loadRes(results), "bbox")
    ev.params.imgIds = list(ids)
    with redirect_stdout(StringIO()) if quiet else redirect_stdout(sys.stdout):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def evaluate(
    model: Path,
    ids: Sequence[int] | None = None,
    split: str = "val",
    jackknife: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """mAP50-95 and mAP50 of an ONNX model on `ids` (default: all of val2017).

    `jackknife=B` adds `map50_95_se`: the delete-d jackknife standard error from B random 80%
    subsets (pycocotools cannot resample with replacement). Detection runs once.
    """
    from pycocotools.coco import COCO

    gt = COCO(str(annotation_file(split)))
    ids = sorted(gt.getImgIds()) if ids is None else list(ids)
    cat_ids = sorted(gt.getCatIds())
    det: Detector = OrtDetector(model, conf=EVAL_CONF, iou=EVAL_IOU, device=device)
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
    ev = _score(gt, results, ids, quiet=False)
    out: dict[str, Any] = {
        "map50_95": float(ev.stats[0]),
        "map50": float(ev.stats[1]),
        "images": len(ids),
        "per_class_ap": per_class_ap(ev.eval["precision"], [gt.cats[c]["name"] for c in cat_ids]),
    }
    if jackknife:
        rng = random.Random(0)
        keep = round(0.8 * len(ids))
        thetas = [
            float(_score(gt, results, rng.sample(list(ids), keep), quiet=True).stats[0])
            for _ in range(jackknife)
        ]
        mean = sum(thetas) / len(thetas)
        # delete-d jackknife: Var = (n - d) / (d * B) * sum((theta_b - mean)^2), here (n-d)/d = 4
        var = (keep / (len(ids) - keep)) * sum((t - mean) ** 2 for t in thetas) / len(thetas)
        out["map50_95_se"] = var**0.5
    return out
