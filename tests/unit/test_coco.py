from pathlib import Path

import pytest

from rtod import coco


def test_lists_are_pinned_and_sized():
    val, train = coco.read_ids("val200"), coco.read_ids("train10k")
    assert len(val) == 200 and len(set(val)) == 200
    assert len(train) == 10_000 and len(set(train)) == 10_000
    assert train[:5000] == coco.read_ids("train10k")[:5000]  # first 5k is the calibration set


def test_lists_ship_inside_the_package():
    # regression: they once sat at the repo root, so a pip-installed copy (the pod) lost them
    assert coco.LISTS.is_relative_to(Path(coco.__file__).parent)


def test_per_class_ap_averages_valid_cells_and_flags_absent_classes():
    import numpy as np

    from rtod.coco import per_class_ap

    prec = np.full((2, 3, 2, 4, 3), -1.0)  # T=2 IoUs, R=3 recalls, K=2 classes, A=4 areas, M=3
    prec[:, :, 0, 0, 2] = [[1.0, 0.5, -1.0], [0.5, 0.0, -1.0]]  # valid cells: 1, .5, .5, 0
    out = per_class_ap(prec, ["car", "ghost"])
    assert out["car"] == pytest.approx(0.5)
    assert out["ghost"] != out["ghost"]  # nan: no ground truth for this class
