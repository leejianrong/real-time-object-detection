from rtod import coco


def test_lists_are_pinned_and_sized():
    val, train = coco.read_ids("val200"), coco.read_ids("train10k")
    assert len(val) == 200 and len(set(val)) == 200
    assert len(train) == 10_000 and len(set(train)) == 10_000
    assert train[:5000] == coco.read_ids("train10k")[:5000]  # first 5k is the calibration set
