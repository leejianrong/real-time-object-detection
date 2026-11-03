import numpy as np

from rtod.detect import INPUT_SIZE, letterbox, nms, postprocess


def test_letterbox_shape_and_padding():
    img = np.full((200, 400, 3), 7, dtype=np.uint8)
    blob, ratio = letterbox(img)
    assert blob.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert ratio == INPUT_SIZE / 400
    assert blob[0, 0, -1, 0] == 114  # padded bottom row
    assert blob[0, 0, 0, 0] == 7


def row(cx, cy, w, h, obj, cls, n=80):
    r = np.zeros(5 + n, dtype=np.float32)
    r[:5] = cx, cy, w, h, obj
    r[5 + cls] = 1.0
    return r


def test_postprocess_round_trips_coordinates_to_original_pixels():
    # a 100x100 box centred at (208,208) in a 416 letterbox of an 832x832 frame (ratio 0.5)
    raw = np.stack([row(208, 208, 100, 100, 0.9, 3)])
    d = postprocess(raw, 0.5, (832, 832, 3))
    np.testing.assert_allclose(d.boxes[0], [316, 316, 516, 516])
    assert d.class_ids[0] == 3 and d.scores[0] > 0.89


def test_nms_drops_overlap_same_class_keeps_other_class():
    raw = np.stack(
        [
            row(100, 100, 50, 50, 0.9, 1),
            row(102, 100, 50, 50, 0.8, 1),
            row(102, 100, 50, 50, 0.7, 2),
        ]
    )
    d = postprocess(raw, 1.0, (416, 416, 3))
    assert sorted(d.class_ids.tolist()) == [1, 2]


def test_nms_keeps_disjoint_boxes():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    assert nms(boxes, np.array([0.5, 0.9], dtype=np.float32), 0.45) == [1, 0]


def test_low_confidence_filtered():
    assert len(postprocess(np.stack([row(50, 50, 20, 20, 0.1, 0)]), 1.0, (416, 416, 3))) == 0


def test_coco_names_are_the_80_classes_in_model_order():
    from rtod.coco_names import COCO_NAMES

    assert len(COCO_NAMES) == 80 and len(set(COCO_NAMES)) == 80
    assert [COCO_NAMES[i] for i in (0, 2, 5, 7, 9, 79)] == [
        "person", "car", "bus", "truck", "traffic light", "toothbrush"
    ]  # fmt: skip
