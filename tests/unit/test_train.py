import numpy as np

from rtod.train import MAX_LABELS, prepare_sample


def img(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_targets_are_center_wh_in_letterboxed_pixels():
    # 832x416 image -> ratio 0.5. box x=100,y=50,w=200,h=100 -> centre (100+100, 50+50)*0.5
    boxes = np.array([[7, 100, 50, 200, 100]], dtype=np.float32)
    blob, t = prepare_sample(img(832, 416), boxes, flip=False)
    assert blob.shape == (3, 416, 416) and t.shape == (MAX_LABELS, 5)
    np.testing.assert_allclose(t[0], [7, 100, 50, 100, 50])
    assert t[1:].sum() == 0  # padding rows are all zero, which the loss treats as "no object"


def test_flip_mirrors_box_horizontally():
    boxes = np.array([[0, 0, 0, 100, 100]], dtype=np.float32)  # top-left in a 400x400 image
    _, t = prepare_sample(img(400, 400), boxes, flip=True)
    cx = (400 - 50) * (416 / 400)
    np.testing.assert_allclose(t[0, 1], cx, rtol=1e-5)
