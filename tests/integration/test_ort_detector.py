import urllib.request

import cv2
import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]  # downloads weights + a COCO image
pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from rtod.detect import Detector, OrtDetector  # noqa: E402
from rtod.models import export_onnx, load_official_weights  # noqa: E402

COCO_CATS = "http://images.cocodataset.org/val2017/000000039769.jpg"
COCO_PERSON, COCO_CAT, COCO_REMOTE = 0, 15, 65


def test_two_cats_and_remotes_found_through_protocol(tmp_path):
    onnx = export_onnx(load_official_weights(), tmp_path / "fp32.onnx")
    urllib.request.urlretrieve(COCO_CATS, tmp_path / "cats.jpg")
    img = cv2.imread(str(tmp_path / "cats.jpg"))
    det: Detector = OrtDetector(onnx)
    d = det.detect(img)
    assert (d.class_ids == COCO_CAT).sum() == 2
    assert (d.class_ids == COCO_REMOTE).sum() >= 1
    assert np.all(d.boxes[:, [0, 2]] <= img.shape[1]) and np.all(d.boxes[:, [1, 3]] <= img.shape[0])
