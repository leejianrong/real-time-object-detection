---
license: apache-2.0
library_name: onnx
pipeline_tag: object-detection
tags: [yolox, quantization, int8, qat, onnx, coco]
base_model: Megvii-BaseDetection/YOLOX
datasets: [detection-datasets/coco]
---

# YOLOX-Tiny, quantized (INT8 PTQ and QAT)

Quantized variants of [YOLOX-Tiny](https://github.com/Megvii-BaseDetection/YOLOX) (Megvii, Apache-2.0),
416 px, 80 COCO classes. Code, benchmarks and the full write-up:
<https://github.com/leejianrong/real-time-object-detection>.

The model output is `(1, 3549, 85)` with box decoding inside the graph. NMS runs outside the model
(`rtod.detect.postprocess` in the repo). Input is a letterboxed 416x416 BGR image, float32, 0 to 255, NCHW.

| File | What | Size | mAP50-95 (COCO val_test) | CPU latency, 1 thread |
|---|---|---|---|---|
| `yolox_tiny_fp32.onnx` | Official weights exported to ONNX | 20.3 MB | 33.47 ± 0.49 | 115.6 ms |
| `yolox_tiny_ptq_int8_calib32.onnx` | ORT static INT8, 32 calibration images | 5.6 MB | 32.75 ± 0.45 | 45.4 ms |
| `yolox_tiny_qat_int8_pathA.onnx` | QAT-trained fake-quant graph as QDQ | 20.5 MB | 32.81 ± 0.52 | 61.2 ms |
| `yolox_tiny_qat_lr2e-6.pt` | QAT checkpoint (folded weights + observer ranges) | 21 MB | | |

`val_test` is a held-out half of COCO val2017 (2,476 images); `±` is a jackknife standard error. Latency is
ONNX Runtime on one thread of an Intel i7-13620H, all models measured in one session. Differences under
about 1 mAP point are within noise. QAT did not beat well-calibrated PTQ here; see the repo's
`docs/RESULTS.md`.

## Use

```python
from rtod.hub import fetch_model

path = fetch_model("ptq-int8")  # downloads once, verifies sha256
```

## Training data and licences

Starting weights: official YOLOX-Tiny COCO checkpoint (Apache-2.0). Quantization calibration and QAT
fine-tuning used images from COCO train2017 (annotations CC BY 4.0; images under their Flickr licences).
These files are derivatives of the YOLOX weights and carry the same Apache-2.0 licence.
