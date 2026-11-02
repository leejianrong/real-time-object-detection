# Results: quantizing YOLOX-Tiny, measured and reported as found

Everything here is on COCO val2017. Choices (recipe, calibration size, which QAT run) were made on the
`val_dev` half (2,476 images); the headline table is the **held-out `val_test` half** (2,476 other
images), scored once after the choices were fixed. `±` is a delete-d jackknife standard error from 10
random 80% subsets of the evaluation images (pycocotools cannot resample with replacement); treat
differences under about 1 point between two models as noise. Latency is one ONNX Runtime CPU thread on
an Intel i7-13620H (WSL2), all models measured back to back in one session; never compare latency
across sessions or machines.

## Headline (held-out `val_test`)

| Model | mAP50-95 | vs FP32 | mAP50 | Latency (model only) | vs FP32 | Size |
|---|---|---|---|---|---|---|
| FP32 | 33.47 ± 0.49 | | 50.82 | 115.6 ms | 1.00x | 20.3 MB |
| PTQ INT8, original recipe (5000 calibration images) | 31.23 ± 0.46 | -2.24 | 48.62 | 45.5 ms | 0.39x | 5.6 MB |
| **PTQ INT8, tuned (32 calibration images)** | **32.75 ± 0.45** | **-0.72** | 50.49 | **45.4 ms** | **0.39x** | **5.6 MB** |
| PTQ INT8, 128 calibration images | 32.43 ± 0.52 | -1.04 | 50.16 | 46.3 ms | 0.40x | 5.6 MB |
| QAT INT8, trained graph (path A) | 32.81 ± 0.52 | -0.65 | 50.64 | 61.2 ms | 0.53x | 20.5 MB |
| QAT weights through the ORT PTQ pipeline (path B) | 31.35 ± 0.50 | -2.11 | 48.97 | 46.1 ms | 0.40x | 5.6 MB |

Raw rows: `results/final/*.json` (schema v1, with hardware fingerprint and git sha).

**What this says**

1. INT8 gives a 2.5x CPU speedup and a 3.6x smaller file for 0.7 to 2.2 points of mAP, depending on how
   it is calibrated.
2. **Calibration moved accuracy more than QAT did.** The same ORT PTQ pipeline goes from -2.24 to -0.72
   points just by calibrating on 32 images instead of 5000.
3. **QAT did not beat tuned PTQ.** Path A scores the same (32.81 vs 32.75, far inside the noise), but its
   graph is slower (0.53x vs 0.39x of FP32) and larger (20.5 MB vs 5.6 MB) in ONNX Runtime.
4. **The QAT-trained weights are not what helped.** Path B (the same weights through the identical
   pipeline as PTQ) scores 31.35, no better than the original PTQ (31.23).

The project's first PTQ baseline (5000-image calibration) made QAT look better than it is. That is
corrected here, and the earlier numbers are kept in `results/history/` for the record.

## What each variant is

- **PTQ (ORT static, QDQ, per-channel weights, min/max activation ranges).** `rtod quantize`. The
  decode tail (everything after the last conv) stays float, because the output mixes pixel-scale boxes
  with 0-1 scores and one INT8 scale rounded every score to zero (a bug we hit and fixed, with a test).
- **QAT (`rtod.qat`, `rtod qat`).** BatchNorm is folded into each conv, then every conv fake-quantizes
  its input, its per-channel int8 weights and its output (torch.ao FX cannot trace YOLOX's head, so
  this is hand-written). Fine-tuned from the official weights: SGD momentum 0.9, cosine schedule,
  batch 16, 1250 steps on a fixed 10k-image train2017 subset, flip-only augmentation, lr 2e-6, gradient
  clip 100, seed 0.
- **Path A (`rtod export-qdq`).** The trained fake-quant network exported as a QDQ graph. It quantizes
  only conv inputs, weights and outputs, so SiLU, Add, Concat and pooling stay float, and its weights
  are stored as float with Q/DQ nodes around them (hence the 20.5 MB file). We did not profile why it
  runs slower than the PTQ graph; the partial coverage and the unquantized biases are the likely causes.
- **Path B.** The QAT-trained float weights exported as ONNX and quantized by the same `rtod quantize`
  as plain PTQ. Same graph, speed and size as PTQ.

## How we got here (`val_dev`, 2,476 images, standard error about 0.6 for every row)

FP32 reference on this half: **32.81 ± 0.63**.

**1. The first QAT runs lost accuracy before any quantization.** A plain float fine-tune, fake-quant
switched off, scored 30.46 at lr 5e-5. BatchNorm is folded before training, so the weights carry the BN
scale, and the same step size moves them much further than in the original network.

| Float fine-tune (no quantization) | mAP50-95 | vs FP32 |
|---|---|---|
| lr 5e-5, 1250 steps | 30.46 | -2.35 |
| lr 5e-5, 250 steps | 29.62 | -3.19 |
| lr 5e-5 + EMA decay 0.999 | 29.27 | -3.54 |
| lr 5e-5 + EMA decay 0.9998 | 27.10 | -5.71 |
| lr 1e-5 | 32.48 | -0.33 |
| lr 1e-5 + EMA decay 0.9998 | 32.78 | -0.03 |

Data size (10k vs 40k images) made no measurable difference. A 0-step fine-tune reproduces FP32
exactly (38.1189 vs 38.12 on a 200-image check), so BN folding and ONNX export are lossless.

**2. With a working learning rate, QAT does what it should, mechanically.**

| QAT run | float weights | INT8, path A | INT8, path B |
|---|---|---|---|
| lr 1e-5 | 32.31 | 31.79 | |
| lr 5e-6 | 32.66 | 31.96 | |
| lr 2e-6 (used for the headline) | 32.73 | 32.03 | 30.75 |
| lr 1e-5, 40k images | 32.42 | 31.81 | |
| lr 5e-5, 40k images (the broken recipe) | 30.87 | 30.28 | |

Quantizing the trained network costs about 0.6 points (float to path A), versus about 2.1 for the first
PTQ baseline.

**3. But PTQ with sensible calibration is just as good.** Full-coverage ORT PTQ, min/max, different
numbers of calibration images:

| Calibration images | mAP50-95 (each row is a different subset of train10k) |
|---|---|
| 32 | 32.05, 31.93, 31.99, 31.73 |
| 64 | 31.75 |
| 128 | 31.76, 31.60, 31.88 |
| 256 | 31.43 |
| 512 | 31.19 |
| 5000 | 30.68 |
| moving-average calibrator, 128 / 1000 | 30.71 / 30.88 |
| conv-only coverage (like path A), 5000 / 128 | 30.99 / 31.67 |

Min/max only ever widens an activation range as images are added, so accuracy fell steadily with more
data. The spread across calibration subsets (about 0.3) is far smaller than the effect (about 1.4).
A moving-average calibrator is not a fix: its ranges lean on the first images and undershoot.
Percentile and entropy calibration were not tried at scale because ORT keeps every intermediate tensor
in memory for them, which does not fit on an 8 GB machine.

Where path A's advantage over the original PTQ came from: about 0.3 from quantizing only the convs,
about 0.7 from tighter ranges, and 0.1 to 0.4 (inside the noise) from the QAT training itself.

## Caveats

- Differences below about 1 point between two models are not significant at this sample size.
- Path A's speed and size could improve with a deployment step that constant-folds the weights
  (not done); the 0.53x here is the graph as exported.
- Latency is single-thread CPU on one laptop under WSL2, and varies between sessions (the FP32 model
  measured 78 to 128 ms on different days). Only the ratios within one session are meaningful.
- The noise-free alternative to the 4 noisy 32-image subsets would be a calibration set chosen on
  purpose; we did not search for one.
- The training subset was already seen in the official weights' pretraining (it is part of
  train2017), so QAT here is recovery fine-tuning, not new learning.
- GPU cost: eight short RunPod sessions on an RTX 2000 Ada at $0.24/h. The ledger (`results/spend.json`)
  charges at a $0.40/h ceiling and shows $2.08 of the $5 cap; real spend is lower.

## Reproduce

```bash
make install
# PTQ, the tuned recipe (about 1 minute)
rtod quantize yolox_tiny_fp32.onnx yolox_tiny_ptq.onnx --calib 32
rtod bench yolox_tiny_ptq.onnx --variant yolox-tiny/ptq-int8-calib32 --eval val_test --jackknife 10
# QAT (needs a GPU; the pod job is `rtod job qat`, see docs/adr/0004)
rtod qat --lr 2e-6 --steps 1250 --device cuda --no-ptq --out qat_out
rtod export-qdq qat_out/qat_model.pt qat_pathA.onnx
rtod plot qat_out/metrics.jsonl training.png
```

`results/qat-final/` holds the training metrics, plot and config of the run used above. The training
curves of the earlier, broken recipe (lr 2e-4) are in `results/history/qat-run4/`.
