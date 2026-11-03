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

## Pruning (structured channels, then recovery training)

`rtod prune` removes whole channels with torch-pruning (magnitude importance, channel counts rounded
to 16; unaligned widths gave no CPU speedup at all), keeping the 80 + 4 + 1 output channels. Accuracy
is on `val_dev` (2,476 images, so about 0.5 standard error); speed, size and memory are one CPU thread
of the i7-13620H, measured back to back in one session (`results/prune-final/`).

| Model | MACs kept | mAP50-95 (`val_dev`) | Latency | Size | Peak RAM added |
|---|---|---|---|---|---|
| FP32 | 100% | 32.81 ± 0.63 | 83.5 ms | 20.3 MB | 103 MB |
| Pruned 30% of MACs, recovered | 68% | 18.44 ± 0.47 | 60.2 ms | 16.0 MB | 88 MB |
| Pruned 50% of MACs, recovered | 48% | 12.39 ± 0.38 | 45.4 ms | 10.7 MB | 69 MB |
| INT8 PTQ (for comparison) | 100% | 32.75 (`val_test`) | 31.6 ms | 5.6 MB | 66 MB |

**Pruning costs real accuracy at the training budget we could afford, and INT8 beats it on every axis.**

- Pruned models score **0.00 mAP before any training**, so recovery training is not optional.
- The first recovery recipe (the QAT one: BatchNorm folded in, lr 1e-5, 1250 steps, about 0.17 of a COCO
  epoch) left all five pruned models at 0.00 (`results/prune-dev/`): the loss only fell from 16 to 10.7
  against 4.4 for the unpruned model.
- Ordinary training fixed that (BatchNorm kept, lr 2.5e-3, weight decay 5e-4, warm-up, EMA 0.999, 6000
  steps on the 10k subset, about 0.8 of a COCO epoch): 18.4 and 12.4 mAP. Training loss was back near the
  original (4.0 against 4.4) but the held-out score is not, and 30% to 50% of the MACs is a lot to remove
  from a model this small. Much longer training would close part of the gap; we did not run it.
- The speedup is real but smaller than the MAC count suggests (50% of the MACs gave 0.54x the latency).
  Peak memory falls modestly (103 to 69 MB), because most of it is the runtime, not the weights.
- Pruned+QAT was not run (the recovered models are too far from FP32 for stacking to be worth it).

## GPU (ONNX Runtime CUDA)

One RTX 2000 Ada (driver 595.91.07, 16 GB), onnxruntime-gpu 1.22 with CUDA 12, batch 1, 500 timed runs,
accuracy on `val_test` (`results/gpu/`). FP16 is the FP32 graph converted with the box decode kept in float.

| Model | mAP50-95 | Latency (model only) | End to end | Size |
|---|---|---|---|---|
| FP32 | 33.47 ± 0.49 | 3.57 ms | 4.42 ms | 20.3 MB |
| **FP16** | **33.47 ± 0.49** | **3.12 ms** | **3.94 ms** | **10.2 MB** |
| INT8 PTQ (QDQ) | 32.83 ± 0.53 | 5.56 ms | 6.29 ms | 5.6 MB |
| INT8 QAT (QDQ) | 32.75 ± 0.50 | 5.06 ms | 5.77 ms | 20.5 MB |

- **FP16 is the free GPU win:** same accuracy, 13% lower latency, half the size.
- **INT8 is slower than FP32 on the plain CUDA provider.** It has no INT8 convolution for these QDQ
  graphs, so it dequantizes around float kernels and adds quantize/dequantize work. INT8 is a CPU win
  (2.6x faster there) and only a GPU win through TensorRT.
- At 3 to 6 ms a frame, the model is not the bottleneck on a GPU: preprocessing and NMS on the CPU
  are about a fifth of the end-to-end time.
- Numbers are one GPU and one library stack. Do not compare them with the CPU table.

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
- GPU cost: about a dozen short RunPod sessions (RTX 2000 Ada and 4000 Ada at $0.24/h). The ledger
  (`results/spend.json`) charges at a $0.40/h ceiling; real spend is well under the $5 cap.

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
