# QAT export spike (V2, REA-8)

Question: can YOLOX-Tiny be trained with fake-quant and deployed as an INT8 ONNX that runs in ONNX
Runtime, and by which path?

**Why not torch.ao FX.** YOLOX's head branches on `self.training` and needs targets in the training
branch, so `prepare_qat_fx` cannot trace it. `rtod.qat` instead folds BN into each conv and wraps all
83 convs in `QConv` (fake-quant on input, per-channel int8 weights, output).

## Findings (CPU, 64 real calibration images, one thread, same-run timings)

| Path | Works | model p50 | size | notes |
|---|---|---|---|---|
| A. Export the fake-quant graph (QDQ) directly | yes | 41.7 ms | 20.5 MB | ORT matches the torch fake-quant model (corr 0.999 on objectness), but weights stay float and biases are not quantized, so it is slower and 4x larger than PTQ |
| B. Train with fake-quant, export the folded **float** weights, quantize with the existing `rtod quantize` | to be confirmed in REA-9 | = PTQ path (31.6 ms, 5.6 MB) | 5.6 MB | needs only the folding export, which is exact (max diff 1.3e-3) |
| FP32 / PTQ (reference, same run) | | 83.4 / 31.6 ms | 20.3 / 5.6 MB | |

## Decision

Path B is the deployment route: fine-tune with `QConv` fake-quant, export folded float weights, run
the PTQ pipeline. Path A is kept as a numerical cross-check of the fake-quant simulation (a test
asserts ORT agrees with torch). Risk for REA-9: the ORT calibrator (MinMax) differs from the
training observers (moving-average MinMax); if QAT then fails to beat PTQ, fall back to path A with
ORT's optimized-model save to constant-fold weights, or to TensorRT's own QDQ handling for GPU.
