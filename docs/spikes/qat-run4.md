# QAT run 4: results and what they mean (REA-9)

Recipe: official YOLOX-Tiny weights, BN folded, fake-quant on all 83 convs (`rtod.qat`), SGD lr 2e-4
cosine, momentum 0.9, grad-norm clip 10, batch 16, 1250 steps on the 10k train subset, flip-only
augmentation. Trained on an RTX 2000 Ada (about 19 min, 1.0 it/s fetch aside, 4 it/s training).
Deployed by path B (docs/spikes/qat-export.md): folded float weights, then `rtod quantize`
(ORT static QDQ, MinMax, 5000 calibration images). Full val2017 for every row.

| Variant | mAP50-95 | vs FP32 | mAP50 | model p50 | size |
|---|---|---|---|---|---|
| fp32 | 32.88 | | 50.26 | 78.7 ms | 20.3 MB |
| ptq-int8 | 30.69 | -2.18 | 48.16 | 29.6 ms | 5.6 MB |
| qat-int8 (path B) | 29.73 | -3.15 | 47.09 | 29.8 ms | 5.6 MB |
| (diagnostic) fine-tuned folded float | 31.65 | -1.23 | 49.01 | | 20.3 MB |

Latency: Intel i7-13620H, 1 thread, ORT 1.31, back-to-back in one session. Never compare across sessions.

## Reading

QAT did **not** beat PTQ (target: at least PTQ). Two effects add up:

1. **The fine-tune itself costs 1.23 pt** in float (31.65 vs 32.88). Likely contributors: flip-only
   augmentation on 10k images, lr 2e-4, and a clip threshold of 10 that fires on every step
   (grad norms 40-2300), so the clip, not the gradient, sets the step size. Not yet separated.
2. **Quantization still costs 1.92 pt on top** of the fine-tuned float model (31.65 -> 29.73), only
   0.26 less than plain PTQ costs on the original (2.18). If the deployed quantizer matched the one
   trained for, this gap should be small. It is not: ORT quantizes more tensors (SiLU Mul/Sigmoid,
   Add, Concat, pooling) than the fake-quant model simulates (conv in/out only).

Speed and size are the same for PTQ and QAT (same graph), as expected.

## Next (run 5)

Save the full QAT state (observer ranges), so both deployment paths can be tried locally for free:
path A (export the trained fake-quant graph as QDQ, which ORT matched at corr 0.999 in the spike)
and path B. Make lr and the clip threshold flags and try a gentler recipe (lr 5e-5, clip 100).
