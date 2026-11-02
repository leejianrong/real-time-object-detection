# ADR-0003: PTQ, then QAT, plus structured pruning in the order prune, fine-tune, QAT

- Status: Accepted
- Date: 2026-11-01
- Deciders: Jian Lee, Claude

## Context

The showcase must include quantization-aware training, and pruning must show real latency effect, not just sparsity.

## Decision

PTQ: ORT static quantization (QDQ), calibrated on the first 5k images of the fixed subset. QAT: fake-quant fine-tune from FP32 on the 10k subset in PyTorch, exported to QDQ ONNX. Pruning: dependency-aware structured channel pruning with torch-pruning (MIT), fine-tune, then QAT on the pruned model. The variants ladder is FP32, PTQ, QAT, pruned, pruned+QAT.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Unstructured pruning | Zero weights do not speed up dense kernels; misleading as a headline |
| Quantization only | The user explicitly wants pruning |
| Prune after QAT | Pruning a quantized graph breaks channel dependencies and scales; the order above is standard |

## Consequences

Pruning changes channel counts, so pruned models need their own ONNX export. QAT export is the main technical risk (see PLAN risks). Fine-tuning data is a subset of train2017 that the pretrained model already saw, which is acceptable for recovery but must be stated in the notebook.

Update 2026-11-03: QAT was built and measured; see ADR-0007 and `docs/RESULTS.md`. It ties tuned PTQ rather than beating it.
