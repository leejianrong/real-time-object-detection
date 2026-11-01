# ADR-0002: Use ONNX (QDQ) as the interchange format and ONNX Runtime as the common runtime

- Status: Accepted
- Date: 2026-11-01
- Deciders: Jian Lee, Claude

## Context

We need one artifact that runs on CPU, CUDA and TensorRT and that the indah gallery can run with its existing onnxruntime dependency. ONNX is not in conflict with TensorRT: TensorRT parses ONNX, including QDQ INT8 graphs, and ORT has TensorRT and CUDA execution providers.

## Decision

Every variant is exported to ONNX (QDQ for INT8). CPU and CUDA FP16 use ORT; GPU INT8 uses TensorRT built from the same QDQ ONNX (ORT TensorRT EP or trtexec). Decode and NMS stay outside the graph, in shared numpy code.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Native PyTorch for each backend | The demo runtime would need torch, too heavy for the 1 GB Fly box |
| OpenVINO / TFLite | Better on some CPUs but split the story and the gallery already ships ORT |
| NMS inside the graph | Differs per runtime, hurts comparability and hides the latency split |

## Consequences

One artifact serves notebook, benchmark and gallery. TensorRT engines are GPU- and version-specific, so they are built on the benchmark pod and never shipped; only the ONNX is. ORT's QDQ CPU kernels may lag best-case INT8 speed on some CPUs, and we report what we measure.
