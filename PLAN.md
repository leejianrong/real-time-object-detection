# Real-time object detection, quantized and pruned: Plan

Status: draft · Milestone: MVP (public repo + Colab showcase + indah gallery demo)

## Problem

Small detectors are easy to download and hard to trust once squeezed. Quantization and pruning tutorials usually stop at "it got smaller", skip quantization-aware training, and report speed on one unnamed machine. Practitioners can't see what each technique costs in accuracy and buys in latency, on CPU and GPU, from a single reproducible source.

indah has no real-time model demo either. Its gallery is model-free by policy and its detection demo works on still photos only.

## Solution

A public, Apache-2.0 repo that takes YOLOX-Tiny and produces a ladder of variants: FP32, PTQ INT8, QAT INT8, pruned, pruned+QAT, plus GPU FP16 and TensorRT INT8. Every variant is benchmarked against the original for accuracy (COCO mAP), latency, throughput and size, with hardware named in every number.

Two public artifacts come out of it:

1. A Colab showcase notebook. It walks through the process, shows QAT and the benchmark results, and runs a live video demo of the final model on CPU and GPU through indah.
2. A gallery demo contributed to indah: a bundled road-traffic clip, boxes drawn in real time by the INT8 model on the shared Fly box, with live fps/latency graphs. If the box cannot keep up, the same UI plays back precomputed detections.

## Users and actors

1. **ML practitioners** reading the notebook and repo. They win when anything conflicts.
2. **indah gallery visitors**, who want a quick demo and never touch code.
3. **Agents and CI**: the CLI emits JSON and uses exit codes so a coding agent can drive jobs, and CI gates on small-subset accuracy deltas.

## Scope

**In this milestone.**
- YOLOX-Tiny (416x416, COCO 80 classes) from the official Apache-2.0 weights.
- PTQ (ORT static, QDQ), QAT (PyTorch, exported to QDQ ONNX), structured pruning with fine-tuning, and their combination.
- Runtimes: ONNX Runtime CPU, ORT CUDA FP16, TensorRT INT8 via ONNX.
- Benchmarks on named hardware: local CPU, RunPod GPU, a Colab T4, and a throttled replica of the Fly box.
- `rtod` Python package + CLI, results JSON, weights on Hugging Face Hub with a model card.
- Showcase notebook with a live-or-checkpoint switch for QAT and pruning fine-tune.
- A local indah app (live + replay modes) in this repo, ready to be ported into the indah PR.
- Licence Apache-2.0, `NOTICE` crediting YOLOX (Megvii) and other dependencies.

**Out.**
- The indah repo PR itself (a later milestone; this repo stays separate). We deliver the app, replay data, thumbnail and notes the PR needs.
- Training from scratch, other architectures, mobile/edge targets, tracking, custom datasets, webcam, user uploads in the gallery demo.
- Unstructured pruning as a result (mentioned as a caveat only, it gives no real speedup).
- Running training or QAT on Colab. Colab only replays or lightly re-runs.

## Requirements

| ID | Requirement | Status |
|----|-------------|--------|
| R0 | Show, with reproducible numbers, what quantization and pruning do to a permissively-licensed detector's accuracy and speed | Core goal |
| R1 | Every variant is benchmarked against FP32 on the same hardware with a documented procedure | Must-have |
| R2 | Include quantization-aware training and show it beating PTQ on accuracy | Must-have |
| R3 | Include structured pruning, including its real latency effect | Must-have |
| R4 | GPU results: FP16, and INT8 via TensorRT | Must-have |
| R5 | Public Colab notebook that runs end to end without heavy jobs and shows a live indah demo on CPU and GPU | Must-have |
| R6 | Gallery-ready indah demo of the final model on a bundled clip, live or replay, on the shared Fly instance | Must-have |
| R7 | Repo follows dev-playbook gates; all heavy jobs on RunPod within a $5 hard cap | Must-have |
| R8 | Every shipped asset (weights, clip, data lists) has a clean redistribution licence | Must-have |

Targets (reported results, not pass/fail): INT8 CPU latency at most 0.6x FP32; QAT mAP50-95 drop at most 1.0 pt; PTQ drop at most 2.5 pt; INT8 size at most 0.3x; Fly demo at least 8 fps live.

## Shape

| Part | Mechanism | ADR |
|------|-----------|-----|
| S1 | **Model zoo**: variant ids like `yolox-tiny/int8-qat` map to a recipe + an ONNX artifact with id, git sha and sha256 in metadata | 0001, 0002 |
| S2 | **Eval harness**: one pre/post-process path (letterbox, decode, NMS in numpy outside the graph) shared by every runtime; pycocotools mAP on val2017 or a pinned subset | 0002, 0006 |
| S3 | **Bench harness**: pinned threads, 20 warmup, at least 200 timed iterations, p50/p95, model-only and end-to-end (with pre/post) split, hardware fingerprint recorded; JSON out | 0004 |
| S4 | **PTQ**: ORT static quantization, QDQ format, calibrated on the first 5k subset images | 0003 |
| S5 | **QAT**: PyTorch fake-quant fine-tune from the FP32 weights on the 10k subset, exported to QDQ ONNX | 0003 |
| S6 | **Pruning**: dependency-aware structured channel pruning (torch-pruning), fine-tune, then optionally QAT. Order is prune, fine-tune, QAT | 0003 |
| S7 | **GPU path**: ORT CUDA FP16 and TensorRT INT8 from QDQ ONNX, built on RunPod | 0002 |
| S8 | **Job runner**: scripts around the runpod-jobs skill: spend ledger, dead-man's-switch teardown, artifacts pushed to HF Hub, results JSON pulled back | 0004 |
| S9 | **Notebook**: reads results JSON + checkpoints from HF Hub; `LIVE_TRAIN=False` by default; live CPU (and GPU if present) demo through indah | 0004 |
| S10 | **indah app**: frame source (bundled clip decoded server-side) → detector slot (one global semaphore) → `ImageOverlay` + `Stat`/`Chart` graphs; source is `live` or `replay` | 0005 |
| S11 | **Replay data**: per-frame boxes/scores/latency JSON captured from the real model on the real clip | 0005 |

## Affordances

**UI (indah app and notebook).**

| Affordance | Place | Wires to |
|------------|-------|----------|
| Play / pause / restart | demo header | S10 frame loop |
| Variant picker (FP32, INT8, pruned...) in notebook; fixed INT8 in gallery | demo header | S1 |
| Live vs replay badge, with reason ("busy", "box too slow") | demo header | S10 slot state |
| Video + boxes with class and score | main pane | `ImageOverlay` |
| fps, p50 latency, detections per frame | stat cards + line chart | S3-style rolling stats |
| FP32 vs INT8 side by side (notebook) | split pane | two detector sessions |

**Non-UI.**

| Affordance | Kind | Wires to |
|------------|------|----------|
| `rtod export / quantize / qat / prune / eval / bench` | CLI, `--json`, exit codes | S1-S6 |
| `results/*.json` (`schema_version`) | committed store | S3, S9 |
| spend ledger | file, updated by job runner | S8 |
| HF Hub model repo + card | artifact store | S1 |

## Implementation decisions

- Modules: `rtod.models` (variant registry), `rtod.data` (COCO subset lists and loaders), `rtod.infer` (runtime backends behind one `Detector` protocol), `rtod.metrics`, `rtod.bench`, `rtod.quant`, `rtod.prune`, `rtod.jobs`. The demo app imports only `rtod.infer` and the replay reader, so the indah PR can vendor the small subset it needs.
- The `Detector` protocol is `detect(frame_bgr) -> Detections`; every backend (ORT CPU, ORT CUDA, TRT) implements it, and every test and the demo use it. This is the main testability seam.
- Post-processing (decode + NMS) stays outside the ONNX graph so it is bit-identical across runtimes, and so latency can be reported with and without it.
- Results JSON fields: variant id, runtime, device, cpu/gpu model, threads, batch, input size, p50/p95 ms, fps, mAP50-95, mAP50, size bytes, git sha, `schema_version`.
- The gallery app can only use indah components and strict CSP: frames travel as images through indah signals rather than a raw `<video>` tag. If that proves too heavy, a spike in V6 evaluates a client-side video element synced to replay data.
- The gallery model is the INT8 ONNX downloaded from HF Hub once and cached, as the existing detection demo does. The clip and replay JSON are bundled (target at most 3 MB).
- Notebook pins an exact `indah` version; the gallery PR pins nothing extra beyond onnxruntime, already in the gallery image.

## Testing approach

Test at three seams: the `Detector` protocol (any backend, fixed images with expected boxes), the CLI (`--json` contracts), and the indah app through its HTTP/SSE surface. Good tests check external behaviour: boxes on a known image, mAP delta on a 200-image pinned subset, JSON schema, never internals. Heavy accuracy and GPU tests are marked `slow`/`gpu` and run on RunPod or manually; pre-push and CI run unit + integration on CPU with a tiny pinned subset. Per-slice plans are in `SLICES.md`.

## Assumed defaults

| ID | Assumed | Cost if wrong |
|----|---------|---------------|
| 8.3 | QAT through torch.ao/torchao FX-or-PT2E exports cleanly to QDQ ONNX for YOLOX-Tiny | Medium. V2 is where it shows. Fallback is ORT-side PTQ with better calibration, QAT as a smaller showcase. |
| 10.1 | Targets above | Low. They are reported either way. |
| D11 | 10k subset is enough for QAT recovery and prune fine-tune | Medium. Larger subsets cost more GPU time against the $5 cap. |
| 6.2 | Frames via images through indah signals are fast enough for 8+ fps | Medium. Falls back to replay mode, same UI. |
| 5.1 | One global inference slot is enough for gallery traffic | Low. Extra viewers get replay. |
| 8.6 | A clean-licence clip exists (user sourcing it) | Low until V6. A CC0 placeholder keeps earlier slices moving. |

## Open risks

- **QAT to QDQ ONNX export** for YOLOX (Focus stem, SiLU) may not be clean. Revealed in V2.
- **Fly box speed.** One shared vCPU may only manage a few fps. Revealed in V1 by measuring on a throttled replica, confirmed on the real instance in V6.
- **$5 cap.** QAT plus prune fine-tune plus TensorRT builds must fit. Revealed by the ledger from V2 on; the runner refuses to start a job projected to exceed the cap.
- **Latency noise on shared RunPod/Colab CPUs.** Mitigated by p95 reporting and named hardware; never compare numbers across machines.
- **Clip licence.** Revealed in V6, not earlier.
