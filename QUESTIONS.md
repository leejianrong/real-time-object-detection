# Decision register

Status: FORK (asked in round 1) · ASSUMED (default, change by saying so) · DEFERRED.
Facts checked against `~/projects/abang-ai/indah-python-ui` (v0.11.0, Apache-2.0).

## Facts that shaped the answers

- indah has `Camera` + `Live` (WebSocket, Tier 1). Colab's proxy blocks WebSockets, so Colab falls to Tier 0 (~1 fps webcam snapshots).
- indah already ships an `object_detection.py` gallery demo (SSD-MobileNetV1, still photos). The video-detection card IND-62 is parked.
- The gallery is **one 1 GB shared CPU Fly process**; `docs/GALLERY-APPS-NEXT.md` sets the rule "model-free, bounded work per session". A live-inference video demo breaks that rule.
- Each gallery card needs: catalog entry, thumbnail, Colab notebook (`make colab`), e2e smoke test, CHANGELOG line; strict CSP (no inline script/style).

## 1 Users
| # | Question | Status | Answer |
|---|---|---|---|
| 1.1 | Primary audience | ASSUMED | ML practitioners who want a reproducible, honest quant/prune walkthrough; secondary: indah visitors who want a cool demo. |
| 1.2 | Human vs agent | ASSUMED | Humans via notebook/UI; repo CLI is agent-friendly (`--json`, exit codes) for benchmarks. |

## 2 Scope
| 2.1 | Which model | ANSWERED F1 | YOLOX-Tiny (Apache-2.0). |
| 2.2 | Pruning in or out | ANSWERED F2 | In: structured pruning (torch-pruning) + fine-tune. |
| 2.3 | GPU INT8 (TensorRT) | ASSUMED (re-scoped) | ONNX works with TensorRT (ORT TensorRT EP, or trtexec on QDQ ONNX). Jobs run on RunPod where we control the image, so TRT INT8 is in scope as a slice after the CPU path; Colab only replays saved results / runs FP16 live. |
| 2.4 | Colab "real-time" source | ANSWERED F4 | No webcam; user-sourced vehicle/road footage, server-side decode. |
| 2.5 | Gallery hosting shape | ANSWERED F5 | Shares the gallery Fly instance. Live INT8 ONNX if a spike shows it fits; otherwise replay of precomputed boxes + live graphs. One UI, two sources. |
| 2.6 | Notebook trains live or loads checkpoint | ANSWERED F6 | Both behind a switch. Heavy jobs run on RunPod; notebook is showcase only. |
| 2.7 | Out of scope | ASSUMED | Training from scratch, new architectures, mobile/edge targets, tracking, custom datasets, webcam on Colab. |

## 3 Data model / identity
| 3.1 | Entities | ASSUMED | Model variant (id like `yolox-tiny/int8-qat`), benchmark run (hardware, runtime, batch, warmup, p50/p95 latency, fps, mAP), artifact (ONNX file + sha256). Variants are addressed by a string id; results are JSON files. |
| 3.2 | Identity across export | ASSUMED | Every ONNX artifact carries its variant id + git sha in metadata_props. |

## 4 Storage
| 4.1 | Where artifacts live | ASSUMED | Code in git; results JSON committed under `results/`; weights on Hugging Face Hub (model repo + model card), sha256 pinned in code. The gallery demo downloads once and caches, like the existing object_detection demo. |
| 4.2 | Datasets | ASSUMED | COCO val2017 (5k imgs) for mAP, never committed; QAT/calibration uses a fixed 5k-image train2017 subset listed by image id in git. |

## 5 Concurrency
| 5.1 | Gallery: many viewers, one CPU | ASSUMED | One global inference slot (semaphore); extra viewers get replay mode automatically. |
| 5.2 | Benchmarks vs noisy neighbours | ASSUMED | Pin threads, warmup 20, ≥200 timed iters, report p50/p95 and CPU model; Colab numbers labelled "shared hardware". |

## 6 Interfaces
| 6.1 | Surfaces | ASSUMED | Python package `rtod` with CLI (`export`, `quantize`, `prune`, `eval`, `bench`), notebook, indah app. One library write path; CLI and notebook both call it. |
| 6.2 | indah dependency | ASSUMED | Public `pip install indah`; no local path deps in published artifacts. Demo app lives in the indah repo as a PR; this repo holds the same app under `examples/` for the notebook. |

## 7 Failure behaviour
| 7.1 | Quantized mAP regresses | ASSUMED | Report it, don't hide it; CI gate is on a small-subset mAP delta threshold, not the full run. |
| 7.2 | Demo model download fails / viewer limit hit | ASSUMED | Clear in-UI message; never a blank page. Busy viewers get a queue/"try again" message or fall back to a pre-recorded annotated clip, labelled as such. |
| 7.3 | No GPU in Colab | ASSUMED | Notebook detects and skips GPU cells with a note. |

## 8 Dependencies (licence · offline · fallback)
| 8.1 | Detector | ANSWERED F1 | YOLOX-Tiny. |
| 8.2 | ONNX Runtime | ASSUMED | MIT. Common runtime for FP32/INT8 on CPU and CUDA EP on GPU. Offline. |
| 8.3 | Quantization | ASSUMED | PTQ via ORT static quantization (QDQ); QAT in PyTorch (torch.ao / torchao, BSD-3), exported to QDQ ONNX. Verify export path in slice 1. |
| 8.4 | Pruning | ANSWERED F2 | `torch-pruning` (MIT). |
| 8.5 | Video I/O | ASSUMED | PyAV or OpenCV-headless (Apache-2.0) for decoding; demo clip must be CC0/CC-BY with attribution, ≤10 s. |
| 8.6 | Test clip | ASSUMED | Pexels/Pixabay-style licence is NOT clean for redistribution in a repo; use CC0 / Wikimedia Commons CC-BY with credit. |

## 9 Runtime
| 9.0 | Heavy jobs (train/QAT/prune/bench) | ASSUMED | RunPod via runpod-jobs skill: dead-man's-switch teardown, artifacts to HF Hub, results JSON committed. |
| 9.1 | Notebook | ASSUMED | Colab free tier (T4 GPU optional), Python 3.12 default; <= 40 min end to end with checkpoints cached. |
| 9.2 | Demo runtime | ANSWERED F5 | onnxruntime CPU in the existing gallery image (already a dependency of the SSD demo). |
| 9.3 | Repo dev env | ASSUMED | `uv`, Python 3.11+, Makefile gates per dev-playbook (ruff, pytest with markers unit/integration/e2e, pre-push + CI). |

## 10 Success
| 10.1 | Numbers | ASSUMED | Report, vs FP32 on same box: INT8 CPU latency ≤ 0.6x (target; not a promise), mAP50-95 drop ≤ 1.0 pt for QAT and ≤ 2.5 pt for PTQ, size ≤ 0.3x. Demo: ≥ 8 fps on the Fly box with overlay latency shown on screen. Missing a target is a reported result, not a failure. |

## 11 Security
| 11.1 | Secrets | ASSUMED | HF/Fly tokens only via env/CI secrets (handling-secrets skill). Demo accepts no uploads (bundled clip only) so there's no untrusted input path. |

## 12 Versioning
| 12.1 | Result schema | ASSUMED | `schema_version` in results JSON; model artifacts are immutable, new ones get new ids. |
| 12.2 | Pin indah | ASSUMED | Notebook pins `indah==<tested version>`. |

## Decided in rounds 2-3

| ID | Question | Status | Answer | Landed |
|---|---|---|---|---|
| D1 | Detector | DECIDED | YOLOX-Tiny, Apache-2.0 | ADR-0001 |
| D2 | Pruning | DECIDED | In scope, structured | ADR-0003 |
| D3 | ONNX vs TensorRT | DECIDED | ONNX is the interchange; TRT consumes it for GPU INT8 | ADR-0002 |
| D4 | Colab role | DECIDED | Showcase only; heavy jobs on RunPod | ADR-0004 |
| D5 | Video source | DECIDED | No webcam; user-sourced clip (CC0/CC-BY/own) | ADR-0006 |
| D6 | Gallery hosting | DECIDED | Shared gallery Fly instance; live if the spike passes, else replay | ADR-0005 |
| D7 | Notebook training | DECIDED | Live-or-checkpoint switch | PLAN S9 |
| D8 | Repo relationship | DECIDED | Separate repo; indah PR later | PLAN Scope |
| D9 | Licence | DECIDED | Apache-2.0, NOTICE credits YOLOX | PLAN Scope |
| D10 | RunPod budget | DECIDED | Hard cap $5 total | ADR-0004 |
| D11 | Dataset | ASSUMED | COCO 2017; val2017 for mAP, fixed 10k train2017 subset for calibration/QAT/prune fine-tune | ADR-0006 |

## Open forks

None. Round 1 closed all forks (rounds used: 3 of 3 cap, final one was confirmation only).

## Coverage

| Category | Covered by |
|---|---|
| Primary user and actors | 1.1, 1.2 |
| Scope boundary | 2.1-2.7, D8, D9 |
| Data model and identity | 3.1, 3.2 |
| State and storage | 4.1, 4.2, D11 |
| Concurrency and conflict | 5.1, 5.2 |
| Interfaces and contracts | 6.1, 6.2 |
| Failure behaviour | 7.1-7.3 |
| External dependencies | 8.1-8.6 |
| Runtime and deployment | 9.0-9.3, D6 |
| Measurable success | 10.1 |
| Security and secrets | 11.1 |
| Versioning and migration | 12.1, 12.2 |
