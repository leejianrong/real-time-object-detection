# Real-time object detection: Slices

Vertical increments. Each ends in something you can demonstrate. Slice 1 confronts the two riskiest unknowns: does an INT8 model hold accuracy, and can the Fly box run it. Heavy-GPU slices are RunPod jobs (ADR-0004); everything else runs on CPU.

## V1: Baseline, PTQ and the Fly feasibility spike

**Delivers:** R0, R1, R7 (partial), R8 (weights/data)

**Build plan**

1. Scaffold the repo per dev-playbook: `uv`, ruff, pytest markers, pre-push + CI, Apache-2.0 `LICENSE`, `NOTICE` crediting YOLOX.
2. Vendor the YOLOX-Tiny definition at a pinned commit, load official weights, export FP32 ONNX.
3. `Detector` protocol + ORT CPU backend; shared letterbox/decode/NMS in numpy.
4. COCO subset tooling (image-id lists, fetch by id) and pycocotools eval.
5. PTQ with ORT static quantization (QDQ) calibrated on 5k images.
6. Bench harness (S3) and `rtod bench --json`; hardware fingerprint.
7. Run FP32 vs PTQ on local CPU and in a container throttled like the Fly box (1 shared vCPU, 1 GB). Write the feasibility note: live fps, RSS.
8. Run the same on the real gallery Fly machine if an access path exists; otherwise defer to V6.

**Demo:** `rtod eval` and `rtod bench` on FP32 and PTQ print a table: mAP50-95, p50/p95 ms, size, hardware. A one-page note states whether live mode clears 8 fps on the Fly-like box.

**Rests on assumptions:** 10.1 targets (reported either way); D11 calibration subset size.

### Test plan

#### End-to-end

- `rtod eval yolox-tiny/fp32` on val2017 reproduces published mAP50-95 within 0.5 pt (about 32.8).
- `rtod bench` on FP32 and PTQ writes valid results JSON and exits 0; PTQ file size is at most 0.35x FP32.
- On the throttled container, the feasibility note records a measured fps and peak RSS (not an estimate).

#### Integration

- FP32 ONNX and PyTorch outputs match on 20 fixed images within 1e-3 before decode.
- Eval on the pinned 200-image subset is deterministic across two runs.
- Bench refuses to run with fewer than 200 timed iterations unless `--quick` is set, and labels the result.

#### Unit

- Letterbox and un-letterbox round-trip box coordinates.
- NMS drops overlapping same-class boxes and keeps different-class ones.
- Results JSON validates against schema v1; unknown `schema_version` is rejected.

## V2: QAT on RunPod

**Delivers:** R2, R7

**Build plan**

1. Job runner (S8): spend ledger, projected-cost check against the $5 cap, dead-man's-switch teardown, artifact upload to HF Hub. Dry-run on CPU first.
2. QAT fine-tune loop on the 10k subset; spike QDQ ONNX export on a 100-image CPU run before renting a GPU.
3. Run QAT on a T4/L4-class pod; evaluate on val2017; compare to PTQ.
4. Publish weights and model card to HF Hub.

**Demo:** Results table now shows FP32, PTQ and QAT; QAT has lower mAP loss than PTQ. The ledger shows total spend so far and the pod is confirmed terminated.

**Rests on assumptions:** 8.3 (QAT exports cleanly). If wrong, V2 becomes "better PTQ" and the notebook shows QAT training curves only, with a note.

### Test plan

#### End-to-end

- QAT ONNX loads in ORT CPU and its mAP50-95 is within 1.0 pt of FP32 (reported, not gated, if missed).
- QAT mAP50-95 is at least as high as PTQ's.
- After the job, the runner's ledger shows no running pods and cumulative spend at or below $5.

#### Integration

- Dry-run of the job runner on CPU with a 50-image subset produces a loadable QDQ ONNX.
- The runner refuses to launch when projected spend would exceed the cap.
- Dead-man's-switch is armed before the training command starts (checked in a fake-provider test).

#### Unit

- Ledger arithmetic: hourly rate x duration, cumulative total, cap check.
- Subset list loader returns the same 10k ids and the first 5k are a prefix.

## V3: Structured pruning

**Delivers:** R3

**Build plan**

1. Dependency-aware channel pruning with torch-pruning at several ratios (for example 20, 35, 50 percent MACs).
2. Fine-tune each on the subset on RunPod; export pruned ONNX (own channel counts).
3. Add pruned+QAT variant.
4. Bench all on CPU; plot size, mAP, latency against pruning ratio.

**Demo:** A trade-off chart of mAP versus CPU latency over the variant ladder, including pruned and pruned+QAT, with one chosen "final" model.

**Rests on assumptions:** D11 (10k subset recovers pruned accuracy); the budget fits several fine-tunes.

### Test plan

#### End-to-end

- At least one pruned variant has lower p50 CPU latency than FP32 by a measured margin with mAP drop reported.
- The chosen final variant is recorded in `results/final.json` and its ONNX hash is on HF Hub.

#### Integration

- Pruned ONNX loads in ORT and runs through the shared `Detector` path.
- Parameter count and MACs reported by the prune step match what the exported ONNX reports within 2 percent.

#### Unit

- Pruning ratio config rejects values outside (0, 1).
- Detection head output channels remain 85 (4 box + 1 objectness + 80 classes) after pruning.

## V4: GPU path, FP16 and TensorRT INT8

**Delivers:** R4

**Build plan**

1. ORT CUDA FP16 backend implementing `Detector`.
2. Build TensorRT INT8 engine from the QDQ ONNX on the pod; wrap as a `Detector`.
3. Benchmark all GPU variants at batch 1 and a larger batch on one named GPU.
4. Record engine/driver/TensorRT versions with the results.

**Demo:** GPU results table: FP32, FP16, TRT INT8 with mAP, latency and fps on the named GPU.

**Rests on assumptions:** Q8.3 (QDQ ONNX is accepted by TensorRT). If TRT refuses, GPU INT8 is reported as not achieved with the error, and FP16 stays the GPU headline.

### Test plan

#### End-to-end

- GPU bench writes results JSON with GPU model and TensorRT version.
- TRT INT8 mAP50-95 is within 1.5 pt of FP32 (reported either way).

#### Integration

- A `gpu`-marked test runs the same 20 fixed images through all backends and matches boxes within IoU 0.9 for confident detections.
- Backends without a GPU are skipped with a clear message.

#### Unit

- Backend selection falls back to CPU with a warning when CUDA is unavailable.

## V5: Local indah app, side-by-side and graphs

**Delivers:** R6 (local), R5 (live demo part)

**Build plan**

1. indah app: frame source decodes the bundled clip server-side; detector via `asyncio.to_thread`; `ImageOverlay` for boxes; `Stat` and `Chart` for fps, latency, detections.
2. Variant picker (FP32 vs INT8 side by side) and a CPU/GPU toggle when a GPU exists.
3. Capture replay JSON from the real model on the real clip.
4. Placeholder CC0 clip until the real clip is chosen.

**Demo:** `python examples/demo.py`, open the URL, press play: the clip plays with boxes and fps/latency graphs, switching variants live.

**Rests on assumptions:** 6.2 (frames-as-images via signals reach 8+ fps locally); a clean clip exists (8.6).

### Test plan

#### End-to-end

- Headless browser test: load the app, press play, and within 10 s see at least one box element and a nonzero fps stat.
- Variant switch changes the displayed model name and keeps playing.

#### Integration

- App serves under a sub-path (as in the gallery mount) and passes under `csp="strict"`.
- A second concurrent session does not start a second inference.

#### Unit

- Rolling fps/latency window computes p50 correctly over fixed samples.
- Frame loop drops late frames instead of queueing them.

## V6: Gallery-ready demo: slot, replay fallback, Fly

**Delivers:** R6, R8 (clip)

**Build plan**

1. Global inference slot and automatic fallback to replay with a visible reason badge.
2. Replay mode renders the same boxes and graphs from the captured JSON.
3. Final clip chosen (self-filmed or CC0/CC-BY), cut to at most 15 s, attribution in `NOTICE`.
4. Measure on the real Fly gallery instance; decide live-by-default or replay-by-default with live opt-in.
5. Prepare the indah PR bundle: example file, catalog entry text, thumbnail, CHANGELOG line, e2e smoke test notes. The PR itself is outside this milestone.

**Demo:** Two browser tabs open at once: the first is "live", the second automatically "replay (busy)". Both look the same and graphs keep moving. Kill the model file and both show "replay (model unavailable)".

**Rests on assumptions:** ADR-0005 spike result from V1/V6 (box speed); 5.1.

### Test plan

#### End-to-end

- Two concurrent sessions: exactly one live, one replay, with correct badges.
- With a missing model file the app still plays and shows the reason.
- On the real Fly instance, measured live fps is recorded; if below 8, default is replay and the doc says so.

#### Integration

- Replay output boxes equal the live-run boxes for the same frames within IoU 0.9 (confirms replay is honest).
- The slot releases on session end, including abrupt disconnect.

#### Unit

- Slot semaphore grants one holder and queues none.
- Fallback predicate: busy, low fps and load failure each yield the right reason string.

## V7: Colab showcase notebook and public release

**Delivers:** R5, R8, R0

**Build plan**

1. Notebook sections: motivation; the variant ladder; PTQ walkthrough; QAT (live switch off by default, checkpoint load by default); pruning; benchmark charts from results JSON; live indah demo on CPU and, if present, GPU.
2. `LIVE_TRAIN` switch with a clearly labelled GPU requirement; checkpoints from HF Hub.
3. Pin `indah` and dependency versions; test on a fresh Colab runtime (CPU and T4).
4. README per readme-playbook, model card, `NOTICE`, CHANGELOG; tag release; badge linking the notebook.

**Demo:** Open the Colab badge on a fresh runtime, Run all with defaults: it finishes without training, shows result charts, and launches the indah demo that plays the clip with boxes on CPU (and GPU on a T4 runtime).

**Rests on assumptions:** 9.1 (Colab free tier timing); indah's Colab proxy path (Tier 0 SSE) handles server-pushed frames well enough.

### Test plan

#### End-to-end

- Run all on a fresh Colab CPU runtime completes in under 15 minutes with no manual edits and shows the live demo.
- On a T4 runtime, the same run shows GPU latency lower than CPU in the live stats.
- Every number in the notebook charts matches `results/*.json` (a script diffs them).

#### Integration

- Notebook executes headlessly in CI (nbclient) with `LIVE_TRAIN=False` against a tiny pinned subset.
- `LIVE_TRAIN=True` on CPU with a 50-image subset runs one step without error (smoke test).

#### Unit

- Notebook helper that loads results JSON rejects an unknown `schema_version`.
- `NOTICE` check: every shipped asset (weights, clip) has a licence line.
