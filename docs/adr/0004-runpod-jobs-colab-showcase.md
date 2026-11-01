# ADR-0004: Run all heavy jobs on RunPod under a $5 hard cap; Colab is showcase only

- Status: Accepted
- Date: 2026-11-01
- Deciders: Jian Lee, Claude

## Context

Colab has no API for an agent to drive. RunPod does, through the runpod-jobs skill. The user set a total GPU budget of $5.

## Decision

QAT, pruning fine-tune, TensorRT builds and GPU benchmarks run on RunPod pods (T4/L4-class, spot or community where possible) launched by a job runner that keeps a spend ledger, refuses to start a job whose projected cost would exceed the cap, and always sets a dead-man's-switch teardown. Outputs go to HF Hub and `results/`. The notebook only loads those artifacts, with `LIVE_TRAIN=False` by default and an opt-in live QAT run for anyone with a GPU.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Train in Colab | No automation, session limits, not reproducible by an agent |
| Local GPU | None assumed available |
| Larger budget | The user asked for $5 |

## Consequences

Reproducibility lives in the repo and RunPod scripts, not in the notebook. The cap is tight, so each job is scripted and tested on CPU with a tiny subset before a GPU is rented. A failed run burns real budget; the ledger makes that visible.
