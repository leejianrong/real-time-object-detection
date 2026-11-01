# rtod: agent brief

Quantize and prune YOLOX-Tiny, benchmark vs the original, ship a Colab showcase and an indah gallery demo.
Source of truth: `PLAN.md`, `SLICES.md`, `docs/adr/`, decisions in `QUESTIONS.md`. Trust the code over the docs if they disagree, and say so.

## Status
Scaffold only (V1 not started). Update this section as slices land.

## Commands
`make` lists targets. `make install`, `make check` (lint + mypy + unit, also the pre-push hook), `make integration`, `make hooks` (once per clone).

## Conventions
- Branch per slice off fresh `main`, PR only, CI green before merge. Never push to `main`.
- Tests: `tests/unit` needs no infra and no network; `tests/integration` is CPU only; mark `slow` and `gpu` tests and keep them out of CI.
- Every bug gets a failing test first. Tests assert external behaviour through the `Detector` protocol or the CLI, not internals.
- Licences: only MIT/Apache/BSD code and weights (ADR-0001). Never commit weights, datasets, `.env` or tokens. Shipped assets get a line in `NOTICE`.
- Heavy jobs (QAT, prune fine-tune, TensorRT) run on RunPod via the runpod-jobs skill, hard cap $5 total (ADR-0004). Check the spend ledger before renting.
- Numbers always name their hardware. Never compare results across machines.
