# rtod: agent brief

Quantize and prune YOLOX-Tiny, benchmark vs the original, ship a Colab showcase and an indah gallery demo.
Source of truth: `PLAN.md`, `SLICES.md`, `docs/adr/`, decisions in `QUESTIONS.md`. Trust the code over the docs if they disagree, and say so.

## Status
V1 and V2 complete (V2's QAT ties tuned PTQ, see `docs/RESULTS.md` and ADR-0007). Next: HF publish (REA-10), pruning (V3), GPU path (V4), indah demo (V5-V6), notebook (V7). Slow tests (`-m slow`) download weights; CI skips them.

## Commands
`make` lists targets. `make install`, `make check` (lint + mypy + unit, also the pre-push hook), `make integration`, `make hooks` (once per clone).

## Conventions
- Branch per slice off fresh `main`, PR only, CI green before merge. Never push to `main`.
- Tests: `tests/unit` needs no infra and no network; `tests/integration` is CPU only; mark `slow` and `gpu` tests and keep them out of CI.
- Every bug gets a failing test first. Tests assert external behaviour through the `Detector` protocol or the CLI, not internals.
- Licences: only MIT/Apache/BSD code and weights (ADR-0001). Never commit weights, datasets, `.env` or tokens. Shipped assets get a line in `NOTICE`.
- Heavy jobs (QAT, prune fine-tune, TensorRT) run on RunPod via the runpod-jobs skill, hard cap $5 total (ADR-0004). Check the spend ledger before renting.
- Numbers always name their hardware. Never compare results across machines.
- QAT: BN is folded before fine-tuning, so use lr <= 1e-5 (5e-5 loses ~2.4 pt). Choose recipes on `val_dev`; report on `val_test` once. Min/max PTQ calibration: fewer images (32-128) beat 5000.
- Do not switch branches or edit `src/` while background jobs (evals, quantize) are running; they import the working tree.
