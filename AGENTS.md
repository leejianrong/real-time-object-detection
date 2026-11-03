# rtod: agent brief (AGENTS.md; CLAUDE.md imports it)

Quantize and prune YOLOX-Tiny, benchmark vs the original, ship a Colab showcase and an indah gallery demo.
Source of truth: `PLAN.md`, `SLICES.md`, `docs/adr/`, decisions in `QUESTIONS.md`. Trust the code over the docs if they disagree, and say so.

## Status
v0.1.0 released: PTQ, QAT, pruning, GPU (CUDA; TensorRT rejected our INT8 graphs), Colab notebook, live indah dashboard (`make dashboard`), models on the Hub. Open: the indah gallery PR (REA-20 to REA-25, a PR into the indah repo, needs the Fly measurement). To release: bump `REF` in `notebooks/make_notebook.py`, regenerate, tag. Board: Pandan 40 (key REA). Slow tests (`-m slow`, `-m notebook`) download weights; CI skips them except the `notebook` job on main.

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
- Pruning: channel counts round to a multiple of 16 (`prune.ROUND_TO`); unaligned widths gave no CPU speedup. MACs targets are coarse, `prune_to_macs` returns the closest. Report peak RAM and CPU with `rtod bench --resources`.
- Git commits and PR text carry no attribution lines. Use a `git worktree` for side work while a pod driver or eval runs.

## Launching a RunPod job
Follow the runpod-jobs skill. The key is in `~/.runpod/config.toml` as `apikey = '...'` in SINGLE quotes; load it by reference, never print it: `export RUNPOD_API_KEY="$(awk -F"'" '/^apikey/{print $2}' ~/.runpod/config.toml)"` (a sed that keeps the quotes gives a 401 that looks like a revoked key). Put `~/.claude/skills/runpod-jobs/scripts` (`rp`) and `.venv/bin` on PATH, then run `rtod job qat [--sweep NAME]` detached (`setsid nohup`, from a launch script) against a commit already on `origin/main`. `--dry-run` checks the cap. The driver prints a status line every 2 min and goes fatal on 15 min without TRAIN/DATA/PRUNE lines in `train` (heartbeats alone prove nothing); it warns on an idle GPU. `kill` on the driver tears the pod down. Afterwards `runpodctl pod list` must show no `rtod-*` pod (other pods on the account are not ours). Ledger: `results/spend.json`.
