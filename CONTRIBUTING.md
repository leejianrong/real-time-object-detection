# Contributing

```bash
make install && make hooks     # once per clone; the pre-push hook runs `make check`
make check                     # lint, types, unit tests (no network, no GPU)
make integration               # CPU integration tests
make notebook                  # rebuild and run the showcase notebook (notebooks/out/)
```

- Branch per change off fresh `main`; PR only; CI green before merge. Never push to `main`.
- Every bug gets a failing test first. Tests assert behaviour through the `Detector` protocol or the CLI.
- Mark tests that download weights `slow` and GPU tests `gpu`; CI skips both.
- Licences: only MIT, Apache or BSD code and weights. Never commit weights, datasets or tokens; shipped
  assets get a line in `NOTICE`.
- Heavy jobs (QAT, pruning recovery, GPU benchmarks) run on RunPod through `rtod job`, under a hard $5 cap.
  The notebook source is `notebooks/make_notebook.py`; edit that, not the `.ipynb`.
- Numbers always name their hardware and carry a standard error. Never compare across machines.

[AGENTS.md](AGENTS.md) holds the same conventions for coding agents.
