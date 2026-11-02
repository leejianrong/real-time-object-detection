# ADR-0007: Ship the QAT code and report its results as measured, including that it does not beat well-calibrated PTQ

- Status: Accepted
- Date: 2026-11-03
- Deciders: Jian Lee, Claude

## Context

ADR-0003 chose PTQ then QAT, expecting QAT to beat PTQ. Eight short GPU sessions and about forty
evaluations later (all numbers in `docs/RESULTS.md`, scored on the `val_dev` half of val2017 unless
stated, standard error about 0.6 mAP points per model) the evidence is:

- **A plain float fine-tune lost 2.4 points at lr 5e-5 and nothing at lr 1e-5.** Cause: BatchNorm is
  folded into the convs before training, so the weights carry the BN scale and the same step size
  moves them much further. Data size (10k vs 40k), EMA, run length and gradient clipping were not
  the cause.
- **QAT does work mechanically.** Quantizing the trained fake-quant network costs about 0.6 points
  (32.4 float -> 31.8 INT8), versus about 2.1 for the first PTQ baseline.
- **It does not beat a well-calibrated PTQ.** Full-coverage ORT PTQ with min/max calibration on 32
  images scores 32.05, the same as the best QAT graph (31.8 to 32.0). The original PTQ baseline
  (30.68) was handicapped by 5000-image min/max calibration, which only ever widens ranges; accuracy
  fell steadily as images were added.
- **QAT weights are not what helps.** The same QAT-trained weights through the identical ORT PTQ
  pipeline (path B) score 30.75, equal to plain PTQ.
- **The trained graph quantizes less.** Path A (the fake-quant graph exported as QDQ) quantizes only
  conv inputs, weights and outputs; ORT PTQ also quantizes SiLU, Add, Concat and pooling.

## Decision

1. Ship `rtod.qat`, `rtod.train`, `rtod qat` and `rtod export-qdq` as part of the project and describe
   them as what they are: a working QAT implementation whose results tie, not beat, tuned PTQ here.
2. Report every variant on the same footing: FP32, PTQ (the original and the tuned calibration), QAT
   path A, QAT path B, with a standard error, the same eval and same-session latency.
3. Choose recipes and calibration settings on `val_dev` only; report the finalists once on the
   held-out `val_test` half.
4. Keep the defaults that the evidence supports: lr 1e-5, clip 100. Do not tune further to make QAT
   win.
5. The deployed demo model is chosen on speed and size as well as accuracy (ADR-0005), so it is the
   fully quantized PTQ graph unless path A is shown to be competitive after its own deployment step.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Keep tuning QAT until it wins | The goal is an honest benchmark; the project owner explicitly does not need QAT to win |
| Drop QAT from the showcase | QAT is a stated deliverable and the lessons (BN folding, calibration) are the interesting part |
| Quantization-aware training through torch.ao FX | Cannot trace YOLOX's head (ADR-0003 spike) |
| Percentile/entropy calibration | ORT keeps every intermediate tensor in memory for these; did not fit the 8 GB dev machine beyond a handful of images |

## Consequences

- The notebook tells a more useful story: QAT works, calibration matters more than training here, and
  the original baseline was wrong in a way that flattered QAT. It also shows the failed runs.
- Two deployment paths exist (A: accurate, partly quantized, slower and larger in ORT; B: same graph
  as PTQ, no accuracy gain). Path A needs its own deployment step (constant-folded weights) before
  any speed claim.
- `val_dev` choices carry noise: min/max calibration depends on which images are used, so the
  calibration subset is itself measured (`docs/RESULTS.md`), not assumed.
- Evaluation numbers before 2026-11-02 (the first `val200` triage) are not used for decisions: the
  jackknife standard error of one model on 200 images is about 2.1 points.
