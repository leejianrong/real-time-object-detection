# Fly box feasibility (V1 spike, REA-6)

Question (ADR-0005): can the shared gallery box run YOLOX-Tiny INT8 live at 8 fps or more?

The gallery runs on `shared-cpu-1x`, 1 GB (`indah-python-ui/deploy/fly/build_gallery.sh`). I did **not**
touch that machine. These numbers come from a Docker container with 1 GB RAM on the i7-13620H used for
the other results, python:3.12-slim, onnxruntime 1.31, 1 intra-op thread, 200 timed iterations, a
640x480 seeded-noise frame (decode/NMS cost is lower than on a real scene), end to end
(letterbox + model + decode/NMS).

| CPU quota | variant | e2e p50 ms | e2e p95 ms | fps | peak RSS |
|---|---|---|---|---|---|
| 1.0 vCPU | fp32 | 116.9 | 135.3 | 8.6 | 162 MB |
| 1.0 vCPU | ptq-int8 | 49.1 | 61.5 | 20.3 | 139 MB |
| 0.25 vCPU | fp32 | 505.1 | 621.7 | 2.0 | 163 MB |
| 0.25 vCPU | ptq-int8 | 202.9 | 290.1 | 4.9 | 139 MB |

## Reading

- **Memory is fine.** About 140 MB peak for one INT8 session, well inside 1 GB, but the gallery
  already hosts other demos in that process, so the real headroom is unmeasured.
- **A full vCPU clears 8 fps for INT8 with margin (20 fps), FP32 only just.**
- **`shared-cpu-1x` is not a full vCPU.** Fly gives it a small baseline share plus burst credits, so
  sustained inference can throttle. The 0.25-CPU rows are a pessimistic stand-in, not a measurement of
  Fly. Whether live mode holds 8 fps over a 15 s clip on the real box is decided by REA-24, not here.
- **Verdict: not yet known.** Live clears 8 fps with a full vCPU (20 fps) and misses it at a quarter
  vCPU (4.9 fps). Plan for live on a burst and replay as the safe default until REA-24 measures the
  real instance; the app's fps-floor fallback (ADR-0005) covers the throttled case.
- Replay mode stays a required part of the design (ADR-0005), not an optional extra.
