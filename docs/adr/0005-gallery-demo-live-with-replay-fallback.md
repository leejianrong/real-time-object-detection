# ADR-0005: Gallery demo shares the Fly instance: live INT8 inference with replay fallback in one UI

- Status: Accepted
- Date: 2026-11-01
- Deciders: Jian Lee, Claude

## Context

The owner wants the model on the shared gallery instance (1 GB, shared CPU). The gallery rule is "model-free, bounded work per session", written before this demo. onnxruntime is already in the image because of the SSD photo demo.

## Decision

Ship one app with two frame sources behind the same UI. Live: the INT8 ONNX runs on a bundled clip through a single global inference slot (semaphore). Replay: precomputed detections and latencies from the same model drawn over the same clip, with the same graphs. A viewer is in replay mode when the slot is taken, when rolling live fps falls below a floor, or when the model fails to load, and the badge says why. V1 measures feasibility before anything is built on it.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Live only | One busy second viewer or slow box makes the demo broken |
| Replay only | Cheap and robust, but not a real-time model demo; kept as the fallback only |
| Separate Fly app | Owner chose the shared instance; remains the escape hatch if the spike fails |

## Consequences

Bounded work holds: one inference at a time, no uploads, fixed clip. This amends the gallery rule for this card and needs a line in the indah PR. Replay data must be regenerated if the model changes.

## Update 2026-11-03: measured on a Fly machine of the gallery's shape, replay is the default

A throwaway Fly app (`shared-cpu-1x`, 1 GB, `sin`) ran the demo at about 38 fps for roughly 55 seconds, then throttled
to about 2.2 fps once the shared CPU's burst credit was spent; peak memory 152 MB (`gallery/pr/PR.md`). Below 8 fps
sustained, so the page opens on the replay and "Run live" runs the model for at most 30 s. The slot and the fallback
stay as decided; the default and the cap are new. Not measured: the full gallery process with its other demos.

