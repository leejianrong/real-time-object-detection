# Add the yolox-traffic demo: a quantized detector on a traffic clip

**Not opened yet.** Everything here is ready to apply to an indah checkout with `gallery/pr/apply.sh`.

## What it adds

`examples/yolox_traffic.py` plus `examples/yolox_traffic/` (clip frames, recorded detections, attribution),
a catalog entry, a thumbnail, the Colab notebook, a test, the build-script lines and a CHANGELOG line.

A 15-second road-traffic clip loops with live boxes, an fps readout and a per-frame latency strip. The page opens
on a **replay** (recorded detections from the same model on the same frames). **Run live on this server** runs the
INT8 YOLOX-Tiny here for up to 30 s, one visitor at a time; a run that turns out too slow drops back to the replay.
A badge says which one you are looking at and why. No uploads, no camera, no OpenCV (frames are bundled JPEGs; Pillow
and numpy do the rest).

## Why replay is the default (measured, not assumed)

A throwaway Fly app with the gallery's machine shape (`shared-cpu-1x`, 1 GB, `sin`), fresh machine, 2026-11-03:

| | fps | per frame |
|---|---|---|
| Live demo, first ~55 s | about 38 | 26 ms |
| After the burst credit is spent | about 2.2 | 480 ms |

Peak memory was 152 MB. The shared CPU holds full speed for roughly a minute and then throttles to about 1/18 of that.
Below 8 fps sustained means replay by default; the 30 s cap keeps a live run from spending the burst credit the other
demos need. (Measured on its own machine, not inside the full gallery process: memory headroom and CPU sharing with the
other demos are unmeasured, so one re-measure after this merges would close that.) The app was destroyed afterwards.

## Gallery rule

The catalog says demos are model-free with bounded work per session. This one runs a model, so it needs an explicit
amendment: the model is 5.6 MB and downloaded once (prefetched in the image, like the SSD demo's); inference is bounded
by a global one-holder slot, a 30 s cap per live run and a fallback to replay; there are no uploads. Suggested wording
for `docs/GALLERY-APPS-NEXT.md`: "A demo may run a model if it is small, fetched at build time, bounded by a global slot
and has a recorded fallback."

## Checks run

- On a clone of indah with the bundle applied: the whole unit and integration suite, **1830 passed, 5 skipped**.
- The example's own tests (12, no network): slot grants one holder and queues nobody, only the holder can release it,
  fallback reasons, the slow-server fallback, the 30 s cap, a missing model, a session closing frees the slot, the
  asset download when only the `.py` file is copied, and the demo serving under a sub-path mount with a strict CSP.
- Replay honesty: recorded boxes equal the live model's on the same frames within IoU 0.9 (runs in the source repo,
  which holds the model; not part of the indah test).

## Caveats

- The clip is CC BY 3.0 (Sounds of Changes, attribution in `examples/yolox_traffic/ATTRIBUTION.txt`), trimmed and downscaled.
- The model lives on the Hugging Face Hub (`jianlee/yolox-tiny-quantized`); if it is unavailable the demo still plays in replay.
- The thumbnail is a composed image (a real frame with the model's recorded boxes), not a browser screenshot; replace it
  with `scripts/capture_thumbnails.py` once the demo is in the gallery.
- The Colab and Space copies fetch the clip and recording from `raw.githubusercontent.com/leejianrong/indah/main/...`,
  which only resolves after this merges.
- The assets add about 2.6 MB to the repo.
