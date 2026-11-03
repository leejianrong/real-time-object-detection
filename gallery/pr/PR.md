# Add the yolox-traffic demo: a quantized detector on a traffic clip

**Not opened yet.** Everything here is ready to apply to an indah checkout with `gallery/pr/apply.sh`.

## What it adds

`examples/yolox_traffic.py` plus `examples/yolox_traffic/` (clip frames, recorded detections, attribution),
a catalog entry, a thumbnail, the Colab notebook, a test, the build-script lines and a CHANGELOG line.

A 15-second road-traffic clip loops with boxes, an fps readout and a per-frame latency strip. A **Recorded | Live**
control picks where the boxes come from. The page opens on **Recorded** (detections computed earlier by the same
model on the same frames). **Live** runs the INT8 YOLOX-Tiny on this server for up to 30 s, one visitor at a time; a
run that turns out too slow, or a busy server, drops back to Recorded and says why. A confidence-threshold slider is
fixed on the recording and free in Live, where lowering it shows boxes no recording contains, which is how a visitor
can tell the model is really running. No uploads, no camera, no OpenCV (frames are bundled JPEGs; Pillow
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

## Gallery rule (the owner's call)

`docs/GALLERY-APPS-NEXT.md` said each app is model-free, needs no network at run time and has bounded work per
session. That was already out of date (`object_detection` and `upload_classify` run ONNX models), and this demo runs
one too, so the PR amends that line instead of relaxing it silently: "A demo may run a model if it is small, fetched
at image build time (the Hugging Face cache is baked into the image and pinned to a commit), bounded by a global
slot, and has a recorded fallback." Reword it, or turn it into an ADR, if you prefer. "No network at run time" holds
only because the image prefetches the model; the download is pinned to a Hub commit and the file is checked against
the checksum the recording was made with.

## Cost of Recorded sessions

Measured on a laptop (13th Gen Intel i7-13620H), not on Fly, so only the ratios mean anything: one Recorded session
costs about 5% of a core and 62 MB; five cost about 23% of a core and 76 MB. Every session pushes about 12 frames a
second, which is where the CPU goes. The clip and the encoded frames are shared by all sessions (before that fix, five
sessions held 109 MB). It has not been measured on the Fly box with the other demos running.

## Checks run

- On a clone of indah with the bundle applied: the whole unit and integration suite, **1830 passed, 5 skipped**.
- The example's own tests (about 25, no network): slot grants one holder and queues nobody, only the holder can release it,
  fallback reasons, the slow-server fallback, the 30 s cap, a missing model, a model error frees the slot and falls back, the class names are the 80 COCO classes, one shared clip, the
  pinned download, a session closing frees the slot, the
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
- The assets add about 4 MB to the repo.
