# ADR-0001: Use YOLOX-Tiny, and only permissively licensed models

- Status: Accepted
- Date: 2026-11-01
- Deciders: Jian Lee, Claude

## Context

The project is public and must be redistributable. Ultralytics YOLO is AGPL-3.0 and YOLO-NAS weights are non-commercial. We want an architecture that quantizes and prunes cleanly with real speedups on CPU.

## Decision

Use YOLOX-Tiny (Megvii, Apache-2.0 code and released COCO weights, about 5M params, 32.8 mAP at 416). Any future model must be MIT/Apache-2.0/BSD for both code and weights. `NOTICE` credits YOLOX.

## Alternatives considered

| Option | Why not |
|--------|---------|
| RT-DETR / RF-DETR | Apache-2.0, but transformer attention and layernorm quantize poorly and the QAT story is fragile |
| torchvision SSDlite | BSD and simple, but a weak baseline that makes a boring showcase |
| Ultralytics YOLO | AGPL-3.0 forces the whole public repo and demo under AGPL |

## Consequences

All-convolutional, so INT8 and channel pruning both pay off. The baseline is modest, so absolute mAP is not impressive; the story is the deltas. The official YOLOX repo is somewhat dated, so we vendor only the model definition and weights loader, pinned to a commit.
