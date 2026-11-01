# ADR-0006: COCO 2017 subsets for data; shipped clip must be CC0, CC-BY or self-filmed

- Status: Accepted
- Date: 2026-11-01
- Deciders: Jian Lee, Claude

## Context

We need accuracy numbers, calibration/QAT data and a demo clip, and everything shipped publicly needs a redistribution licence. UA-DETRAC, BDD100K, KITTI and MOT17 are non-commercial or share-alike.

## Decision

Evaluation: COCO val2017 (5k images) with pycocotools, plus a pinned 200-image subset for CI. Calibration, QAT and prune fine-tune: a fixed 10k-image train2017 subset stored as an image-id list in git (first 5k for calibration). Images are fetched by id and never committed. The demo clip is self-filmed or CC0/CC-BY with attribution in `NOTICE`. Non-redistributable video datasets may be used privately for sanity checks and are never shipped.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Full train2017 (18 GB) | Download time and storage cost on a $5 budget with no accuracy benefit for recovery |
| A road-traffic dataset for QAT | Licences block redistribution and the model is trained on COCO classes |
| Pexels/Pixabay clips | Free to use but bundling the file in a repo is a grey area |

## Consequences

Numbers are comparable to the published YOLOX COCO result. COCO images carry mixed Flickr licences, which is why they are fetched not committed. Domain shift to traffic footage is not measured by mAP, so the clip is a qualitative demo only.
