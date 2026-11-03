#!/usr/bin/env bash
# Stage the reported model files under their published names and upload them to the Hub.
# Needs a logged-in `hf` (uvx --from huggingface_hub hf auth login). Run from the repo root.
set -euo pipefail
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp .cache/yolox_tiny_fp32.onnx "$stage/yolox_tiny_fp32.onnx"
cp .cache/ptq_full32.onnx "$stage/yolox_tiny_ptq_int8_calib32.onnx"
cp .cache/qat_lr2e6_pathA.onnx "$stage/yolox_tiny_qat_int8_pathA.onnx"
cp .cache/jobs/qat_artifact/artifact3/qat_lr2e6/qat_model.pt "$stage/yolox_tiny_qat_lr2e-6.pt"
cp docs/hub/README.md "$stage/README.md"
cp LICENSE NOTICE "$stage/"
( cd "$stage" && sha256sum *.onnx *.pt )
uvx --from huggingface_hub hf upload jianlee/yolox-tiny-quantized "$stage" . \
  --repo-type model --commit-message "Add YOLOX-Tiny FP32, PTQ INT8, QAT"
