#!/usr/bin/env bash
# Runs on the RunPod pod (see rtod.jobs). The pod's dead-man's-switch is armed by `rp` before this runs.
# Usage: RTOD_SHA=<git sha> scripts/job_qat.sh   (RP_JOB_CMD downloads and runs this)
set -euo pipefail
pip install -q "rtod[torch,onnx,coco] @ git+https://github.com/leejianrong/real-time-object-detection@${RTOD_SHA}"
rtod qat --device cuda --steps "${QAT_STEPS:-1250}" --out /workspace/qat_out
# full val2017 mAP on the pod, so the local machine stays free; ~minutes of pod time
rtod eval /workspace/qat_out/yolox_tiny_qat_folded.onnx --json > /workspace/qat_out/eval_folded.json
rtod eval /workspace/qat_out/yolox_tiny_qat_int8.onnx --json > /workspace/qat_out/eval_int8.json
tar -C /workspace -czf /workspace/qat_out.tgz qat_out
runpodctl send /workspace/qat_out.tgz 2>&1 | tee /workspace/send.log
sed -n 's/.*\(runpodctl receive [^ ]*\).*/RP_ARTIFACT_CODE=\1/p' /workspace/send.log
