#!/usr/bin/env bash
# Pod-side QAT job. Started by `rtod job qat` through rp, so the pod's dead-man's-switch is already armed.
# Observability contract (the driver-side monitor in rtod.jobs reads these markers):
#   "RTOD phase=<name> ..."   a phase started          "RTOD_HB ..."  heartbeat every 30 s (liveness + GPU/RAM/disk)
#   "TRAIN ..." / "DATA ..."  progress from the python  "RTOD_FAIL ..." / "RTOD_FATAL ..." failure, with a log tail
#   "RP_ARTIFACT_CODE=..."    artifact ready to receive "RTOD_DONE"     success
set -Eeuo pipefail
: "${RTOD_SHA:?}" ; QAT_STEPS="${QAT_STEPS:-1250}" ; W=${RTOD_WORKDIR:-/workspace} ; OUT=$W/qat_out ; LOGS=$W/logs
mkdir -p "$OUT" "$LOGS" ; START=$(date +%s) ; PHASE=boot

say()   { printf 'RTOD t+%04ds %s\n' $(( $(date +%s) - START )) "$*" ; }
phase() { PHASE=$1 ; echo "$1" > "$W/.phase" ; say "phase=$1 ${2:-}" ; }
on_err() { local rc=$? ; say "RTOD_FAIL phase=$PHASE line=${BASH_LINENO[0]} exit=$rc cmd=$BASH_COMMAND" ; say "last log lines:" ; tail -n 40 "$LOGS"/*.log 2>/dev/null || true ; exit "$rc" ; }
trap on_err ERR

heartbeat() { while sleep 30 ; do
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo na)
  printf 'RTOD_HB t+%04ds phase=%s gpu(util%%,memMiB,totMiB,C)=%s ram_free_gb=%s disk_free_gb=%s load=%s\n' $(( $(date +%s) - START )) "$(cat "$W/.phase" 2>/dev/null || echo ?)" "$gpu" \
    "$(free -g | awk '/Mem:/{print $7}')" "$(df -BG --output=avail $W | tail -1 | tr -d ' G')" "$(cut -d' ' -f1 /proc/loadavg)"
done ; }
heartbeat & HB=$! ; trap 'kill $HB 2>/dev/null || true' EXIT

phase boot "host=$(hostname) cpus=$(nproc) ram_gb=$(free -g | awk '/Mem:/{print $2}')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { say "RTOD_FATAL phase=boot no nvidia-smi: not a GPU pod" ; exit 3 ; }
df -h $W | tail -1 ; python3 --version

phase install "sha=$RTOD_SHA"
if [ -z "${RTOD_SKIP_INSTALL:-}" ]; then
  pip install -q "rtod[torch,onnx,coco,plot] @ git+https://github.com/leejianrong/real-time-object-detection@${RTOD_SHA}" 2>&1 | tee "$LOGS/install.log"
fi

phase preflight
python3 - <<'PY' 2>&1 | tee "$LOGS/preflight.log"
import sys, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    print("RTOD_FATAL phase=preflight CUDA is not available to torch", flush=True); sys.exit(3)
x = torch.randn(2048, 2048, device="cuda"); torch.cuda.synchronize()
print("gpu matmul ok:", (x @ x).sum().item() == (x @ x).sum().item(), torch.cuda.get_device_name(0), flush=True)
PY

phase smoke "3 steps on 16 val images; a failure here costs minutes, not the whole run"
rtod qat --subset val200 --steps 3 --batch 4 --workers 2 --device cuda --out "$W/smoke" 2>&1 | tee "$LOGS/smoke.log"
test -s "$W/smoke/yolox_tiny_qat_int8.onnx"

publish() {  # publish() <code-base> <tarball>: the sender must stay up until the driver has received it
  runpodctl send --code "$1" "$2" > "$LOGS/send_$1.log" 2>&1 &
  local pid=$! code=""
  for _ in $(seq 30) ; do code=$(grep -oE "$1[-0-9]+" "$LOGS/send_$1.log" | head -1 || true) ; [ -n "$code" ] && break ; sleep 1 ; done
  say "RTOD_ARTIFACT $2 sha256=$(sha256sum "$2" | cut -d' ' -f1) bytes=$(stat -c %s "$2")"
  say "RP_ARTIFACT_CODE=$code"
  wait $pid
}

# One config per artifact, published as soon as it finishes so a later failure cannot lose it.
# QAT_CONFIGS is "name:flags,name:flags"; unset means a single config built from QAT_LR/QAT_CLIP.
CONFIGS="${QAT_CONFIGS:-run:}"
n=0
for cfg in $(echo "$CONFIGS" | tr ',' '\n' | tr ' ' '\001') ; do
  cfg=$(echo "$cfg" | tr '\001' ' ') ; name="${cfg%%:*}" ; flags="${cfg#*:}" ; n=$((n+1))
  OUT=$W/runs/$name ; mkdir -p "$OUT"
  phase train "config=$name flags='$flags' steps=$QAT_STEPS lr=${QAT_LR:-1e-5} clip=${QAT_CLIP:-100}"
  # shellcheck disable=SC2086  (flags are intentionally word-split)
  rtod qat --subset train10k --steps "$QAT_STEPS" --lr "${QAT_LR:-1e-5}" --clip "${QAT_CLIP:-100}" \
    --seed 0 $flags --device cuda --no-ptq --out "$OUT" 2>&1 | tee "$LOGS/train_$name.log"
  test -s "$OUT/yolox_tiny_qat_folded.onnx"
  rtod plot "$OUT/metrics.jsonl" "$OUT/training.png" || say "plot failed (non-fatal)"
  cp "$LOGS/train_$name.log" "$OUT/"
  phase send "config=$name -> artifact $n"
  tar -C "$W/runs" -czf "$W/qat_$name.tgz" "$name"
  publish "rtodqat$n" "$W/qat_$name.tgz"
done
phase done ; say "RTOD_DONE"
