#!/bin/bash
# Bolt 74B-A9B W4A16 export runner: waits until every BF16 shard is on disk, checks
# that torch sees both B70 cards, runs the converter on both cards, then judges the
# result from the log and the output (never from the exit code alone).
#
# usage:
#   SRC=/models/bolt74b-bf16 OUT=/models/bolt74b-autoround-w4a16 \
#   CALIB_CORPUS=/path/to/corpus PY=/path/to/venv/bin/python \
#   setsid nohup ./run_bolt_export.sh > run_bolt_export.log 2>&1 < /dev/null & disown
#
# env (defaults in brackets):
#   WORK          scratch dir for logs and the AutoRound workspace [directory of this script]
#   SRC           BF16 source checkpoint (required)
#   OUT           export directory; the artifact lands in $OUT/<src-name>-w4g128 (required)
#   CALIB_CORPUS  directory of *.txt.gz calibration text (required)
#   PY            python of the export venv [python3]
#   LOGF          converter log [$WORK/export_logs/bolt-w4a16.log]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HERE}
SRC=${SRC:?set SRC to the BF16 checkpoint directory}
OUT=${OUT:?set OUT to the export directory}
CALIB_CORPUS=${CALIB_CORPUS:?set CALIB_CORPUS to a directory of *.txt.gz files}
VENV=${PY:-python3}
LOGF=${LOGF:-$WORK/export_logs/bolt-w4a16.log}
export WORK SRC CALIB_CORPUS
mkdir -p "$(dirname "$LOGF")"
# Bolt's remote modeling code hard-imports mamba_ssm's rmsnorm_fn (CUDA-only
# package). shims/ provides a bitwise-checked pure-torch rmsnorm_fn; the CUDA
# fast path stays off (is_mamba_2_ssm_available() is False), so torch_forward runs.
export PYTHONPATH=$HERE/shims${PYTHONPATH:+:$PYTHONPATH}

while true; do
  if [ -f "$SRC/model.safetensors.index.json" ]; then
    need=$($VENV - <<'PY'
import json, os
src = os.environ["SRC"]
wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
sh = sorted({v for v in wm.values() if v.endswith(".safetensors")})
missing = [s for s in sh if not os.path.exists(os.path.join(src, s))]
print(len(missing))
PY
)
    echo "[bolt-runner $(date -u +%H:%M:%S)] shards missing: $need"
    [ "$need" = "0" ] && break
  else
    echo "[bolt-runner $(date -u +%H:%M:%S)] index.json not present yet"
  fi
  sleep 120
done

# Both cards: AutoRound spreads each block over xpu:0 + xpu:1 (device_map "0,1").
# One card is not enough: layer 21 (432 experts x 2560) hit driver ENOMEM ->
# DEVICE_LOST, and 9 MoE layers are that big or bigger.
# Do NOT set ZE_AFFINITY_MASK together with ONEAPI_DEVICE_SELECTOR (renumbering
# made torch see zero devices and silently run on CPU).
unset ZE_AFFINITY_MASK
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
$VENV -c "import torch,sys; sys.exit(0 if torch.xpu.device_count()==2 else 1)" 2>/dev/null \
  || { echo "[bolt-runner $(date -u +%H:%M:%S)] FATAL: both cards not visible to torch.xpu"; exit 4; }
echo "[bolt-runner $(date -u +%H:%M:%S)] starting Bolt W4A16 (gpus 0,1)"
flock /tmp/b70-gpu0.lock flock /tmp/b70-gpu1.lock -c "\
  $VENV $HERE/convert_nemotron_omni.py --src $SRC --bits 4 --out $OUT --corpus $CALIB_CORPUS \
  --iters 200 --nsamples 128 --seqlen 1536 --batch-size 1 --device 0,1" \
  > "$LOGF" 2>&1
rc=$?
size_gb=$(( $(du -sb "$OUT" 2>/dev/null | cut -f1) / 1000000000 ))
status=ok
[ "$rc" = 0 ] || status="python rc=$rc"
grep -q -E 'Traceback|DEVICE_LOST|OUT_OF_RESOURCES' "$LOGF" && status="error in log"
grep -q 'Summary: quantized' "$LOGF" || status="no quantization summary"
[ "$size_gb" -ge 30 ] || status="$status; artifact only ${size_gb} GB"
find "$OUT" -maxdepth 2 -name config.json | grep -q . || status="$status; no config.json"
echo "[bolt-runner $(date -u +%H:%M:%S)] Bolt W4A16 status=$status size=${size_gb}GB"
# No AutoRound verify_artifact.py here: it misses the language_model.backbone alias and
# would re-copy every LM tensor in BF16 (OOM on a 60 GB host). The artifact lands in
# $OUT/<src-name>-w4g128 and is checked by check_artifact.py.
echo "[bolt-runner $(date -u +%H:%M:%S)] done"
[ "$status" = ok ]
