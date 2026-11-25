#!/usr/bin/env bash
# Serve Inkling 2 Mini with the TensorFold XPU research build (tag b70-inkling2-mini-20261121) on one Arc Pro B70.
#
#   TF=<dir with tf/ and out/ from build-tensorfold-xpu.sh> MODEL=<dir>/Inkling-2-Mini-UD-Q4_K_XL.gguf \
#   META=<dir with tokenizer.json and chat_template.jinja> benchmarks/inkling2-mini/launch-inkling2-mini-tensorfold.sh
#
# Research build: numbers will change; not upstream TensorFold. No INK2_* variable is set: the measured cell used
# the defaults of commit e47dc7ae8. Needs Python 3 with transformers (measured: 5.15.0, tokenizers 0.22.2).
set -euo pipefail
: "${TF:?build directory}"
: "${MODEL:?path to Inkling-2-Mini-UD-Q4_K_XL.gguf}"
: "${META:?directory with the release tokenizer.json and chat_template.jinja}"
GPU=${GPU:-0}
PORT=${PORT:-18452}
CTX=${CTX:-16384}   # the measured cell; prompt + reply must fit
export ZE_AFFINITY_MASK=$GPU
exec python3 "$TF/tf/tools/xpu/inkling2_serve.py" --runner "$TF/out/bin/xpu-inkling2" --model "$MODEL" \
  --meta "$META" --port "$PORT" --ctx "$CTX"
