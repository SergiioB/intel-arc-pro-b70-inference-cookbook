#!/usr/bin/env bash
# Nanbeige4.3-33B — one Arc Pro B70, llama.cpp SYCL, measured 2026-10-25.
#
# Usage:  MODEL=/data/models/nanbeige43-33b/Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf \
#         GPU=1 CTX=32768 PORT=8084 ./llama-server-sycl.sh
#
# What the flags do (all measured or closed by measurement, see the recipe page):
#   -fa on -ctk q8_0 -ctv q4_1   fleet KV; F16 KV measured the same speed, so keep capacity
#   -ub 2048                     +43 % pp8192 over -ub 512, decode unchanged
#   --jinja                      use the GGUF template; its markers select the Qwen3-Coder parser
#   --reasoning-format deepseek  thoughts in reasoning_content, not content
#   --reasoning-preserve         keep prior turns' reasoning (vendor: preserve_thinking=true
#                                for multi-turn tool use, office and code-agent work)
#   --temp/--top-p               vendor agentic sampling; override TEMP= and TOPP= for a chat profile
#                                (T=0.6) or pass temperature=0 per request for benchmark cells
#   GGML_SYCL_ENABLE_GRAPH=0     SYCL graphs measured ~±1 % on dense models; declared off
#
# Optional decode lever (measured +73 % on repeated code, neutral-to-+10 % elsewhere):
#   SPEC="--spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 7"
#   BUDGET: add --reasoning-budget 1500 (or REASONING=off) when emitting code into a fixed
#   max_tokens: unbounded thinking can consume the whole answer budget and return nothing.
# Keep size_m at 7: size_m 8 makes a 9-wide verify batch and is 31 % slower than plain.
set -euo pipefail

: "${MODEL:?set MODEL to the Q4_K_M gguf path}"
: "${GPU:=1}"          # the card WITHOUT the display
: "${CTX:=32768}"
: "${PORT:=8084}"
: "${BIN:=${HOME}/llama.cpp/build-sycl/bin/llama-server}"
: "${SPEC:=}"           # see the optional block in the header
# Vendor card (Nanbeige/Nanbeige4.3-33B): agentic/tool use T=1.0 top_p=0.95 max_new 65536;
# reasoning/chat T=0.6 top_p=0.95 max_new 262144. These are the agentic defaults.
: "${TEMP:=1.0}"
: "${TOPP:=0.95}"

pgrep -af 'llama-server|llama-bench' | grep -v pgrep || true
# setvars.sh is not `set -u` safe and exits non-zero when already initialised
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -u

export ZE_AFFINITY_MASK="${GPU}"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export GGML_SYCL_ENABLE_OPT=1
export GGML_SYCL_PRIORITIZE_DMMV=0
export GGML_SYCL_ENABLE_ESIMD=1
export GGML_SYCL_ENABLE_GRAPH=0
unset SYCL_CACHE_PERSISTENT

exec "${BIN}" -m "${MODEL}" \
  --device SYCL0 --split-mode none -ngl 99 \
  -fa on -ctk q8_0 -ctv q4_1 \
  -c "${CTX}" -b 2048 -ub 2048 -t 8 -tb 8 -np 1 \
  --mmap --jinja --reasoning-format deepseek \
  --temp "${TEMP}" --top-p "${TOPP}" --reasoning-preserve \
  ${SPEC} \
  --host 127.0.0.1 --port "${PORT}" --no-webui
