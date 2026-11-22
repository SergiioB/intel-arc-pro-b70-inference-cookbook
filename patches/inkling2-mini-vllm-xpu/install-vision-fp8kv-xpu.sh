#!/usr/bin/env bash
set -euo pipefail

# Install XPU Vision Tower (23-layer ViT + bidirectional SDPA) and FP8 E4M3 KV
# cache support (--kv-cache-dtype fp8) into an existing vllm-inkling2-xpu venv.
#
# Usage:
#   VENV=/path/to/vllm-inkling-venv ./install-vision-fp8kv-xpu.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VENV:+${VENV}/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PLUGIN_DIR="$("$PYTHON_BIN" -c 'import vllm_inkling2_xpu, pathlib; print(pathlib.Path(vllm_inkling2_xpu.__file__).resolve().parent)')"

cp "$SCRIPT_DIR/vision.py" "$PLUGIN_DIR/vision.py"
cp "$SCRIPT_DIR/ops_vision.py" "$PLUGIN_DIR/ops/vision.py"

# Break uv hardlinks before patching so shared caches stay untouched
for rel in model.py attention.py dspark.py ops/attention_prefill.py; do
  f="$PLUGIN_DIR/$rel"
  cp "$f" "${f}.bak" && rm -f "$f" && cp "${f}.bak" "$f" && rm -f "${f}.bak"
done

if ! grep -q "_FP8_PREFILL_SCRATCH" "$PLUGIN_DIR/attention.py"; then
  patch -d "$(dirname "$PLUGIN_DIR")" -p1 < "$SCRIPT_DIR/vllm-inkling2-xpu-vision-fp8kv.patch"
fi

echo "Installed Inkling 2 Mini XPU Vision Tower + FP8 KV into: $PLUGIN_DIR"
echo "  - Enable Vision: omit --language-model-only and pass --limit-mm-per-prompt '{\"image\":4}'"
echo "  - Enable FP8 KV: pass --kv-cache-dtype fp8"
