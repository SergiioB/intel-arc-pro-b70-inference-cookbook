#!/usr/bin/env bash
set -euo pipefail

# Build and install the opt-in AOT bmg_g31 SYCL M=1 Decode Attention extension
# (_inkling_attn_xe2) into an existing vllm-inkling2-xpu Python environment.
# Zero external header dependencies beyond Intel oneAPI icpx + PyTorch XPU!
#
# Usage:
#   VENV=/path/to/vllm-inkling-venv ./build-sycl-decode-attn-xe2.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VENV:+${VENV}/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AOT_TARGET="${AOT_TARGET:-bmg_g31}"
BUILD_DIR="${BUILD_DIR:-/tmp/inkling-attn-xe2-build}"
mkdir -p "$BUILD_DIR"

PY_INC="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
EXT_SUFFIX="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
TORCH_DIR="$("$PYTHON_BIN" -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent)')"
PLUGIN_DIR="$("$PYTHON_BIN" -c 'import vllm_inkling2_xpu, pathlib; print(pathlib.Path(vllm_inkling2_xpu.__file__).resolve().parent)')"
ABI_FLAG="$("$PYTHON_BIN" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
OUT_SO="$BUILD_DIR/_inkling_attn_xe2${EXT_SUFFIX}"

icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device ${AOT_TARGET} -options -ze-opt-large-register-file" \
  -O3 -std=c++20 -fPIC -shared \
  -D_GLIBCXX_USE_CXX11_ABI="${ABI_FLAG}" \
  -DTORCH_EXTENSION_NAME=_inkling_attn_xe2 \
  -I"$PY_INC" -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  "$SCRIPT_DIR/attn_decode_xe2.cpp" \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_xpu -ltorch_xpu \
  -o "$OUT_SO"

cp "$OUT_SO" "$PLUGIN_DIR/ops/_inkling_attn_xe2${EXT_SUFFIX}"

f="$PLUGIN_DIR/ops/attention.py"
cp "$f" "${f}.bak" && rm -f "$f" && cp "${f}.bak" "$f" && rm -f "${f}.bak"
if ! grep -q "INK2_SYCL_DECODE_ATTN" "$f"; then
  patch -d "$(dirname "$PLUGIN_DIR")" -p1 < "$SCRIPT_DIR/vllm-inkling2-xpu-sycl-decode-optin.patch"
fi

echo "Installed AOT bmg_g31 SYCL M=1 Decode Attention to: $PLUGIN_DIR/ops/_inkling_attn_xe2${EXT_SUFFIX}"
echo "Enable at runtime with: INK2_SYCL_DECODE_ATTN=1  (default 0 preserves exact stock Triton)"
