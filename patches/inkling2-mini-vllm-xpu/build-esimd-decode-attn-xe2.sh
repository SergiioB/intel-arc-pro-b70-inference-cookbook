#!/usr/bin/env bash
set -euo pipefail

# Build and install the opt-in AOT bmg_g31 ESIMD XMX Paged Decode Attention
# extension (_inkling_esimd_attn_xe2) into an existing vllm-inkling2-xpu
# Python environment. Zero external header dependencies beyond Intel oneAPI
# icpx + PyTorch XPU; the spir64_gen device target also needs ocloc on PATH.
#
# Usage:
#   VENV=/path/to/vllm-inkling-venv ./build-esimd-decode-attn-xe2.sh
#   INSTALL=0 BUILD_DIR=/tmp/esimd-build VENV=... ./build-esimd-decode-attn-xe2.sh  # build only

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VENV:+${VENV}/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AOT_TARGET="${AOT_TARGET:-bmg_g31}"
BUILD_DIR="${BUILD_DIR:-/tmp/inkling-esimd-attn-xe2-build}"
INSTALL="${INSTALL:-1}"
mkdir -p "$BUILD_DIR"

command -v icpx  >/dev/null || { echo "error: icpx not on PATH (source /opt/intel/oneapi/setvars.sh)" >&2; exit 1; }
command -v ocloc >/dev/null || { echo "error: ocloc not on PATH (install the intel-ocloc package matching your compute-runtime)" >&2; exit 1; }

PY_INC="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
EXT_SUFFIX="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
TORCH_DIR="$("$PYTHON_BIN" -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent)')"
PLUGIN_DIR="$("$PYTHON_BIN" -c 'import vllm_inkling2_xpu, pathlib; print(pathlib.Path(vllm_inkling2_xpu.__file__).resolve().parent)')"
ABI_FLAG="$("$PYTHON_BIN" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
OUT_SO="$BUILD_DIR/_inkling_esimd_attn_xe2${EXT_SUFFIX}"

icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device ${AOT_TARGET} -options -ze-opt-large-register-file" \
  -O3 -std=c++20 -fPIC -shared \
  -D_GLIBCXX_USE_CXX11_ABI="${ABI_FLAG}" \
  -DTORCH_EXTENSION_NAME=_inkling_esimd_attn_xe2 \
  -I"$PY_INC" -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  "$SCRIPT_DIR/attn_esimd_xe2.cpp" \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_xpu -ltorch_xpu \
  -o "$OUT_SO"

if [[ "$INSTALL" != "1" ]]; then
  echo "Built $OUT_SO (INSTALL=0: not copied into the plugin, no patch applied)"
  exit 0
fi

cp "$OUT_SO" "$PLUGIN_DIR/ops/_inkling_esimd_attn_xe2${EXT_SUFFIX}"

for rel in attention.py ops/attention.py; do
  f="$PLUGIN_DIR/$rel"
  cp "$f" "${f}.bak" && rm -f "$f" && cp "${f}.bak" "$f" && rm -f "${f}.bak"
done
if ! grep -q "INK2_ESIMD_DECODE_ATTN" "$PLUGIN_DIR/ops/attention.py"; then
  patch -d "$(dirname "$PLUGIN_DIR")" -p1 < "$SCRIPT_DIR/inkling2-hybrid-sdpa-esimd.patch"
fi

echo "Installed AOT bmg_g31 ESIMD XMX Paged Decode Attention to: $PLUGIN_DIR/ops/_inkling_esimd_attn_xe2${EXT_SUFFIX}"
echo "Enable at runtime with: INK2_ESIMD_DECODE_ATTN=1  (default 0 preserves exact stock Triton)"
echo "The same patch also adds INK2_HYBRID_PREFIX_SDPA=1 chunked oneDNN prefix SDPA (default 0)."
