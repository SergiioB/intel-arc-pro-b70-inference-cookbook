#!/usr/bin/env bash
set -euo pipefail

# Build and install the opt-in Native NVFP4 CUTLASS Xe2 (AOT bmg_g31) SYCL extension
# into an existing vLLM XPU Python environment with vllm-inkling2-xpu installed.
#
# Prerequisites:
#   - Intel oneAPI icpx (2025.x+) on PATH (source /opt/intel/oneapi/setvars.sh)
#   - Python venv with PyTorch XPU, vLLM XPU, vllm-xpu-kernels, and vllm-inkling2-xpu
#   - Local clones of intel/sycl-tla (@7b56b936b87f799f80dde60c2fd8d11c1128cebb) and
#     vllm-project/vllm-xpu-kernels (@e9b6e4b24a69886a8b384edc06e7a631ad282705)
#
# Usage:
#   VENV=/path/to/vllm-venv \
#   SYCL_TLA_DIR=/path/to/sycl-tla \
#   VLLM_XPU_KERNELS_DIR=/path/to/vllm-xpu-kernels \
#   ./build-nvfp4-cutlass-xe2.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VENV:+${VENV}/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SYCL_TLA_DIR="${SYCL_TLA_DIR:?Set SYCL_TLA_DIR to a checkout of intel/sycl-tla@7b56b936}"
VLLM_XPU_KERNELS_DIR="${VLLM_XPU_KERNELS_DIR:?Set VLLM_XPU_KERNELS_DIR to a checkout of vllm-project/vllm-xpu-kernels@e9b6e4b2}"
AOT_TARGET="${AOT_TARGET:-bmg_g31}"
BUILD_DIR="${BUILD_DIR:-/tmp/inkling-nvfp4-xe2-build}"

mkdir -p "$BUILD_DIR"

PY_INC="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
EXT_SUFFIX="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
TORCH_DIR="$("$PYTHON_BIN" -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent)')"
PLUGIN_DIR="$("$PYTHON_BIN" -c 'import vllm_inkling2_xpu, pathlib; print(pathlib.Path(vllm_inkling2_xpu.__file__).resolve().parent)')"
ABI_FLAG="$("$PYTHON_BIN" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"

OUT_SO="$BUILD_DIR/_inkling_nvfp4_xe2${EXT_SUFFIX}"

icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device ${AOT_TARGET} -options -ze-opt-large-register-file" \
  -O3 -std=c++20 -fPIC -shared \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED \
  -D_GLIBCXX_USE_CXX11_ABI="${ABI_FLAG}" \
  -I"$PY_INC" \
  -I"$TORCH_DIR/include" \
  -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -I"$SYCL_TLA_DIR/include" \
  -I"$VLLM_XPU_KERNELS_DIR/csrc/xpu/cutlass_kernels" \
  -I"$VLLM_XPU_KERNELS_DIR/csrc" \
  "$SCRIPT_DIR/nvfp4_xe2_ext.cpp" \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_xpu -ltorch_xpu \
  -o "$OUT_SO"

cp "$OUT_SO" "$PLUGIN_DIR/ops/_inkling_nvfp4_xe2${EXT_SUFFIX}"
cp "$SCRIPT_DIR/moe_cutlass.py" "$PLUGIN_DIR/ops/moe_cutlass.py"

if ! grep -q "INK2_CUTLASS_MOE" "$PLUGIN_DIR/moe.py"; then
  patch -d "$(dirname "$PLUGIN_DIR")" -p1 < "$SCRIPT_DIR/vllm-inkling2-xpu-cutlass-optin.patch"
fi

echo "Installed Native NVFP4 CUTLASS Xe2 extension to: $PLUGIN_DIR/ops/_inkling_nvfp4_xe2${EXT_SUFFIX}"
echo "Enable at runtime with: INK2_CUTLASS_MOE=1  (default 0 preserves exact stock plugin)"
