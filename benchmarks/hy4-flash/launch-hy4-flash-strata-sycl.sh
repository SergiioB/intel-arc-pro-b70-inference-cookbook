#!/usr/bin/env bash
# Serve Hy4-Flash native .strata packs on two Intel Arc Pro B70s with strata-hy4-sycl
# (Niko1221/Strata v0.2.20 + strata-hy4-b70-opt.patch).
#
# Usage:
#   STRATA_BIN=/path/to/strata-hy4-sycl \
#   NATIVE_PACK=/path/to/Hy4-Flash-SQ-64.strata \
#   ENGRAM_PACK=/path/to/Hy4-Flash-Engram-Q8.strata \
#     benchmarks/hy4-flash/launch-hy4-flash-strata-sycl.sh
#
# Verified dual-B70 operating shapes (override via env):
#   1. Symmetric serial (17 GiB each card):   PEER_VRAM_EXP=17 DRAFT=off
#   2. Asymmetric serial (default below):    PEER_VRAM_EXP=25 DRAFT=off   # +7-9% decode, +9% prefill
#   3. Asymmetric + drafter (opt-in):        PEER_VRAM_EXP=25 DRAFT=dfly2 # 2.2x decode @28K; parity unverifiable
#   4. Same-binary stock control:            HY4S_PEER_PF_LEGACY=1 HY4S_PEER_DEC_SG32=1 PEER_VRAM_EXP=-1
#
# VRAM floors measured at 32,768 ctx: peer card >= 4.6 GiB free at PEER_VRAM_EXP=25;
# primary card ~1.3 GiB free at VRAM_EXP=17. Do NOT raise VRAM_EXP to 19+ (transient 0 MiB).
# Contexts deeper than 32,768 are configurable via CTX but were not measured in this
# generation; KV stays device-resident while ctx <= KV_HOST_ABOVE.
set -euo pipefail

: "${STRATA_BIN:?path to patched strata-hy4-sycl binary (build_hy4_sycl.sh bmg)}"
: "${NATIVE_PACK:?path to Hy4-Flash-SQ-64.strata (or another native .strata pack)}"
: "${ENGRAM_PACK:?path to Hy4-Flash-Engram-Q8.strata (mandatory EngramShard profile)}"

GPUS="${GPUS:-1,0}"                  # engine GPU order: primary,peer (PCI-id order via ZE_ENABLE_PCI_ID_DEVICE_ORDER)
VRAM_EXP="${VRAM_EXP:-17}"           # GiB of experts on the primary card; keep <= 17
PEER_VRAM_EXP="${PEER_VRAM_EXP:-25}" # GiB of experts on the peer card; -1 = fall back to VRAM_EXP (stock behavior)
RAM_EXP="${RAM_EXP:-36}"             # GiB expert arena in host RAM
CTX="${CTX:-32768}"
DRAFT="${DRAFT:-off}"                # off | dfly2 (opt-in; parity unverifiable)
THREADS="${THREADS:-8}"
KV_HOST_ABOVE="${KV_HOST_ABOVE:-131072}"
MEM_CAP_GB="${MEM_CAP_GB:-52}"       # host RAM safety scope; 0 disables the systemd wrapper

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"

# The SYCL Level Zero adapter needs libumf.so.1 plus compiler/MKL runtime libs.
if [[ -n "${ONEAPI_ROOT:-}" ]]; then
  export LD_LIBRARY_PATH="${ONEAPI_ROOT}/compiler/latest/lib:${ONEAPI_ROOT}/umf/latest/lib:${ONEAPI_ROOT}/mkl/latest/lib:${LD_LIBRARY_PATH:-}"
fi

ARGS=(
  --serve
  --native "$NATIVE_PACK"
  --engram "$ENGRAM_PACK"
  --max-context "$CTX"
  --draft "$DRAFT"
  --threads "$THREADS"
  --gpus "$GPUS"
  --vram-experts-gb "$VRAM_EXP"
  --ram-experts-gb "$RAM_EXP"
  --kv-host-above "$KV_HOST_ABOVE"
)
if [[ "$PEER_VRAM_EXP" != "-1" ]]; then
  ARGS+=(--peer-vram-experts-gb "$PEER_VRAM_EXP")
fi
if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

if [[ "$MEM_CAP_GB" != "0" ]] && command -v systemd-run >/dev/null 2>&1; then
  exec systemd-run --user --scope -p "MemoryMax=${MEM_CAP_GB}G" -p MemorySwapMax=0 \
    -- "$STRATA_BIN" "${ARGS[@]}"
fi
exec "$STRATA_BIN" "${ARGS[@]}"
