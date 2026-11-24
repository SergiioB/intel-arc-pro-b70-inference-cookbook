#!/usr/bin/env bash
set -euo pipefail

# Install the opt-in exact-tail single-slot prefix cache
# (INK2_RING_PREFIX_CACHE=1) into an existing vLLM XPU + vllm-inkling2-xpu
# environment.
#
# What it does:
#   1. Copies ring_prefix.py / prefix_tail.py / ring_prefix_worker.py into
#      the vllm_inkling2_xpu package.
#   2. Patches vllm/v1/core/kv_cache_manager.py (5 flag-gated RingPrefix
#      hooks: limit()/covers() around find_longest_cache_hit, scheduled()
#      in allocate_slots, finish() in free, hash reset in
#      reset_prefix_cache).
#   3. Patches vllm_inkling2_xpu/model.py (flag validation, worker
#      construction, forward() entry/exit calls, requires_raw_input_tokens
#      only under the flag).
#
# Every change is inert unless INK2_RING_PREFIX_CACHE=1 is exported at
# runtime (default 0 preserves the exact stock code path). The flag also
# requires --enable-prefix-caching and --max-num-seqs 1; the model raises a
# clear ValueError otherwise.
#
# The script runs an atomic `patch --dry-run --batch --forward` preflight on
# BOTH pending patches before modifying any file, then breaks `uv` hardlinks
# on the patch targets so the shared package cache is not modified. Running
# it a second time is a no-op.
#
# Usage:
#   VENV=/path/to/vllm-inkling-venv ./install-ring-prefix-cache.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VENV:+${VENV}/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PLUGIN_DIR="$("$PYTHON_BIN" -c 'import vllm_inkling2_xpu, pathlib; print(pathlib.Path(vllm_inkling2_xpu.__file__).resolve().parent)')"
SITE_DIR="$(dirname "$PLUGIN_DIR")"
MODEL="$PLUGIN_DIR/model.py"
KVMGR="$SITE_DIR/vllm/v1/core/kv_cache_manager.py"
KVMGR_PATCH="$SCRIPT_DIR/vllm-ring-prefix-cache-kvmanager.patch"
OPTIN_PATCH="$SCRIPT_DIR/vllm-inkling2-xpu-ring-prefix-optin.patch"

for f in "$MODEL" "$KVMGR"; do
  [[ -f "$f" ]] || { echo "error: expected file missing: $f" >&2; exit 1; }
done

NEED_MODEL=1
NEED_KVMGR=1
grep -q "ring_prefix_worker" "$MODEL" && NEED_MODEL=0
grep -q "inkling_history" "$KVMGR" && NEED_KVMGR=0

# Atomic preflight: dry-run every pending patch before touching any file.
if (( NEED_KVMGR )); then
  if ! patch --dry-run --batch --forward -d "$SITE_DIR" -p1 < "$KVMGR_PATCH" >/dev/null; then
    echo "error: vllm-ring-prefix-cache-kvmanager.patch dry-run failed — the pinned vLLM base" >&2
    echo "       (vllm-0.34.0rc1.dev1133+g647a76866.xpu) is required. No files were modified." >&2
    exit 1
  fi
fi
if (( NEED_MODEL )); then
  if ! patch --dry-run --batch --forward -d "$SITE_DIR" -p1 < "$OPTIN_PATCH" >/dev/null; then
    echo "error: vllm-inkling2-xpu-ring-prefix-optin.patch dry-run failed — the pinned plugin" >&2
    echo "       (vllm-inkling2-xpu@2e9d28b04ff23f5d9996a221ffaa0ae3a440d8f9) is required." >&2
    echo "       No files were modified." >&2
    exit 1
  fi
fi

# Copy the three mechanism modules. rm first so a pre-existing destination
# (e.g. a hardlinked wheel file) is replaced with a fresh inode.
for mod in ring_prefix.py prefix_tail.py ring_prefix_worker.py; do
  rm -f "$PLUGIN_DIR/$mod"
  cp "$SCRIPT_DIR/$mod" "$PLUGIN_DIR/$mod"
done

# Break `uv` hardlinks on the two patch targets: copy -> remove -> copy back
# yields a fresh inode so patching never writes through to the shared
# package cache.
for f in "$MODEL" "$KVMGR"; do
  cp "$f" "${f}.bak" && rm -f "$f" && cp "${f}.bak" "$f" && rm -f "${f}.bak"
done

if (( NEED_KVMGR )); then
  patch --batch --forward -d "$SITE_DIR" -p1 < "$KVMGR_PATCH"
fi
if (( NEED_MODEL )); then
  patch --batch --forward -d "$SITE_DIR" -p1 < "$OPTIN_PATCH"
fi

if (( NEED_MODEL == 0 && NEED_KVMGR == 0 )); then
  echo "Already installed: exact-tail ring prefix cache in $SITE_DIR (idempotent no-op)"
else
  echo "Installed exact-tail single-slot ring prefix cache into: $SITE_DIR"
fi
echo "Enable at runtime with: INK2_RING_PREFIX_CACHE=1 + --enable-prefix-caching --max-num-seqs 1"
echo "Default (unset/0) preserves the exact stock path; the launcher exports it via RING_PREFIX_CACHE=1."
