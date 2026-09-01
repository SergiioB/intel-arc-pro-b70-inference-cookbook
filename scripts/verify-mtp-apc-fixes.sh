#!/usr/bin/env bash
# verify-mtp-apc-fixes.sh -- dry-run of the hybrid MTP + prefix-caching
# correctness patches WITHOUT a GPU.
#
# Boots `docker run -i --rm --entrypoint bash` (no --device: CPU-only; the
# patches are pure text transforms on a disposable copy of the image), mounts
# this repo's patches/ read-only, optionally applies the rest of the Qwen
# apply-list first, then applies the three correctness fixes and checks:
#
#   1. every patch exits 0;
#   2. every touched file still compiles (python -m py_compile) in BOTH vllm
#      trees these images ship (/workspace/vllm/vllm and site-packages —
#      `vllm serve` imports site-packages while `python -c "import vllm"`
#      from the WORKDIR imports /workspace, so we verify both);
#   3. each fix is idempotent (a second apply changes no file);
#   4. content greps: markers present in both trees, exactly 3
#      backward-copy guards in mamba_utils.py (postprocess fused kernel +
#      precopy fused kernel + copy_meta), both eagle-drop hunks in
#      MambaManager's finder, and the accept-sync wait + gather hunks in
#      gpu_model_runner.py.
#
# Usage:
#   scripts/verify-mtp-apc-fixes.sh
#       # default image: Qwen3.8 champion digest; override with IMAGE=...
#   IMAGE=vllm/vllm-openai-xpu@sha256:2c427ef... scripts/verify-mtp-apc-fixes.sh
#   APPLY_SERGIO=0 scripts/verify-mtp-apc-fixes.sh   # only the three fixes
#
# Exit 0 = green; non-zero = a patch or invariant failed.

set -u
IMAGE=${IMAGE:-vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f}
APPLY_SERGIO=${APPLY_SERGIO:-0}   # 1 = also apply Sergio's full Qwen chain first
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PATCH_DIR="$REPO_ROOT/patches"

[[ -d "$PATCH_DIR" ]] || { echo "ERROR: run from a repo checkout (no $PATCH_DIR)" >&2; exit 2; }
for f in patch_fix_accepted_sync.py patch_fix_backward_copy.py patch_fix_eagle_drop.py; do
  [[ -f "$PATCH_DIR/$f" ]] || { echo "ERROR: missing patches/$f" >&2; exit 2; }
done

echo "image  : $IMAGE"
echo "patches: $PATCH_DIR"

docker run -i --rm \
  -e APPLY_SERGIO="$APPLY_SERGIO" \
  -e B70_MTP_BF16_DRAFT=1 -e B70_DRAFT_LMHEAD_INT4=1 -e B70_DRAFT_MTP_INT4=1 \
  -v "$PATCH_DIR:/patches:ro" \
  --entrypoint bash "$IMAGE" -s <<'INHEREDOC'
set -u
LIVE=/opt/venv/lib/python3.12/site-packages/vllm   # what `vllm serve` imports
ROOTS="$LIVE /workspace/vllm/vllm"
FIXES="patch_fix_backward_copy patch_fix_eagle_drop patch_fix_accepted_sync"
TOUCHED="v1/core/single_type_kv_cache_manager.py v1/worker/mamba_utils.py v1/worker/gpu_model_runner.py"

if [ "${APPLY_SERGIO:-1}" = "1" ]; then
  for p in patch_mtp_nightly patch_mtp_boundary patch_gdn_mixed_split_v5 \
           patch_draft_lmhead_int4 patch_draft_mtp_int4; do
    echo "== sergio apply-list: $p"
    python /patches/$p.py || { echo "FAIL (pre-existing chain) $p"; exit 1; }
  done
fi

for f in $FIXES; do
  echo "== $f"
  python /patches/$f.py >/tmp/$f.out 2>&1 || { tail -5 /tmp/$f.out; echo "FAIL apply $f"; exit 1; }
  tail -2 /tmp/$f.out
  sig1=$(for r in $ROOTS; do for t in $TOUCHED; do [ -f "$r/$t" ] && md5sum "$r/$t"; done; done | md5sum)
  python /patches/$f.py >/dev/null 2>&1 || { echo "FAIL reapply $f"; exit 1; }
  sig2=$(for r in $ROOTS; do for t in $TOUCHED; do [ -f "$r/$t" ] && md5sum "$r/$t"; done; done | md5sum)
  [ "$sig1" = "$sig2" ] || { echo "FAIL not idempotent: $f"; exit 1; }
  echo "   idempotent OK"
done

for r in $ROOTS; do
  [ -d "$r" ] || continue
  for pair in "v1/worker/gpu_model_runner.py B70_ACCEPT_SYNC" \
              "v1/worker/mamba_utils.py B70_BACKWARD_COPY_GUARD" \
              "v1/core/single_type_kv_cache_manager.py B70_EAGLE_DROP"; do
    set -- $pair
    [ -f "$r/$1" ] || continue
    grep -q "$2" "$r/$1" || { echo "FAIL marker $2 missing in $r/$1"; exit 1; }
    python -m py_compile "$r/$1" || { echo "FAIL py_compile $r/$1"; exit 1; }
  done
  if [ -f "$r/v1/worker/mamba_utils.py" ]; then
    n=$(grep -c "if src_block_idx > dest_block_idx:\|if src_col > dst_col:" "$r/v1/worker/mamba_utils.py")
    [ "$n" = "3" ] || { echo "FAIL backward guards $n/3 in $r"; exit 1; }
    echo "OK 3/3 backward-copy guards in $r/v1/worker/mamba_utils.py"
  fi
  if [ -f "$r/v1/core/single_type_kv_cache_manager.py" ]; then
    m=$(grep -c "B70_EAGLE_DROP_COARSE_HUNK\|B70_EAGLE_DROP_FINE_HUNK" "$r/v1/core/single_type_kv_cache_manager.py")
    [ "$m" = "2" ] || { echo "FAIL eagle-drop hunks $m/2 in $r"; exit 1; }
    echo "OK 2/2 eagle-drop hunks in $r (rama fina + gruesa)"
  fi
done
echo "ALL CHECKS PASSED"
INHEREDOC
