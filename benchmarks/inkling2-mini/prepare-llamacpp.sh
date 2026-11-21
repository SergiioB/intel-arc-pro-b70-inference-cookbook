#!/usr/bin/env bash
# Build the exact llama.cpp source tree of the Inkling 2 Mini recipe and prove it (GPU-free).
#
#   benchmarks/inkling2-mini/prepare-llamacpp.sh <llama.cpp clone> <new worktree dir>
#
# Steps: upstream master 919794500 -> merge PR #32695 head 6196d980 (looped layers) -> merge PR #32218
# head 78d77942 (DSpark drafter) -> git am the six local patches. After each stage the script compares the
# git tree hash with the tree the measured build used, and exits 1 on any difference.
# A matching tree proves source compatibility only: not runtime correctness, stability or speed.
set -euo pipefail

SRC=${1:?llama.cpp clone (origin = github.com/ggml-org/llama.cpp)}
DST=${2:?new worktree directory}
HERE=$(cd "$(dirname "$0")" && pwd)
PATCHES=$(cd "$HERE/../../patches/inkling2-mini-llamacpp" && pwd)
SRC=$(cd "$SRC" && pwd)

BASE=919794500475862f6768c353255469dc661cb734
PR_LOOP=6196d9802e0757a5218f16f91d9a6830793ba800    # ggml-org/llama.cpp PR #32695
PR_DSPARK=78d779424d83e17a450807fe42e28445510134b4  # ggml-org/llama.cpp PR #32218
TREE_MERGED=9998cc28d9e06bc342397065fd324d2d6c4589c9
TREE_PATCHED=b1d97b7bfd0f2b997c53db4a96f2b76f1dec4597

# Absolute target path: `git -C` would otherwise resolve it inside the clone.
mkdir -p "$(dirname "$DST")"
DST="$(cd "$(dirname "$DST")" && pwd)/$(basename "$DST")"
[ ! -e "$DST" ] || { echo "FAIL $DST exists"; exit 1; }

# Fetch every pinned commit by its SHA. A PR head moves (PR #32218 did on 2026-11-21), so fetching the PR ref
# alone can miss the pinned commit; GitHub serves a commit by SHA while any ref still reaches it.
have() { git -C "$SRC" cat-file -e "$1^{commit}" 2>/dev/null; }
pin() {  # pin <sha> <fallback ref>
  have "$1" || git -C "$SRC" fetch -q origin "$1" || git -C "$SRC" fetch -q origin "$2" || true
  have "$1" || { echo "FAIL commit $1 not fetchable (by SHA or through $2)"; exit 1; }
}
pin "$BASE" master
pin "$PR_LOOP" pull/32695/head
pin "$PR_DSPARK" pull/32218/head

git -C "$SRC" worktree add --detach "$DST" "$BASE"
cd "$DST"
ID=(-c user.name=recipe -c user.email=recipe@localhost)
git "${ID[@]}" merge --no-edit -q "$PR_LOOP"
git "${ID[@]}" merge --no-edit -q "$PR_DSPARK"

check() {
  local got
  got=$(git rev-parse 'HEAD^{tree}')
  if [ "$got" = "$2" ]; then echo "OK   $1 tree $got"; else echo "FAIL $1 tree $got, expected $2"; exit 1; fi
}
check "merged base" "$TREE_MERGED"
( cd "$PATCHES" && sha256sum -c SHA256SUMS )
git "${ID[@]}" am -q "$PATCHES"/000*.patch
check "patched (0001-0006)" "$TREE_PATCHED"
