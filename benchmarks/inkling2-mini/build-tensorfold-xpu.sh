#!/usr/bin/env bash
# Build the Inkling 2 Mini TensorFold XPU runner from the public tag and check it against the measured binary (GPU-free).
#
#   ZIG=<zig 0.17.0> benchmarks/inkling2-mini/build-tensorfold-xpu.sh <new dir>
#
# Needs Zig 0.17.0, Intel's ocloc (measured with intel-ocloc 26.05.37020.3) and objcopy (binutils) on PATH. About 2 minutes.
# The whole-file sha256 differs from the measured binary because build paths land in the debug sections;
# the code (.text), constants with both kernel binaries (.rodata) and .data must match the hashes below.
set -euo pipefail
DIR=${1:?new directory}
mkdir -p "$DIR"; DIR=$(cd "$DIR" && pwd)   # absolute: the build runs inside $DIR/tf
ZIG=${ZIG:-zig}
TAG=b70-inkling2-mini-20261121
COMMIT=e47dc7ae8e06c9cdefb846e01cf4671bad6cb3db

git clone -q --depth 1 --branch "$TAG" https://github.com/SergiioB/TensorFold.git "$DIR/tf"
[ "$(git -C "$DIR/tf" rev-parse HEAD)" = "$COMMIT" ] || { echo "FAIL tag does not point at $COMMIT"; exit 1; }
( cd "$DIR/tf" && "$ZIG" build xpu-inkling2 -Doptimize=ReleaseFast --prefix "$DIR/out" )
BIN="$DIR/out/bin/xpu-inkling2"

declare -A WANT=(
  [.text]=f766626a2b07e6acdd08ce2f866796c8601627957f48098ea35dd019f2a250dd
  [.rodata]=fef235af923140e8c303b7ad1086abfa1c93f8af9b91f88341ada4cadba00348
  [.data]=80b19f955e72a2fbc2891bd5686d4180a2969036ed7a8004c59e153a34a0e5ee
)
rc=0
for s in .text .rodata .data; do
  got=$(objcopy -O binary --only-section="$s" "$BIN" /dev/stdout | sha256sum | cut -c1-64)
  if [ "$got" = "${WANT[$s]}" ]; then echo "OK   $s $got"; else echo "DIFF $s $got (measured ${WANT[$s]})"; rc=1; fi
done
[ $rc = 0 ] || echo "A different ocloc or Zig build changes these; the binary is then not the measured one."
exit $rc
