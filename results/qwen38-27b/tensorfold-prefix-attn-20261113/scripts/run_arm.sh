#!/usr/bin/env bash
# Usage: run_arm.sh NAME PREF(0|CH) DEPTHS REPS [GPU=1] [BINDIR=src-prefix]
set -euo pipefail
NAME=$1; PREF=$2; export DEPTHS=$3 REPS=$4; GPU=${5:-1}; SRC=${6:-src-prefix}
R=$RUN
export TF_BIN=$TF/$SRC/zig-out/xpu/bin/tensorfold-native
[ "$PREF" = 0 ] || export XPU_ATTN_PREF=$PREF
export CANARY=${CANARY-}
bash $BENCH/tf_depth_run.sh $R/$NAME $GPU
