#!/usr/bin/env bash
# The kv256k campaign after run a: each line is its own GPU-lock hold. Usage: chain.sh (from the kv256k folder)
cd ~/B70-DOCS/results/2026/11/b70-breakthrough-20261118/wave2/kv256k
L=$WORK/gpu.lock
hold() { local name=$1; shift; echo "$(date +%T) queued $name"; flock $L "$@" > runs/$name.console 2>&1; echo "exit $?" >> runs/$name.console; echo "$(date +%T) done $name"; }
hold c-tp2-bf16-262k scripts/kv_run.sh runs/c-tp2-bf16-262k 2 bf16 262144 0
hold hold_w scripts/hold_w_warm.sh runs
hold hold_f scripts/hold_f_fresh.sh runs
hold hold_g scripts/hold_g_131k.sh runs
echo "$(date +%T) chain done"
