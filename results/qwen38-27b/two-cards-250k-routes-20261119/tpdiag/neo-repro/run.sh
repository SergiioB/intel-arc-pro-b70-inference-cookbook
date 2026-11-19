#!/usr/bin/env bash
# Queue job (both cards, exclusive): build the reproducer, then run it 2 devices default / 2 devices with implicit
# counter-based events off / 1 device default / 2 devices direct submission off. Each run is bounded (alarm 30 s + timeout).
source ~/B70-DOCS/results/2026/11/b70-breakthrough-20261118/tp/tpdiag/scripts/common.sh
src=$(cd "$(dirname "$0")" && pwd)
out=$work/raw/repro; mkdir -p "$out"; cd "$out"
ocloc compile -file "$src/repro.cl" -device bmg-g31 -output repro > ocloc.log 2>&1
g++ -std=c++20 -O2 -Wall -Wextra -Wno-missing-field-initializers "$src/repro.cpp" -lze_loader -o repro
dpkg-query -W 'intel-opencl-icd*' 'libze-intel-gpu*' 'libze1*' 2>/dev/null
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
idle
neo=$work/neo-26.40/root   # compute-runtime 26.40.40196.8 + IGC 2.42.6, extracted side by side (not installed)
newlib=(LD_LIBRARY_PATH="$neo/usr/lib/x86_64-linux-gnu:$neo/usr/local/lib")
for v in 2-default 2-nocb 1-default 1-card1 2-nods 2-default-again 2-default-2640 2-nocb-2640; do
  case $v in
    2-default|2-default-again) e=(); args=(2) ;;
    2-nocb) e=(NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0); args=(2) ;;
    1-default) e=(); args=(1) ;;
    1-card1) e=(ZE_AFFINITY_MASK=1); args=(1) ;;
    2-nods) e=(NEOReadDebugKeys=1 EnableDirectSubmission=0); args=(2) ;;
    2-default-2640) e=("${newlib[@]}"); args=(2) ;;
    2-nocb-2640) e=("${newlib[@]}" NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0); args=(2) ;;
  esac
  rc=0; run_bounded 120 env "${e[@]}" ./repro repro_bmg.bin "${args[@]}" 3 128 > "$v.log" 2>&1 || rc=$?
  echo "$v rc=$rc $(tail -1 "$v.log")"
done
