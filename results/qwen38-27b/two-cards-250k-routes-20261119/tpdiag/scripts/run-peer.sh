#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cd "$work"
shopt -s nullglob
image=( tpdiag-peer*.bin )
(( ${#image[@]} == 1 ))
[[ tpdiag-ze -nt "$(dirname "$0")/peer.cpp" || tpdiag-ze -nt ~/B70-DOCS/results/2026/11/b70-breakthrough-20261118/tp/tpdiag/scripts/peer.cpp ]] || { echo "tpdiag-ze is older than peer.cpp: rebuild first"; exit 3; }
idle
cool
record
trap record EXIT
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export NEO_CACHE_DIR=$work/cache
unset ZE_AFFINITY_MASK
status=0
for test in "$@"; do
  # default = the runtime's own choice (what the engines run with); off/on force NEO direct submission.
  for mode in default off on; do
    case $mode in
      default) setup=() ;;
      off) setup=(NEOReadDebugKeys=1 EnableDirectSubmission=0) ;;
      on) setup=(NEOReadDebugKeys=1 EnableDirectSubmission=1 DirectSubmissionOverrideComputeSupport=1 DirectSubmissionOverrideBlitterSupport=1) ;;
    esac
    record
    if run_bounded 150 env "${setup[@]}" ./tpdiag-ze "${image[0]}" "$test" > "raw/$test-$mode.csv" 2> "raw/$test-$mode.log"; then
      echo "$test $mode pass"
    else
      rc=$?
      echo "$test $mode exit $rc"
      cat "raw/$test-$mode.log"
      status=$rc
    fi
    if [[ $test == host ]]; then
      run_bounded 60 strace -f -qq -e trace=ioctl -o "raw/host-$mode.ioctl" \
          env "${setup[@]}" ./tpdiag-ze "${image[0]}" host \
          > /dev/null 2> "raw/host-$mode-ioctl.log" || echo "ioctl diagnostic $mode exit $?"
    fi
  done
done
exit "$status"
