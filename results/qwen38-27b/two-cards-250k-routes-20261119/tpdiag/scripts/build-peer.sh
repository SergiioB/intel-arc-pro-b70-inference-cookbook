#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
src=$(cd "$(dirname "$0")" && pwd)
cd "$work"
ocloc compile -file "$src/peer.cl" -device bmg-g31 -options '-cl-std=CL2.0' -output tpdiag-peer
g++ --version | sed -n '1p'
g++ -std=c++20 -O3 -Wall -Wextra -Werror "$src/peer.cpp" -lze_loader -o tpdiag-ze
find "$work" -maxdepth 1 -name 'tpdiag-peer*.bin' -print
sha256sum tpdiag-ze tpdiag-peer*.bin "$src"/peer.{cpp,cl}
mkdir -p isa
ocloc disasm -file "$(find . -maxdepth 1 -name 'tpdiag-peer*.bin' -print -quit)" -device bmg-g31 -dump isa
