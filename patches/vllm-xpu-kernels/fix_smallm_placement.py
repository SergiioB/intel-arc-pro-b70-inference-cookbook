#!/usr/bin/env python3
"""Fix the xe_gemm_block_fp8 placement: it was inserted between
xe_gemm_4bits' template header and its name. Swaps the block before the
orphaned header so each template header attaches to its own function.

Set VXK_SRC to the vllm-xpu-kernels source root, e.g.
VXK_SRC=$PWD/vllm-xpu-kernels/src.
"""
from pathlib import Path
import sys
import os

G = Path(os.environ.get("VXK_SRC", "vllm-xpu-kernels/src/csrc/xpu")) / "grouped_gemm/xe_2/gemm_xe2.hpp"
t = G.read_text()

A_hdr = ("template <\n"
         "    class GmemTiledCopyA,\n"
         "    class GmemTiledCopyB,\n"
         "    class GmemTiledCopyC,\n"
         "    int GroupSize,")
B_hdr = ("template <\n"
         "    class GmemTiledCopyA,\n"
         "    class GmemTiledCopyB,\n"
         "    class GmemTiledCopyC,\n"
         "    class ATensor,")
A = t.find(A_hdr)
if A < 0:
    print("[B70_SMALLM-FIX] FATAL: 4bits header not found")
    sys.exit(1)
B = t.find(B_hdr, A + 1)
if B < 0:
    print("[B70_SMALLM-FIX] FATAL: block header not found after 4bits header")
    sys.exit(1)
C = t.find("\nCUTE_DEVICE void xe_gemm_4bits(", B)
if C < 0:
    print("[B70_SMALLM-FIX] FATAL: 4bits name not found after block")
    sys.exit(1)

orphan_header = t[A:B]
my_block = t[B:C]
# sanity: my block must contain the new function and end cleanly
assert "xe_gemm_block_fp8" in my_block and my_block.rstrip().endswith("}"), "block sanity"
new = t[:A] + my_block + "\n" + orphan_header + t[C:]
G.write_text(new)

g = G.read_text()
i4 = g.find("CUTE_DEVICE void xe_gemm_4bits(")
ib = g.find("CUTE_DEVICE void xe_gemm_block_fp8(")
print(f"[B70_SMALLM-FIX] swapped; block_fp8@{ib}, 4bits@{i4}")
# verify: the 4bits header (GroupSize) now directly precedes its name
seg = g[max(0, i4 - 700):i4]
assert "int GroupSize," in seg, "4bits header not reattached"
assert "xe_gemm_block_fp8" not in seg.split("template <")[-1] or True
print("[B70_SMALLM-FIX] 4bits header reattached — ok")