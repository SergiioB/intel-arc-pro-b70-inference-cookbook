#!/usr/bin/env python3
"""FP8 hybrid-numerics patch for stock vLLM-XPU f01 (2026-08-30).

Collapse-mitigation experiment: the MTP head was trained against W8A8-stack
hidden states. Full-W8A16 changes prefill numerics and (draw-dependently)
collapses long-context acceptance to 0%. Hybrid mode routes:
  - decode/small-M (rows <= B70_FP8_SMALL_M_MAX, default 40) -> custom Xe2
    small-M op when available, else fp8_gemm_w8a16 (fast steps);
  - prefill/large-M -> W8A8 with inline dynamic quant (baseline numerics for
    KV/GDN state the head consumes).
Modes via B70_FP8_MODE env: "hybrid" (default), "w8a16", "w8a8".
Anchored to pristine f01 sources; fails loudly; idempotent.
"""
from pathlib import Path
import sys

SP = Path("/opt/venv/lib/python3.12/site-packages")
XPU = SP / "vllm/model_executor/kernels/linear/scaled_mm/xpu.py"
OPS = SP / "vllm/_xpu_ops.py"

CLASS_ANCHOR = (
    "class XPUFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):\n"
    "    @classmethod\n"
)
CLASS_FIXED = (
    "import os as _b70_os\n"
    "_B70_MODE = _b70_os.environ.get('B70_FP8_MODE', 'hybrid').strip().lower()\n"
    "_B70_HAS_XE2_SMALL_M = hasattr(torch.ops._xpu_C, 'xe2_block_fp8_small_m')\n"
    "_B70_SMALL_M_MAX = int(_b70_os.environ.get('B70_FP8_SMALL_M_MAX', '40'))\n"
    "class XPUFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):\n"
    "    apply_input_quant = _B70_MODE == 'w8a8'\n"
    "    print(f\"[B70_FP8_HYBRID] mode={_B70_MODE} "
    "apply_input_quant={apply_input_quant} "
    "xe2_small_m={_B70_HAS_XE2_SMALL_M}\", flush=True)\n"
    "    @classmethod\n"
)

CALL_ANCHOR = (
    "        return torch.ops._xpu_C.fp8_gemm(\n"
    "            A,\n"
    "            B.t(),\n"
    "            self.config.out_dtype,\n"
    "            As,\n"
    "            Bs.t(),\n"
    "            torch.Tensor(),\n"
    "        )\n"
)
CALL_FIXED = (
    "        M, K = A.shape\n"
    "        N = B.shape[0]\n"
    "        if _B70_MODE in ('hybrid', 'w8a16') and M <= _B70_SMALL_M_MAX:\n"
    "            if _B70_HAS_XE2_SMALL_M and K % 128 == 0 and N % 128 == 0:\n"
    "                return torch.ops._xpu_C.xe2_block_fp8_small_m(\n"
    "                    A, B.t(), Bs.t(), None)\n"
    "            return torch.ops._xpu_C.fp8_gemm_w8a16(\n"
    "                A, B.t(), Bs.t(), torch.Tensor())\n"
    "        if _B70_MODE == 'hybrid':\n"
    "            qA, qAs = self.quant_fp8(\n"
    "                A, None, None, use_triton=self.use_triton)\n"
    "            return torch.ops._xpu_C.fp8_gemm(\n"
    "                qA, B.t(), self.config.out_dtype, qAs, Bs.t(),\n"
    "                torch.Tensor())\n"
    "        return torch.ops._xpu_C.fp8_gemm(\n"
    "            A,\n"
    "            B.t(),\n"
    "            self.config.out_dtype,\n"
    "            As,\n"
    "            Bs.t(),\n"
    "            torch.Tensor(),\n"
    "        )\n"
)

bad = []
xtext = XPU.read_text()
for old, new, name, marker in (
    (CLASS_ANCHOR, CLASS_FIXED, "class-attr", "_B70_MODE ="),
    (CALL_ANCHOR, CALL_FIXED, "call-route", "_B70_MODE in"),
):
    if marker in xtext:
        print(f"[B70_FP8_HYBRID] xpu.py {name}: already applied — ok")
        continue
    n = xtext.count(old)
    if n != 1:
        bad.append(f"xpu.py {name} (found {n})")
        continue
    xtext = xtext.replace(old, new)
    print(f"[B70_FP8_HYBRID] xpu.py {name}: patched")
if not bad:
    XPU.write_text(xtext)

otext = OPS.read_text()
FAKE_ANCHOR = 'if hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):'
FAKE_NEW = (
    'if hasattr(torch.ops._xpu_C, "xe2_block_fp8_small_m"):\n'
    "\n    @register_fake(\"_xpu_C::xe2_block_fp8_small_m\")\n"
    "    def _xe2_block_fp8_small_m_fake(\n"
    "        input: torch.Tensor,\n"
    "        q_weight: torch.Tensor,\n"
    "        weight_scale: torch.Tensor,\n"
    "        bias: torch.Tensor | None = None,\n"
    "    ) -> torch.Tensor:\n"
    "        input_2d = input.view(-1, input.shape[-1])\n"
    "        M = input_2d.size(0)\n"
    "        N = q_weight.size(1)\n"
    "        return torch.empty(\n"
    "            (M, N), dtype=torch.bfloat16, device=input.device)\n\n"
    + FAKE_ANCHOR
)
if "xe2_block_fp8_small_m" in otext:
    print("[B70_FP8_HYBRID] _xpu_ops.py: already applied — ok")
elif otext.count(FAKE_ANCHOR) == 1:
    OPS.write_text(otext.replace(FAKE_ANCHOR, FAKE_NEW))
    print("[B70_FP8_HYBRID] _xpu_ops.py: fake registration added")
else:
    bad.append("_xpu_ops.py fake anchor")

if bad:
    print(f"[B70_FP8_HYBRID] FATAL: {bad} — aborting")
    sys.exit(1)