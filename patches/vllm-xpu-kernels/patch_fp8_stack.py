#!/usr/bin/env python3
"""FP8 stack patch for stock vLLM-XPU f01 (final form, 2026-08-30).

Applies to the pinned image at container start:
  1) kernels/linear/scaled_mm/xpu.py:
     - W8A16 reroute (BF16 activations direct, apply_input_quant=False),
     - M<=40 dispatch to the custom torch.ops._xpu_C.xe2_block_fp8_small_m
       Xe2 kernel when available (hasattr-guarded),
     - B70_FP8_FORCE_W8A8=1 env fallback (re-enables dynamic act quant),
     - one-time engagement print (class-body, trace-safe).
  2) _xpu_ops.py: fake-tensor registration for the new op (compile mode).
Fails loudly on anchor mismatch; idempotent.
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
    "_B70_USE_W8A16 = _b70_os.environ.get('B70_FP8_FORCE_W8A8', '0') != '1'\n"
    "_B70_HAS_XE2_SMALL_M = hasattr(torch.ops._xpu_C, 'xe2_block_fp8_small_m')\n"
    "class XPUFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):\n"
    "    apply_input_quant = not _B70_USE_W8A16\n"
    "    print(f\"[B70_FP8_STACK] apply_input_quant={apply_input_quant} "
    "use_w8a16={_B70_USE_W8A16} xe2_small_m={_B70_HAS_XE2_SMALL_M}\", flush=True)\n"
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
    "        if _B70_USE_W8A16:\n"
    "            M, K = A.shape\n"
    "            N = B.shape[0]\n"
    "            if (_B70_HAS_XE2_SMALL_M and M <= 40\n"
    "                    and K % 128 == 0 and N % 128 == 0):\n"
    "                return torch.ops._xpu_C.xe2_block_fp8_small_m(\n"
    "                    A, B.t(), Bs.t(), None)\n"
    "            return torch.ops._xpu_C.fp8_gemm_w8a16(\n"
    "                A,\n"
    "                B.t(),\n"
    "                Bs.t(),\n"
    "                torch.Tensor(),\n"
    "            )\n"
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
    (CLASS_ANCHOR, CLASS_FIXED, "class-attr", "_B70_USE_W8A16 ="),
    (CALL_ANCHOR, CALL_FIXED, "call-route", "xe2_block_fp8_small_m("),
):
    if marker in xtext:
        print(f"[B70_FP8_STACK] xpu.py {name}: already applied — ok")
        continue
    n = xtext.count(old)
    if n != 1:
        bad.append(f"xpu.py {name} (found {n})")
        continue
    xtext = xtext.replace(old, new)
    print(f"[B70_FP8_STACK] xpu.py {name}: patched")
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
    print("[B70_FP8_STACK] _xpu_ops.py: already applied — ok")
elif otext.count(FAKE_ANCHOR) == 1:
    OPS.write_text(otext.replace(FAKE_ANCHOR, FAKE_NEW))
    print("[B70_FP8_STACK] _xpu_ops.py: fake registration added")
else:
    bad.append("_xpu_ops.py fake anchor")

if bad:
    print(f"[B70_FP8_STACK] FATAL: {bad} — aborting")
    sys.exit(1)