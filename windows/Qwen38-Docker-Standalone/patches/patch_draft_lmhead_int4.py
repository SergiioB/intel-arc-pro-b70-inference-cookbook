#!/usr/bin/env python3
"""B70 Fase S: draft MTP LM head cuantizado a INT4 g128 sym (GPTQ format).

El LM head fp16 compartido (5120x248320 = 2.54 GB) se lee 5x/paso MTP4
(4 drafts + 1 target) = 12.7 GB/step a ~98% del pico de DRAM (21.3 ms,
41% del tiempo de GEMM). Cuantizar SOLO la copia del draft a INT4 g128 sym
(0.66 GB/paso incl. scales) reduce las 4 pasadas del draft a ~4x0.66 GB
→ -7.6 GB/step ≈ -13 ms/step → objetivo ~108 tok/s.

Lossless: el draft MTP usa SU PROPIO lm_head (DraftModelProposer._maybe_share
_lm_head es no-op — no comparte el del target). Cuantizar la copia del draft
no toca el target de verificacion (fp16) y los tokens del draft se verifican
contra el target: la secuencia emitida es identica SOLO en C1/TP1 (greedy).
NO usar con TP>1: bajo `--tensor-parallel-size 2` la verificacion del target
no protege la salida emitida (issue #9).

Formato del peso INT4 = exactamente el que consume la op YA existente
``torch.ops._xpu_C.int4_gemm_w4a16`` (el que usan los layers INC/GPTQ del
cuerpo): qweight int32 [K/8, N] NT (nibbles secuenciales LSB-first,
valor almacenado = q + 8), scales fp16 [K/128, N], qzeros int8 [1] = 8.

Env-gated: B70_DRAFT_LMHEAD_INT4=1 (default off = comportamiento identico).
Anclas verificadas contra la imagen vllm/vllm-openai-xpu@2c427ef (vLLM
0.26.1.dev457.gc810e5ee9).
"""
from __future__ import annotations

import os
import sys

MARKER = "B70_DRAFT_LMHEAD_INT4"

HELPER_MODULE = "b70_draft_lmhead_int4.py"

HELPER_SOURCE = '''\
"""B70 Fase S runtime helper: draft MTP LM head INT4 g128 sym.

Escribido por patch_draft_lmhead_int4.py dentro del contenedor. Provee la
cuantizacion one-time del lm_head fp16 compartido a GPTQ INT4 g128 sym y el
ruteo de las 4 pasadas del draft por ``int4_gemm_w4a16``. El target queda
fp16 (lossless).
"""
from __future__ import annotations

import os

import torch


def quantize_lmhead_to_int4(weight: torch.Tensor, group_size: int = 128):
    """Quantiza un lm_head fp16 [N, K] a GPTQ INT4 g128 sym.

    Returns (qweight, scales, qzeros, group_size):
      qweight: int32 [K//8, N] en layout NT (strides[-2] == 1), nibbles
               secuenciales LSB-first, valor almacenado = q + 8 (q in [-8, 7])
      scales:  fp16 [K//group_size, N]
      qzeros:  int8 tensor([8])  -> rama simetrica de int4_gemm_w4a16
    """
    device = weight.device
    N, K = weight.shape
    num_groups = K // group_size
    chunk = 4096
    shifts = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
    )
    parts = []
    scale_parts = []
    for i in range(0, N, chunk):
        wc = weight[i : i + chunk].float()  # [c, K] fp32 (chunked: no 5 GB temp)
        wg = wc.view(wc.shape[0], num_groups, group_size)
        maxabs = wg.abs().amax(dim=-1)  # [c, g]
        scale = maxabs / 7.0
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        stored = q + 8  # 0..15
        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)
        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(
            wc.shape[0], K // 8
        )
        parts.append(packed)
        scale_parts.append(scale.half())
    qweight_contig = torch.cat(parts, dim=0)  # [N, K//8] int32
    scales_contig = torch.cat(scale_parts, dim=0)  # [N, g] fp16
    # Layout NT requerido por la op (strides[-2] == 1) + scales contiguas
    qweight = qweight_contig.t()  # [K//8, N], strides (1, K//8)
    scales = scales_contig.t().contiguous()  # [g, N]
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def int4_lmhead_logits(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Logits [.., vocab] via int4_gemm_w4a16 (mismo formato que el cuerpo)."""
    flat = x.reshape(-1, x.shape[-1])
    logits = torch.ops._xpu_C.int4_gemm_w4a16(
        flat, qweight, None, scales, qzeros, group_size, None
    )
    return logits.reshape(*x.shape[:-1], qweight.shape[1])


@torch.no_grad()
def build_draft_lmhead_int4(model) -> None:
    """Cuantiza el lm_head fp16 compartido del draft (one-time, no-op si no
    hay env gate o si ya se construyo). Almacena en model._b70_lmhead_int4."""
    if os.environ.get("B70_DRAFT_LMHEAD_INT4") != "1":
        return
    if getattr(model, "_b70_lmhead_int4_tp_blocked", False):
        return
    parallel_config = getattr(
        getattr(model, "vllm_config", None), "parallel_config", None
    )
    tp_size = getattr(parallel_config, "tensor_parallel_size", None)
    if tp_size is None:
        model._b70_lmhead_int4_tp_blocked = True
        print("[B70] draft LM head INT4: TP unknown; skipping "
              "(fail-closed, issue #9)", flush=True)
        return
    if tp_size > 1:
        model._b70_lmhead_int4_tp_blocked = True
        print("[B70] draft LM head INT4: TP>1 (tp=%d) detected; skipping "
              "(C1/TP1 only, issue #9)" % tp_size, flush=True)
        return
    if getattr(model, "_b70_lmhead_int4", None) is not None:
        return
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None:
        print("[B70] draft LM head INT4: lm_head.weight no disponible; "
              "draft sigue por fp16", flush=True)
        return
    print("[B70] draft LM head INT4: cuantizando lm_head fp16 "
          f"{tuple(weight.shape)} -> INT4 g128 sym (one-time)", flush=True)
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(
        weight.detach()
    )
    model._b70_lmhead_int4 = (qweight, scales, qzeros, group_size)
    fp16_bytes = weight.numel() * weight.element_size()
    int4_bytes = qweight.numel() * qweight.element_size() + (
        scales.numel() * scales.element_size()
    )
    print(f"[B70] draft LM head INT4: listo. {fp16_bytes/1e9:.2f} GB fp16 -> "
          f"{int4_bytes/1e9:.2f} GB INT4 (ahorro "
          f"{(fp16_bytes - int4_bytes)/1e6:.1f} MB/lectura)", flush=True)


def draft_lmhead_int4_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Logits del draft via la copia INT4 (4 pasadas/paso -> 0.66 GB c/u)."""
    qweight, scales, qzeros, group_size = model._b70_lmhead_int4
    logits = int4_lmhead_logits(
        hidden_states, qweight, scales, qzeros, group_size
    )
    org = getattr(getattr(model, "logits_processor", None), "org_vocab_size", None)
    if org is not None and logits.shape[-1] > org:
        logits = logits[..., :org]
    return logits
'''

QWMTP_OLD = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "        spec_step_idx: int = 0,\n"
    "    ) -> torch.Tensor | None:\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n"
)
QWMTP_NEW = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "        spec_step_idx: int = 0,\n"
    "    ) -> torch.Tensor | None:\n"
    "        if os.environ.get(\"B70_DRAFT_LMHEAD_INT4\") == \"1\" and not getattr(self, \"_b70_lmhead_int4_tp_blocked\", False):\n"
    "            from vllm.model_executor.models.b70_draft_lmhead_int4 import (\n"
    "                build_draft_lmhead_int4,\n"
    "                draft_lmhead_int4_logits,\n"
    "            )\n"
    "\n"
    "            if getattr(self, \"_b70_lmhead_int4\", None) is None:\n"
    "                build_draft_lmhead_int4(self)\n"
    "            if getattr(self, \"_b70_lmhead_int4\", None) is not None:\n"
    "                return draft_lmhead_int4_logits(self, hidden_states)\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n"
)


def _write_helper(vllm_dir: str) -> str:
    models_dir = os.path.join(vllm_dir, "model_executor", "models")
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, HELPER_MODULE)
    existing = None
    if os.path.exists(path):
        existing = open(path).read()
    if existing == HELPER_SOURCE:
        print(f"helper already present {path}")
        return path
    with open(path, "w") as f:
        f.write(HELPER_SOURCE)
    print(f"helper written {path}")
    return path


def _patch_qwen3_5_mtp(vllm_dir: str) -> None:
    path = os.path.join(vllm_dir, "model_executor", "models", "qwen3_5_mtp.py")
    text = open(path).read()
    if MARKER in text:
        print(f"already patched {path}")
        return
    if QWMTP_OLD not in text:
        sys.exit(f"anchor not found in {path}: compute_logits")
    text = text.replace(QWMTP_OLD, QWMTP_NEW, 1)
    if "\nimport os\n" not in text and not text.startswith("import os\n"):
        text = text.replace("import torch\n", "import os\nimport torch\n", 1)
        if "\nimport os\n" not in text and not text.startswith("import os\n"):
            sys.exit(f"could not inject import os in {path}")
    compile(text, path, "exec")
    open(path, "w").write(text)
    print(f"patched {path}")


def main() -> None:
    import vllm

    vllm_dir = os.path.dirname(vllm.__file__)
    _write_helper(vllm_dir)
    _patch_qwen3_5_mtp(vllm_dir)


if __name__ == "__main__":
    main()
