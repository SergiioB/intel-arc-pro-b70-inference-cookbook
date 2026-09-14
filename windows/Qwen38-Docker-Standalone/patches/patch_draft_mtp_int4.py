#!/usr/bin/env python3
"""B70 Fase M1: MTP layer (draft) cuantizado a INT4 g128 sym (formato GPTQ).

El modulo MTP del draft (Qwen3_5MultiTokenPredictor: fc + full_attention
decoder layer) esta en BF16 (forzado por B70_MTP_BF16_DRAFT=1) y se lee
~0.85 GB/paso (4 pasadas/paso = 3.4 GB). Cuantizar sus 5 linears
(fc, qkv_proj, o_proj, gate_up_proj, down_proj) a INT4 g128 sym
(~0.21 GB incl. scales) reusa el kernel YA existente
``torch.ops._xpu_C.int4_gemm_w4a16`` (formato GPTQ del cuerpo) y elimina
~2.6 GB/paso de lecturas de DRAM.

Lossless: el target de verificacion no se toca; los tokens del draft se
verifican contra el target (greedy) -> la secuencia emitida es identica
SOLO en C1/TP1. NO usar con TP>1: bajo `--tensor-parallel-size 2` la
verificacion no protege la salida emitida (issue #9).
GATE de aceptacion: el draft INT4 puede proponer distinto; si la aceptacion
baja >0.03 la ganancia de bytes puede perderse -> medir ANTES vs DESPUES.

Mecanismo: los linears del MTP tienen ``quant_method`` (UnquantizedLinearMethod)
cuyo ``apply(layer, x, bias)`` hace el matmul. Se intercambia ``quant_method``
por un metodo duck-typed que rutea por int4_gemm_w4a16 con la copia INT4.
El resto del layer (attention, rope, norms, silu) queda intacto.

IMPORTANTE (compilacion): el hook va en ``Qwen3_5MTP.load_weights`` (eager,
al cargar el modelo) y NO en forward: el forward de Qwen3_5MTP esta
decorado con @support_torch_compile (AOT fullgraph) y una construccion
lazy en forward rompe el trace de dynamo ("Failed to trace builtin operator
print" en warmup). La construccion en load_weights corre antes de cualquier
trace -> el grafo compilado ya ve los quant_method INT4.

Env-gated: B70_DRAFT_MTP_INT4=1 (default off = comportamiento identico).
Anclas verificadas contra la imagen vllm/vllm-openai-xpu@2c427ef (vLLM
0.26.1.dev457.gc810e5ee9).
"""
from __future__ import annotations

import os
import sys

MARKER = "B70_DRAFT_MTP_INT4"

HELPER_MODULE = "b70_draft_mtp_int4.py"

HELPER_SOURCE = '''\
"""B70 Fase M1 runtime helper: draft MTP layer INT4 g128 sym.

Escribido por patch_draft_mtp_int4.py dentro del contenedor. Provee la
cuantizacion one-time de los 5 linears del modulo MTP del draft a GPTQ
INT4 g128 sym y el ruteo por ``int4_gemm_w4a16``. El target queda intacto.
"""
from __future__ import annotations

import os

import torch


def quantize_to_int4(weight: torch.Tensor, group_size: int = 128):
    """Quantiza un peso [N, K] a GPTQ INT4 g128 sym (formato int4_gemm_w4a16).

    Returns (qweight, scales, qzeros, group_size):
      qweight: int32 [K//8, N] layout NT (strides[-2] == 1), nibbles
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
        wc = weight[i : i + chunk].float()
        wg = wc.view(wc.shape[0], num_groups, group_size)
        maxabs = wg.abs().amax(dim=-1)
        scale = maxabs / 7.0
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        stored = q + 8
        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)
        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(
            wc.shape[0], K // 8
        )
        parts.append(packed)
        scale_parts.append(scale.half())
    qweight_contig = torch.cat(parts, dim=0)
    scales_contig = torch.cat(scale_parts, dim=0)
    qweight = qweight_contig.t()
    scales = scales_contig.t().contiguous()
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def _collect_linears(predictor) -> list[tuple[str, torch.nn.Module]]:
    """Lista (name, linear) de los 5 linears del MTP predictor a INT4."""
    found: list[tuple[str, torch.nn.Module]] = []
    found.append(("fc", predictor.fc))
    for li, layer in enumerate(predictor.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            found.append((f"layers.{li}.self_attn.qkv_proj", attn.qkv_proj))
            found.append((f"layers.{li}.self_attn.o_proj", attn.o_proj))
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            gate_up = getattr(mlp, "gate_up_proj", None)
            down = getattr(mlp, "down_proj", None)
            if gate_up is not None:
                found.append((f"layers.{li}.mlp.gate_up_proj", gate_up))
            if down is not None:
                found.append((f"layers.{li}.mlp.down_proj", down))
    return found


class _B70MTPInt4LinearMethod:
    """Duck-typed quant_method: apply() rutea por int4_gemm_w4a16."""

    def __init__(self, qweight, scales, qzeros, group_size):
        self.qweight = qweight
        self.scales = scales
        self.qzeros = qzeros
        self.group_size = group_size

    def create_weights(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, *args, **kwargs):
        pass

    def apply(self, layer, x, bias):
        flat = x.reshape(-1, x.shape[-1])
        if flat.dtype != torch.float16:
            flat = flat.to(torch.float16)
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            flat, self.qweight, None, self.scales, self.qzeros,
            self.group_size, None,
        )
        return out.reshape(*x.shape[:-1], self.qweight.shape[1])


@torch.no_grad()
def build_draft_mtp_int4(model) -> None:
    """Cuantiza los linears del MTP predictor (one-time en load_weights; no-op
    si no hay env gate o si ya se construyo). Almacena en
    model._b70_mtp_int4_built."""
    if os.environ.get("B70_DRAFT_MTP_INT4") != "1":
        return
    if getattr(model, "_b70_mtp_int4_tp_blocked", False):
        return
    parallel_config = getattr(
        getattr(model, "vllm_config", None), "parallel_config", None
    )
    tp_size = getattr(parallel_config, "tensor_parallel_size", None)
    if tp_size is None:
        model._b70_mtp_int4_tp_blocked = True
        print("[B70] draft MTP INT4: TP unknown; skipping "
              "(fail-closed, issue #9)", flush=True)
        return
    if tp_size > 1:
        model._b70_mtp_int4_tp_blocked = True
        print("[B70] draft MTP INT4: TP>1 (tp=%d) detected; skipping "
              "(C1/TP1 only, issue #9)" % tp_size, flush=True)
        return
    if getattr(model, "_b70_mtp_int4_built", False):
        return
    predictor = getattr(model, "model", None)
    if predictor is None or not hasattr(predictor, "layers"):
        print("[B70] draft MTP INT4: no Qwen3_5MultiTokenPredictor; "
              "MTP queda en BF16", flush=True)
        return
    linears = _collect_linears(predictor)
    if not linears:
        print("[B70] draft MTP INT4: no linears encontrados; skip", flush=True)
        return
    print(f"[B70] draft MTP INT4: cuantizando {len(linears)} linears "
          f"del MTP -> INT4 g128 sym (one-time)", flush=True)
    total_fp16 = 0
    total_int4 = 0
    for name, lin in linears:
        w = getattr(lin, "weight", None)
        if w is None:
            continue
        orig_shape = tuple(w.shape)
        qweight, scales, qzeros, gs = quantize_to_int4(w.detach())
        lin._b70_mtp_int4 = _B70MTPInt4LinearMethod(
            qweight, scales, qzeros, gs
        )
        lin.quant_method = lin._b70_mtp_int4
        fp16_bytes = w.numel() * w.element_size()
        int4_bytes = qweight.numel() * qweight.element_size() + (
            scales.numel() * scales.element_size()
        )
        total_fp16 += fp16_bytes
        total_int4 += int4_bytes
        with torch.no_grad():
            lin.weight.set_(
                torch.empty(0, dtype=w.dtype, device=w.device)
            )
        print(f"[B70] draft MTP INT4: {name} {orig_shape} "
              f"{fp16_bytes/1e6:.0f} MB -> {int4_bytes/1e6:.0f} MB "
              f"(fp16 liberado)", flush=True)
    model._b70_mtp_int4_built = True
    print(f"[B70] draft MTP INT4: listo. {total_fp16/1e9:.2f} GB BF16 -> "
          f"{total_int4/1e9:.2f} GB INT4 (ahorro "
          f"{(total_fp16 - total_int4)/1e6:.0f} MB/lectura)", flush=True)
'''

QWMTP_FORWARD_OLD = (
    "        loader = AutoWeightsLoader(self)\n"
    "        return loader.load_weights(remap_weight_names(weights))\n"
)
QWMTP_FORWARD_NEW = (
    "        loader = AutoWeightsLoader(self)\n"
    "        result = loader.load_weights(remap_weight_names(weights))\n"
    "        if os.environ.get(\"B70_DRAFT_MTP_INT4\") == \"1\":\n"
    "            from vllm.model_executor.models.b70_draft_mtp_int4 import (\n"
    "                build_draft_mtp_int4,\n"
    "            )\n"
    "\n"
    "            build_draft_mtp_int4(self)\n"
    "        return result\n"
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
    if QWMTP_FORWARD_OLD not in text:
        sys.exit(f"anchor not found in {path}: Qwen3_5MTP.forward")
    text = text.replace(QWMTP_FORWARD_OLD, QWMTP_FORWARD_NEW, 1)
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
