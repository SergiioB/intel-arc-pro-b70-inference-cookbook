#!/usr/bin/env python3
"""v4: native XpuFusedMoe for GPTQ MoE — correct int8 pack + is_int4 dtype.

Root cause of v3 `ptr_A.size(1) must match ptr_B.size(1)`:
  C++ int4 path: is_B_int4 = (B_dtype == kChar/int8) && scales
  v3 left weights as uint8 after implement_zp → kernel treated B as BF16 [E,K,N]
  and compared A_K (2048) to B.size(1) (N=1024).

v4:
  1. Patch _get_weights_dtype: int8 (+ 3D scales) counts as int4 (C++ kChar).
  2. Lean expert-by-expert implement_zp → int8 storage (no empty_like peak).
  3. Route MoeWNA16Method.apply → XpuFusedMoe on XPU GPTQ-int4.
"""
from pathlib import Path


def patch_fused_moe_interface():
    path = Path("/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/fused_moe_interface.py")
    if not path.exists():
        cands = list(Path("/opt/venv/lib").rglob("fused_moe_interface.py"))
        if not cands:
            raise SystemExit("fused_moe_interface.py not found")
        path = cands[0]
    t = path.read_text()
    old = """def _get_weights_dtype(weight, scales):
    weight_dtype = weight.dtype
    is_fp8 = weight_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    is_int4 = weight_dtype == torch.uint8
    is_mxfp4 = weight_dtype == torch.float4_e2m1fn_x2"""
    new = """def _get_weights_dtype(weight, scales):
    weight_dtype = weight.dtype
    is_fp8 = weight_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    # C++ cutlass path detects int4 via at::kChar (torch.int8) + scales.
    # GPTQ WNA16 loads as uint8; after implement_zp we store int8.
    is_int4 = (
        weight_dtype in (torch.uint8, torch.int8)
        and scales is not None
        and getattr(scales, "ndim", 0) == 3
    )
    is_mxfp4 = weight_dtype == torch.float4_e2m1fn_x2"""
    if old not in t:
        if "weight_dtype in (torch.uint8, torch.int8)" in t:
            print("fused_moe_interface already v4-patched dtype")
        else:
            raise SystemExit("could not find _get_weights_dtype block to patch")
    else:
        t = t.replace(old, new, 1)
        path.write_text(t)
        print(f"patched _get_weights_dtype in {path}")

    # Skip implement_zp when weights are already int8 (pre-converted by moe_wna16).
    # Tensor attrs like w13.xpu_fused_moe do NOT survive Parameter.data reassignment.
    t = path.read_text()
    old_prep = """        if is_int4 and not hasattr(w13, 'xpu_fused_moe'):
            w13_tmp = torch.empty_like(w13).to(torch.int8)
            w2_tmp = torch.empty_like(w2).to(torch.int8)
            for i in range(num_experts):
                w13_tmp[i] = implement_zp(w13[i])
                w2_tmp[i] = implement_zp(w2[i])
            w13_tmp = w13_tmp.contiguous()
            w2_tmp = w2_tmp.contiguous()
            w13.data = w13_tmp
            w2.data = w2_tmp
            w13.xpu_fused_moe = True"""
    # Also match lean prep if a previous patch applied it
    old_prep_lean = """        if is_int4 and not hasattr(w13, 'xpu_fused_moe'):
            # Lean path: convert expert-by-expert into preallocated int8 (lower peak).
            w13_tmp = torch.empty(w13.shape, dtype=torch.int8, device=w13.device)
            w2_tmp = torch.empty(w2.shape, dtype=torch.int8, device=w2.device)
            for i in range(num_experts):
                w13_tmp[i].copy_(implement_zp(w13[i].contiguous()).to(torch.int8))
                w2_tmp[i].copy_(implement_zp(w2[i].contiguous()).to(torch.int8))
            w13.data = w13_tmp.contiguous()
            w2.data = w2_tmp.contiguous()
            w13.xpu_fused_moe = True
            w2.xpu_fused_moe = True"""
    new_prep = """        # Already int8 ⇒ pre-packed by moe_wna16.process_weights_after_loading.
        # Fresh uint8 ⇒ convert here (legacy path).
        if is_int4 and w13.dtype == torch.uint8 and not getattr(w13, 'xpu_fused_moe', False):
            w13_tmp = torch.empty(w13.shape, dtype=torch.int8, device=w13.device)
            w2_tmp = torch.empty(w2.shape, dtype=torch.int8, device=w2.device)
            for i in range(num_experts):
                w13_tmp[i].copy_(implement_zp(w13[i].contiguous()).to(torch.int8))
                w2_tmp[i].copy_(implement_zp(w2[i].contiguous()).to(torch.int8))
            w13 = w13_tmp.contiguous()
            w2 = w2_tmp.contiguous()
        elif is_int4 and w13.dtype == torch.int8:
            # ensure contiguous int8 storage for cutlass kChar path
            w13 = w13.contiguous()
            w2 = w2.contiguous()"""
    replaced = False
    for cand in (old_prep_lean, old_prep):
        if cand in t:
            t = t.replace(cand, new_prep, 1)
            replaced = True
            break
    if "Already int8 ⇒ pre-packed" in t and not replaced:
        print("XpuFusedMoe prep already v4b")
    elif replaced:
        path.write_text(t)
        print("patched XpuFusedMoe int4 prep (skip if int8)")
    else:
        raise SystemExit("XpuFusedMoe prep block not found")


def patch_moe_wna16():
    root = Path("/opt/vllm/vllm")
    moe_path = root / "model_executor/layers/quantization/moe_wna16.py"
    t = moe_path.read_text()

    old_apply_start = t.find("    def apply(\n        self,\n        layer: FusedMoE,")
    if old_apply_start < 0:
        raise SystemExit("apply not found")
    rest = t[old_apply_start + 4 :]
    cuts = [rest.find("\n    def "), rest.find("\n    @staticmethod")]
    cut = min(c for c in cuts if c >= 0)

    new_block = '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """One-time GPTQ→XPU int4 pack: uint8 nibble → int8 signed nibble (kChar)."""
        import os
        from vllm.platforms import current_platform
        if (
            current_platform.is_xpu()
            and self.quant_config.weight_bits == 4
            and not self.quant_config.has_zp
            and os.environ.get("VLLM_XPU_MOE_TRITON", "0") not in ("1", "true", "TRUE")
        ):
            from vllm_xpu_kernels.fused_moe_interface import implement_zp
            w13 = layer.w13_qweight
            w2 = layer.w2_qweight
            if getattr(w13, "xpu_fused_moe", False):
                return
            # Allocate int8 destinations once (same nbytes as uint8 pack).
            w13_i8 = torch.empty(w13.shape, dtype=torch.int8, device=w13.device)
            w2_i8 = torch.empty(w2.shape, dtype=torch.int8, device=w2.device)
            for i in range(w13_i8.size(0)):
                w13_i8[i].copy_(implement_zp(w13.data[i].contiguous()).to(torch.int8))
                w2_i8[i].copy_(implement_zp(w2.data[i].contiguous()).to(torch.int8))
            # Bit-preserving uint8→int8 view (C++ only checks kChar; reads bits as u8).
            w13.data = w13_i8.contiguous()
            w2.data = w2_i8.contiguous()
            # Flag MUST live on the Tensor (.data), because XpuFusedMoe gets .data
            for tens in (w13, w13.data, w2, w2.data):
                tens.xpu_fused_moe = True
            layer._xpu_int4_prepared = True
            # scales already [E, N, group_num] — matches cutlass int4 checks
            layer.w13_scales.data = layer.w13_scales.data.contiguous()
            layer.w2_scales.data = layer.w2_scales.data.contiguous()
            print(
                f"[B70] v4 native int4 prep done w13={tuple(w13.data.shape)} "
                f"dtype={w13.data.dtype} scales={tuple(layer.w13_scales.shape)}",
                flush=True,
            )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        import os
        from vllm.model_executor.layers.fused_moe import fused_experts
        from vllm.platforms import current_platform

        assert layer.activation == MoEActivation.SILU, (
            f"Only SiLU activation is supported, not {layer.activation}."
        )

        use_native = (
            current_platform.is_xpu()
            and self.quant_config.weight_bits == 4
            and not self.quant_config.has_zp
            and os.environ.get("VLLM_XPU_MOE_TRITON", "0") not in ("1", "true", "TRUE")
            and getattr(layer, "_xpu_int4_prepared", False)
        )
        if use_native:
            from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe
            if not hasattr(layer, "_xpu_fused_moe_impl"):
                act = layer.activation
                act_name = act.value if hasattr(act, "value") else str(act)
                if "SILU" in act_name.upper():
                    act_name = "silu"
                print(
                    f"[B70] Using native XpuFusedMoe int4 v4 "
                    f"w13={tuple(layer.w13_qweight.shape)} dtype={layer.w13_qweight.dtype} "
                    f"scales={tuple(layer.w13_scales.shape)}",
                    flush=True,
                )
                layer._xpu_fused_moe_impl = XpuFusedMoe(
                    w13=layer.w13_qweight.data,
                    w13_scales=layer.w13_scales.data.contiguous(),
                    w13_bias=None,
                    w2=layer.w2_qweight.data,
                    w2_scales=layer.w2_scales.data.contiguous(),
                    w2_bias=None,
                    n_experts_per_token=int(topk_ids.size(-1)),
                    activation=act_name,
                    num_experts=int(layer.w13_qweight.size(0)),
                    ep_rank=0,
                    ep_size=1,
                    expert_map=layer.expert_map,
                )
            out = torch.empty_like(x)
            layer._xpu_fused_moe_impl.apply(
                output=out,
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids.to(dtype=torch.int32, device=x.device),
                expert_map=layer.expert_map,
            )
            return out

        return fused_experts(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=not self.moe.disable_inplace,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )
'''

    # Strip any previous process_weights_after_loading we inserted
    while "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:" in t:
        idx = t.find("    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:")
        # find next def at class indent
        rest2 = t[idx + 4 :]
        nd = rest2.find("\n    def ")
        ns = rest2.find("\n    @staticmethod")
        cands = [c for c in (nd, ns) if c >= 0]
        if not cands:
            break
        t = t[:idx] + rest2[min(cands) :]

    # Re-find apply after stripping
    old_apply_start = t.find("    def apply(\n        self,\n        layer: FusedMoE,")
    if old_apply_start < 0:
        raise SystemExit("apply not found after strip")
    rest = t[old_apply_start + 4 :]
    cuts = [rest.find("\n    def "), rest.find("\n    @staticmethod")]
    cut = min(c for c in cuts if c >= 0)
    t = t[:old_apply_start] + new_block + rest[cut:]
    moe_path.write_text(t)
    compile(t, str(moe_path), "exec")
    print("patched moe_wna16 v4 OK")


def main():
    patch_fused_moe_interface()
    patch_moe_wna16()
    print("v4 patch complete")


if __name__ == "__main__":
    main()
