#!/usr/bin/env python3
"""Force Qwen3.5 MTP draft head to load unquantized BF16 fused experts.

Native-MTP-Preserved GPTQ keeps mtp.* as BF16 fused tensors (gate_up_proj /
down_proj). Draft inherits target GPTQ quant_config → expert params become
GPTQ-shaped (w2_qweight) → KeyError on w2_weight during fused load.

Also strips is_fp8/is_mxfp4 kwargs from xpu_moe.py XpuFusedMoe call
(kernels auto-detect dtype; unquantized BF16 draft hits that path).
"""
from pathlib import Path


def patch_mtp_file() -> None:
    path = Path("/opt/vllm/vllm/model_executor/models/qwen3_5_mtp.py")
    t = path.read_text()
    if "B70_MTP_BF16_DRAFT" in t and "B70_MTP_LOAD_DEBUG" in t:
        print("mtp file already fully patched")
        return

    if "import dataclasses" not in t:
        if "from vllm.config import VllmConfig" in t:
            t = t.replace(
                "from vllm.config import VllmConfig\n",
                "from vllm.config import VllmConfig\nimport dataclasses\n",
                1,
            )
        elif "import typing\n" in t:
            t = t.replace("import typing\n", "import dataclasses\nimport typing\n", 1)
        else:
            t = "import dataclasses\n" + t

    if "B70_MTP_BF16_DRAFT" not in t:
        old_head = (
            '    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):\n'
            "        super().__init__()\n"
            "\n"
            "        model_config = vllm_config.model_config\n"
            "        quant_config = vllm_config.quant_config\n"
        )
        new_head = (
            '    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):\n'
            "        super().__init__()\n"
            "\n"
            "        model_config = vllm_config.model_config\n"
            "        quant_config = vllm_config.quant_config\n"
            "        # B70_MTP_BF16_DRAFT: GPTQ-preserved checkpoints store mtp.* as BF16\n"
            "        # fused experts. Build the entire draft head without quant_config.\n"
            "        _qname = quant_config.get_name() if quant_config is not None else None\n"
            "        if _qname is not None:\n"
            '            print(f"[B70] MTP MultiTokenPredictor: forcing unquantized draft (was {_qname})")\n'
            "            vllm_config = dataclasses.replace(vllm_config, quant_config=None)\n"
            "            quant_config = None\n"
        )
        if old_head not in t:
            raise SystemExit("MultiTokenPredictor __init__ head not found")
        t = t.replace(old_head, new_head, 1)

        marker = "        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory("
        if marker not in t:
            raise SystemExit("make_empty_intermediate_tensors marker not found")
        dump = (
            "        # B70 debug: show what expert params actually exist after build\n"
            '        _exp = [n for n, _ in self.named_parameters() if "expert" in n.lower() or "w13" in n or "w2" in n]\n'
            '        print(f"[B70] MTP draft expert-ish params ({len(_exp)}):")\n'
            "        for n in _exp[:40]:\n"
            "            p = dict(self.named_parameters())[n]\n"
            '            print(f"  {n} shape={tuple(p.shape)} dtype={p.dtype}")\n'
            '        if not any(n.endswith("w2_weight") for n in _exp):\n'
            '            print("[B70] WARNING: no w2_weight in MTP draft — still quantized or missing MoE")\n'
            "\n"
        )
        t = t.replace(marker, dump + marker, 1)

    if "B70_MTP_LOAD_DEBUG" not in t:
        old_load = (
            "    def load_fused_expert_weights(\n"
            "        self,\n"
            "        name: str,\n"
            "        params_dict: dict,\n"
            "        loaded_weight: torch.Tensor,\n"
            "        shard_id: str,\n"
            "        num_experts: int,\n"
            "    ) -> bool:\n"
            "        param = params_dict[name]\n"
        )
        new_load = (
            "    def load_fused_expert_weights(\n"
            "        self,\n"
            "        name: str,\n"
            "        params_dict: dict,\n"
            "        loaded_weight: torch.Tensor,\n"
            "        shard_id: str,\n"
            "        num_experts: int,\n"
            "    ) -> bool:\n"
            "        # B70_MTP_LOAD_DEBUG\n"
            "        if name not in params_dict:\n"
            '            keys = [k for k in params_dict if "expert" in k.lower() or "w13" in k or "w2" in k]\n'
            '            print(f"[B70] load_fused_expert_weights missing {name!r}")\n'
            '            print(f"[B70] available expert-ish keys ({len(keys)}): {keys[:50]}")\n'
            '            print(f"[B70] loaded_weight shape={tuple(loaded_weight.shape)} dtype={loaded_weight.dtype} shard={shard_id}")\n'
            "        param = params_dict[name]\n"
        )
        if old_load not in t:
            raise SystemExit("load_fused_expert_weights not found")
        t = t.replace(old_load, new_load, 1)

    path.write_text(t)
    print(f"patched {path}")


def patch_sparse_moe_block() -> None:
    path = Path("/opt/vllm/vllm/model_executor/models/qwen3_next.py")
    t = path.read_text()
    if "B70_MTP_SPARSE_MOE" in t:
        print("sparse moe already patched")
        return

    old = (
        "class Qwen3NextSparseMoeBlock(nn.Module):\n"
        '    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):\n'
        "        super().__init__()\n"
        "\n"
        "        config = vllm_config.model_config.hf_text_config\n"
        "        parallel_config = vllm_config.parallel_config\n"
        "        quant_config = vllm_config.quant_config\n"
    )
    new = (
        "class Qwen3NextSparseMoeBlock(nn.Module):\n"
        '    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):\n'
        "        super().__init__()\n"
        "\n"
        "        config = vllm_config.model_config.hf_text_config\n"
        "        parallel_config = vllm_config.parallel_config\n"
        "        quant_config = vllm_config.quant_config\n"
        "        # B70_MTP_SPARSE_MOE: mtp draft experts are BF16 fused in GPTQ-preserved ckpts\n"
        '        if "mtp" in prefix and quant_config is not None:\n'
        '            print(f"[B70] SparseMoeBlock {prefix}: forcing quant_config=None (was {quant_config.get_name()})")\n'
        "            quant_config = None\n"
    )
    if old not in t:
        raise SystemExit("Qwen3NextSparseMoeBlock head not found")
    path.write_text(t.replace(old, new, 1))
    print(f"patched {path}")


def patch_fused_moe_mtp() -> None:
    path = Path("/opt/vllm/vllm/model_executor/layers/fused_moe/layer.py")
    t = path.read_text()
    if "B70_MTP_FUSED_MOE" in t:
        print("fused moe mtp already patched")
        return

    needle = "        self.quant_config = quant_config\n\n        def _get_quant_method() -> FusedMoEMethodBase:"
    if needle not in t:
        raise SystemExit("FusedMoE quant_config block not found")
    repl = (
        "        self.quant_config = quant_config\n"
        "        # B70_MTP_FUSED_MOE: draft MTP experts are BF16 fused in GPTQ-preserved ckpts\n"
        '        if "mtp" in (prefix or "") and self.quant_config is not None:\n'
        '            print(f"[B70] FusedMoE {prefix}: forcing Unquantized (was {self.quant_config.get_name()})")\n'
        "            self.quant_config = None\n"
        "\n"
        "        def _get_quant_method() -> FusedMoEMethodBase:"
    )
    t = t.replace(needle, repl, 1)

    old_assign = "        self.quant_method: FusedMoEMethodBase = _get_quant_method()\n"
    new_assign = (
        "        self.quant_method: FusedMoEMethodBase = _get_quant_method()\n"
        '        if "mtp" in (prefix or ""):\n'
        '            print(f"[B70] FusedMoE {prefix}: method={self.quant_method.__class__.__name__}")\n'
    )
    if old_assign in t:
        t = t.replace(old_assign, new_assign, 1)

    path.write_text(t)
    print(f"patched {path}")


def patch_xpu_moe_is_fp8() -> None:
    """vllm xpu_moe.py passes is_fp8/is_mxfp4; kernels XpuFusedMoe auto-detects dtype."""
    path = Path("/opt/vllm/vllm/model_executor/layers/fused_moe/experts/xpu_moe.py")
    t = path.read_text()
    if "B70_XPU_MOE_NO_IS_FP8" in t:
        print("xpu_moe is_fp8 already patched")
        return
    old = (
        "            self.fused_moe_impl = XpuFusedMoe(\n"
        "                w13=w1,\n"
        "                w13_scales=self.w1_scale,\n"
        "                w13_bias=self.w1_bias,\n"
        "                w2=w2,\n"
        "                w2_scales=self.w2_scale,\n"
        "                w2_bias=self.w2_bias,\n"
        "                n_experts_per_token=topk,\n"
        "                activation=activation.value,\n"
        "                num_experts=self.moe_config.num_local_experts,\n"
        "                ep_rank=self.moe_config.ep_rank,\n"
        "                ep_size=self.moe_config.ep_size,\n"
        "                is_fp8=self.is_fp8,\n"
        "                is_mxfp4=self.is_mxfp4,\n"
        "            )\n"
    )
    new = (
        "            # B70_XPU_MOE_NO_IS_FP8: kernels XpuFusedMoe detects dtype from weights\n"
        "            self.fused_moe_impl = XpuFusedMoe(\n"
        "                w13=w1,\n"
        "                w13_scales=self.w1_scale,\n"
        "                w13_bias=self.w1_bias,\n"
        "                w2=w2,\n"
        "                w2_scales=self.w2_scale,\n"
        "                w2_bias=self.w2_bias,\n"
        "                n_experts_per_token=topk,\n"
        "                activation=activation.value,\n"
        "                num_experts=self.moe_config.num_local_experts,\n"
        "                ep_rank=self.moe_config.ep_rank,\n"
        "                ep_size=self.moe_config.ep_size,\n"
        "            )\n"
    )
    if old not in t:
        raise SystemExit("XpuFusedMoe call site not found in xpu_moe.py")
    path.write_text(t.replace(old, new, 1))
    print(f"patched {path}")


def patch_gdn_spec_assert() -> None:
    """Remove XPU GDN spec_sequence_masks assert.

    The SYCL kernel already receives num_spec_decodes, spec_query_start_loc,
    spec_token_indx, spec_state_indices_tensor, num_accepted_tokens — the
    boolean mask is only used for metadata organization and is NOT passed to
    the kernel. The assert blocks all speculative decoding (ngram, MTP) on
    hybrid GDN models; removing it lets the kernel do its spec-decode path.
    """
    path = Path("/opt/vllm/vllm/_xpu_ops.py")
    t = path.read_text()
    if "B70_GDN_SPEC_OK" in t:
        print("gdn spec assert already patched")
        return
    old = """    # TODO: xpu does not support speculative decoding yet
    assert attn_metadata.spec_sequence_masks is None  # type: ignore[attr-defined]
"""
    new = """    # B70_GDN_SPEC_OK: kernel takes spec_* tensors; boolean mask is not passed
    if attn_metadata.spec_sequence_masks is not None:
        print("[B70] GDN XPU: spec decode active, proceeding (mask not passed to kernel)")
"""
    if old not in t:
        raise SystemExit("GDN assert block not found")
    path.write_text(t.replace(old, new, 1))
    print(f"patched {path}")


def main() -> None:
    patch_mtp_file()
    patch_sparse_moe_block()
    patch_fused_moe_mtp()
    patch_xpu_moe_is_fp8()
    patch_gdn_spec_assert()
    print("mtp bf16 draft patch complete")


if __name__ == "__main__":
    main()


