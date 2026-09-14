#!/usr/bin/env python3
"""B70 nightly port of patch_mtp_bf16_draft.py.

Nightly qwen3_5_mtp.py already has the upstream mechanism (null
vllm_config.quant_config before building MTP layers) but it only triggers
when the checkpoint's quantization_config.dynamic has a "-:mtp" exclusion
pattern. Our GPTQ-preserved checkpoint has dynamic=null, so we extend the
condition with an env gate: B70_MTP_BF16_DRAFT=1.

Only one injection point needed in the new architecture: nulling
vllm_config.quant_config before Qwen3_5DecoderLayer construction makes all
draft MoE/linear layers build unquantized, matching the BF16 mtp.* weights.
"""
import os
import sys


def main() -> None:
    import vllm

    vllm_dir = os.path.dirname(vllm.__file__)
    path = os.path.join(vllm_dir, "model_executor", "models", "qwen3_5_mtp.py")
    if not os.path.exists(path):
        sys.exit(f"qwen3_5_mtp.py not found at {path}")

    t = open(path).read()
    if "B70_MTP_NIGHTLY_DRAFT" in t:
        print("already patched")
        return

    old = (
        "        original_quant = vllm_config.quant_config\n"
        '        if quant_config and quant_config.get_name() not in ("modelopt_fp4",):\n'
        '            hf_qc = getattr(model_config.hf_config, "quantization_config", None)\n'
        "            if isinstance(hf_qc, dict):\n"
        '                dynamic = hf_qc.get("dynamic", {})\n'
        '                if any(k.startswith("-:") and "mtp" in k for k in dynamic):\n'
        "                    vllm_config.quant_config = None\n"
    )
    new = (
        "        original_quant = vllm_config.quant_config\n"
        '        if quant_config and quant_config.get_name() not in ("modelopt_fp4",):\n'
        '            hf_qc = getattr(model_config.hf_config, "quantization_config", None)\n'
        '            if os.environ.get("B70_MTP_BF16_DRAFT") == "1":\n'
        '                print("[B70] MTP draft: forcing unquantized build '\
        '(env B70_MTP_BF16_DRAFT=1)")\n'
        "                vllm_config.quant_config = None\n"
        "            elif isinstance(hf_qc, dict):\n"
        '                dynamic = hf_qc.get("dynamic", {})\n'
        '                if any(k.startswith("-:") and "mtp" in k for k in dynamic):\n'
        "                    vllm_config.quant_config = None\n"
        "        # B70_MTP_NIGHTLY_DRAFT\n"
    )
    if old not in t:
        sys.exit("anchor not found — nightly qwen3_5_mtp.py changed")
    t = t.replace(old, new, 1)

    if "\nimport os\n" not in t and not t.startswith("import os\n"):
        t = t.replace("import torch\n", "import os\nimport torch\n", 1)
        if "\nimport os\n" not in t and not t.startswith("import os\n"):
            sys.exit("could not inject import os")

    open(path, "w").write(t)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
