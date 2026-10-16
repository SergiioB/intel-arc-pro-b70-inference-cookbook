#!/usr/bin/env python3
"""Tiny random Bolt-shaped (nemotron_h_omni + nemotron_h_puzzle) checkpoint, real key naming.

Needs only the small files of the real BF16 repo (config, remote code, tokenizer, chat template
and model.safetensors.index.json), not the weight shards. Use it to run the converter on CPU in
under a minute before the real export.

usage: build_tiny_bolt.py --src $SRC --out $WORK/tiny
       (or SRC=... TINY=... build_tiny_bolt.py)
"""
import argparse, json, os, re, shutil, collections, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.environ.get("SRC"), help="real BF16 repo dir (small files suffice)")
ap.add_argument("--out", default=os.environ.get("TINY"), help="output dir for the tiny checkpoint (wiped)")
_a = ap.parse_args()
if not _a.src or not _a.out:
    ap.error("--src and --out (or SRC and TINY) are required")
SRC = os.path.abspath(_a.src)
OUT = os.path.abspath(_a.out)

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)
for fn in os.listdir(SRC):
    if fn.endswith(".py") or fn.endswith(".json") and fn != "model.safetensors.index.json" or fn.endswith(".jinja"):
        shutil.copy(os.path.join(SRC, fn), os.path.join(OUT, fn))

cfg = json.load(open(os.path.join(SRC, "config.json")))
llm = cfg["llm_config"]
# all K dims divisible by 128 (converter default group_size=128, same as real run)
llm.update(
    hidden_size=256, head_dim=64, num_attention_heads=4, num_key_value_heads=2,
    mamba_num_heads=4, mamba_head_dim=64, n_groups=2, ssm_state_size=16, chunk_size=64,
    moe_latent_size=128, moe_shared_expert_intermediate_size=256, intermediate_size=256,
    use_mamba_kernels=False,
)
llm["block_configs"] = [
    {"block_type": "mamba"},
    {"block_type": "moe", "moe_intermediate_size": 128, "n_routed_experts": 8, "num_experts_per_tok": 3},
    {"block_type": "attention"},
    {"block_type": "moe", "moe_intermediate_size": 256, "n_routed_experts": 12, "num_experts_per_tok": 3},
    {"block_type": "mamba"},
    {"block_type": "moe", "moe_intermediate_size": 384, "n_routed_experts": 16, "num_experts_per_tok": 3},
]
llm["layers_block_type"] = [b["block_type"] for b in llm["block_configs"]]
llm["num_hidden_layers"] = len(llm["block_configs"])
for k in ("n_routed_experts", "moe_intermediate_size", "num_experts_per_tok", "num_experts", "num_local_experts"):
    llm.pop(k, None)
cfg["vision_config"].update(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, max_img_size=224, image_size=224,
    min_num_patches=4, max_num_patches=196,
)
cfg["vision_config"]["args"].update(max_num_patches=196, min_num_patches=4, cpe_max_size=224)
cfg["vit_hidden_size"] = 64
cfg["projector_hidden_size"] = 128
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)

from transformers import AutoConfig, AutoModelForCausalLM
conf = AutoConfig.from_pretrained(OUT, trust_remote_code=True)
torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(conf, trust_remote_code=True, torch_dtype=torch.bfloat16)
sd = model.state_dict()
print("model class", type(model).__name__, "LM", type(model.language_model).__name__, "tensors", len(sd))

import tempfile
from safetensors.torch import load_file as _lf
_tmp = tempfile.mkdtemp(dir=os.path.dirname(OUT))
model.save_pretrained(_tmp)
sd = {}
for _f in os.listdir(_tmp):
    if _f.endswith(".safetensors"):
        sd.update(_lf(os.path.join(_tmp, _f)))
shutil.rmtree(_tmp)
print("save_pretrained tensors", len(sd))
real = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
real_dt = {}
# dtype of special tensors in the real source (header read)
import struct
def hdr(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

def disk_name(k):
    k = k.replace("language_model.model.", "language_model.backbone.", 1)
    k = k.replace("backbone.embeddings.weight", "backbone.embeddings.weight")
    k = re.sub(r"^vision_projector\.mlp1\.norm\.", "mlp1.0.", k)
    k = re.sub(r"^vision_projector\.mlp1\.linear1\.", "mlp1.1.", k)
    k = re.sub(r"^vision_projector\.mlp1\.linear2\.", "mlp1.3.", k)
    return k

tensors = {}
for k, v in sd.items():
    t = v.detach().clone()
    if t.is_floating_point():
        t = (torch.randn_like(t, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    dk = disk_name(k)
    if dk.endswith("e_score_correction_bias"):
        t = torch.randn(t.shape, dtype=torch.float32) * 0.01
    elif dk.endswith(("A_log",)):
        t = torch.log(torch.arange(1, t.numel() + 1, dtype=torch.float32)).to(torch.bfloat16)
    elif dk.endswith(".D") or "norm" in dk.split(".")[-2]:
        t = torch.ones_like(t)
    tensors[dk] = t.contiguous()

pat = lambda k: re.sub(r"\.\d+\.", ".N.", k)
rp = collections.Counter(pat(k) for k in real)
tp = collections.Counter(pat(k) for k in tensors)
only_real = sorted(set(rp) - set(tp))
only_tiny = sorted(set(tp) - set(rp))
print("patterns real", len(rp), "tiny", len(tp))
print("ONLY_IN_REAL", only_real)
print("ONLY_IN_TINY", only_tiny)

from safetensors.torch import save_file
keys = sorted(tensors)
half = len(keys) // 2
wm = {}
for i, chunk in enumerate((keys[:half], keys[half:]), 1):
    fn = f"model-{i:05d}-of-00002.safetensors"
    save_file({k: tensors[k] for k in chunk}, os.path.join(OUT, fn), metadata={"format": "pt"})
    wm.update({k: fn for k in chunk})
json.dump({"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())}, "weight_map": wm},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=1)
print("saved", len(tensors), "tensors; bias dtype",
      {k: str(v.dtype) for k, v in tensors.items() if k.endswith("e_score_correction_bias")})
