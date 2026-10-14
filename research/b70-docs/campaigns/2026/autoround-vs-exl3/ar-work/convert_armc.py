#!/usr/bin/env python3
"""Arm C local AutoRound — CAM-2026-AUTOROUND-VS-EXL3.
Fixes vs the handoff draft, all from the Ornith run ledger
(research/2026/ornith15-autoround-vs-gptq-20260822.md §3):
- scheme as string "W4A16" (0.14.2: sym=True G128 default), not ctor bit kwargs
- ignore_layers as comma string; mtp.* / visual / lm_head / embed stay bf16
- calib samples = {"input_ids": int64 tensor[1,seqlen]} dicts (MLLM path
  forwards dicts verbatim into model(**data))
- ar.model_context.is_model_patched = True post-init (keeps per-block release;
  without it low_cpu_mem silently off -> host-RAM ratchet -> OOM kill)
- quantize_and_save(inplace=True) -> immediate streaming shard save
- model pre-loaded with device_map="auto" across both B70s (54.6G bf16 cannot
  fit one 32.6G card); low_cpu_mem_usage keeps host RAM low during load
"""
import glob, gzip, json, os, random, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

ITERS = int(os.environ.get("AR_ITERS", "2"))
NSAMPLES = int(os.environ.get("AR_NSAMPLES", "8"))
OUTDIR = os.environ.get("AR_OUT", "/out/smoke")
SRC = "/src"
SEQLEN = 1536

tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(SRC, trust_remote_code=True)

files = sorted(glob.glob("/corpus/*.txt.gz"))
random.Random(20260609).shuffle(files)
corpus_texts = []
for fp in files:
    with gzip.open(fp, "rt", errors="ignore") as fh:
        text = fh.read(400000)
    corpus_texts += [text[i:i+6000] for i in range(0, len(text), 6000)]
corpus_texts = corpus_texts[:NSAMPLES]
samples = [{"input_ids": tok(t, return_tensors="pt", truncation=True,
                             max_length=SEQLEN).input_ids.long()}
           for t in corpus_texts]
print(f"calib: {len(samples)} samples", flush=True)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    SRC, torch_dtype=torch.bfloat16, device_map="cpu",
    low_cpu_mem_usage=True, trust_remote_code=True)
print(f"model loaded in {time.time()-t0:.0f}s; devices:",
      {str(v.device) for _, v in model.named_parameters()}, flush=True)


# --- local fix: auto_round 0.14.2 revert_checkpoint_conversion_mapping strips ALL
# (...) groups from source patterns while keeping backrefs in target -> re.error
# ("invalid group reference") on qwen3_5's language_model<->model.layers mapping.
# Robust revert: try unstripped (backrefs intact), then stripped, then unchanged.
import re as _re
import auto_round.utils.common as _arc
import auto_round.compressors.shard_writer as _arsw
import auto_round.compressors.base as _arb

def _revert_ckpt(name, key_mapping):
    if "," in name:
        return ",".join(_revert_ckpt(p, key_mapping) for p in name.split(","))
    for source_pattern, target_patterns in key_mapping.items():
        if isinstance(target_patterns, str):
            target_patterns = [target_patterns]
        for target_pattern in target_patterns:
            for src in (source_pattern.lstrip("^"),
                        _re.sub(r"\(.*\)", "", source_pattern.lstrip("^"))):
                try:
                    new_name, n = _re.subn(src, target_pattern, name)
                except _re.error:
                    continue
                if n > 0:
                    return new_name
    return name

_arc.revert_checkpoint_conversion_mapping = _revert_ckpt
_arsw.revert_checkpoint_conversion_mapping = _revert_ckpt
_arb.revert_checkpoint_conversion_mapping = _revert_ckpt

from auto_round import AutoRound
ar = AutoRound(
    model, tokenizer=tok, processor=proc, scheme="W4A16",
    layer_config=None,
    dataset=samples,
    iters=ITERS, seqlen=SEQLEN, nsamples=NSAMPLES,
    batch_size=1, gradient_accumulate_steps=1,
    low_gpu_mem_usage=True, low_cpu_mem_usage=True,
    ignore_layers="lm_head,mtp,visual,embed_tokens",
    seed=42, device_map=0)

# Ornith fix #4: qwen3_5 absent from MODEL_CONFIG -> low_cpu_mem_usage gate is
# silently disabled unless we assert the flag.
try:
    ar.model_context.is_model_patched = True
    print("is_model_patched forced True", flush=True)
except Exception as e:
    print("is_model_patched set failed:", e, flush=True)

model_q, folders = ar.quantize_and_save(OUTDIR, format="auto_gptq", inplace=True)
print("SAVED", json.dumps({"folders": folders if isinstance(folders, list) else [folders],
                           "iters": ITERS, "nsamples": NSAMPLES}), flush=True)
