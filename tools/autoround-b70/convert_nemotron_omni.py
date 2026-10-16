#!/usr/bin/env python3
"""AutoRound W4A16 / W3A16 export of an NVIDIA Nemotron-H Omni checkpoint (nemotron_h_omni).

Used for Nemotron 3.5 Bolt 74B-A9B (remote nemotron_h_puzzle LM) and Nemotron 3.5 Super VL.

Scope (identical for both bit widths):
  quantized : language-model nn.Linear only (routed experts, shared experts, latent
              projections, Mamba2 in_proj/out_proj, attention q/k/v/o_proj)
  BF16      : vision tower, vision projector, embeddings, lm_head, every norm,
              router gate/e_score_correction_bias, mamba A_log/D/dt_bias, conv1d,
              and any MTP layer (no module builds it; copied verbatim from source)

Usage (see run_bolt_export.sh for the full two-card run):
  PYTHONPATH=shims ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
    python convert_nemotron_omni.py --src $SRC --out $OUT --corpus $CORPUS \
      --bits 4 --iters 200 --nsamples 128 --seqlen 1536 --batch-size 1 --device 0,1

Paths:
  --corpus / $CALIB_CORPUS  directory of *.txt.gz text files used for calibration
  $WORK                     scratch root; AutoRound workspace defaults to $WORK/ar_workspace
"""
import argparse
import glob
import gzip
import json
import os
import random
import sys
import time

import torch

CORPUS = os.environ.get("CALIB_CORPUS", "corpus")
WORK = os.environ.get("WORK", os.getcwd())


def log(msg):
    print(f"[convert {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_calib(tokenizer, nsamples, seqlen, corpus, seed=20260609):
    files = sorted(glob.glob(os.path.join(corpus, "*.txt.gz")))
    if not files:
        raise SystemExit(f"no *.txt.gz calibration files in {corpus}")
    rnd = random.Random(seed)
    rnd.shuffle(files)
    texts = []
    for fp in files:
        with gzip.open(fp, "rt", errors="ignore") as fh:
            t = fh.read(4_000_000)
        texts += [t[i : i + 8000] for i in range(0, len(t), 8000)]
    rnd.shuffle(texts)
    texts = texts[:nsamples]
    samples = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=seqlen).input_ids.long()
        samples.append({"input_ids": ids})
    log(f"calibration: {len(samples)} samples x <= {seqlen} tokens from {corpus}")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("SRC"), required="SRC" not in os.environ)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=CORPUS, help="directory of *.txt.gz calibration text")
    ap.add_argument("--bits", type=int, default=4, choices=[3, 4])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=1536)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--torch-compile", action="store_true", help="leave torch.compile enabled during tuning")
    args = ap.parse_args()

    os.environ.setdefault("AR_WORK_SPACE", os.path.join(WORK, "ar_workspace"))
    # The BF16 weights cannot be CPU-resident on a 60 GB host: force the meta-skeleton
    # build plus per-block materialization from the checkpoint.
    os.environ["AR_DISK_STREAM_MODEL"] = "1"
    os.makedirs(os.environ["AR_WORK_SPACE"], exist_ok=True)

    from transformers import AutoProcessor, AutoTokenizer
    # Bolt's remote image processor subclasses transformers.image_processing_backends.TorchvisionBackend,
    # which transformers 5.12.1's ProcessorMixin class check rejects (written for 5.8.1).
    from transformers.processing_utils import ProcessorMixin

    ProcessorMixin.check_argument_for_proper_class = lambda self, name, arg: None
    # Bolt's inner LM is model_type nemotron_h_puzzle; no checkpoint rename is registered for it,
    # so disk-stream name resolution finds no checkpoint tensor for any LM param. Reusing the
    # nemotron_h "backbone."->"model." rule is wrong here: its reverse matches the "model." inside
    # "language_model." first. Register the full-prefix rename instead.
    from transformers.conversion_mapping import WeightRenaming, register_checkpoint_conversion_mapping

    register_checkpoint_conversion_mapping(
        "nemotron_h_puzzle",
        [WeightRenaming("language_model.backbone.", "language_model.model.")],
        overwrite=True,
    )

    from auto_round import AutoRound

    idx = os.path.join(args.src, "model.safetensors.index.json")
    if not os.path.exists(idx):
        log(f"FATAL: {idx} missing (download incomplete / index not fetched)")
        return 2
    with open(idx) as f:
        weight_map = json.load(f)["weight_map"]
    shards = {v for v in weight_map.values() if v.endswith(".safetensors")}
    missing = [s for s in sorted(shards) if not os.path.exists(os.path.join(args.src, s))]
    if missing:
        log(f"FATAL: {len(missing)} shard(s) missing, e.g. {missing[:3]}")
        return 2
    log(f"source complete: {len(weight_map)} tensors in {len(shards)} shards")

    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(args.src, trust_remote_code=True)
        log(f"processor: {type(processor).__name__}")
    except Exception as exc:
        processor = None
        log(f"AutoProcessor unavailable ({type(exc).__name__}: {exc}); continuing with tokenizer only")

    samples = build_calib(tokenizer, args.nsamples, args.seqlen, args.corpus)

    ar = AutoRound(
        args.src,
        tokenizer=tokenizer,
        processor=processor,
        scheme=f"W{args.bits}A16",
        group_size=args.group_size,
        iters=args.iters,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        seed=42,
        dataset=samples,
        ignore_layers="vision_model,vision_projector,lm_head,embeddings",
        device_map=args.device,
        low_gpu_mem_usage=True,
        low_cpu_mem_usage=True,
        enable_torch_compile=True if args.torch_compile else False,
        trust_remote_code=True,
    )
    mc = ar.model_context
    log(f"is_model_patched={mc.is_model_patched} disk_stream_index={'yes' if mc._disk_stream_index else 'no'}")
    if not mc.is_model_patched and mc._disk_stream_index is None:
        log("FATAL: arch patch did not apply and no disk-stream index -> full CPU materialization")
        return 3
    if mc._disk_stream_index is None:
        log("FATAL: meta skeleton not built -> full CPU materialization")
        return 3

    t0 = time.time()
    ar.quantize_and_save(args.out, format="auto_gptq", inplace=True)
    log(f"quantization+export wall time: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
