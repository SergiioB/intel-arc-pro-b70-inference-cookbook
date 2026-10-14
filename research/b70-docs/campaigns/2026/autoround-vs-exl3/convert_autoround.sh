#!/bin/bash
# Local AutoRound conversion — CAM-2026-AUTOROUND-VS-EXL3 arm C.
# Changed variables vs the removed GPTQ artifact: converter (AutoRound signSGD,
# not GPTQModel) + calibration corpus (our real-workload E1 corpus, code/agent
# heavy — not wikitext-2). Same target: W4A16 sym G128, mtp.* BF16 preserved,
# desc_act false. Runs in the pinned vLLM XPU image on GPU1 (ZE_AFFINITY_MASK=1)
# OUTSIDE measurement windows (thermal coupling rule, RL-077).
#
# NOTE setvars/pipefail workaround: source via bash -lc, never inside set -e script.
set -euo pipefail

SRC=/mnt/<nvme-disk>/b70-bf16-qwen38-20260926
OUT=/mnt/<sata-disk>/b70-autoround-local-qwen38-20260926
CORPUS=~/B70-DOCS/results/2026/09/b70-improvement-probes-20260926/shortlist-rebuild-ours-20260926/corpus
IMG=vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f

mkdir -p "$OUT"

docker run --rm --entrypoint bash --dns 1.1.1.1 --dns 8.8.8.8 \
  -v /mnt/<sata-disk>/ar-wheels:/wheels:ro \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -e ZE_AFFINITY_MASK=1 -e HF_HUB_OFFLINE=1 \
  -v "$SRC":/src:ro -v "$OUT":/out -v "$CORPUS":/corpus:ro "$IMG" \
  -c '
    pip install --quiet --no-index --find-links /wheels auto-round==0.14.2 && python - <<"PY"
from auto_round import AutoRound
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob, gzip, json, random
files = sorted(glob.glob("/corpus/*.txt.gz"))
random.Random(20260609).shuffle(files)
# 128 texts x ~1536 tokens, mirroring the Ornith contract scale
corpus = []
for fp in files[:64]:
    with gzip.open(fp, "rt", errors="ignore") as f:
        text = f.read(200000)
    corpus += [text[i:i+6000] for i in range(0, len(text), 6000)][:6]
corpus = corpus[:128]
tok = AutoTokenizer.from_pretrained("/src", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "/src", torch_dtype="bfloat16", device_map="xpu:0", trust_remote_code=True)
ar = AutoRound(model, tokenizer=tok, bits=4, group_size=128, sym=True,
               iters=200, asym=False, round_vals="auto",
               calib_data=corpus, seqlen=1536)
ar.quantize()
ar.save_quantized("/out", format="auto_gptq")
print("SAVED", json.dumps({"texts": len(corpus)}))
PY'
echo "conversion done -> $OUT"
