# Engine setup recipes — reproduce every lane from zero

This page is the step-by-step guide to reproduce all four measurement lanes in
[ENGINE-COMPARISON-CTX-20260914.md](ENGINE-COMPARISON-CTX-20260914.md).
Every command is complete and self-contained. The two-number summaries exist
so you know what a working setup should produce; the full data lives in the
charts and the raw JSON files under `results/qw38-ov-mtp-20260914/`.

Hardware used: 2× Intel Arc Pro B70 (32 GB each) on the cookbook runner.
One-card lanes use a single card; the FP8 lane needs both.

---

## 0. One-time environment

```bash
# Cookbook clone (patches live here)
git clone https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook
cd intel-arc-pro-b70-inference-cookbook
export COOKBOOK="$PWD"

# GPU permissions for Docker
export RENDER_GID=$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')

# Intel oneAPI (llama.cpp and python tooling)
source /opt/intel/oneapi/setvars.sh
```

---

## 1. vLLM XPU — GPTQ-Int4 + MTP4 + fp8 KV + graph capture

What it is: vLLM serving Qwen3.8-27B with 4-bit weights, a 4-token MTP
speculative head, fp8 KV cache, and XPU graph capture enabled.

Weights (pinned): `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
rev `9d189a60` — local copy at
`/mnt/models2/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-nogidx`.
Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff...` (tag `f01e24f6c7ff`).

```bash
docker run -d --name qw38 -p 8000:8000 \
  --device /dev/dri --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro \
  -v /mnt/models2/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-nogidx:/model:ro \
  -v $COOKBOOK/patches/patch_mtp_nightly.py:/patch_mtp.py:ro \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash f01e24f6c7ff -lc \
  'set -e; python /patch_mtp.py; exec vllm serve /model \
   --quantization gptq --dtype float16 --max-model-len 100000 \
   --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 --port 8000 \
   --max-num-seqs 1 --max-num-batched-tokens 8192 \
   --no-enable-prefix-caching --served-model-name qwen38 \
   --language-model-only \
   --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"'

# Wait for readiness, then check
curl -f http://127.0.0.1:8000/health
```

What a working setup looks like:
- Cold start (patches + first compile + graph capture): about 3 minutes
  (170–190 s observed).
- If first init OOMs during compile: add `TORCHINDUCTOR_COMPILE_THREADS=1`
  (slower, 3–4 min).
- Graph capture is on by `VLLM_XPU_ENABLE_XPU_GRAPH=1`; measured +8.7%
  decode at 512, +1.8% at 8K versus off.

---

## 2. OpenVINO GenAI — int8-ov + MTP5

What it is: OpenVINO's native pipeline for the 27B Qwen in int8 weights, with
a 5-token static MTP head (draft model loaded separately, kept at f16).

Weights: `OpenVINO/Qwen3.8-27B-int8-ov` on the HF org (VL split export).
Runtime: openvino 2026.5.0-22967 / genai 2026.5.0.0-3412 nightly.

```python
import openvino_genai as og

MODEL = '/mnt/models/OpenVINO/Qwen3.8-27B-int8-ov'

sc = og.SchedulerConfig()
sc.max_num_seqs = 1
sc.enable_prefix_caching = False      # required: MTP verifier is linear-attention

draft = og.draft_model(MODEL, 'GPU.1')   # draft stays f16
pipe = og.VLMPipeline(MODEL, 'GPU.1', draft_model=draft, scheduler_config=sc)

cfg = og.GenerationConfig()
cfg.apply_chat_template = False       # render the ChatML template yourself
cfg.num_assistant_tokens = 5          # static MTP depth
cfg.do_sample = False
```

Request loop: in-process `pipe.generate(prompt, cfg)`; TTFT from
`perf_metrics.get_ttft()`; decode = (generated−1)/(wall−ttft).

Long context behavior (all measured, details in the tuning guide):
- Up to 48K: MTP + f16 KV.
- 64K: mixed KV — main cache u8, draft untouched
  (`VLMPipeline(..., KV_CACHE_PRECISION='u8')`).
- 96K: MTP off (u4 or f16 weights).
- 131K does not fit one card.

---

## 3. llama.cpp — Q8_0 + native draft-MTP

What it is: the flashnext llama.cpp fork (SYCL backend) serving the 27B in
Q8_0, with speculative decoding enabled through the MTP tensors that ship
inside the same GGUF file — no second model needed.

Weights: `unsloth/Qwen3.8-27B-Q8_0.gguf` (27.04 GiB).
Build: flashnext tree `52d4268` + patches, `build-sycl/bin/llama-server`.

```bash
source /opt/intel/oneapi/setvars.sh
/mnt/flashnext-nvme/llamacpp-main-20260910/build-sycl/bin/llama-server \
  -m /mnt/models2/Qwen3.8-27B-Q8_0.gguf \
  -ngl 99 -fa on --spec-type draft-mtp \
  -c 131072 -b 2048 -ub 512 --port 8901

# health
curl http://127.0.0.1:8901/health
```

What a working setup looks like:
- Model load including the MTP draft context: 41 s.
- Without `--spec-type draft-mtp` you get the flat 14 tok/s floor. With it,
  decode roughly triples (52→38 across 512→64K using the same file).
- Client: `/completion` streaming; TTFT = first content chunk; decode =
  (n_predict−1)/(wall−TTFT).
- Cold prefill rerun note: the server caches prompts across requests, so
  prefill numbers need cache eviction between runs to be valid.

---

## 4. vLLM XPU — FP8 W8A16 + MTP8 + tensor parallel ×2 (needs two B70s)

What it is: vLLM serving the 27B with full 8-bit FP8 weights (30.9 GB) split
across both B70s via tensor parallelism, 8-token MTP head. This is the
highest-fidelity configuration; it physically needs two cards because
30.9 GB of weights do not fit in one 32 GB card.

Weights: `Qwen/Qwen3.8-27B-FP8` (per-layer safetensors, 66 files).
Image: same `f01e24f6c7ff`.
Patches: worker-affinity (`patch_vllm_worker_affinity.py`) + oneCCL
contract from [DUAL-B70-TP2.md](../DUAL-B70-TP2.md).

```bash
# 1. Download the weights (30.9 GB)
mkdir -p /mnt/models2/Qwen3.8-27B-FP8
# iterating the 66 files from the HF API, e.g.:
#   aria2c -c -x8 -s8 -k4M -d /mnt/models2/Qwen3.8-27B-FP8 \
#     "https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/main/<file>"
# 2. Verify: every file in model.safetensors.index.json exists (66/66)

# 3. Serve (both cards, TP2)
docker run -d --name qw38-fp8 -p 8005:8000 \
  --device /dev/dri --group-add "$RENDER_GID" \
  --ipc=host --cap-add SYS_PTRACE --workdir / \
  -v /dev/dri:/dev/dri:ro \
  -v /mnt/models2/Qwen3.8-27B-FP8:/model:ro \
  -v $COOKBOOK/patches/patch_vllm_worker_affinity.py:/patch_affinity.py:ro \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_ALLTOALL_TMP_BUF=1 \
  --entrypoint bash f01e24f6c7ff -lc \
  'set -e; python /patch_affinity.py; exec vllm serve /model \
   --quantization fp8 --dtype bfloat16 --tensor-parallel-size 2 \
   --max-model-len 100000 --async-scheduling --block-size 64 \
   --mamba-ssm-cache-dtype float16 --max-num-batched-tokens 4096 \
   --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --port 8000 \
   --max-num-seqs 1 --no-enable-prefix-caching --served-model-name qwen38 \
   --language-model-only \
   --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":8}"'

curl -f http://127.0.0.1:8005/health
```

What a working setup looks like:
- Cold start: 306 s (two-card compile + graph capture).
- Do not add `CCL_ZE_IPC_EXCHANGE`, `CCL_ATL_TRANSPORT`, `FI_PROVIDER`, or
  `CCL_ATL_SHM` — per the TP2 infra guide they break the topology.
- The worker-affinity patch must run before `vllm serve`; the oneCCL
  thresholds above are the validated values for non-P2P dual-B70.

---

## 5. Run the sweep (same protocol for every lane)

The measurement script lives in
`results/qw38-ov-mtp-20260914/` (`ctx-sweep-*.py`). It:

1. Builds the shared prompt: Eiffel-Tower filler repeated to the target
   length (512 / 2K / 8K / 32K / 64K / 96K), then the counting task.
2. Warms up once, then runs n=3 requests, greedy, 128 output tokens,
   one streaming request each.
3. Reports median TTFT, prefill tok/s, decode tok/s (post-first-token).

```bash
# Example: vLLM Int4 lane (server already listening on :8000)
/mnt/models/.ov-genai-venv/bin/python \
  /home/sergio/B70-DOCS/scripts/tmp/ov-mtp-20260914/ctx_sweep_vllm_v2.py
```

The JSON it writes is the source of truth that feeds the charts.

---

## Cheat sheet (what to expect)

| Lane | cold start | decoded behavior |
|---|---:|---|
| vLLM Int4 · 1× B70 | ≈3 min | 76 → 51 tok/s, flat to 96K |
| OpenVINO int8 · 1× B70 | 60–70 s | 62 → 25 tok/s (KV ceiling at 64K+) |
| llama.cpp Q8_0 + MTP · 1× B70 | 41 s | 52 → 38 tok/s |
| vLLM FP8 · 2× B70 | 306 s | 73 → 57 tok/s, no decay |