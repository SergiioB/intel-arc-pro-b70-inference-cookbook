# Engine setup recipes — exact steps per engine (2026-09-14)

Companion to [ENGINE-COMPARISON-CTX-20260914.md](ENGINE-COMPARISON-CTX-20260914.md).
Every config used in the final ranking, as run. Hardware: 1× Arc Pro B70 per
engine. Weights are NOT the same quant — that asymmetry is deliberate and
labeled: **vLLM = GPTQ-Int4 (19 GB)**, OpenVINO = int8 (25.8 GB),
llama.cpp = Q8_0 (27 GB).

## 1. vLLM XPU — GPTQ-Int4 + MTP4 + fp8 KV + graph capture ON

Weights: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` pinned rev
`9d189a60` (local copy at `/mnt/models2/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-nogidx`).
Image: `f01e24f6c7ff` (vllm/vllm-openai-xpu).

```bash
RENDER_GID=$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')
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
curl -f http://127.0.0.1:8000/health
```

- **Graph capture: ON** (`VLLM_XPU_ENABLE_XPU_GRAPH=1`) — all sweep numbers
  are with graphs. A/B measured: +8.7% decode @512, +1.8% @8K vs OFF.
- **Cold start incl. patches + graph capture: ≈3 min** (observed 170–190 s).
  Second engine init on a hot cache is faster; fresh compile with parallel
  inductor workers can OOM on Level Zero — add
  `TORCHINDUCTOR_COMPILE_THREADS=1` (slower, ~3–4 min) if it dies.
- Client: OpenAI API, single streaming request per run (`stream: true`,
  `stream_options.include_usage`), TTFT = first content chunk, decode =
  (completion_tokens−1)/(wall−TTFT).

## 2. OpenVINO GenAI — int8-ov + MTP nat5

Weights: `OpenVINO/Qwen3.8-27B-int8-ov` (VL split export → VLMPipeline only).
Runtime: openvino 2026.5.0-22967 / genai 2026.5.0.0-3412 nightly,
venv `/mnt/models/.ov-genai-venv`.

```python
import openvino_genai as og
MODEL = '/mnt/models/OpenVINO/Qwen3.8-27B-int8-ov'
sc = og.SchedulerConfig()
sc.max_num_seqs = 1
sc.enable_prefix_caching = False          # MTP hard requirement (linear-attn verifier)
draft = og.draft_model(MODEL, 'GPU.1')    # draft stays f16 — see tuning guide §7b
pipe = og.VLMPipeline(MODEL, 'GPU.1', draft_model=draft, scheduler_config=sc)
cfg = og.GenerationConfig()
cfg.apply_chat_template = False           # render ChatML yourself (Minja bug)
cfg.num_assistant_tokens = 5              # static MTP depth
cfg.do_sample = False
```

- **Pipeline load incl. draft: ~60–70 s** (68.6 s measured with u8 KV).
- Long context: ≤48K MTP+f16 KV; 64K needs mixed KV
  (`VLMPipeline(..., KV_CACHE_PRECISION='u8')`, draft untouched); 96K needs
  no-MTP. 131K blocked on one card (VRAM / compressed-KV decode defect).
- Client: in-process `pipe.generate()`, TTFT from `perf_metrics.get_ttft()`,
  decode = (gen−1)/(wall−ttft).

## 3. llama.cpp — Q8_0 + native draft-MTP

Weights: `unsloth/Qwen3.8-27B-Q8_0.gguf` (27.04 GiB, MTP tensors INCLUDED —
no second model file needed). Build: flashnext 52d4268 + patches,
`build-sycl/bin/llama-server` (oneAPI env required).

```bash
source /opt/intel/oneapi/setvars.sh
/mnt/flashnext-nvme/llamacpp-main-20260910/build-sycl/bin/llama-server \
  -m /mnt/models2/Qwen3.8-27B-Q8_0.gguf \
  -ngl 99 -fa on --spec-type draft-mtp \
  -c 131072 -b 2048 -ub 512 --port 8901
# health: curl http://127.0.0.1:8901/health
```

- **Model load incl. MTP draft context: 41 s** (measured, log timestamp).
- The spec lever is OFF by default: without `--spec-type draft-mtp` decode is
  a flat 14 tok/s; with it, 52→38 tok/s (512→64K). Same GGUF file.
- This fork's llama-bench has no `-c` flag (context follows `-p`) and its
  server caches prompts across requests — for cold-prefill numbers evict
  the cache between runs (follow-up pending; sweep prefill column is INVALID).
- Client: `/completion` with `stream: true`, TTFT = first content chunk,
  decode = (n_predict−1)/(wall−TTFT).

## Client protocol (identical for all three, v2)

- Prompt: shared Eiffel-Tower filler repeated to length + counting task
  ("integers 1 through 30, comma-separated").
- n=3 runs per length (512/2K/8K/32K/64K; OV 64K/96K points n=1–2, marked
  provisional), greedy/temperature 0, 128 output tokens, single streaming
  request per run. Median reported.
- Power: `energy1_input` µJ counter on the CORRECT card
  (GPU.0 `0000:0b:00.0` for vLLM/llama, GPU.1 `0000:0f:00.0` for OV),
  delta/wall = mean W. 230 W cap during measurement, restore 150 W after.

## Startup cost summary

| Engine | cold start | notes |
|---|---:|---|
| vLLM (graph capture ON) | ≈3 min | patches + compile + capture; A/B: graphs +8.7% @512 |
| OpenVINO | 60–70 s | pipeline + draft load, in-process |
| llama.cpp server | 41 s | includes MTP draft context build |
