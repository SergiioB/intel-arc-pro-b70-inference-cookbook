# Nemotron 3.5 Bolt 74B-A9B on two B70: vLLM XPU TP2, AutoRound W4A16

[Family index](README.md) · [dual-B70 topology and oneCCL setup](../DUAL-B70-TP2.md) · [image/patch matrix](../IMAGE-AND-PATCH-MATRIX.md)

This route serves an Intel AutoRound int4 quantization of the BF16 model on two
Arc Pro B70 cards with tensor parallel 2. It is a second route next to the
[one-card llama.cpp route](NEMOTRON-BOLT-B70-LLAMACPP.md). The two routes use
different weights, engines and patches. Do not mix them.

Evidence tier E2: self-reported, one host, two cards, not independently
reproduced. Speed numbers are C1, median of 5 reps, both cards capped at 180 W.

## Stack

| Item | Value |
|---|---|
| Artifact | [`SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym`](https://huggingface.co/SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym) (weights commit `31f574676ecc`, card commit `213539444c7a`). License OpenMDW 1.1. |
| Source model | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16` (139.5 GiB) |
| Quantization | Intel AutoRound W4A16, int4, group 128, symmetric, GPTQ packing. 41 GB (40.65 GiB), 9 shards. No MTP layer. How it was made and checked: [AUTOROUND-W4A16-PROCESS.md](AUTOROUND-W4A16-PROCESS.md). |
| Image | `vllm/vllm-openai-xpu@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7` (vLLM 0.31.0) |
| Patches, in order | 1. [`patches/patch_vllm_worker_affinity.py`](../../patches/patch_vllm_worker_affinity.py) 2. [`patches/patch_xpu_moe_topk_pad.py`](../../patches/patch_xpu_moe_topk_pad.py). Both applied at container start by [`tp2-entry.sh`](../../benchmarks/nemotron35-bolt-74b/tp2-entry.sh). |
| Env | `ZE_AFFINITY_MASK=0,1 B70_WORKER_AFFINITY=1 ONEAPI_DEVICE_SELECTOR=level_zero:0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 CCL_SYCL_ALLTOALL_TMP_BUF=1 HF_HUB_OFFLINE=1`, plus `CCL_KVS_IFACE=lo FI_TCP_IFACE=lo VLLM_HOST_IP=127.0.0.1` |
| Docker | `--network host --device /dev/dri --group-add <render gid> --ipc=host --shm-size 8g --cap-add SYS_PTRACE --entrypoint bash ... /tp2-entry.sh serve /model` |
| Serve flags | `--served-model-name nemotron-bolt --host 127.0.0.1 --port 8000 --trust-remote-code --tensor-parallel-size 2 --dtype bfloat16 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN --max-model-len 65536 --max-num-batched-tokens 8192 --block-size 64 --max-num-seqs 4 --gpu-memory-utilization 0.85 --limit-mm-per-prompt {"image":1,"video":0} --enable-prefix-caching --mamba-cache-mode align` |
| Context | 65,536 by default. 262,144 (the model maximum) is opt-in: `MAX_LEN=262144`. |
| Graph mode | XPU graph capture on (no `--enforce-eager`) |

## Launch

```bash
bash benchmarks/nemotron35-bolt-74b/launch-nemotron-bolt-vllm-tp2.sh /path/to/model-dir 8000
```

The launcher refuses to start if another inference process or container runs.
It binds 127.0.0.1 only and mounts the model directory read-only. It prints the
health, logs and stop commands.

Ready check: the log must show `Using XPUExpertsWNA16` on both ranks. Load time
depends on disk; about 12-15 min on our USB SSD, including graph capture.

Default context is 65,536. `MAX_LEN=262144` raises it to the model maximum.
That loads and works, but a 250K-token prompt takes about 35 min to read, so it
is not the default. At 262,144 max length the KV cache is 0.92-0.94 GiB per
card, which holds 409,375 fp8 tokens. That is a max concurrency of 1.56x for one
full-length request. Weights take 20.0 GiB per card.

## Sampling and thinking

The [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16)
recommends text sampling with temperature `1.0` and top_p `0.95`. The launcher
sets `--override-generation-config '{"temperature":1.0,"top_p":0.95}'` so these
are the server defaults even when the exported quantization's generation
config omits them. Clients such as omp that send no sampling settings rely
on the server defaults. Explicit client settings still take precedence.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron-bolt","messages":[{"role":"user","content":"What is 17*23?"}],
       "temperature":1.0,"top_p":0.95,"max_tokens":16384,
       "chat_template_kwargs":{"enable_thinking":true}}'
```

Set `enable_thinking` to `false` for thinking off. For medium effort, use
`{"enable_thinking":true,"medium_effort":true}`. Reasoning and answer share
the `max_tokens` budget. The historical greedy results below are unchanged.

## The three fixes

1. **Worker affinity** (`patch_vllm_worker_affinity.py`). Each worker gets its
   own `ZE_AFFINITY_MASK` at spawn time, plus a device remap. See
   [DUAL-B70-TP2.md](../DUAL-B70-TP2.md). On vLLM 0.31.0 the Mamba pointer wrap
   part is already upstream, so that anchor miss is expected. The wrapper
   checks the spawn and worker markers instead.
2. **MoE top-k pad** (`patch_xpu_moe_topk_pad.py`). The vllm-xpu-kernels
   `remap_hidden_states` kernel supports TopK 1, 2, 4, 6, 7, 8, 10, 12 and 16
   only. Bolt routes 9 experts per token, so the first forward fails with
   `Unsupported TopK value`. The patch pads to 10 by repeating the last expert
   with router weight 0. The output is unchanged. The cost is about 11% more
   routed-expert work.
3. **Loopback pinning** (`CCL_KVS_IFACE=lo FI_TCP_IFACE=lo VLLM_HOST_IP=127.0.0.1`
   with `--network host`). Without it the two ranks paired endpoints on
   different interfaces (wifi and the docker bridge) and hung at startup.

## Results

### Speed (C1)

Server at `--max-model-len 262144`, prefix caching on, graph mode, one request
at a time, greedy, thinking off, 128 output tokens, a unique random prefix per
request. One discarded warmup, then 5 reps per shape; the table shows the
median and the min-max range. The prefix-cache hit count did not change over
the run (delta 0), so every prompt was cold input. Both cards capped at 180 W;
average draw over the 1,178 s run was 170.8 W (card 0) and 172.3 W (card 1).

| Prompt tokens | Prefill tok/s (prompt/TTFT), median [min-max] | TTFT s | Decode tok/s, median [min-max] |
|---:|---:|---:|---:|
| 886 | 2,925 [2,895-2,943] | 0.30 | 60.5 [60.2-60.7] |
| 8,192 | 2,014 [1,993-2,026] | 4.07 | 50.7 [49.9-50.7] |
| 32,000 | 871 [835-873] | 36.7 | 33.5 [32.0-33.5] |
| 64,000 | 486 [463-487] | 131.7 | 23.0 [23.0-23.0] |

- Graph mode matters. In an earlier check at a 16K config, eager mode
  (`--enforce-eager`) decoded at 10.1 tok/s, about 6x slower.
- Reference on the same host: llama.cpp with NVIDIA's 24 GB GGUF on one card
  decodes at 28.2 tok/s, pp8192 1,431 tok/s.

### Quality and function

262K config, prefix cache on. These are function checks, not speed claims.

- 6-question gate: 5/6. The miss is vision with the question text before the
  image: on our 512x512 red-circle image it answers only "42" (deterministic).
  With the image first it answers "Red circle with 42". In a separate 6-image
  vision battery both orders pass 6/6. Put the image before the text.
- Exact-answer battery: 8/8 (thinking off), including `sum(i*i for i in range(4))` = 14.
- GSM8K: 192/200 = 96.0% (first 200 of the test split, greedy, thinking off).
  This is not a full-benchmark claim.
- Multi-turn with prefix cache on: 3/3. Cache on vs off: 6/6 identical outputs.
  4 concurrent requests: 4/4 correct and identical to serial.
- Needle retrieval (one hidden fact, greedy): found at 31,153 tokens (90%
  depth), 127,072 (10%) and 127,078 (50%).
- Full-context check: one needle at 50% depth in a 249,212-token prompt was
  found. The request took 34.6 min. This is a functional check only.

Quantization static check against BF16: 27,920/27,920 quantized Linear layers,
no NaN, Inf or zero scales. Every unquantized tensor is identical. The export
steps, fixes and checks are in [AUTOROUND-W4A16-PROCESS.md](AUTOROUND-W4A16-PROCESS.md).

## Claim boundary

- Artifact: the AutoRound W4A16 repo above at weights commit `31f574676ecc`.
  Other quantizations are not covered.
- Engine: the pinned vLLM 0.31.0 XPU image plus the two patches and the
  loopback env. Other images or kernel builds are not covered.
- Two B70 cards, TP2, one host, both capped at 180 W. Speed numbers are C1
  only, up to 64,000 prompt tokens.
- Evidence tier E2. Public record `nemotron35-bolt-74b-vllm-tp2-c1-20261016`
  in [`data/benchmarks.v1.json`](../../data/benchmarks.v1.json); evidence in
  [`results/nemotron35-bolt-74b/vllm-tp2-20261016-summary.json`](../../results/nemotron35-bolt-74b/vllm-tp2-20261016-summary.json).

## Not tested yet

- Task benchmarks beyond the first 200 GSM8K questions.
- LiLiCorr DFlash drafter.
- Tool calling.
- Video.
- Concurrency beyond 4.
