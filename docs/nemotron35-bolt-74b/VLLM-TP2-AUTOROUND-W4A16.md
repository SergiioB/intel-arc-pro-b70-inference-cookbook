# Nemotron 3.5 Bolt 74B-A9B on two B70: vLLM XPU TP2, AutoRound W4A16

[Family index](README.md) · [dual-B70 topology and oneCCL setup](../DUAL-B70-TP2.md) · [image/patch matrix](../IMAGE-AND-PATCH-MATRIX.md)

This route serves an Intel AutoRound int4 quantization of the BF16 model on two
Arc Pro B70 cards with tensor parallel 2. It is a second route next to the
[one-card llama.cpp route](NEMOTRON-BOLT-B70-LLAMACPP.md). The two routes use
different weights, engines and patches. Do not mix them.

Evidence tier E2: self-reported, one host, two cards, not independently
reproduced. Speed numbers are a **screen** (median of 3), not a confirmed result.

## Stack

| Item | Value |
|---|---|
| Artifact | [`SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym`](https://huggingface.co/SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym) (weights commit `31f574676ecc`, card commit `213539444c7a`). License OpenMDW 1.1. |
| Source model | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16` (139.5 GiB) |
| Quantization | Intel AutoRound W4A16, int4, group 128, symmetric, GPTQ packing. 41 GB (40.65 GiB), 9 shards. No MTP layer. |
| Image | `vllm/vllm-openai-xpu@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7` (vLLM 0.31.0) |
| Patches, in order | 1. [`patches/patch_vllm_worker_affinity.py`](../../patches/patch_vllm_worker_affinity.py) 2. [`patches/patch_xpu_moe_topk_pad.py`](../../patches/patch_xpu_moe_topk_pad.py). Both applied at container start by [`tp2-entry.sh`](../../benchmarks/nemotron35-bolt-74b/tp2-entry.sh). |
| Env | `ZE_AFFINITY_MASK=0,1 B70_WORKER_AFFINITY=1 ONEAPI_DEVICE_SELECTOR=level_zero:0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 CCL_SYCL_ALLTOALL_TMP_BUF=1 HF_HUB_OFFLINE=1`, plus `CCL_KVS_IFACE=lo FI_TCP_IFACE=lo VLLM_HOST_IP=127.0.0.1` |
| Docker | `--network host --device /dev/dri --group-add <render gid> --ipc=host --shm-size 8g --cap-add SYS_PTRACE --entrypoint bash ... /tp2-entry.sh serve /model` |
| Serve flags | `--served-model-name nemotron-bolt --host 127.0.0.1 --port 8000 --trust-remote-code --tensor-parallel-size 2 --dtype bfloat16 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN --max-model-len 262144 --max-num-batched-tokens 8192 --block-size 64 --max-num-seqs 4 --gpu-memory-utilization 0.85 --limit-mm-per-prompt {"image":1,"video":0} --enable-prefix-caching --mamba-cache-mode align` |
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

At 262,144 max length the KV cache is 0.92-0.94 GiB per card, which holds
409,375 fp8 tokens. That is a max concurrency of 1.56x for one full-length
request. Weights take 20.0 GiB per card. At the 16K config each card uses
26.6 GiB.

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

### Speed screen (C1)

16K config, graph mode, prefix cache off, unique prefixes, 128 output tokens,
greedy, thinking off. Median of 3 after 1 warmup. Card 0 capped at 180 W,
card 1 at 230 W. This is a **screen**, not a confirmed result.

| Prompt tokens | Prefill tok/s (prompt/TTFT) | Decode tok/s |
|---:|---:|---:|
| 886 | 2,866 | 59.5 |
| 8,192 | 2,005 | 49.8 |
| 15,000 | 1,446 | 43.5 |

- Eager mode (`--enforce-eager`) decodes at 10.1 tok/s. Graph mode is about 6x
  faster on decode.
- Reference on the same host: llama.cpp with NVIDIA's 24 GB GGUF on one card
  decodes at 28.2 tok/s, pp8192 1,431 tok/s.
- Card 1 was later capped at 180 W (VRAM temperature 88 C). The screen numbers
  were taken before that change.

### Quality and function

262K config, prefix cache on.

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
  depth), 127,072 (10%) and 127,078 (50%). A 127K-token prompt takes about
  9 min (about 240 tok/s prefill at that length).

Quantization static check against BF16: 27,920/27,920 quantized Linear layers,
no NaN, Inf or zero scales. Every unquantized tensor is identical.

## Claim boundary

- Artifact: the AutoRound W4A16 repo above at weights commit `31f574676ecc`.
  Other quantizations are not covered.
- Engine: the pinned vLLM 0.31.0 XPU image plus the two patches and the
  loopback env. Other images or kernel builds are not covered.
- Two B70 cards, TP2, one host. Speed numbers are C1 only.
- Evidence tier E2. Public record `nemotron35-bolt-74b-vllm-tp2-screen-20261016`
  in [`data/benchmarks.v1.json`](../../data/benchmarks.v1.json); evidence in
  [`results/nemotron35-bolt-74b/vllm-tp2-20261016-summary.json`](../../results/nemotron35-bolt-74b/vllm-tp2-20261016-summary.json).

## Not tested yet

- 5-rep speed confirmation.
- Needles at 250K tokens.
- LiLiCorr DFlash drafter.
- Tool calling.
- Video.
- Concurrency beyond 4.
