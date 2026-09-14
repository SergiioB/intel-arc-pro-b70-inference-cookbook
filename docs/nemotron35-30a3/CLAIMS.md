# Nemotron family — claims from observations

Copy numbers from this page. If a sentence is not here, do not invent it.

LocalMaxxing `APPROVED` means the payload was accepted into the public leaderboard.

The speed card and 120K capacity ladder are documented in
[NEMOTRON-DFLASH-B70.md](NEMOTRON-DFLASH-B70.md).
No-spec graph n=5 is a separate campaign on the same target / image family.

**Hardware / stack**

- One Intel Arc Pro B70 32 GB, C1 (`max_num_seqs=1`)
- Configured cap **150 W**
- Cache **explicitly off** (`--no-enable-prefix-caching`)
- Public image `vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57`
- vLLM `0.26.1rc1.dev668+g3ee2df303`, kernels `0.1.12.3`
- Timing: client monotonic SSE
  - decode = `(completion_tokens - 1) / (end - first)`
  - input = `prompt_tokens / TTFT` (**cold input rate**, not isolated prefill,
    not llama-bench `pp`)

## Allowed headlines

| Claim | Exact wording allowed |
|---|---|
| Representative DFlash decode | **186.61 t/s median** C1 client post-first at **p2048/g128**, n=5, range 174.60–201.83 |
| DFlash long-prompt decode | **157.92 t/s median** at p8192/g128 n=5 (143.50–170.25) |
| DFlash Lane 1 input | **7160 t/s median** cold input at **p8192/g1** n=5 (7117–7226) |
| DFlash short-prompt input | **6456 t/s median** at p2048/g1 n=5 |
| Window acceptance | **1830 / 3521 = 52.0%** at `n_spec=7` |
| Matched-except-speculation | **1.81×** at **p8192/g128 only**: DFlash 157.92 vs no-spec 87.25, both n=5 |
| No-spec graph floor | **93.00 / 87.25** t/s at p512/p8192 g128 n=5 (eager was 21.8) |
| VRAM after DFlash load | **5826 MiB** `visible_avail` at **16K speed card** / U=0.90 (Run 36) |
| DFlash capacity ceiling (completed) | **119,904** tokens at `max-model-len=120000` (p119872+g32, n=1). After-load **5328 MiB**, KV **295,000** (Run 38) |
| Draw (DFlash n=5 windows) | cell averages ~149–160 W; peak interval-average **179.3 W**; pkg max **68.0 °C** |

## Forbidden / withdrawn

| Do not say | Why |
|---|---|
| “DFlash 10k prefill” | 10,371 → **10,349** is **no-spec n=3** cold input from p8192/**g128** TTFT |
| Headline **194.6** | p512/g128 median, but family range **140.20–220.01** |
| Isolated n=3 214 / 185 | Another inference container respawned mid-load; superseded |
| Native MTP works | Acceptance historically **0%** on this stack |
| Faster than a 5090 / Spark / H100 | No matched C1 protocol on those SKUs |
| Verified by LocalMaxxing | Record `cmsr9po4w000ams01e4fc5qhj` is an accepted leaderboard payload |
| 128K Nemotron vLLM | **Not run.** Isolated DFlash completed **119,904** tokens at `max-model-len=120000` (Run 38). Speed card is still the 16K n=5 matrix |
| Capacity decode rates (g32) | Run 38 TTFT / post-first numbers are diagnostics on n=1 g32 cells |
| Isolated engine prefill = 7160 | 7160 is prompt/TTFT |
| “16K budget → 9874 / 10k prefill” | n=3 screen only. Fresh n=5 at batched=16384 was **6096** (−15% vs 7160) and decode dropped to 167. Keep batched=8192 |
| A local Docker tag is required | Public digest is the reproduce default. Kernel `at::zeros` is still [vllm-xpu-kernels#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524) + `0001`. Do not apply `0002`. |

## HF ids (keep)

- `SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym`
- `SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16`

Do not rename. They already state the conversion contract.
