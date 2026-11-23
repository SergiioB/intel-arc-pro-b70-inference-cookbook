# Evidence: Inkling 2 Mini on vLLM XPU — Opt-in Chunked oneDNN Prefix SDPA + ESIMD XMX Decode

- **Date:** 2026-11-23
- **Hardware:** 1x Intel Arc Pro B70 (32 GB, non-display card, `xe` driver, as-found 230 W power cap)
- **Model:** `thinkingmachines/Inkling-2-Mini` @ `ed40e7da625f594fc9c4990f5f66be24873c22d7` (native NVFP4 L0–19 / MXFP8 L20–23 experts, BF16 dense/attention/`lm_head`, FP8 E4M3 KV cache)
- **Engine:** `vllm-0.34.0rc1.dev1133+g647a76866.xpu` (`647a76866b4559a74e27d1c6b55fbffe85c07ce4`) + `mgaruccio/vllm-inkling2-xpu` @ `2e9d28b04ff23f5d9996a221ffaa0ae3a440d8f9` + [`patches/inkling2-mini-vllm-xpu/`](../../../patches/inkling2-mini-vllm-xpu/APPLY.md), PyTorch `2.15.0+xpu`, Triton `3.9.0`, `icpx 2026.1` AOT `bmg_g31`
- **Observer:** client-side streaming OpenAI `/v1/chat/completions` observer, hardware energy counter (`energy1_input`), live VRAM sampling, `dmesg` watch; same-binary env-flag controls (flag off = byte-identical stock path), unique entropy-first prompts, zero prefix-cache hits

Two independent opt-ins measured in two same-session A/B campaigns:

1. **`INK2_HYBRID_PREFIX_SDPA=1`** — chunked oneDNN flash SDPA (8192-token windows, persistent 16 MiB BF16 scratch) over the bias-free KV prefix of full-attention layers, fp32-LSE-merged with the Triton relative-bias tail. Prefill-only mechanism; decode codepath untouched.
2. **`INK2_ESIMD_DECODE_ATTN=1`** — AOT `bmg_g31` ESIMD/XMX `PagedXmx<M>` two-pass split-KV decode attention (FP8 E4M3 + BF16 KV, `-ze-opt-large-register-file`), gated to `mql<=8`, GQA group `3`, `head_dim 128`, single request. Fixed per-window-type splits (`SWA=16`, `FULL=128`) keep it graph-capture safe. `INK2_SYCL_DECODE_ATTN`'s M=1 route retains precedence.

Both flags default `0` = exact stock Triton behavior.

## Files

| File | Contents |
|---|---|
| `hybrid-prefix-sdpa-20261123.json` | Same-binary A/B of `INK2_HYBRID_PREFIX_SDPA`: kernel microbench (`1.42x` one-shot @ 64K; `1.94x`/`1.99x` chunked 8K oneDNN @ 128K/245K vs Triton rel-attn), live cold prefill `2,706 -> 3,048` / `2,149 -> 2,575` / `1,586 -> 2,023 tok/s` (+12.6%/+19.9%/+27.6%), decode-untouched cells, 5/5 quality gate, safety + closed arms (BF16 SWA ring launch-bound `1.94 vs 1.95 ms`; one-shot >32K oneDNN SDPA trap `0.70x`/`0.46x`) |
| `esimd-decode-20261123.json` | Same-binary A/B of `INK2_ESIMD_DECODE_ATTN` (hybrid SDPA active in both arms): 24/24 kernel parity (`rel_l2 <= 2.63e-3`), isolated per-layer `0.742`/`0.804`/`1.435 ms` at 64K/128K/256K vs Triton `1.37–1.55`/`1.72–1.73`/`3.32–3.34`, live decode `54.1 -> 68.0` / `59.6 -> 76.1` / `37.6 -> 56.6 tok/s` (+11–12%/+19–20%/+32–33% acceptance-normalized step/s), `-22%` to `-34%` J/token, split sweep, quality gate, safety incl. `ctrl-esimd0` incident |

## Honest incident note (control arm, flag-inactive stock path)

The first ESIMD control launch (`ctrl-esimd0`) safety-aborted during its first 64K prefill cell: free VRAM sampled **357 MiB** (< 400 MiB abort threshold) alongside a recovered `bcs` engine reset (`guc_id=16`). The control path is byte-identical stock (flag off), so this is the known ~450 MiB-headroom regime under `gpu_memory_utilization=0.98` + `mbt=2048` + hybrid SDPA scratch — **not an ESIMD effect**; the prior campaign recorded the same signature in its pre-final candidate iterations. The arm was rerun as `ctrl-esimd0-r2` (`vram_abort=250`) and completed the full matrix cleanly at the previously observed **437 MiB** floor, `dmesg` clean, VRAM restored to 32,615 MiB after teardown.

## Displayed headline pairs (provenance)

| Pair | Cells used |
|---|---|
| Prefill `2,706 -> 3,048` @64K | flag-off control e0.9 (`2705.6`) vs flag-on e0.9 replication (`3048.1`); same-session candidate was `3050.1` |
| Prefill `2,149 -> 2,575` @128K | flag-off control e0.2 (`2148.7`) vs flag-on e0.2 replication (`2574.9`); candidate `2576.2` |
| Prefill `1,586 -> 2,023` @245K | flag-off control e0.2 (`1585.9`) vs flag-on e0.2 replication (`2022.9`); candidate `2023.5` |
| Decode `54.1 -> 68.0` @64K | e0.9 same-session cells (`54.06`/`68.00`), +12% step-normalized |
| Decode `59.6 -> 76.1` @128K | e0.9 needle-request cells (`59.57`/`76.10`), +19% step-normalized |
| Decode `37.6 -> 56.6` @245K | e0.9 same-session cells (`37.58`/`56.57`), +33% step-normalized |
| Energy `-22%..-34%` | 128K e0.9 `3.847 -> 3.009` J/tok; 245K e0.2 `5.639 -> 4.112`; 245K e0.9 `6.115 -> 4.061` |

Step-normalized (`step/s = tok/s ÷ mean DSpark acceptance`) is the honest delta for decode: raw tok/s gains (+26%/+28%/+51%) are inflated by run-to-run acceptance variance; the ~0K decode rows (-5%/-24%) are the same noise floor where attention share is ~0. See `esimd-decode-20261123.json` for all cells.

## Reproduce

```bash
# build + install (requires icpx + ocloc on PATH; no GPU needed for compile)
source /opt/intel/oneapi/setvars.sh
VENV=./vllm-inkling-venv patches/inkling2-mini-vllm-xpu/build-esimd-decode-attn-xe2.sh

# serve with both opt-ins
INK2_HYBRID_PREFIX_SDPA=1 INK2_ESIMD_DECODE_ATTN=1 \
  KV_DTYPE=fp8 DSPARK=1 INK2_CUTLASS_MOE=1 CTX=262144 MAX_BATCHED=2048 GPU_MEM_UTIL=0.98 \
  VLLM_BIN=./vllm-inkling-venv/bin/vllm MODEL=thinkingmachines/Inkling-2-Mini \
  benchmarks/inkling2-mini/launch-inkling2-mini-vllm-xpu.sh
```
