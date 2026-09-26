# Qwen3.8-27B Quant Format Showdown — one Arc Pro B70 (2026-09-13/14)

> **PROVISIONAL self-report** (single lab machine, raw samples retained, claims-audited;
> not independently reproduced). Engine-native and client-side metrics are labeled per
> row and never mixed. Cross-engine rows are directional only — engine, KV
> implementation, kernels, and metric family all differ.

## Question

For dense hybrid-GDN Qwen3.8-27B (`qwen3_5`: 48 gated-delta-net + 16 full-attention
layers) on **one** Intel Arc Pro B70 (32 GB): how do classic GGUF quant formats —
freshly quantized from the same verified BF16 source — compare against the registered
GPTQ-INT4 (W4A16, sym G128) vLLM route, and what do the power cap and execution mode
actually cost?

## Provenance

- Source: official `Qwen/Qwen3.8-27B` BF16, all files SHA256-verified against HF LFS
  metadata before conversion; converted to F16 GGUF, then `llama-quantize` per arm
  (Q4_K_M, Q5_K_M, Q8_0, and a mixed variant: 48× `attn_qkv`/`ssm_out`→Q6_K,
  `attn_gate`→Q8_0, `token_embd`→q6_K over the Q4_K_M base).
- llama.cpp SYCL commit `1692f9e50` (build 361). Fleet flags: `-ngl 99 -fa on
  -ctk q8_0 -ctv q4_1 -t 8 -b 8192 -ub 4096`, one GPU, `llama-bench` r=5
  (statistic = mean of 5 repetitions), synthetic prompts (cache-immune).
- vLLM: pinned XPU image `vllm/vllm-openai:xpu@sha256:f01e24f6…`, TP1/PP1, one GPU,
  compile-only (torch.compile on, XPU graphs off), prefix cache off (config-verified;
  per-cell counter snapshots retained), max-model-len 9216, max-num-seqs 8,
  recommended non-thinking sampling (temp 0.7 / top_p 0.8 / top_k 20 / presence 1.5),
  `ignore_eos`, unique entropy-first prompts (wikitext-derived), exact rendered token
  counts (+12 chat-template tokens), 1 discarded same-shape warmup + n=5 measured per
  cell, all `finish_reason=length`.

## GGUF format matrix (llama-bench, engine-native, 150 W cap)

| Arm | Bytes | pp512 | pp2048 | pp8192 | tg128 @ p512 | tg128 @ p8192 |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 16.81 GB | 611.7 | 848.0 | 844.3 | **20.01** | **20.06** |
| mixed Q4/Q6/Q8 | 18.61 GB | 622.7 | 849.0 | 847.5 | 18.85 | 18.72 |
| Q5_K_M | 19.54 GB | 677.1 | 871.3 | 856.8 | 16.97 | 16.90 |
| Q8_0 | 29.05 GB | 664.1 | 864.6 | 853.1 | 13.62 | 13.54 |

Readings:

- **Decode tracks streamed weight bytes** (20.0 → 18.9 → 17.0 → 13.6 tok/s). Effective
  streaming: Q4_K_M ≈336 GB/s, Q5_K_M ≈332, Q8_0 ≈396 — the simpler Q8_0 dequant
  path utilizes GDDR6 better per byte, but cannot overcome its byte count.
- **Long-prompt prefill is byte-flat** (pp2048/pp8192 within 844–871 across
  16.8–29 GB): kernel/compute-bound, not bandwidth-bound. The pp512 anchor
  (611–677) sits in the launch-bound short-prompt regime — do not mix it into
  bandwidth statements.
- **Mixed-precision "rescue" of GDN/attn tensors is a double negative as tested**:
  +10.7 % bytes bought −5.8 % decode and no measurable quality gain. The Q4_K_M
  preset's own tensor mixture already allocates precision well on this architecture.

### Decode dispatch at occupied depth (screen, 2026-09-26)

The matrix above is measured at short occupied depth. At long depth the quantized-KV
attention scan dominates, and llama.cpp [#26689](https://github.com/ggml-org/llama.cpp/pull/26689)
(merged 2026-08-28, **after** the pinned `1692f9e50` build) matters: on Xe2/BMG it
dispatches quantized-KV single-token decode to the TILE kernel instead of VEC.

Twin builds of `1692f9e50` differing only in that 6-line hunk, `llama-bench -n 128
-d 32768` (tg at occupied 32K), ABAB interleaved n=3/arm on one B70:

| Build | tg128 @ depth 32K |
|---|---:|
| pinned dispatch (VEC) | 11.57 tok/s median |
| #26689 dispatch (TILE) | **13.07 tok/s median (+13.0 %)** |

Every interleaved pair positive (+8.8/+13.3/+15.9 %). Lab screen (see
[WHAT-WORKED](../WHAT-WORKED.md)); external ~2x figures were measured on Q4_0-K cells.
Unchecked before this becomes the pinned recipe: MTP/speculative serving, where
verification was already TILE upstream and one external full-serving test saw ~0 %.

## vLLM GPTQ-INT4 route (client-side, C1, clean-host canonical set)

| Cell | Spec-off | + MTP-4 | Multiplier |
|---|---:|---:|---:|
| p512/g128 post-first decode | 24.41 tok/s (TPOT 41.0 ms) | **45.72 tok/s** | +87 % |
| p8192/g128 post-first decode | 23.89 tok/s (TPOT 42.0 ms) | **37.11 tok/s** | +55 % |
| p2048/g1 cold input | 1048 tok/s | 1018 tok/s | — |
| p8192/g1 cold input | 1047 tok/s | 1028 tok/s | — |

- Clean-host confirmation run (third independent n=5 matrix): 24.32 / 24.21 —
  canonical set confirmed within 0.8 %. Measured window draw 146 W avg (150 W cap),
  package max 73 °C.
- MTP-4 acceptance (server-log windows): mean accepted length ≈3.1 (range 2.5–3.9).
  MTP throughput on this model is strongly **content-driven** (±10 % across prompt
  draws) — never quote an MTP cell without its acceptance context.
- **Metric semantics**: under MTP-4, 128 completion tokens arrive in 33–49 SSE
  chunks. Rates above use endpoint completion tokens over the client decode window;
  a naive chunk-counting client undercounts ~3.4×. 45.7 tok/s is completion
  throughput, not smooth streaming rate.
- **`--enforce-eager` is a 2.36× decode trap** on this stack (10.35 vs 24.41 tok/s,
  matched config): compile-only is mandatory for representative XPU numbers.

## Power cap A/B (150 W → 230 W stock cap, fresh servers, n=5)

| Cell | 150 W | 230 W | Delta |
|---|---:|---:|---:|
| Spec-off p512/g128 decode | 24.41 | 24.25 | −0.7 % (cap-insensitive) |
| Spec-off p8192/g128 decode | 23.89 | 24.34 | +1.9 % |
| Spec-off p8192/g1 cold input | 1039 | **1586** | **+53 %** |
| MTP-4 p8192/g1 cold input | 1026 | **1493** | **+45 %** |

Decode is launch-gap-bound (~75–110 W draw) and does not care about the cap;
**prefill is heavily clipped at 150 W**. Measured window draw at 230 W: 201–202 W
avg, package max 82 °C. The B4 p512 decode cell moved 45.7 → 41.1 across caps with
a flat no-spec control and ~flat acceptance — that is content variance, not power.

## Quality: do NOT rank quant formats with `llama-perplexity` on hybrid-GDN models

Wikitext-2 PPL (`-c 4096`, 72 chunks, unquantized KV) came back non-monotonic in
precision: Q4_K_M 6.5474 < mixed 6.6528 < Q8_0 6.6779 < Q5_K_M 6.7943.

Root cause (bounded follow-up session):

- **The full-F16 reference scored 6.5545 — worse than Q4_K_M on 7 of 8 chunks** and
  statistically identical to Q8_0. Full precision cannot legitimately lose to 4-bit
  weights ⇒ the metric is confounded; quantization kernels are exonerated (F16 has
  none).
- Chunk-geometry sensitivity (same corpus prefix): PPL **grows** with chunk length
  (Q4_K_M: 6.29 → 6.44 → 7.15 at c2048/c4096/c8192) and the Q8−Q4 gap widens
  +0.09 → +0.56 — consistent with recurrent-state accumulation dominating the
  measurement on linear-attention hybrids.

Upstream issue with full evidence: [ggml-org/llama.cpp#28879](https://github.com/ggml-org/llama.cpp/issues/28879).
If a quality axis is needed for this model class, use per-position logit parity
against an F16 reference — not perplexity.

## Repro sketch

```bash
# GGUF cells (llama-bench, one GPU):
llama-bench -m qw38-27B-Q4_K_M-fresh.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q4_1 \
  -p 512,2048,8192 -n 128 -r 5 -b 8192 -ub 4096 -t 8 -dev SYCL0

# vLLM (compile-only; MTP-4 variant adds):
#   --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

Private raw evidence (per-request JSONL, cell JSONs, PPL logs, monitor CSVs,
counter snapshots, audit report) retained in the lab result root; available on
request. Deterministic-replay smokes passed on the Q4_K_M and Q8_0 arms.
