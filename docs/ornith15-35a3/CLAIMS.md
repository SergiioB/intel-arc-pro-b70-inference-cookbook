# Ornith-1.5-35B-A3B measured numbers

Self-reported E2 with raw evidence. Isolated C1, cache off, greedy diagnostic.
Not independently reproduced. LocalMaxxing `APPROVED` means accepted
self-report, not independent reproduction.

Recipe: [ORNITH-VLLM-XPU.md](ORNITH-VLLM-XPU.md). Conversion:
[MIXEDCAL-V2.md](MIXEDCAL-V2.md).

## Stack (do not substitute)

| Component | Exact tested value |
|---|---|
| GPU | Intel Arc Pro B70 32 GB, `xe` driver |
| Image | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` |
| vLLM | `0.27.2rc1.dev77+gac7509e2b` |
| `vllm-xpu-kernels` | `0.1.12.3` |
| Dtype | float16 |
| MoE backend | XPU WNA16 (`int_wna16`) |
| Model | [`SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2`](https://huggingface.co/SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2) |
| Cache | `--no-enable-prefix-caching` |
| Timing | client SSE post-first; cold input = actual endpoint tokens / client TTFT |
| Scope | C1 only |
| Power | configured **150 W** unless a cell names another cap |

Same digest as Qwen3.8-27B. Not Qwen3.6 Pi `2c427ef`, not the v0.21
native-int4moe image that produced Qwen3.6 MTP4 204.6.

## Artifact

| Artifact | RTN fallback | Contract |
|---|---|---|
| WikiText GPTQ (control) | 7,605 / 30,720 = 24.76% | 30,720 expert `qweight`, 785 MTP BF16, 0 MTP leaks |
| MixedCal-v2 | 3,186 / 30,720 = 10.37% | same contract; 24,454,916,052 bytes |

Calibration: 128 × 1,536 = 196,608 tokens, seed `15035260819`, corpus SHA-256
`e88ccd5f999a05a3c78bb9ec86a82256255a9ad97068c787dfc564907a377381`.
Layer 39: 297/768 = 38.7% → 75/768 = 9.8%. Per-projection MixedCal split
1,062 / 1,062 / 1,062. Relative RTN cut −58%. Do not call MixedCal-v2 faster.

## No-spec 32K, 150 W, n=5, instance-median of 3 loads

| Cell | Metric | WikiText GPTQ | MixedCal-v2 |
|---|---|---:|---:|
| p512/g128 | client post-first tok/s | 70.80 | 70.74 |
| p8192/g128 | client post-first tok/s | 64.86 | 64.95 |
| p2048/g1 | cold input tok/s | 6935 | 6968 |
| p8192/g1 | cold input tok/s | 6926 | 6963 |

Parity, not faster.

## MTP1 16K, 150 W, n=5 (BF16 draft)

| Artifact | p512/g128 | p8192/g128 | Acceptance |
|---|---:|---:|---|
| WikiText GPTQ | 97.88 | 91.06 | 78.5% (604/769) |
| MixedCal-v2 | 96.43 | 89.85 | 77.0% (595/773) |

## MixedCal-v2 MTP1 at 131,072, U=0.90, n=5

| Cell | client post-first | Acceptance |
|---|---:|---|
| p512/g128 | 98.16 | 85.3% (617/723) |
| p8192/g128 | 95.25 | same window |

## Exact-token capacity (MixedCal-v2)

| Serve | Exact cell | n | client post-first median |
|---|---|---:|---:|
| 65,536 no-spec | p65408/g128 | 3 | 54.49 |
| 131,072 no-spec | p130944/g128 | 3 | 45.84 |
| 131,072 MTP1 + boundary patch | p130944/g128 | 3 | 70.25 |
| 262,144 no-spec, `--kv-cache-memory 6623879680` | p262016/g128 | 3 | 35.35 |

Finish reason `length` on every retained sample. A successful 262K serve is
not a 262K quality claim.

## MTP2 / MTP4, 16K, 150 W, n=5 (BF16 draft)

| Artifact | Depth | p512/g128 | p8192/g128 | Acceptance |
|---|---|---:|---:|---|
| WikiText GPTQ | MTP2 | 82.03 | 77.47 | 41.1% (604/1468) |
| MixedCal-v2 | MTP2 | 84.16 | 75.86 | 41.7% (610/1464) |
| WikiText GPTQ | MTP4 | 64.35 | 59.57 | 20.4% (602/2948) |
| MixedCal-v2 | MTP4 | 66.27 | 59.69 | 22.1% (628/2848) |

MTP4 is **slower than no-spec** (~70.7 p512). Winner remains **MTP1**.
Per-position acceptance on a depth-4 probe: **81 / 15 / 2.5 / 0.5%**.

## DraftINT4 overlay, MixedCal-v2, 16K, 150 W, n=5

Runtime INT4 of **draft** `lm_head` + MTP linears. Target verify stays higher
precision. Shards stay BF16 MTP. Default on (`DRAFT_INT4=1`).

| Serve | p512/g128 | p8192/g128 | Acceptance |
|---|---:|---:|---|
| MTP1 DraftINT4 | 106.27 | 97.16 | 81.9% (602/735) |
| MTP1 BF16 draft | 96.43 | 89.85 | 77.0% (595/773) |

About +10 tok/s vs BF16 draft at 150 W. Not a MixedCal conversion win.

## Prefill lever — paired 150↔230 W, MixedCal-v2, no-spec 32K

One warm server, cache off, three alternating rounds. Matched except
configured cap.

| Round | cap | p2048/g1 median | p8192/g1 median |
|---|---:|---:|---:|
| 1 | 150 W | 7271 | 7036 |
| 1 | 230 W | 9748 | 9647 |
| 2 | 150 W | 7212 | 7050 |
| 2 | 230 W | 9713 | 9683 |
| 3 | 150 W | 7055 | 7062 |
| 3 | 230 W | 9771 | 9670 |

230 W recovers **~9.7k**. 150 W stays **~7.1k**. Prefill lever is **configured
power cap**, not MixedCal. Campaign-window draw **~152 W** vs **~222 W**.
Clocks not pinned.

After each 230→150 drop, the first retained p2048 sample is ~8.7k before later
samples settle ~6.9k. Do not treat that first sample as 150 W sustained.

## Combined 230 W — MTP1 + DraftINT4, 32K

Host harness, same load:

| Cell | n | median | range |
|---|---:|---:|---|
| p512/g128 client post-first | 5 | **106.64** | 104.72–109.16 |
| p2048/g1 cold input | 5 | **9403** | 9359–9424 |

## LocalMaxxing (APPROVED = accepted self-report)

| Id | Serve | Prompt / gen | Cap | `tokSOut` | `tokSPrefill` |
|---|---|---|---:|---:|---:|
| `cmt2tdx5q0hy0mv01koh4xwpw` | MTP1 + DraftINT4, 32K | long prompt, g128 | 230 W | **108.4** | **9072.9** @ 2906 tokens |
| `cmt2sr6gq0himmv01ogieh0c8` | no-spec, 32K | long prompt, g128 | 230 W | **69.9** | **9780** |
| `cmt2sl6eg0hdcmv01gre5o3ub` | MTP1 BF16 draft, 16K | p32/g256 | 150 W | **94.1** | 654.7 |

These are different speculation / prompt / cap cells. Combined `tokSPrefill`
9073 is below no-spec 9780 because MTP first-token work is inside TTFT.

Never publish g=1 `tokSOut` **19153.8**.

## Not a path

- DFlash / DFlash2 — no Ornith hidden-2048 draft
- GDN mixed-split v5 on C1
- Qwen3.6 native-int4moe 204.6 / MTP4 170.91 as Ornith numbers
- BF16 logit/KL or task-quality suite
