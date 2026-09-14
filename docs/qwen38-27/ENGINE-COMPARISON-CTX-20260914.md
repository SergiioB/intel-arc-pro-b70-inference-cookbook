# Qwen3.8-27B engine context sweep — OpenVINO vs vLLM vs llama.cpp (2026-09-14)

Same model family, one Arc Pro B70 per engine, same filler text and lengths
(512→98K/128K), greedy, n=5 (≤8K) / n=3 (≥32K), C1.

## Decode tok/s (median post-first-token)

| len | OpenVINO GenAI¹ | vLLM XPU² | llama.cpp SYCL³ |
|---:|---:|---:|---:|
| 512 | **62.1** | 58.1 | 14.0 |
| 2K | **56.5** | 34.5 | 14.0 |
| 8K | **50.8** | 13.2 | 14.0 |
| 32K | **41.3** | 3.4 | 14.0 |
| 64K | **25.0**⁴ | 1.4 | 14.0 |
| 96K | 15.2⁵ | 0.8 | **14.0** |
| 128K | — | — | init-crash⁶ |

## Prefill tok/s (input / TTFT)

| len | OpenVINO | vLLM | llama.cpp |
|---:|---:|---:|---:|
| 512 | **1537** | 1155 | 230 |
| 2K | **1815** | 1170 | 227 |
| 8K | **1643** | 1103 | 224 |
| 32K | **1119** | 924 | 220 |
| 64K | **780** | 753 | 213 |
| 96K | 587 | **632** | — |

¹ int8-ov VL split, VLMPipeline, MTP nat5, KV f16 ≤48K
² GPTQ-Int4 sym G128, MTP4 BF16-draft, fp8 KV, 0.27.2rc1, docker
³ Q8_0 GGUF (unsloth), flashnext 52d4268, fa on, no spec decode
⁴ mixed KV fix: main u8 + draft f16 (see OPENVINO-B70-TUNING.md §7b)
⁵ no-MTP u8-KV path (MTP ceiling ~64-80K on 32GB)
⁶ 27.04 GiB weights + 128K f16 KV exceeds 32 GB at model init

## Power (energy-counter means)

| Engine | mean W | cap | card |
|---|---:|---:|---|
| OpenVINO MTP | 202–211 | 230 W | GPU.1 |
| vLLM MTP4 | ~149 | 150 W | GPU.0 |
| llama.cpp Q8_0 | ~90 | 150 W | GPU.0 |

Note: OV ran at its 230 W measurement cap; vLLM/llama ran at the host default
150 W cap on the other card. Neither vLLM nor llama was power-limited (both
well under cap), so the cap difference does not explain the decode gaps.

## Protocol notes

- llama.cpp measured via llama-bench pp/tg (engine metrics, fresh process
  per length); OV and vLLM measured client-side post-first-token. llama-bench
  tg is engine decode rate — comparable at C1.
- vLLM served via docker (OpenAI API, streaming TTFT); OV/llama native.
- Watt measurement: `energy1_input` µJ counter. Earlier drafts showed ~45 W
  for vLLM/llama — that was the idle card (wrong PCI device); corrected here.
- OV ceiling work (KV u8, mixed draft precision) documented in
  OPENVINO-B70-TUNING.md; raw JSONs in B70-DOCS `qw38-ov-mtp-20260914/`.

## Reading

- **≤32K chat/RAG workloads: OpenVINO + MTP is the clear pick** — 3-4x the
  decode of the alternatives, prefill 1.4-7x llama.cpp.
- **Long-context tail (96K+): llama.cpp** — flat 14 tok/s at every length,
  never falls over; OV no-MTP still beats it at 96K (15.2) but caps out.
- **vLLM MTP4 on this stack collapses with context** (58→0.8 tok/s at 96K);
  at short context it matches OV's decode within 7% but with Int4 weights.
- 128K on one card requires ≤Int4 weights (llama Q8_0 init-crashes; OV needs
  no-MTP u8 KV; vLLM configured max 100K).
