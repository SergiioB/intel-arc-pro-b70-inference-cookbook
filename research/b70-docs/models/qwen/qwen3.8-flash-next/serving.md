---
status: living
title: "Serving"
type: model-dossier
date: 2026-09-10
---

---
schema_version: 1
model_id: Qwen/Qwen3.8-Flash-Next
model_slug: qwen3.8-flash-next
document: serving
as_of: 2026-09-10
status: research_foothold
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
engines: [llama.cpp-sycl]
---

# Serving

## Route

Dual Arc Pro B70, llama.cpp SYCL, community M64 GGUF, PLE mmap on CPU/NVMe, compute on both cards (`--tensor-split 1,1 --split-mode layer`). Independent stacks / stock vLLM TP2 are not this recipe (ADR-0008).

Campaign cap **195 W**; restore the card's pre-campaign observed cap. C1 only (`-np 1`). KV `q8_0` K + `q4_1` V. `-fa on` mandatory. Do not use `--no-mmap`/`--mlock` on this ~88 GiB GGUF.

## Runtime pin (2026-09-10 MTP campaign)

| Field | Value |
|---|---|
| Tree | `/mnt/<nvme-disk>/llamacpp-main-20260910` (upstream main `52d42686560a9e8f441f9b9780c8890c37d2802d`) |
| qwen4exp MTP port | dzannotti `qwen4exp-mtp-draft-head.patch` (4/5 hunks, fuzz) + manual hunk #3 (MTP-block tensor walk, graph_mtp ctor, DECODER_MTP routing) + 2 drift fixes: null-guard `mctx_hyb` in `build_layer_attn`; full-width `t_h_nextn` capture (qwen3next masked/unmasked pattern) |
| Multi-token verify kernel | local `mul_mat_vec_q_moe_mt` + `ggml_sycl_mul_mat_vec_q_id_mt` (mmvq.cpp/hpp) + fused-wrapper multi_token branch; M in 2..8, ids [k,M], token-major dst |
| FP32 decode binary | `build-sycl/bin/llama-server` (`GGML_SYCL_F16=OFF`) |
| F16 prefill binary | `build-sycl-f16/bin/llama-server` (`GGML_SYCL_F16=ON`; no _mt kernel yet) |
| GGUF | AtomicChat AD-4.27bpw Q4_K_M M64 (main, no nextn tensors) + dzannotti MTP draft Q4_K_M SHA-256 `1f2a6991…f4c02` |
| IntelLLVM | 2026.0.0 |

## Flags (current best)

### Decode recipe (FP32 binary + MTP)

```text
--device SYCL0,SYCL1 --tensor-split 1,1 --split-mode layer -ngl 99
-ot per_layer_token_embd=CPU -fa on -t 6 -tb 14 -np 1
--cache-type-k q8_0 --cache-type-v q4_1 --no-warmup --no-cache-prompt
-md <mtp-draft.gguf> -ngld 999
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75
--spec-draft-backend-sampling --spec-draft-device SYCL1
LLAMA_ATTN_ROT_DISABLE=1   # mandatory with quantized KV (upstream #21038)
IMM=0, DMMV unset
```

### Prefill recipe (F16 binary, no spec)

Same common flags without `-md`/spec, binary `build-sycl-f16`, `-c 16384 -b/-ub 3072`.

n-max 2 measured worse (23.86); deeper fixed drafting is slower (each drafted token runs the head's MoE); draft on the split (no `-devd`) is break-even; no backend sampling is break-even.

## Numbers (engine C1 timings; PROVISIONAL — NOT FOR PUBLIC HEADLINE)

| Cell | n | Decode | Cold input | Binary/config |
|---|---:|---:|---:|---|
| c8 p512/g128 no-spec | 3 | 26.86 (26.31–26.97) | 183.5 | FP32 52d4268 (old pinned build n=5 was 23.38 → +15% free) |
| c8 p512/g128 no-spec | 3 | 26.21 | 183.9 | F16 (decode wash; keep FP32) |
| c8 p512/g128 +MTP tuned | 3 | 29.90 (28.13–33.03) | 168.7 | FP32, n-max3+bsampling+draft@SYCL1 |
| c8 p512/g128 +MTP tuned | 5 | 26.51 (23.43–32.39) | 170.0 | same, fresh server (high variance band) |
| c8 p512/g128 +MTP+_mt kernel | 3 | 29.51 (25.42–29.76) | 172.1 | multi-token verify kernel (correct, within noise) |
| c16 p9096 +MTP | 3 | 21.38 | 426.1 | FP32+MTP (prefill pays spec overhead) |
| c16 p9096 no-spec | 3 | 22.96 (22.63–22.98) | **585.9** | F16 b3072 (prefill leader; old-build n=5 594.49) |

Acceptance (natural prose+code, whole phase): 13,156/14,996 = 87.7%, 7,158 rounds, ≈2.84 committed tokens/round. Screens: 0.77–1.00 per request, mean len 2.5–3.8.

Quality verdict: PASS — see `results/2026/09/qwen38-flashnext-mtp-20260910T1624Z/quality/QUALITY-VERDICT.md` (42/42 code tests both modes; greedy divergence inside the OFF/OFF envelope).

## Bottleneck statement (2026-09-10)

Base decode ≈ 13% of the 608 GB/s serial-split roofline — execution overhead, not bandwidth. MTP rounds cost ≈ 2.4× a bare step: each draft step ≈ 20–23 ms (vs 37 ms for the whole 48-layer token) — fresh graph schedule + host-bounced h-handover + per-step sync. Verify batches now have a fused multi-token kernel (_mt, correct) but the gain is inside screen noise: the draft step is the binding constraint. Next lever: draft schedule reuse / on-device handover (architect rank 3), then GDN/HC fusion; persistent draft kernel deferred (IQ3_S/IQ4_NL are not DPAS-native formats).

GPU1 pkg loaded-idle ≈ 62–63 °C at 195 W; staged thermal gate 60–62 °C used throughout (2026-09-01 precedent).
