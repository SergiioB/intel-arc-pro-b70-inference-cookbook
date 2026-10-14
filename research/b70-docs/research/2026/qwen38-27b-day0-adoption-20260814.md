---
title: "Qwen3.8-27B Day-0 B70 Adoption — 2026-08-14"
type: research
date: 2026-08-14
status: final
---

# Qwen3.8-27B Day-0 B70 Adoption — 2026-08-14

The complete record of release-day adoption: llama.cpp GGUF + LocalMaxxing
submissions before dinner, a self-made GPTQ artifact with preserved MTP head
by night, and a vLLM XPU MTP4 decode champion of **81.26 t/s** (n=5 median,
+17% over Qwen3.6-27B's 69.3 on the matched class). Includes every failure
that got us there — one host OOM reboot, one ENOSPC, and five rejected
"breakthrough" levers with evidence.

Evidence root: `results/2026/08/qwen38-gguf-20260814T150404/`
Skill: `.agents/skills/qwen38-adoption-zeroday/SKILL.md`

## 1. Timeline (all times local B70, 2026-08-14)

| Time | Event |
|---|---|
| 15:00:01Z | `Qwen/Qwen3.8-27B` completes: 18 shards, 55.59 GB BF16, tag `qwen3_5`, multimodal, Apache-2.0 |
| 15:00:44Z | Automation-fired session observes release (placeholder 40 min earlier) |
| ~15:05 | Architecture diff vs 3.6-27B: ZERO structural deltas; same MTP head tensor names |
| ~15:07 | unsloth GGUF quants live; Q4_K_M (17.11 GB) chosen, downloaded in ~3 min |
| ~15:10 | GPU blocked by H3 ComfyUI campaign9 (user's workload — left untouched; drained during download) |
| ~15:15 | llama-server load @128K: 10,165 MiB free; correctness rungs 1-4 pass |
| 15:41:52Z | LocalMaxxing submit #1: llama.cpp tg128 mean 17.78 (n=5, 150W) |
| 15:50:32Z | LocalMaxxing submit #2: llama.cpp tg128 mean 22.30 (n=5, 230W) |
| ~15:55 | vLLM path decided (no community GPTQ exists; FP8/OV/MLX/NVFP4 all unusable on XPU) |
| ~16:00 | BF16 split-download: 13 shards → /mnt/<models-disk>, 5 → /mnt/<models-disk>, symlink dir |
| ~16:05 | gptqmodel 7.3.2 XPU container built (torch 2.13.0+xpu untouched, deps installed no-deps) |
| 16:14 | Quant run1 → **host OOM reboot** (54 GB BF16 materialized into 30 GB RAM) |
| 16:42 | Run2 with 24 GB swapfile on models2 SSD + watchdog → quantizes fine, **save dies ENOSPC** (swapfile ate the output budget) |
| 17:57 | Run3: output rerouted to `/` → SUCCESS in 63+6 min |
| 19:13 | Contract verified: 400 qweights sym G128, 15 mtp tensors BF16, 0 leaks, 19.56 GB |
| ~19:15 | vLLM no-spec probe (missed `-p` port publish first; relaunched) |
| ~19:30 | no-spec decode 24.4 t/s flat across anchors; cold input 1,540/1,146 @150W eager |
| ~19:40 | 3.6-matched setup: 230W + XPU graphs + `--max-num-seqs 64` (GDN Mamba-cache caps seqs at 72) |
| ~19:55 | MTP1 47.5 → MTP2 64.3 → MTP4 80.6 (n=3), acceptance 99/97% |
| ~20:15 | MTP4 n=5: **81.26 / 77.04 median**, cold input 1,486/1,731 |
| ~20:30 | Breakthrough push, all rejected: MTP8 71.0, async 75.8, 16K budget 78.5, newest nightly 78.9, W4A8 corrupt |
| ~20:45 | Champion confirmed; state restored (150W, engines down) |

## 2. Architecture verdict (why day-0 was possible)

Config-diff of `Qwen/Qwen3.8-27B` vs our proven
`Qwen3.6-27B-MTP-Preserved-GPTQ-Int4`:

- `model_type qwen3_5_text`, `Qwen3_5ForConditionalGeneration` (nested `text_config`)
- 64 layers: 3× linear_attention (GDN) + 1× full_attention repeating; head_dim 256;
  24 q-heads / 4 kv; mrope_interleaved [11,11,10], partial rotary 0.25, θ=1e7
- `mtp_num_hidden_layers: 1`; identical 15 `mtp.*` tensor names
- **Only diffs:** our 3.6 checkpoint's own GPTQ block, `pad_token_id`, vision
  `model_type` string cosmetic, one dropped duplicate top-level rope_theta
- Normalized tensor-family diff: empty both directions
- The 2.4T-A95B flagship (released 08-08) kept `qwen3_5_moe_text` — the whole
  3.8 generation stayed in the qwen3_5 family

Consequence: every 3.6 artifact/patch/recipe transferred. This was the
single biggest de-risker of the day.

## 3. Artifacts produced

### 3.1 unsloth GGUF (community) — llama.cpp path
`unsloth/Qwen3.8-27B-GGUF` @ `430473d9`, Q4_K_M 17,106,773,984 B,
SHA-256 `7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`.
GGUF v3, `general.architecture=qwen35`. MTP head stripped by converter
(no llama.cpp draft artifact exists day-0).

### 3.2 Self-quantized GPTQ-Int4 + BF16 MTP — vLLM path (ours)
`/qw38-gptq-out/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
- gptqmodel 7.3.2 **on the B70 XPU itself** (torch 2.13.0+xpu native)
- bits=4, sym=true, group_size=128, desc_act=false, lm_head=false — the
  exact 3.6 MTP-Preserved contract
- calibration: wikitext-2-raw-v1, 128 texts
- mtp.* excluded via `QuantizeConfig(dynamic={"-:.*mtp.*": {}}, offload_to_disk=False)`
- verified: 400 qweight tensors, 15 mtp BF16 tensors, 0 quantization leaks,
  19.56 GB total
- quantize.py: `/mnt/<models-disk>/qw38-quant/quantize.py` (self-verifying)
- NOT yet uploaded to HF (SergiioB upload is a pending authorized gate)

## 4. Correctness ladder (both paths)

| Rung | GGUF/llama.cpp | GPTQ/vLLM |
|---|---|---|
| 1 metadata/load | PASS (clean, qwen35) | PASS (native Triton warmup qwen3_5_text) |
| 2 coherent smoke | PASS 5/5 (vowel count 11 correct) | PASS 5/5 (Spanish correct; reasoning text lands in `content` — qwen3_5 template trait on this stack) |
| 3 deterministic replay | PASS (md5 identical) | PASS (md5 identical) |
| 4 token exactness | PASS (g128 ignore_eos → length) | PASS |
| 5 reference parity | NOT RUN | NOT RUN |
| 6 task quality | NOT RUN | NOT RUN |

## 5. Performance results

### llama.cpp (b10255+ production build, q8_0/q4_1 KV, FA on, C1, llama-bench)
| Cell | 150W n=5 | 230W n=5 |
|---|---|---|
| tg128 | 17.78 ±0.17 | **22.30 ±0.01** |
| pp2048 | 785 ±21 | 1116 ±0.96 |
| pp4096 | 808 ±2.1 | 1137 ±1.9 |
| pp8192 | 787 ±1.7 | 1121 ±1.9 |

Matched-cap note: 3.6 Q4_K_M @150W ran tg128 21.3 (n=3, Run 9) — 3.8 is
~16% slower on llama.cpp at matched conditions. Power scaling 150→230W
measured +25% decode on 3.8.

LocalMaxxing submissions (both accepted, same engine entry):
- 15:41:52Z — 150W record: tokSOut 17.78, tokSPrefill 787.45
- 15:50:32Z — 230W record: tokSOut 22.30, tokSPrefill 1121.40
- Claims-audit correction applied pre-submit: llama-bench reports **mean ±
  stdev**, not median (auditor caught the label error); 3.6 comparison
  discloses n-mismatch (n=5 vs n=3)

### vLLM XPU (pinned `2c427ef` digest, patches nightly+boundary, XPU graphs,
fp8 KV, batched 8192, max-num-seqs 64, 230W, C1, cache off, exact tokens
via /tokenize, entropy-first unique prefixes)

| Mode | p512/g128 | p8192/g128 | Cold-in p8192 |
|---|---|---|---|
| no-spec (150W, eager) | 24.4 | 24.7 | 1,146 |
| MTP1 (accept 99.0%) | 47.5 | 46.0 | 1,705 |
| MTP2 (97.0%) | 64.3 | 59.5 | 1,740 |
| **MTP4 n=5** | **81.26 median** (max 81.39; one 56.6 dip) | **77.04** (75.3-77.2) | **1,731** |

Spec gain MTP4 vs no-spec: 3.33×. vs 3.6 Run 31 MTP4 (69.3 p512):
**+17%** — the new generation's draft path is stronger, inverting the
llama.cpp "slower than 3.6" story. vs our llama.cpp 230W (22.30): 3.6×.

## 6. The failure ledger (what it took)

1. **Host OOM reboot (quant run1).** gptqmodel materialized the 54 GB BF16
   into 30 GB RAM despite mmap expectations. Lesson: quantization loaders
   materialize; budget RAM+swap ≥ model size or die. Fix: 24 GB swapfile on
   the models2 SSD + a watchdog that kills the container at <4 GB combined
   headroom (kills the job, not the host).
2. **ENOSPC at save (run2).** The swapfile consumed the exact space the
   16 GB output needed — 13/16 GB written, then the in-memory quantized
   weights were lost with the container. Lesson: budget source + swap +
   output **+ reserve on every mount simultaneously**; the artifact lives
   where the space is (`/` in our case), not where the convention says.
3. **vLLM port not published** — first probe served into the void (`-p`
   missing). Diagnosed by "HTTP server started" in log vs connection refused.
4. **Chunk-counter bug in spec streaming** — MTP emits 2 tokens per SSE
   chunk; a naive chunk counter halved the reported rate. Fix: count real
   tokens via `stream_options.include_usage` completion_tokens.
5. **MTP graph capture blocked** — default max_num_seqs 256 vs 72 available
   GDN Mamba-cache blocks. Fix: `--max-num-seqs 64`.
6. **pkill -f self-match** — a watchdog cleanup command killed its own shell
   (the pattern matched the bash -c wrapper). Use pgrep + exact PID kill.

## 7. Rejected levers (the "push further" matrix, all evidence retained)

| Lever | Result | Why rejected |
|---|---|---|
| MTP8 | 71.0/75.1, accept 70.8-81% | single-layer draft acceptance decay beats depth gain |
| async-scheduling | 75.8/74.3 | regresses dense qwen3_5 (Nemotron MoE-only win) |
| 16K batched budget | 78.5/74.5 | no prefill gain, slight decode loss |
| newest nightly ac7509e (v0.27.2rc1.dev77) | 78.9/74.6 | below pinned digest; pinned stays champion |
| W4A8 (env-gated patch) | no prefill gain + **output corruption** ("Paris!!!!!!!") | our artifact carries g_idx (3.6 fast path needed no-g_idx); quality-broken regardless. DEAD for this family |

## 8. Bottleneck verdict (b70-architect, verified 2026-08-14)

| Phase | Primary limit | Secondary | Confidence |
|---|---|---|---|
| no-spec decode | GDDR6 weight-streaming bandwidth | kernel/launch overhead | high |
| MTP4 decode | weight bandwidth + draft/verify overhead | graph/launch efficiency, 1-layer draft cost | medium-high |
| long-prefill | XMX/DPAS compute-path utilization | GDN non-GEMM kernels, register pressure, launches | high |
| short-prefill | XMX tile occupancy + fixed overhead | GDN kernels/launches | high |

### Decode arithmetic
- no-spec 24.4 t/s x 20.43 GB (19.03 GiB upper bound) = **498.5 GB/s = 82% of 608 GB/s**
  → bandwidth-bound; identical rate at p512/p8192 confirms (context-independent).
- The naive "MTP4 = 2 full weight reads per 5 tokens" model gives 109% of bandwidth —
  **physically impossible, therefore disproven**. Empirical model: MTP4 = 3.330x no-spec
  → 5 accepted tokens cost 1.502 target-forward equivalents (0.300/token) → ~498 GB/s
  effective again (~82%). MTP4 is bandwidth-ANCHORED with material draft/verify overhead.
- **95 t/s is NOT proven impossible**: it needs either ~96% of nominal bandwidth at
  current speculative efficiency, or ~14.5% lower MTP step cost (1.502 → 1.284
  forward-equivalents per 5 outputs), or a combination. async/newer-nightly regressions
  (−3 / −2.5 t/s) prove several percent is still software-sensitive.

### Prefill arithmetic
- 1,731 t/s x 2 x 27.32B = **94.6 TOPS effective = 6.4% of the inferred 1,468-TOPS INT4
  peak** (7.9% at base-clock scaling; 12.9% vs the disputed halved 734 figure). NOT
  bandwidth-bound (weights amortize to ~4.3 GB/s at p8192). The wall is end-to-end XMX
  utilization: GDN state/conv/norm launches lower XMX duty cycle between GEMMs.
- **3,000 t/s needs 163.9 TOPS (11.2% of inferred peak) = 1.73x pipeline efficiency**,
  or active compute cut to ~15.8B/token. Silicon has the headroom; no config lever we
  tested delivers it. Named software levers (untested): -ze-opt-large-register-file build
  flag, DPAS tile verification for XPUwNa16, GDN gate/state/conv fusion, nonstandard
  head-dim attention kernel paths, per-op prefill profile.
- MoE reality check: 8.3K t/s is only 52.7% of the ideal 9.1x active-param scaling —
  grouped-GEMM MoE does NOT achieve better end-to-end XMX utilization than dense
  (49.8 vs 94.6 TOPS effective); its prefill win is almost entirely fewer active params.

### Measurement gaps before calling anything a hard ceiling
1. MTP4 phase split (draft vs verify vs sampling vs graph-replay time)
2. device memory-read counters per mode (82% vs 90% vs 96% bandwidth utilization)
3. touched-weight inventory (vision tower bytes are NOT read by text decode)
4. isolated vLLM MTP0/MTP4 150W-vs-230W A/B (llama.cpp's +25% does NOT transfer)
5. PMU GT-clock sampling by phase; prefill per-op breakdown; DPAS busy counters

## 9. Open gates (all require explicit authorization)

1. exact-128K staging + boundary-patch validation at 128K
2. task-quality screen before any production recommendation
3. claims audit of the 81.26/1,731 wording before public use
4. HF upload of the GPTQ artifact to SergiioB
5. new llama.cpp build (1692f9e5, GDN/q4_K-FFN fusions) A/B vs 0804
6. MoE-track day-0 when a 32GB-fit Qwen3.8 MoE quant lands (the 3K-prefill
   machine)
7. benchmark-history.md Run entry + portfolio update after audit
