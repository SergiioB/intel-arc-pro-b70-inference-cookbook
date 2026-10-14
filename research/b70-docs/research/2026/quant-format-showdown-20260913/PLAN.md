---
title: "Qwen3.8-27B Quant Format Showdown — Overnight Campaign Plan"
type: research
date: 2026-09-13
status: final
---

# Qwen3.8-27B Quant Format Showdown — Overnight Campaign Plan

Date: 2026-09-13 (overnight 09-13 → 09-14 UTC)
Intent: research → bench (Lane 2 tuning screen → fresh n=5 finalist confirmation)
Status authority: all numbers produced here are **PROVISIONAL — NOT FOR PUBLIC HEADLINE** until the claims audit passes.

---

## 0. Objective

Find the best weight-format tradeoff for the B70 for Qwen3.8-27B (dense): faster
and/or better inference while keeping quality, measured against the registered
AutoRound/GPTQ INT4 G128 sym baseline (vLLM XPU, `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`,
best known 102.61 tok/s p512/g128 with MTP4, recommended sampling, Run48 n=5).

## 1. Background — architect judgment of the proposed plan

The dropped plan (`~/Downloads/b70-quant-challengers-autoround.md`)
proposed a custom `.b70pack` sidecar ABI + custom kernels (mixed Q4_G128+Q6/Q8,
MXFP4 E2M1 G32, NVFP4 E4M3 G16, QAT). The b70-architect review concluded:

- **Reject as executable tonight.** `scripts/b70pack_build.py` does not exist; no
  loader/kernel consumes the sidecar; the bench commands reference nonexistent
  `build-sycl-0717` and `/models/Qwen3.8-Flash-F16.gguf`.
- Hardware corrections: MXFP4 scales are 8-bit E8M0 (not 2-bit); NVFP4 is E2M1
  values + E4M3 block scales (not "E4M3 weights"); G128 is not a DPAS panel
  format and a W4A16 path is not native W4A4 DPAS; M≥8 is not a universal DPAS
  crossover; two B70s are two independent 608 GB/s cards (no P2P, ADR-0008) —
  this campaign uses **one card, serial arms**.
- Scale-traffic math: vs INT4-G128-sym (4.125 bpw), MXFP4-G32 ≈ +3.0% bytes,
  NVFP4-G16 ≈ +9.1% bytes. "8× scale traffic" claims confuse entry count with bytes.
- Recommended realistic ladder: same-engine GGUF format matrix, not new formats.
- One tooling note: the installed quantizer advertises `MXFP4_MOE` (unverified for
  Qwen3.8 on SYCL; optional smoke only).

## 2. Host constraints discovered (2026-09-13 evening)

| Fact | Consequence |
|---|---|
| No F16/BF16 GGUF or safetensors source on host (BF16 dir = dangling symlinks; GLM5 Flash absent) | Download `Qwen/Qwen3.8-27B` BF16 as the conversion source; verify SHA256 |
| AutoRound/GPTQ INT4 baseline checkpoint present (`/mnt/<sata-disk>/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-nogidx/`, config-verified GPTQ bits=4 sym G128 desc_act=false lm_head=false, mtp.* BF16) | vLLM arm is measurable without download |
| FP8 checkpoint absent (`/mnt/<models-disk>/hf/qw38-fp8` gone) | FP8 vLLM arm only if a spare download window appears; else reuse registered numbers as E2 historical, clearly labeled |
| Builds: `~/llama.cpp/build-sycl-latest-qw38` (commit 1692f9e50, 2026-08-14) full toolset | Use for quantize/imatrix/bench/server |
| GPUs idle; GPU0 hwmon5 cap 180 W, GPU1 hwmon6 cap 230 W; temps 54–55 °C | Set **150 W both** at campaign start; restore 150 W at end |
| T-FORCE 358 GiB free; NVMe 300 GiB; `/` only ~26 GiB | All artifacts + evidence on T-FORCE / B70-DOCS |
| RAM ~53 GiB free | llama-quantize via mmap is OK |

## 3. Arms

Conversion source (shared, hash-verified): `src/Qwen3.8-27B-BF16/` (HF `Qwen/Qwen3.8-27B`, exclude `original/*`).

| ID | Arm | Format detail | Purpose |
|---|---|---|---|
| B0 | vLLM AutoRound/GPTQ INT4 G128 sym, spec OFF | exact registered checkpoint | matched serving-route control |
| B4 | vLLM AutoRound/GPTQ INT4 G128 sym + MTP4 | same + validated spec route | deployed-product reference (registered 102.61 cell reused if evidence matches; else re-measured) |
| G0 | fresh Q4_K_M | plain K-quant | same-engine 4-bit workhorse control |
| G1 | fresh Q5_K_M | plain K-quant | quality-step comparator (first drop if night is short) |
| G2 | fresh mixed: attn + router + norms + embed + lm_head at Q6_K (router/norms F32-ish), FFN stays Q4_K_M | `--tensor-type` overrides; narrow enumerated set per QDR | the feasible version of plan §1.1 "protect hot/sensitive tensors" |
| GR | fresh Q8_0 | near-lossless reference | quality anchor + KL reference logits |
| G3 | fresh IQ4_XS + imatrix | only if imatrix completes within capped budget | compact-challenger arm |
| FX | Flash-Next AD 4.27bpw (existing artifact, pinned build) | reuse prior overnight evidence only | directional cross-model note, not a matched cell |

Not attempted tonight (architect + contract): custom sidecars/kernels, NVFP4/W4A4,
QAT, TP2/stock dual-engine, KV sweeps, 128K capacity claims, FP8 re-download unless
time allows, Ornith/GLM/DeepSeek (no weights).

## 4. Cells and protocol mapping

llama.cpp (llama-bench, synthetic, immune to HTTP cache bugs):

- Screen (all GGUF arms): `pp512` + `tg128` (from p512 context) and `pp8192` + `tg128` (from p8192 context), n=3 after one same-shape warmup — Lane 2 two-anchor screen.
- Finalists (fresh, n=5): same two anchors + `pp2048` + one fixed natural workload request via llama-server HTTP (fixed prompt, recorded hash), engine `timings` retained.
- Flags: `-ngl 99 -ncmoe 0 -fa on -ctk q8_0 -ctv q4_1 -t 8 --no-mmap -b 8192 -ub 4096 -c <declared> -dev SYCL0`; env per protocol §5 (`ZE_AFFINITY_MASK=0`, immediate commandlists off, persistent SYCL cache off).
- Statistic: median of measured samples; max labeled "maximum observed"; every raw repetition retained.

vLLM (B0/B4):

- Protocol §9: `vllm bench serve` where it covers the cell; else controlled HTTP with unique entropy-first prefixes, exact endpoint token verification (`len(encoded["input_ids"])`), zero `vllm:prefix_cache_hits` delta for cold cells, `ignore_eos` for fixed gN.
- Cells: p2048/g1, p8192/g1 (cold input anchors), p512/g128, p8192/g128 (decode anchors), n=5 measured + 1 same-shape warmup. B4 adds MTP acceptance counters as diagnostics.
- Sampling: recommended non-thinking (temp 0.7, top_p 0.8, top_k 20, presence 1.5) for representative numbers; greedy runs labeled "greedy synthetic" only.

Quality (independent of engine speed):

- `llama-perplexity` on a fixed deterministic corpus (local wiki-style text; corpus + hash recorded), same chunk geometry for all arms.
- KL divergence of each 4-bit arm against the GR (Q8_0) reference logits (`--kl-divergence-base`), same corpus/chunks.
- Coherent-output smoke per arm: fixed greedy prompt, deterministic replay ×2, JSON/UTF-8 wellformedness on a structured prompt.

## 5. Power, thermals, VRAM

- `configured_cap_W = 150` (both cards) for the whole campaign; discovered hwmon paths recorded (currently hwmon5=GPU0, hwmon6=GPU1 — re-resolve live, never hard-code).
- Average card draw from `energy1_input` (µJ) deltas over each measured interval, monotonic timestamps, 1 s cadence monitor (energy + `temp*_label` sensors) into CSV per arm.
- Post-load `visible_avail` from debugfs recorded per arm; ≥2048 MiB required to proceed with performance cells; <500 MiB abort.
- Empty-GPU gate before every load (processes, containers, ~31,000 MiB visible_avail).
- One engine + one client at a time; serial arms.

## 6. Gates and stop rules

- Promote (per declared objective): speed winner = quality passes (ppl within noise of Q8_0 anchor / KL small) + improvement beyond dispersion on both anchors; quality winner = better quality at explicitly accepted speed cost; capacity winner = a target workload fits that did not before.
- Kill/fallback: if BF16 download fails by ~02:00 local → fallback campaign = re-measure B0 matched cells + re-quantization arms from the existing Q4_K_M **clearly labeled double-quantized diagnostics** + Flash-Next evidence consolidation.
- If vLLM image fails on this stack → B0/B4 fall back to registered E2 evidence, labeled as not re-measured tonight.
- Any OOM/throttle/contamination → sample retained and marked, cell rerun or excluded with reason.

## 7. Evidence layout

```
results/2026/09/qw38-quant-format-showdown-20260913/
  manifest.json                 # identities: host, builds, checkpoints, hashes, env
  download-qw38-27b-bf16.log
  monitor/<arm>.csv             # 1 s energy/temp cadence
  vllm-b0/ vllm-b4/             # server logs, bench serve results, counter snapshots
  gguf-g0/ gguf-g1/ gguf-g2/ gr/ gguf-g3/   # llama-bench raw output, server logs, smokes
  perplexity/<arm>/             # raw ppl + KL artifacts
  conversions/                  # quantize logs + timing per arm
  FINDINGS-POINTER.md
research/2026/quant-format-showdown-20260913/
  PLAN.md (this file)
  FINDINGS.md                   # full evaluation, PROVISIONAL labels
  ARCHITECT-VERDICT.md          # condensed judgment on the dropped plan
```

Artifacts (GGUFs) live on `/mnt/<sata-disk>/quant-showdown-20260913/arms/`.

## 8. Schedule (target, local)

1. ~evening: plan + download start + gates + 150 W.
2. +1–2 h: source lands (verify hashes) → start conversions serially (G0, G2, GR, G1; imatrix in parallel capped 45 min; G3 only if imatrix done).
3. Overlap: while conversions run on CPU, run vLLM B0/B4 on GPU (matched cells).
4. Serial GGUF arms on GPU (screen n=3 → finalists n=5) + perplexity/KL per arm.
5. Early morning: FINDINGS.md + cleanup (stop engines, 150 W, gates re-run, evidence retained).

## 9. Non-goals / publication

No commit/push/publication implied. Everything labeled PROVISIONAL. Registry and
public catalog updates are follow-ups requiring the claims audit and user authorization.
