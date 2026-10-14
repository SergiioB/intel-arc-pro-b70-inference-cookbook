---
title: "b70-architect verdict — quant challengers plan (2026-09-13)"
type: research
date: 2026-09-13
status: final
---

# b70-architect verdict — quant challengers plan (2026-09-13)

Source plan: `~/Downloads/b70-quant-challengers-autoround.md`
Reviewer: b70-architect agent, two consultations (plan judgment + validation ladder).

## Verdict per candidate

| Candidate | Verdict | Key reasoning |
|---|---|---|
| 1.1 Mixed Q4_G128 + Q6/Q8 hot tensors | NEEDS-FIX (best practical quality experiment) | Selective precision is sound, but mixing GPTQ G128 packing with GGUF K-quants needs loader support; the feasible GGUF approximation is `--tensor-type` mixtures. "tok/s flat" is unsupported — raising always-active tensors (attn/GDN runs every token) increases active bytes/token. |
| 1.2 MXFP4 E2M1 G32 | NEEDS-FIX | MXFP4 scale is 8-bit E8M0, not "2-bit exponent". No demonstrated native E2M1×E2M1 DPAS K64 path for W4A16 on this stack. +3.0 % bytes vs G128-INT4. |
| 1.3 NVFP4 E4M3 G16 | WRONG as described | NVFP4 = E2M1 values + E4M3 block scales; "E4M3 weights" is wrong. +9.1 % bytes vs G128-INT4. No converter→loader→kernel route. |
| 1.4 QAT + distillation | NEEDS-FIX (research; not overnight) | Fixes quality at a representation, not bandwidth. Resource estimates unjustified. |
| 1.5 Reject GPTQ-full/AWQ-full/INT2/HQQ | WRONG reasoning, sensible queue exclusion | GPTQ can be sym without act-order; AWQ has no inherent act-order gathers; KV-Q4 KL values say nothing about weight-INT2. Exclude 2-bit routes for support/quality reasons only. |

## Hardware corrections

- Two B70s ≠ one 64 GB GPU: no P2P, independent 608 GB/s cards; TP2 gated (ADR-0008); llama.cpp layer-split is capacity, not 2× C1 speed.
- G128 is not a DPAS panel; W4A16 ≠ native W4A4. DPAS geometries: FP16 N16/K16, INT8 N16/K32, low-bit int N16/K64 (M=repeat 1–8).
- "8192 ops/clock/core" needs a MAC convention and verified basis before use.
- 574/316 GB/s are path-specific observations (94 %/52 % of nominal), not universal ceilings.
- GRF 64 KB/XVE; L2 18 MB; 2 MiB GDN state does not fit one Xe-core SLM.
- `-ctk/-ctv` are llama.cpp KV controls, not the vLLM baseline's KV description — record engine KV separately.
- Quantization method ≠ checkpoint format ≠ kernel layout (AutoRound/GPTQ/AWQ are calibration methods; runtime cost is the exported representation).
- Nibble order `[k0,k4,k1,k5,k2,k6,k3,k7]` is a kernel-specific packing contract, not hardware.
- "Spills/JIT → fix AOT first" conflates two different problems (AOT ≠ register allocation).

## Formats ranked (no custom kernels)

1. Q4_K_M (imatrix if available) — same-engine control, quality/fallback win possible.
2. Selective Q4_K + Q6_K/Q8_0 sensitive tensors — quality-first, budget active-traffic.
3. IQ4_XS + imatrix — compact challenger, conditional on calibration + kernel coverage.
4. Q5_K_M — quality comparator.
5. Q6_K — quality control, higher traffic.
6. Q8_0 — near-lossless anchor (dense fits; not for MoE full-offload).
7. FP8/INT8 vLLM — only from existing compatible checkpoints; no M=1 advantage expected.
8. MXFP4_MOE — conditional smoke only (quantizer advertises it; unverified for qwen35 on SYCL).

bpw reference: INT4-G128-sym 4.125 · IQ4_XS 4.25 · Q4_K 4.5 · Q5_K 5.5 · Q6_K 6.5625 · Q8_0 8.5.

## Red flags confirmed by inventory

`scripts/b70pack_build.py` absent; bench command references nonexistent build and `.gguf` (not the pack); P1 not expressible in GGUF; P2/P3 would hit missing operator dispatch, not slow fallbacks; `-pp/-tg` flags don't exist (correct: `-p`/`-n`); build name proves no model support; F16 calibration is a capacity problem; 20-turn deterministic transcripts are a smoke test, not parity; quantization does not create MTP layers.

## Validation ladder (operator rule: probe before committing hours)

1. Live-server sanity (flat TPOT check, log inspection, no queue/preemption) — DONE, passed.
2. Compile-only A/B: one relaunch, keep everything fixed, one screen cell n=3.
   Thresholds vs 10.35: ≥12.4 retain · 10.9–12.4 inconclusive · ≤10.9 stop-and-diagnose.
   Compile engagement = torch.compile cache/compile logs, not XPU-graph capture. — DONE (23.63, retained).
3. One warmed p8192/g128 observation before full matrix — DONE (23.86).
4. B4 gate: proceed only if ≥15 % over fresh B0, coherent, no pathology — DONE (+59 %).
5. GGUF Lane-0 per arm: `llama-bench -dev SYCL0 -sm none -ngl 99 -ncmoe 0 -fa on -ctk q8_0 -ctv q4_1 -t 8 -mmp 0 -b 512 -ub 512 -p 512 -n 32 -r 2 -o json`; pass = GPU-only placement, reserve ≥2048 MiB, two reps within ~15 %, no fallback warnings. M=1 MMVQ is a vector path, not XMX — do not claim DPAS from throughput.
6. Conversions serial first; MemAvailable guard (abort < 8 GiB); only test 2-way parallelism if serial leaves resources idle AND aggregate improves ≥20 %.
7. Perplexity harness: validate on a short chunk first (NaN/alignment check) — DONE (4-chunk probe, finite PPL, 6.6 s/chunk).
8. Imatrix: small-chunk time probe first; proceed only if calibration+conversion+evaluation fits the night (tonight: skipped).
