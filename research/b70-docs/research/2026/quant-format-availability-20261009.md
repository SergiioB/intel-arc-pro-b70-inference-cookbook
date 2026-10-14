---
title: "New quantization-format availability per B70 serving path + horizon (2026-10-09)"
type: research
date: 2026-10-09
status: provisional
---

# Quant-format availability matrix — five B70 serving paths (source-only, no GPU)

**PROVISIONAL — source research only.** No GPU run, no model load, no local benchmark.
External numbers are class E0 (their hosts). Ledger row RL-126. Follows RL-121/RL-125
(Strata lane) and the external-lever sweep RL-073.

## 0. Pinned identities (all read 2026-10-09)

| Source | Identity |
|---|---|
| llama.cpp pinned local tree | `~/llama.cpp` @ `1692f9e50` (2026-08-14), **dirty**: SYCL fused-gemm/iq3-xxs-reconstruct/mmvq/vecdotq/moe-cache local patches (+615/-43 in ggml-sycl) |
| llama.cpp upstream master | `50e3e3e480` (2026-10-09T16:23Z) |
| Strata local tree | `~/Strata` @ `fb58e0d` = tag **v0.1.41** (2026-10-08); deployed adopted = v0.1.40.3 @ `d5ea713`; releases after adoption: v0.1.40.4 (Pascal-only fix), **v0.1.41** (multi-GPU + short-prompt work; B70 IQ3_S +13%/+12% short prompts, +10% 4K/20K, decode equal) |
| vLLM master | `75e3b17c0` (2026-10-09T16:07Z); XPU platform file read at this SHA |
| vllm-xpu-kernels | `main` README (wheel 0.1.15.4, release v0.1.15 2026-09-22) |
| exl3xpu | `0xSero/exl3xpu` @ `15ded2f` (2026-09-26) |
| OpenVINO GenAI | releases 2026.4.1.0 (2026-10-01), 2026.4.0.0 (2026-09-17); NNCF `master` |
| HF API | author/model queries 2026-10-09 (~18:3x CEST) |

Raw evidence: `research/references/quant-avail-20261009/` (master SYCL dispatch
files, vLLM platform file, exl3xpu README, NNCF algorithm docs) + `captures/`
(sha256-pinned PR pages via `research.py source`).

## 1. Availability matrix (format × engine)

Legend: **A**=available with real kernel path · **P**=partial (works with caveats/flag)
· **—**=absent · **H**=horizon (open PR exists). "today" = the pinned upstream
revisions above, NOT our pinned serving images (our vLLM pin is older).

| Format | (a) llama.cpp SYCL @master | (b) Strata v0.1.41 | (c) vLLM XPU @master | (d) exl3xpu | (e) OV GenAI/OVMS |
|---|---|---|---|---|---|
| K-quants Q2_K–Q6_K | **A** (MMVQ+multicol+reorder; DMMV; prefill dequant+GEMM) | **A** (Q4_K/Q5_K/Q6_K wide kernels; Q6_K the hot one) | — (not a vLLM format) | — | — |
| Q4_0/Q5_0/Q5_1/Q8_0/Q4_1 | **A** (MMVQ+multicol, Q4_0/Q8_0 reorder-ESIMD, Q8_0 DMMV-ESIMD #29186) | **A** (Q8_0/Q2_0; Q4_0 legacy) | — | — | — |
| IQ4_XS / IQ4_NL | **A** (MMVQ; IQ4_XS multicol; IQ4_NL no multicol) | **A** (IQ4_NL wide expert dots, 39/48 layers) | — | — | — |
| IQ3_XXS / IQ3_S / IQ2_S | **A** decode (MMVQ single-col; **IQ3_S multicol #29500 merged 2026-10-07**; our local dirty patch adds IQ3_XXS/IQ3_S multicol — now partially upstream); prefill = dequant fallback; no DMMV | **A** (IQ3_XXS/IQ3_S gate/up dots; the adopted Flash-Next route) | — | — | — |
| IQ2_XXS/XS, IQ1_S/IQ1_M | **P** (MMVQ single-col only, no multicol, no DMMV; sub-4bpw crawl externally corroborated) | **A** (IQ2_XS, IQ1_M Coder: host-mirror streaming rows) | — | — | — |
| TQ1_0/TQ2_0 (ternary) | **—** (no SYCL MMVQ/DMMV/dequant kernels; CPU/HIP/OpenCL only; **H**: PTQ1_0 #29672 open) | — | — | — | — |
| **Q1_0/Q2_0** (new 1-bit pair, ids 41/42) | **A** (MMVQ+multicol; Q1_0 reorder-ESIMD; set_rows #24564) | — | — | — | — |
| **MXFP4** (E2M1+E8M0 g32; gpt-oss) | **A** (MMVQ + **multicol**; get/set_rows; since ≤2026-08-14 base; **H: #29809** MoE arithmetic-decode + reorder, B70-tested, ~2× decode) | — | **A** (gpt-oss-20b/120b on Arc Pro B-series are the *recommended* XPU models; `mxfp4`,`gpt_oss_mxfp4` in platform list; **P flag**: #48946 corrupt under piecewise graph capture TP2 on PVC, open) | — | **P** (NNCF emits MXFP4 IR; OV GGUF frontend parses MXFP4 GGUF; GPU-plugin execution not confirmed) |
| **MXFP8** (E4M3+E8M0) | — | — | **A** per platform list (`mxfp8`); kernels vllm-xpu-kernels | — | **P** (NNCF mode; GPU exec unconfirmed) |
| **NVFP4** (E2M1+E4M3 g16) | **A** storage-wise (type 40; MMVQ `vec_dot_nvfp4_q8_1` + multicol; convert-side mixed-ct import **#28636 merged 2026-10-09**; #27161 closed unmerged, overlapping scope — corr. 2026-10-09 post-audit, GitHub API verified; **H**: #25153 imatrix-aware quant, #26869 complete MXFP4+NVFP4, #28898 fp8/nvfp4 quant scales) | — | **—** on XPU (not in platform list; **H: vllm-xpu-kernels #553** adds NVFP4 B-dtype to Xe2 grouped MoE GEMM; today requires requant to MXFP4 = accuracy loss). CUDA-only native (Blackwell W4A4; older SM via Marlin W4A16) | — | **H/P** (NNCF has `nvfp4` mode at master; GPU exec unconfirmed) |
| EXL3 trellis (QTIP-family) | — | — | — | **A** (4/6 bpw mul1 native; **2/3/5 bpw + mcg/3INST codebooks behind `EXL3_FLAGS=-DEXL3_ALL_CODEBOOKS`**; fp8 KV only; MTP k=3; ESIMD dp4a GEMV + DPAS GEMM) | — |
| GSQ-RCO packed GGUF (ISTA-DASLab) | — | **A** (pinned HF shas in setup.py; native pack `iq3xxs` deployed) | — | — | — |
| GPTQ W4A16 (int4 g64/g128) | — | — | **A** (platform list; our serving record; **P flag**: #52203 all-GPTQ DEVICE_LOST at profile_run on Arc B60, open — does not match our B70 GPTQ serving, watch on re-pin) | — | **A** (GPTQ import via optimum/nncf → int4) |
| AWQ | — | — | **A** per platform list (`awq`,`auto_awq`) — no B70-local validated artifact (RL-034 blocked) | — | **A** (AWQ import) |
| compressed-tensors (int4/fp8/mixed) | — | — | **A** per platform list; NVFP4-layer checkpoints blocked by kernel absence | — | — |
| FP8 (block/per-tensor/per-channel; online) | — | — | **A** (all three granularities + online; our W8A16 record) | KV only (fp8 KV) | **P** (NNCF FP8_E4M3 mode; GPU exec unconfirmed) |
| INT4 sym/asym (OV IR) | — | — | (via gptq/ct import) | — | **A** (NNCF INT4_SYM/ASYM; our validated OV route) |
| NF4 / CB4 codebook | — | — | — | — | **A** (NNCF modes) |
| QAT GGUF artifacts (Qwen-family) | **— artifacts**: no `Qwen/Qwen3.8*` QAT repo exists (base/FP8/Flash-Next/2.4T only); gemma-3/4 QAT GGUFs exist (unsloth re-quants); historical Qwen3 QAT line (2025) not re-found under author=Qwen today — **unconfirmed, absence-of-search caveat** | n/a | n/a | n/a | n/a |
| Unsloth UD dynamic 3.0 | **P** = plain mixed recipes of standard types (UD-IQ1_M…UD-Q6_K_XL verified in repo file list; no new runtime type). Runtime support = per-type kernel coverage above (IQ-heavy UD variants inherit the IQ prefill fallback; UD-Q4/Q5/Q6_K fine) | reads same GGUFs only if types ∈ Strata set (Q4_K/Q5_K/Q6_K/IQ*/Q2_0) | n/a | n/a | via GGUF frontend only |
| Quantized KV | **A** q4_0/q8_0 TILE decode (#26689, +42–169%); q4_1/q5_x VEC behind `GGML_SYCL_FA_ALL_QUANTS` compile flag; iq4_nl KV absent (CUDA got it #27248) | **A** int8/q4_0/k8v4 (k8v4 slower) | fp8 KV (our record); TurboQuant KV dtype routing exists in platform file | fp8 KV only | int8 KV (our record) |

## 2. SYCL/Xe2 kernel coverage — the structural facts (llama.cpp @master `50e3e3e480`)

1. **MMQ is globally disabled**: `ggml_sycl_supports_mmq()` returns `false` for
   every type ("TO-DO: accuracy issues in MMQ"). The PR-29864-class int8-DPAS
   mul_mat is NOT merged. Consequence: every quant GEMM > `MMVQ_MAX_BATCH_SIZE=8`
   columns runs **dequantize→F16 + GEMM**. This is the single biggest SYCL gap
   vs CUDA (which has per-type MMQ incl. NVFP4/MXFP4 tiles on Blackwell).
2. **MMVQ (decode ≤8 cols)** covers all IQ/K/simple quants **plus MXFP4, NVFP4,
   Q1_0, Q2_0**; multi-column MMVQ (spec-decode verify shapes) exists for Q4_0/1,
   Q5_0/1, Q8_0, Q1_0, Q2_0, K-quants, MXFP4, NVFP4, IQ3_S (#29500), IQ4_XS —
   **not** for IQ2_*/IQ1*/IQ3_XXS/IQ4_NL (our local dirty patch adds IQ3_XXS/S).
3. **DMMV** (batch-1 dequant path): Q1_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/K-quants only.
4. **TQ1_0/TQ2_0: no SYCL kernels at all** (no vecdot, no dequantize, no DMMV).
5. Quantized-KV decode: TILE kernel default on BMG (#26689 merged 2026-08-28 —
   after our pinned local base, so our base tree has it only if patched; the
   deployed builds carry it). "Next upstream" for Xe2 KV quant: iq4_nl/q4_1
   default-on (CUDA #27248/#28634 class), prefill-side quantized-KV attention,
   and MX-format KV (CPU scalar ref #20609 open).

## 3. The three highest-value "new format, reachable within weeks" picks

Ranked jointly by availability + **decode feasibility on Xe2 (real kernel path,
not dequant fallback)**:

### Pick 1 — MXFP4 (gpt-oss class) on llama.cpp SYCL, sped up by open PR #29809
- **Format is already A-available at master** (MMVQ + multicol + get/set_rows;
  our ≤2026-08-14 base already carried `vec_dot_mxfp4_q8_1`).
- **Open PR [ggml-org/llama.cpp#29809](https://github.com/ggml-org/llama.cpp/pull/29809)**
  (2026-10-01, updated today): bit-identical arithmetic decoding replaces
  table-based decode + lazy per-expert weight reordering (aligned vector loads),
  reorder-aware decode/multi-token/prefill. **Measured by the author on exactly
  Arc Pro B70** (Windows, oneAPI 2026.0, FA on): gpt-oss-20b one card decode
  55.84→**106.64 tok/s (1.91×, n=5)**; gpt-oss-120b two cards 33.27→67.14 (2.02×,
  n=2); PPL within noise (14.5148→14.5129); MUL_MAT 39/39, MUL_MAT_ID 74/74.
- Decode feasibility: this is **not** a dequant fallback — it upgrades the real
  MMVQ/MoE path. The K-contiguous-INT4-class fused-DP4A/DPAS question: the
  missing generic piece remains SYCL MMQ (disabled), but for MXFP4 MoE #29809's
  arithmetic-decode + reorder is the type-specific answer; a DPAS fusion would
  be follow-on kernel work (handoff: b70-kernel-engineer).
- Weeks-path: merge-watch #29809 → rebuild our SYCL lane → gpt-oss-20b one-B70
  screen. Also v0.1.41-class Strata does not compete here (no MXFP4).

### Pick 2 — NVFP4 dense checkpoints through llama.cpp (convert PR merged TODAY)
- `convert` support for compressed-tensors **mixed-precision NVFP4** checkpoints:
  **#28636 merged 2026-10-09** (validated converting unsloth/Qwen3.8-27B-NVFP4);
  **#27161 closed unmerged** the same day (overlapping scope). Correction
  2026-10-09 post-audit, GitHub API verified — this note originally recorded
  both as merged. MoE stacked experts preserved since #29400 (2026-09-24).
- SYCL runtime side already present: `GGML_TYPE_NVFP4=40`, `vec_dot_nvfp4_q8_1`
  MMVQ **and multi-column MMVQ** (spec-decode shapes) — i.e. real decode kernels,
  prefill dequant-fallback like other quants.
- Artifact exists: community QAT **`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`**
  (QAT-trained NVFP4 for exactly our dense 27B; verified on HF today). Quality
  gate risk: community QAT lineage needs our exact-answer battery (RL-079 class).
- Contrast that CLOSES the NVFP4 question for other engines: vLLM XPU = absent
  (not in `supported_quantization`; #553 open, Xe2 grouped-MoE-GEMM only, and
  today's workaround is requant to MXFP4); native W4A4 = Blackwell CUDA-only;
  NNCF can *emit* NVFP4 IR but OV GPU execution is unconfirmed. NVFP4 is only
  plausibly reachable on B70 via llama.cpp SYCL — confirmed and closed elsewhere.

### Pick 3 — MXFP4 on vLLM XPU (gpt-oss) after capture-safety verification
- **Officially recommended**: the vLLM XPU doc lists `openai/gpt-oss-20b/-120b`
  MXFP4 as recommended models on Arc Pro B-series validated hardware; kernels
  ship in vllm-xpu-kernels ("FP8, MxFP4 quantization and GEMM"; MXFP8 also in
  the platform list — a second new format already in-tree).
- Blocker to clear per our §7 contract: **#48946** (open): gpt-oss-120b TP2
  decode collapses to a repeated token under PIECEWISE XPU graph capture on PVC;
  compile-mode correct. Graph capture is mandatory for our serving — so the
  weeks-path is (i) verify whether BMG single-card shows it (PVC/TP2 report),
  (ii) if present, hand the capture-safe-kernel requirement to the engine
  specialist before any route talk. AutoRound MXFP4 Hadamard on XPU (#54006,
  open) is the quant-production side.

**Honorable mention (not new-format, new-degree):** EXL3 **2/3/5/6 bpw**
all-codebooks on exl3xpu — the fork README now documents
`EXL3_FLAGS=-DEXL3_ALL_CODEBOOKS` (mcg/3INST codebooks) at `15ded2f`; our
RL-094 showed 3bpw quality-first promise, RL-097's patched build failed —
re-trial against the fork's flag is cheap and reuses the champion engine.

## 4. Literature mechanism harvest — recombination candidates for a B70-native format

Post-train (PT) = convertible from existing BF16/FP8 artifacts locally;
QAT-only = needs training. Verified URLs (HTTP 200 on 2026-10-09) except where
marked. Xe2 facts: 256 XMX/DPAS int8+fp16 (no fp8 XMX, no native MX decode),
608 GB/s pin ceiling, GRF pressure, SLM 128 KiB/WG.

| Mechanism (paper/repo) | One-line mechanism | Xe2 collision / exploit | PT? |
|---|---|---|---|
| AQLM [arXiv:2401.06118](https://arxiv.org/abs/2401.06118) | Multi-codebook additive quantization (2×8-bit codebooks ≈ sub-2.5bpw) + finetuned reconstruction | Exploit: LUT fits SLM/registers like IQ codebooks; collision: multi-pass dot products raise instruction count against DPAS-int8 dominance | PT (+ optional finetune) |
| QuIP [arXiv:2307.13304](https://arxiv.org/abs/2307.13304) / QuIP# [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) | Randomized-Hadamard incoherence processing + E8 lattice codebooks | Exploit: Hadamard is cheap on Xe2 FWHT kernels (#29243/#29605 upstream); lattice LUT decode suits vector lanes; collision: K-major weight layout vs row-major MMVQ | PT |
| SqueezeLLM [arXiv:2306.07629](https://arxiv.org/abs/2306.07629) | Dense-and-sparse: nonuniform k-means quant + sensitivity-weighted outlier sparse matrices | Collision: sparse gather indirection defeats coalesced 16B loads (Strata's Q6_K alignment lesson) and DPAS tiling | PT |
| SpQR [arXiv:2306.03078](https://arxiv.org/abs/2306.03078) | Isolate outlier weights (fp16/bits) + coarse residual quant | Exploit: outlier count ~1% → SLM-resident per-tile corrections; collision: two-format kernels double dispatch surface | PT |
| BitNet b1.58 [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) | Ternary {-1,0,+1} weights QAT-trained | Exploit: add-heavy kernels map to Xe2 int8 DPAS beautifully (TQ2_0 8×8 i8mm precedent); collision: **no post-train path** — needs trained checkpoints (llama.cpp TQ kernel maturity is CPU-side; PTQ1_0 #29672 open) | **QAT-only** |
| Trellis/QTIP codebooks (turboderp-org/exllamav3; QTIP arXiv ID **unverified** — search tool down) | Trellis-structured lattice coding across weight rows (EXL3's core) | Already shipping on B70 via exl3xpu ESIMD dp4a+DPAS; exploit proven locally (63.25 tok/s champion) | PT (EXL3 quantizer) |
| imatrix calibration (llama.cpp #4861-class) | Importance-weighted oblique Gram accumulation choosing per-tensor quant granularity | Pure host-side tool; exploits every format above; zero kernel cost | PT |
| OCP MX / E8M0 microscaling ([OCP MX v1.1 spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-1-spec-final-pdf)) | 32-element groups share one 8-bit exponent byte | Exploit: byte-aligned scales suit aligned vector loads (#29809's reorder insight); collision: no MX decode units on Xe2 → arithmetic decode cost is the whole game | PT |
| NVFP4 (E4M3 g16 scales) | Finer scales than MX at same bpw (+9.1% bytes vs INT4-G128 per our architect) | Same as MXFP4; finer groups raise scale-reload frequency — #553's tile_k=16 ladder is the exact Xe2 collision | PT (requant) / QAT (QUASAR) |
| 2025–26 unshipped formats | **Could not survey**: web_search tool unavailable this session (no API key); arXiv API queries returned empty. Only first-party evidence above (PTQ1_0, MXFP8, TurboQuant KV, codebook/CB4) | — | — |

## 5. What I could NOT confirm (absence ≠ proof)

1. **OV GPU-plugin execution** of NNCF MXFP4/MXFP8/NVFP4/F4 IR on Xe2 (NNCF can
   emit them; no device-support statement found in the time available).
2. **Qwen-org QAT artifacts for any Qwen3.8/Flash-Next model** — no repo found
   (only base/FP8 variants); HF author-list search may miss gated/renamed repos.
3. Exact arXiv ID of QTIP (redirect-verified only AQLM/QuIP/QuIP#/SqueezeLLM/
   SpQR/BitNet) and any 2025–2026 format paper beyond first-party repos.
4. Whether #48946 (MXFP4 capture corruption) reproduces on BMG single-card B70 —
   PVC/TP2-only report; explicitly a verify-first item.
5. llama.cpp Q1_0/Q2_0 origin PR (local history is a squashed clone; GitHub
   search surfaced only per-backend follow-ups). Their SYCL kernel coverage is
   verified at master regardless.
6. AWQ-on-XPU serving quality/behavior (platform-list presence only; RL-034
   remains blocked locally).
7. vLLM GPTQ #52203 (Arc B60 DEVICE_LOST) applicability to B70 — our GPTQ
   serving record contradicts it on our pin; re-check at any re-pin.

## 6. Implications for our pins

- llama.cpp lane: our 2026-08-14 dirty base is ~2 months stale; upstream gained
  TILE-KV (#26689), IQ3_S multicol (#29500), grouped MoE XMX GEMM (#29245),
  sparse FA (#28796), Q5_K reorder+GLU (#29375), pinned-ring loading (#29608).
  Any rebuild should diff our local fused-gemm/IQ3 patch set against #29500/#29809
  first (overlap risk).
- Strata lane: v0.1.41 is a straight upgrade candidate (B70 rows positive, no
  Intel regressions listed; engine version already logs 0.1.41).
- vLLM lane: MXFP4/MXFP8 presence at master ≠ our pinned image; NVFP4 stays
  closed on XPU until #553-class work lands.
- Handoffs: #29809 merge-watch + screen → b70-bench-engineer (matched cells);
  MXFP4 MoE DPAS fusion design → b70-kernel-engineer; OV MXFP4 device check →
  engine specialist; NVFP4-QAT artifact quality → b70-correctness-engineer.

## 7. Artifact readiness addendum (2026-10-09 evening, HF metadata only — no downloads)

| Artifact | Repo @ rev | Format | Bytes | Fit vs live free disks |
|---|---|---|---|---|
| Flash-Next GSQ-RCO IQ2_XS | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF @ ed59f920 | 2-split GGUF 39,225,954,592 + 28,800,138,432 | 68.03 GB | part2 sha256 316b46f3…61e113 **identical to local IQ3_XXS part2** → incremental +39.23 GB; T-FORCE ✓ |
| Flash-Next GSQ-RCO Q2_0 | same @ ed59f920 | 37,623,740,192 + 28,800,138,432 | 66.42 GB | incremental +37.62 GB; T-FORCE ✓ |
| Flash-Next GSQ-RCO IQ3_S / IQ3_XXS | same | 83.62 / 75.84 GB | — | IQ3_XXS already local |
| Flash-Next GSQ-RCO **IQ2_XXS / IQ2_S / IQ2_M** | — | **do not exist** in any GSQ-RCO repo (IQ2_M is not a ggml type) | — | — |
| Coder IQ1_M | ISTA-DASLab/…-Coder-GGUF @ 5348543e | IQ1_M only | — | Strata B70 port supports |
| UD-Q4_K_XL Flash-Next | unsloth/Qwen3.8-Flash-Next-GGUF @ 766911a6 | 4-split plain GGUF 111.34 GB | 111.34 GB | 2× rule fails everywhere; also NOT GSQ-RCO layout (Strata pack compat unconfirmed) |
| QUASAR NVFP4 QAT 27B | QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 @ d9af17e2 (2026-10-05) | compressed-tensors `nvfp4-pack-quantized`, W4F16 g16 E4M3 scales + 4-bit dynamic acts, 5 safetensors, apache-2.0 | 20.58 GB | SN770 ✓ (2× = 41.2 ≤ 43, tight) |
| gpt-oss-20b | openai/gpt-oss-20b @ 6cee5e81 | MXFP4 safetensors (3 shards) | 13.77 GB (repo 41.3 GB) | SN770 ✓ |
| gpt-oss-120b | openai/gpt-oss-120b @ b5c939de | MXFP4 safetensors (14 shards) | 65.25 GB | T-FORCE: fits 1×, **fails 2× by ~2.5 GB** |
| PR #29809 branch | Captain-Tripps/llama.cpp branch `sycl-mxfp4` @ d890cc89590 (base: master) | 263-line SYCL MXFP4 MoE PR | — | buildable without cloning mainline history |

Qwen-org QAT re-check: 100 org repos scanned (author=Qwen, 3 pages): **zero QAT repos**; Qwen3.8 family = base/FP8/Flash-Next(+FP8)/2.4T-A95B(+FP8) only. Private/gated-invisible caveat applies (API lists gated repos but not private ones).

Strata ≥ v0.1.40.3 B70 rows: v0.1.40.4 = Pascal-only; **v0.1.41** = one B70 row (IQ3_S: +13%/+12% short prompts, +10% 4K/20K, decode equal) + B70 expert-dequant 16-byte-run fix (+6% prompt, bit-identical output) + two-card mirror fix. IQ2-class B70 evidence lives in INTEL.md, not release tables: **IQ2_XS 58-64 tok/s decode** (host-mirror class, their box, E0), port expert kernels = IQ2_XS/IQ3_XXS/Q2_0 (+ Coder IQ1_M).
