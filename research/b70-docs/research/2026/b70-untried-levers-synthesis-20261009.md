---
title: "B70 untried levers: what the hardware rewards, what was never measured, and the next deciding probes (2026-10-09)"
type: research
date: 2026-10-09
status: provisional
---

# B70 untried levers — synthesis (2026-10-09)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Source-only research. No GPU work, no
download, no route change. Ledger row: RL-127. Builds on the same day's quant-format
hunt (`b70-quant-lever-hunt-20261009-summary.md`, RL-126) instead of repeating it.

Inputs: five delegated surveys retained under
[`b70-untried-levers-20261009/`](b70-untried-levers-20261009/) (tried-vs-untried
inventory, Xe2 datapath, engine/format landscape, lever literature, repository
audit), the ledger RL-001..RL-126, model-status, and primary captures under
`research/references/levers-synthesis-20261009/`.

## 1. What the B70 rewards (corrected datapath)

| Path | Rate per Xe-core per clock | What it means here |
|---|---|---|
| XMX FP16/BF16 | 2048 ops | Every quant decode path on this host ends here: EXL3, Strata, llama.cpp prefill, vLLM W4A16. |
| XMX INT8 (s8×s8) | 4096 ops (367 TOPS = 32 × 4096 × 2.8 GHz) | The only fully exposed 2× path (joint_matrix, oneDNN, sycl-tla). **Unused by every quant path in production on this host** (RL-126 ISA audit). |
| XMX INT4/INT2 (int×int) | 8192 ops | Real in hardware and measured 2.00× per instruction via ESIMD (`xmx-dpas-rate-classes-20260915.md`). Not exposed by joint_matrix. Useful only if activations are also 4-bit; W4A16 and W4A8 gain nothing. |
| FP8 / FP4 DPAS | absent on Xe2 | FP8, MXFP4 and NVFP4 are byte formats here, never rate formats. |
| GDDR6 | 608 GB/s pin | EXL3 M=4 verify reaches 381 GB/s (63 %). External FP16 GEMV on B70 reaches ~570 GB/s (~93 %, Puget, E0). Strata's aligned Q6_K kernel went 160 → 493 GB/s after fixing misaligned 16-byte loads (E0, Strata INTEL.md capture). |

Two consequences drive every recommendation below:

1. **Decode** headroom is instruction issue and alignment, not DRAM: the same 608 GB/s
   card streams ~93 % with trivial dequant and 63 % with trellis decode. A format wins
   decode by bytes per token **and** cheap, aligned dequant.
2. **Prefill** headroom is the unused 2× INT8 path. Strata's upstream fused int8
   prompt kernels (#136) are stubbed in its SYCL port (Strata INTEL.md, "0.1.38 merge").

Source: Intel Tech Tour Xe2 deck (URL in the appendix; >8 MiB, not captured), B70
datasheet, local ESIMD probe.

## 2. What was tried (one line per class)

Full table: [`tried-matrix.md`](b70-untried-levers-20261009/tried-matrix.md).

| Class | Verdict on this host |
|---|---|
| EXL3 4bpw trellis + MTP3 (exl3xpu) | Adopted dense route, 63.25 tok/s (ADR-0010). Kernel probe program operator-stopped (RL-122). |
| GPTQ / AutoRound symmetric INT4 W4A16 (27B) | Closed: weight-level "30" defect (RL-087, RL-092). |
| Block-FP8 W8A16 (27B) | Diagnostic, 48.08 tok/s; below EXL3. |
| Strata IQ3_XXS + expert streaming (Flash-Next) | Adopted, 48.03 tok/s, 989 tok/s 8K prefill (ADR-0011); quality gates open. |
| GPTQ-INT4 MoE + DFlash n7 (Nemotron) | Route, 186.6 tok/s historical; re-measure owed (RQ-11). |
| DFlash2 / DSpark drafters on quantized 27B | Closed with a precise retry condition (`qwen38-dflash2-objective-campaign-20260830.md` §12.4). |
| Stock TP2, PP2, RPC split | Closed (RL-075, RL-051, RL-083); custom copy-only TP2 = RQ-06 (gated). |
| W4A8 prefill, QuaRot→INT3, s4×s4 decode, megakernel | Closed by attribution or pricing (standing rules §3). |
| OpenVINO INT4 (27B) | Alternate route, 28.5 tok/s @150 W, clean quality. |

## 3. Never measured locally, ranked

Ranking = end-to-end benefit × mechanism support × feasibility on today's disks
(three of four volumes ≥84 % full on 2026-10-09). Each live item is now a program
question; gates were set before any result.

| Rank | Lever | Term it moves | Bound / external evidence | Cost and risk | Program |
|---:|---|---|---|---|---|
| 1 | **Flash-Next on two cards with Strata `--layer-split`, all IQ3_XXS experts resident** | Removes the expert-miss stream (~33–37 % of a ~56 ms round, lever audit) | Amdahl ceiling **≤1.53×** decode; boundary handoff ~0.06 ms/round (`research/references/rq18-layer-split-price.json`). External: 2×B60 (456 GB/s, PCIe 3 x8) ran Flash-Next IQ2_XS at 55–61 tok/s with "100 % of the experts resident" | Same artifact, no download. Occupies both cards, so the EXL3 27B service cannot run beside it: operator placement decision. ADR-0011 rejected only the host-staged `--expert-cache-device1` arm, not layer-split | **RQ-18** (gated on RQ-17) |
| 2 | MXFP4 MoE decode, llama.cpp PR #29809 (published on gpt-oss-20b; local target retargeted by the operator to Qwen3.6-35B-A3B MTP-MXFP4_MOE, 21.2 GB) | Dequant instruction cost on MoE MMVQ | Author 55.84 → 106.64 tok/s on one B70 (Windows); in-PR Linux B70 53.72 → 137.7 (E0, gpt-oss shapes) | Branch build + 21.2 GB download (SN770 has 160 GB) | **RQ-19** (overnight P2) |
| 3 | Cheap-dequant 4-bit streaming kernel falsifier (INT4-G128 dual-path) | Instruction tax 37 % of pin | ≥500 GB/s hypothesis vs 381 anchor; FP16 GEMV ~93 % and aligned Q6_K 493 GB/s show the streaming headroom exists | Kernel microbench only; a pass still needs a quality-passing 4-bit artifact for the 27B | **RQ-20** (overnight P3) |
| 4 | Workload-matched speculation: suffix / n-gram vs MTP on the RQ-10 task corpus | Yield on copy-heavy tasks (tool loops, document edits) | SuffixDecoding up to 5.3× on agentic traces; n-gram ~2.4× on summarization (NVIDIA, E0). vLLM lists both methods. Lossless by construction. Local counter-evidence: llama.cpp ngram-4 alone was −31.2 % vs MTP and MTP+ngram stacking −2.1/−2.3 % on 27B dense (2026-07-03, `docs/methods/speculative-decoding-on-b70.md`), measured on non-copy content | Config change on a vLLM-based route; the method must replace MTP per request class, not stack. Wins only where copy rate is high, so the corpus decides | **RQ-21** (gated on RQ-10) |
| 5 | INT8 prompt GEMM for Strata SYCL (port of upstream #136) | Prefill compute on the 2× INT8 path | exl3xpu int8 prefill linears measured 1.89× at +0.17 % NLL (sybil fork, E0); Strata's own int8 kernels exist upstream for CUDA | Kernel port. Price the GEMM share of the 8K prefill first; reject below the 10 % floor | **RQ-22** (deferred) |

Kept as notes, not questions (program stays small):

- **Block alignment of GGUF types on SYCL.** Strata measured 2.3–4.7× from aligned
  loads; IQ4_XS (136 B), IQ3_XXS (98 B) and Q8_0 (34 B) blocks are not 16-byte
  aligned. Treat any IQ4_XS speed estimate (lever audit C2) as unproven until the
  kernel's load path is read. Check before any new GGUF speed claim (standing rules §2).
- **Prefill/decode disaggregation across the two cards.** Useful only under mixed
  arrival; the KV hand-off for an 8K request on the 27B is ~256–310 MB fp8 (32–38 KiB
  per token, RL-085) ≈ 23–27 ms at 11.5 GB/s. Re-enter inside RQ-05 after its C4
  panel, not before.
- **TurboQuant 3/4-bit KV** is merged and XPU-routed in vLLM main, but the exl3xpu
  fork (vLLM 0.26.1) predates it. Long-context only; no current question.
- **llama.cpp Vulkan on B70** loads and answers correctly (Mesa 26.2.2 smoke, off-repo
  `/mnt/<nvme-disk>/vulkan-llama-campaign-20261002-140820/`); no speed data.

## 4. New suggestions rejected on local evidence

| Suggestion (from surveys) | Why not |
|---|---|
| DFlash / EAGLE-3 drafter for the 27B | Pre-rejected until a drafter trained on the quantized target's hidden states shows ≥1.5× native-head acceptance (DFlash2 campaign §12.4: 25 vs 51 tok/s). |
| AWQ W4A16 for the 27B | Same closed W4A16 family (ADR-0010); RL-034 never reached the AWQ loader. Re-entry is an operator decision. |
| NVFP4 on vLLM XPU | No XPU kernel (vllm-xpu-kernels #553 open). NVFP4 only via llama.cpp MMVQ (overnight P5). |
| W4A4 / INT4-XMX prefill | Rate is real (2.00×) but needs 4-bit activations; quality risk on a model that already fails at W4A16. |
| 2:4 sparsity | No sparse DPAS on Xe2; index overhead without GEMM gain. |
| Strata `--peer-device` second-card expert tier | Stubbed in the SYCL port (CUDA calls); the host-staged equivalent measured slower (RL-125). |

## 5. Corrections found while checking

1. The same-day hunt summary says "sub-8-bit INT gives no TOPS multiplier". Precise
   form: s4×s4 is 2.00× per instruction; INT4 **weights with 8- or 16-bit activations**
   gain no rate. The two notes are compatible once the operand pair is named.
2. The root `references/` notes claimed GPTQ W4A16 as the fastest dense format and a
   43 tok/s ceiling that ignores speculation. Archived under
   `archive/root-strays-20261009/`.
3. The repository audit reported three result roots as "canonical evidence lost".
   156 of 197 sync-conflict copies were byte-identical to live files (the roots had moved
   to `results/2026/08/`). All 197 were archived losslessly with a manifest.
4. Model-status and recommended-configs disagreed: Ornith (quality-rejected) and the
   removed GPTQ 27B route were still listed as current routes. Fixed.

## 6. Structure changes

| Change | Where |
|---|---|
| Living per-regime format/engine/lever guide | `docs/reference/b70-format-and-lever-guide.md` |
| Standing rules promoted from a dated note and corrected | `docs/governance/standing-rules.md` |
| Strata engine dossier (adopted engine had none) | `engines/strata-sycl/README.md` |
| Overnight plan items and new levers registered as RQ-17..RQ-22 | `registry/research-program.json` |
| Ledger evidence paths repaired (12 rows), dead TP2 campaign cancelled | registry |
| Stale engine/ops docs aligned to model-status | `engines/`, `docs/operations/` |

## 7. Next deciding probe

RQ-17: close ADR-0011's open quality gates on the incumbent IQ3_XXS route (the
precommitted overnight P0 battery). It gates RQ-18 and costs ~30–45 minutes on one card.

## Appendix — primary sources

- Intel, Xe2 and Lunar Lake GPU deep dive (Tech Tour 2024):
  https://cdrdv2-public.intel.com/824434/2024_Intel_Tech%20Tour%20TW_Xe2%20and%20Lunar%20Lakes%20GPU.pdf (not captured, >8 MiB)
- Intel Arc Pro B70 datasheet: https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf
- Strata 2×B60 layer-split bench: capture `sha256-ca6f0ea9…`
- Strata INTEL.md (alignment fix, stubbed int8 prompt kernels, `--peer-device`): capture `sha256-a394b896…`
- vLLM speculative decoding methods (suffix, n-gram, MTP, EAGLE-3, DFlash): capture `sha256-81d5b2a2…`
- llama.cpp PR #29809 (MXFP4 MoE MMVQ): `research/references/quant-avail-20261009/captures/`
- Puget Systems, 4× Arc Pro B70 vLLM: https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/ (E0)
- SuffixDecoding: https://arxiv.org/abs/2411.04975 (E0, NVIDIA hardware)
