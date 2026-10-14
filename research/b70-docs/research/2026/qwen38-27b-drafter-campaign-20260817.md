---
title: "Qwen3.8-27B C1 decode — executed decision + drafter campaign"
type: research
date: 2026-08-17
status: final
---

# Qwen3.8-27B C1 decode — executed decision + drafter campaign

Date: 2026-08-17
Status: **decision executed**. No new speed number. No on-box SpecForge.
Architect: `b70-architect` this session. Claims standard: do not headline any figure below.

Host snapshot (this session): GPU empty (`visible_avail` 31979 MiB / 32656), no vLLM/llama container, user units inactive, leftover `sglang-xpu-local` waiter **killed**. RAM 30 GiB (23 GiB avail). `/` 27 G free. `/mnt/<models-disk>` 56 G free. Production image restored: `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97`. `sglang-xpu-torch:base` 29.1 GB still on disk.

Teacher that already fits: `/mnt/<models-disk>/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-nogidx` (19 G, `mtp_num_hidden_layers=1`). `/mnt/<models-disk>/Qwen3.8-27B-BF16` is a **23 M stub**, not the 55 GB teacher. Do not download/materialize BF16 27B on this box.

## 1. Where we are (3.8-27B only)

| Item | Value | Class |
|---|---|---|
| Production path | vLLM XPU GPTQ-INT4-sym-G128 + **BF16 1-layer MTP4** | keep |
| Run 39 pinned `2c427ef` | **81.26** t/s median p512/g128 n=5 C1 230 W | E2 provisional |
| Run 40 nightly `f01e24f6` | **83.7** t/s same cell | E2 provisional; **not** the restored pin |
| Run 40 no-spec matched | **32.9** t/s p512/g128 | use this, not 24.4 |
| MTP8 on this 1-layer head | **71** t/s | rejected |
| Closed flags | W4A8 (corrupt), async, 16K batch, SGLang E=1 / n256, ngram | dead |

**Roofline correction (architect, this session).** Day-0 used 24.4 → 81.26 = 3.33× and wrote **1.50 target-forwards / 5 outputs**. That mixed a slow eager no-spec with MTP4. **Matched Run 40 is 32.9 → 83.7 = 2.54×.** With MTP4 accept 95% (γ ≈ 4.75) that is **d ≈ 0.87** extra target-eq per step. MTP weight traffic is only ~0.19 of that. **~0.68 is serial draft + GDN verify + launches**, not the 0.849 GB BF16 head.

Text-decode working set if vision+MTP unread: **~17.78 GB/token**. 32.9 × 17.78 ≈ **585 GB/s ≈ 96% of 608**. C1 decode is at the bus.

`/start_profile` is **404** on the pin. Do not chase a profiler image.

## 2. DSpark / DFlash — closed for this week

- **DSpark** is RadixArk/SpecForge + **NVFP4/SGLang/Blackwell**, not z-lab DFlash and not a B70-native next step. Name collision with DGX Spark is real; the method is still the **wrong stack** for maxing this 27B on 1× B70.
- **No official Qwen3.8 DFlash.** `z-lab/Qwen3.6-27B-DFlash` has no 3.8 successor.
- Community 3.6→3.8 GGUF transplant: author already lost to native MTP.
- Doopeworld DSpark-vLLM on 1× B70 GPTQ: ~52 t/s tg32 vs our MTP4 83. Not our cells; still MTP2-class.

Do not pull those drafts. Do not GPTQ them. Do not stand up SGLang to serve official DSPARK.

## 3. GPTQ the 1-layer MTP head?

**Refuted as a speed lever.** Head is 0.849 GB of 19.55 GB. INT4 saves ~0.64 GB. Extra cost `d=0.87` is mostly **not** those bytes. `B70_MTP_BF16_DRAFT=1` is the pinned path. MTP8 already died on **acceptance compounding**, not BF16 bandwidth.

VRAM-only A/B later, never overwrite the BF16-head artifact.

## 4. Can we train a drafter on this host?

Online SpecForge / Eagle3 / DFlash / DSpark with a **55 GB BF16 teacher: no.** That class of “immense results” is an 80+ GB GPU + BF16 teacher story. This host already **OOM-rebooted** materializing that teacher for GPTQ.

| Recipe | Fits 1× B70 + 30 GB RAM? |
|---|---|
| Online Eagle3/SpecForge, BF16 27B teacher | **No** — 55 GB vs 30 GB RAM |
| Online FP8 27B + draft + Adam + KV | **No** |
| Multi-layer continued pretrain of full 27B | **No** |
| Offline dump from **existing GPTQ**, then train **one extra MTP layer or LoRA** | **Maybe, days.** Only honest on-box path. |
| GPTQ the current MTP head | Not a speed path |

**Success bar before celebrating:** MTP6/8 with **mean accept γ ≥ 6** and `d ≲ 1.2`. Anything less loses to MTP4 (already proven: 1-layer MTP8 = 71 t/s). 95 t/s at R0=32.9 needs `γ/(1+d) ≥ 2.89`. Today’s d=0.87 wants **γ ≥ 5.4**. A fatter serial 3-layer draft (d≈1.5) wants **γ ≥ 7.2**.

The artifact that actually wins is a **multi-layer or parallel next-k** head. Qwen did not ship it. One extra **serial** layer is a research attempt, not a promised 95 t/s.

## 5. What was executed this session

1. Architect review of claims + roofline (Run 40 matched 2.54×, not 3.33×).
2. HF reality check: no official 3.8 DFlash; DSpark rejected as the speed program.
3. Host gates: empty GPU, 56 G on models2, 27 G on `/`, waiter loop killed.
4. This note + offline dump/train scripts under `scripts/tmp/`.
5. **Synthetic dry-run** of the extra-layer trainer (no 27B load).

Not executed (correctly): SpecForge, BF16 teacher download, DSpark pull, GPTQ-MTP, vLLM reload for a 404 profiler, second engine.

## 6. Next execute list (do in order)

1. **Production stays** GPTQ-INT4 + BF16 MTP4 on `2c427ef` (81.26) or Run 40 digest (83.7). Do not mix those images in one A/B.
2. **Do not** train tonight with SpecForge / online teacher / BF16 27B.
3. **Optional 30 min, same image as the number you cite:** log `vllm:spec_decode_*` + wall time on one p512/g128. Closes 1.50 vs 0.87 accounting. Adds **0 t/s**.
4. **Later multi-day campaign only:** `scripts/tmp/b70-qwen38-offline-hidden-dump.py` then `scripts/tmp/b70-qwen38-train-extra-mtp-layer.py`. Kill if γ < 6.

## 7. Offline campaign contract (when authorized)

- Teacher = GPTQ path above. **Never** the 23 M BF16 stub, never a new 55 GB pull.
- One engine. Dump **last hidden (5120 BF16) + next-k token ids only**. Do not dump 64-layer Eagle3 features (TBs).
- Budget: 2k seq × 256 tok × 5120 × 2 B ≈ **5.2 GB** on `/mnt/<models-disk>`. 10k×512 is **52 GB** — only if models2 still has the room *after* the GPTQ stays.
- Then **kill vLLM**. Train extra MTP layer or LoRA, batch 1–2, seq 256–512. Full 0.85 GB + fp32 Adam ≈ 4.3 GB — that part fits.
- A/B: MTP4 vs MTP6/8, p512/g128 + p8192/g128, n=3 screen, cache off, C1. Kill if not above MTP4.
- Do not use swap as a training plan (last time: ENOSPC).
