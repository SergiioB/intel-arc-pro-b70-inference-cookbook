---
title: "Qwen3.8-27B + DFlash 2 on champion XPU — 2026-08-19 smoke"
type: research
date: 2026-08-19
status: final
---

# Qwen3.8-27B + DFlash 2 on champion XPU — 2026-08-19 smoke

**Status:** PROVISIONAL — NOT FOR PUBLIC HEADLINE. Load/smoke only. Not Lane 1.

## Question

Does official [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
(`DFlash2DraftModel`, ~3.85 GB, `method=dflash` `n=7`) work on the Qwen3.8
champion stack, and can acceptance be made non-zero?

## Stack

- Image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
  (vLLM `0.27.2rc1.dev77+gac7509e2b`, kernels `0.1.12.3`)
- Target: `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
- Draft: `/mnt/<models-disk>/Qwen3.8-27B-DFlash2` (no leftover `hf_quant_config.json`)
- Context 16K, U=0.88, fp8 KV, cache off, MBT 8192, configured 230 W
- Overlay: open [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816)
  sources + image `model_cls` / V2-runner hooks (`scripts/tmp/patch_qwen38_dflash2_overlay.py`)
- Required for warmup: `patch_gdn_mixed_split_v5.py`

## Sequence (do not skip the failures)

1. Unpatched image: pydantic `SpeculativeConfig` —
   `Model architectures ['DFlash2DraftModel'] are not supported`. Image has
   `DFlashDraftModel` only. Relabeling as v1 is an architecture mismatch.
2. Registry + files only (workspace tree): same ValidationError until
   **site-packages** was also patched. Then EngineCore built v1
   `DFlashQwen3Model` and died: no module `candidate_selector`.
3. Overlay + `decoder_layer_cls` / `model_cls` + `_is_dflash2_draft()` V2
   force, no v5: target + draft weights loaded. Warmup died:
   `causal_conv1d does not support spec-decode and non-spec (prefill + decode)
   tokens in the same invocation`.
4. Same overlay + v5: `/health` 200, 4645 MiB free after load.
   Raw: `results/2026/08/qwen38-dflash2-smoke-20260819T071612Z/`.
   Greedy `pong` ok. ignore_eos g128: **19.18 tok/s**, **Accepted 0 / Drafted 574**.
5. CUDA Triton selector on XPU: first torch fallback died
   `expand(XPULongType{[8,8]}, size=[8])` (`20260819T073836Z`). Gather rewrite
   then died in XPU graph capture:
   `wait method cannot be used for an event associated with a command graph`
   at `temp[valid] = temps[req_state[valid]]` (`20260819T074459Z`).
6. Graph-safe greedy walk + `--enforce-eager` + `VLLM_XPU_ENABLE_XPU_GRAPH=0`:
   load ok, 5177 MiB free. Drafts 896 / accepts **0**. post-first **8.54 tok/s**.
   Raw: `results/2026/08/qwen38-dflash2-smoke-20260819T075955Z/`.
7. Same + `attention_backend=TRITON_ATTN` + reuse parent `draft_logits`
   (do not replace the tensor the rejection sampler already bound):
   Triton backend selected. Warmup log
   `B70_DFLASH2_LOGITS finite_scores=0 score_max=nan n_valid=56`.
   Serving still **0 / 896**. Raw: `results/2026/08/qwen38-dflash2-smoke-20260819T081114Z/`.
8. NaN sanitizer on selector hidden + scores (`nan_to_num` → -1e4):
   `finite_scores=896 score_max=-10000.0`. Rejection still **0 / 896**.
   post-first **8.54 tok/s**. Raw: `results/2026/08/qwen38-dflash2-smoke-20260819T081653Z/`.

Checkpoint `is_causal: false` with all `sliding_attention` layers. Image
`_dflash_layer_causal` treated SWA as causal; overlay honors `is_causal`.
That was not sufficient. Candidate-selector scores are NaN on this XPU path
(`CandidateSelector` / grouped conv / non-causal FA). Filling -1e4 lets the
kernel run but is not a real proposal distribution, so acceptance stays 0.

## Measured (n=1, not a card)

| Run | load | drafts | accepted | post-first tok/s |
|-----|------|-------:|---------:|-----------------:|
| 071612Z FA + v5, CUDA selector | yes | 574 | 0 | 19.18 |
| 075955Z eager torch selector | yes | 896 | 0 | 8.54 |
| 081114Z Triton + reused logits | yes | 896 | 0 | 8.37 |
| 081653Z Triton + nan_to_num | yes | 896 | 0 | 8.54 |

Output stays coherent (`pong`, twenty `tick`s). Spec is dead weight.

## Decision (morning)

**Does not work as a serving spec on this setup.** Keep cookbook default
MTP4 (optional draft-INT4 overlay). Do not add DFlash 2 to the apply list.
Do not mix Nemotron DFlash patches onto this Qwen image.

A later image that vendors PR #52816 still needs **finite, non-clamped
selector scores** and a non-zero acceptance n=5 card before any speed claim.
The remaining gate is XPU correctness of DFlash2 grouped-conv + non-causal
attention feeding `CandidateSelector`, not registry/load.

## Afternoon ladder (2026-08-19) — chain proven, still not a recipe

NaN bisection (`B70_DFLASH2_DEBUG`) on a later load found two independent
causes for the 0/896 scores:

- **A.** XPU FlashAttention ignores `use_non_causal`; the DFlash2 draft is
  `is_causal: false`, so every draft attention layer returned exact zeros.
- **B.** Draft MLP overflows fp16 → inf → NaN. Checkpoint is bf16; the
  champion server is `--dtype float16`. Global bf16 (fix1) made scores
  finite (896/896, max 4.03) then OOM'd KV init (profiling peak ~7.3 GiB).

Mixed dtype (target fp16, draft bf16) peeled one GEMM/norm boundary per
load (fix4–11). Dummy-run `isfinite` over all draft params during memory
profiling is a silent KV killer (`Available KV cache memory: -1.33 GiB`)
even after dtype was solved — leave the debug patch off for serving.

**First non-zero acceptance (causal fallback, mixed dtype):**
`results/2026/08/qwen38-dflash2-fix12-20260819T144711Z/`. C1, cache off, 16K,
U=0.88, fp8 KV, 230 W cap, `--enforce-eager`, `n=7`. Greedy `pong` ok.
One-shot ignore_eos g128: **68 / 434** accepted draft tokens (**15.7%**),
client post-first **17.40 tok/s**, 5006 MiB free after load, Available KV
**3.47 GiB**. Causal fallback is the *wrong* mask for block diffusion;
it only proves the rest of the chain.

**Non-causal torch SDPA shim (5-layer draft, C1):**
`results/2026/08/qwen38-dflash2-fix13-20260819T145921Z/`. Same stack minus causal
fallback, plus `B70_DFLASH2_SDPA=1`. Greedy `pong` ok. One-shot g128:
**42 / 616** accepted (**6.8%**), client post-first **12.53 tok/s**,
4721 MiB free, Available KV **3.52 GiB**. Non-zero without the wrong mask,
but slower and lower acceptance than causal fallback on this n=1 prompt.

Neither cell is a Lane 1 card, not matched to MTP4 106.7 (different
prompt, n=1, no entropy prefix, lifetime `/metrics` totals include
warmup). Overlay stays research:
`scripts/tmp/patch_qwen38_dflash2_{overlay,fix2,fix13}.py` +
`qwen38-dflash2-fix12.sh` / `fix13.sh`.

## Labeled n=5 screen (2026-08-19T151054Z) — still not a recipe

Fix12 mixed-dtype + causal-fallback overlay, same image, C1, cache off
(`--no-enable-prefix-caching`, log `enable_prefix_caching=False`), 16K,
U=0.88, fp8 KV, `--enforce-eager`, configured **230 W**. Sampling:
greedy `temperature=0`, `enable_thinking=false`, `ignore_eos=true`.
Prompts: Run 40 calibrated JSON
`prompts-184007933656937-512.json` (target 512). Endpoint
`prompt_tokens=476` on every sample (template / thinking-off render;
record actual, do not relabel as p512). Warmup discarded. Timing:
client post-first via `stream_options.include_usage`. Spec counters
diffed per request.

| i | prompt_n | g | finish | post-first tok/s | acc/draft | hits | measured avg W |
|---|---:|---:|---|---:|---:|---:|---:|
| 0 | 476 | 128 | length | 22.412 | 81/336 | 0 | 127.97 |
| 1 | 476 | 128 | length | 23.085 | 82/315 | 0 | 128.10 |
| 2 | 476 | 128 | length | 23.010 | 84/315 | 0 | 127.33 |
| 3 | 476 | 128 | length | 21.924 | 82/322 | 0 | 126.07 |
| 4 | 476 | 128 | length | 19.866 | 77/357 | 0 | 126.56 |

- **median 22.41 tok/s** (max 23.09, min 19.87, n=5)
- acceptance **406 / 1645 = 24.7%** (window sum)
- cache hits delta **0** every measured request
- after load: 5152 MiB free, Available KV 3.31 GiB / 26,014 tokens
- campaign monitor (includes load): pkg max 72.0 °C, vram max 70.0 °C,
  peak ~1 s interval-average **226.2 W**; per-request energy median
  **127.3 W**
- correctness: `coherent_output_smoke_tested` only (warmup/rep heads
  are assistant-ready prose). No token/KL/task-quality vs MTP4.

**Comparison to MTP4 106.7:** `not_comparable`. MTP4 106.7 is a
different spec stack (native MTP4 + draft-INT4 overlay, prefix ON,
champion 131K serve, greedy Run 47). This cell is DFlash2 research
overlay, causal (wrong) mask, 16K, cache off, endpoint p476/g128.
Do not write a multiplier.

**Decision:** keep MTP4 as the serving spec. Do not add DFlash 2 to
the cookbook apply list. Overlay is research-class only.

Harness: `scripts/tmp/qwen38-dflash2-n5.sh` +
`qwen38-dflash2-n5-client.py`. Evidence:
`results/2026/08/qwen38-dflash2-n5-20260819T151054Z/`. Cleanup: 31,640 MiB
visible_avail, 150 W.

## Why 24.7% is low (ranked, 2026-08-19)

This is **not** "DFlash2 is a 25% drafter." The n=5 cell used the
**wrong attention mask**. Potential is real (selector scores finite,
drafts proposed, target verifies some tokens) but the XPU path still
corrupts the draft distribution.

1. **Causal fallback is architecturally wrong (primary).** Checkpoint
   `is_causal: false`, `block_size: 8`, all five layers
   `sliding_attention`. Mask tokens must see the whole 1+N query block
   (right-context). Forcing `causal=True` (fix1 / n=5) makes the draft
   a 5-layer left-only predictor. ~25% is leftover prefix agreement,
   not block-diffusion fidelity. Throughput also suffers: n=5 drafts
   only 315–357 tokens per 128-output request vs ~574–616 on the
   non-causal cells — the verifier rejects earlier, so fewer drafts
   even get proposed.

2. **XPU FA / SDPA cannot do non-causal.** Image FA ignores
   `use_non_causal` (dummy `attn_out` absmax=0). torch SDPA on this
   stack is the same family (fix13 n=1: 6.8% / 12.53 tok/s).

3. **Float32 math non-causal (fix14) still lost.** n=3, same prompts,
   no causal fallback, independent post-RoPE K clone + softmax in fp32:
   median **13.15 tok/s**, accept **140/1701 = 8.2%**. Logs prove
   context attached (`ctx=yes`, propose-time `Tq=16 Tk=34`, attn
   absmax 15–55, finite). Dummy profiling still NaN'd `Tq=64 Tk=2112`
   (padding / empty ctx). So a working non-causal kernel is **not**
   sufficient — remaining distribution bugs sit beside the mask.

4. **Remaining XPU distribution bugs (keep this order):**
   - **Grouped-conv + RMSNorm recast.** Layer traces: `mlp_out`
     absmax 1e3–1e5 in bf16 after every residual. Recast after fused
     RMSNorm hides inf, it does not restore the trained activation
     scale. Candidate logits then sit in a compressed/noisy basin
     (`score_max=4.0` on the first logits log).
   - **RoPE layout.** PR warns a target vs draft `is_neox_style`
     mismatch silently kills acceptance. Overlay copies the target
     style at load; not independently verified on this GPTQ Qwen3.5
     target.
   - **Query vs context packing.** Official path writes context K/V
     into paged KV then attends query-only. Shims concat dense
     `[ctx; query]` with query positions that do not include ctx
     indices. Sliding-window tail (`2048`) vs packed 1+N=8 is easy
     to get slightly wrong; fix14 `Tk=34` on p476 is 26 ctx + 8 query
     — plausible, not proven aligned with slot_mapping.
   - **Selector walk / logits cache.** Torch greedy walk +
     `nan_to_num(-1e4)` is a CUDA-Triton substitute. Rejection
     sampling needs the realized path distribution; a leftover `-inf`
     row is the silent 0% path. Current n=5/n=3 >0 so this is no
     longer the 0% bug, but it can still distort later positions.
   - **Eager / no graphs.** `--enforce-eager` + math attn is a
     research crutch. Even 80% accept would still lose to MTP4 on
     this overlay until FA/graphs work.

5. **What will not raise acceptance:** more draft tokens (`n>7`),
   higher power, prefix cache, or comparing to MTP4 106.7. Optimize
   draft **hidden-state fidelity** (non-causal attn that matches CUDA
   FA + finite MLP without 1e5 residuals), then re-measure accept.

**fix14 n=3 evidence:**
`results/2026/08/qwen38-dflash2-fix14-20260819T152418Z/` (4941 MiB free,
KV 3.35 GiB, cache hits 0, cleanup 31,838 MiB / 150 W). Overlay:
`scripts/tmp/patch_qwen38_dflash2_fix14.py`. **Keep MTP4.**
