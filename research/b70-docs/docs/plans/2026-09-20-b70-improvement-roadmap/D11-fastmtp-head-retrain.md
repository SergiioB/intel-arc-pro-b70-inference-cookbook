---
title: "D11 — FastMTP-style retrain of the shipped Qwen3.8-27B MTP head (program spec)"
type: plan
date: 2026-09-23
status: prep-complete-awaiting-execution-window
---

# D11 — FastMTP-style fine-tune of the native shipped MTP head

**PROVISIONAL — prep/readiness program. No training run has happened; no benchmark claim is made here.
Execution is gated on an owned GPU window (another campaign holds the GPUs) and on the readiness
checklist: `D11-fastmtp-readiness-checklist.md` (same directory).**

Basis: LEVERS D9–D14 — acceptance α is the steep lever (**+8.3…+8.8 % throughput per +0.05 α** at the
measured α ∈ [0.487, 0.568]); every config-level α mover measured and lost (D10 clamp −5.45 %, D12
probabilistic drafting −3.86 % at finalist, D13 penalty audit neutral/keep 1.5, D14 fp8 draft-KV
falsified). Depth is closed (D6/D9). Wave-3 research
(`worklogs/RESEARCH-acceptance-levers-wave3-20260923.md`) identified the one remaining frontier: the
vanilla MTP head's per-step acceptance decay (published: 70 % → 11 % → 2 % at k=1..3 on a 7B vanilla
head vs 81 % / 56 % / 36 % after a FastMTP fine-tune, arXiv:2509.18362). A retrained head is the only
lever whose expected effect (+0.10…+0.20 Δα) exceeds the α resolution floor of our n=5 protocol (RL-048).

## (a) Goal + acceptance targets

**Goal:** fine-tune ONLY the shipped MTP module (`mtp.*`, 424,699,392 params — 1.6 % of the 27B
backbone) with position-shared weights on self-distilled data, swap the retrained head into the draft
checkpoint, re-run the existing draft-side INT4 pair, and measure whether per-step acceptance flattens
enough to move aggregate α at k=4.

**Primary target (adoption gate):** aggregate α at k=4, measured window, **+0.10 or more over the
stock-head arm** (expected band +0.10…+0.20).

**Secondary / diagnostic targets:**
- Per-step α at k=1..4 from `vllm:spec_decode_num_accepted_tokens_total` /
  `vllm:spec_decode_num_draft_tokens_total` counter diffs (our harness's measurement, exactly as in
  `scripts/tmp/d12-finalist-20260923.sh:156-157` and every D-campaign window): the stock head's
  per-step decay curve is measured first (business-case probe, §f step 1); the retrained head must
  flatten step-2+ decay, not just lift step-1.
- Engine throughput at k=4: arithmetic τ projection at champion c_d=1.5 ms / c_v=28.8 ms is
  **+19.7 % (Δα=+0.10) to +44.6 % (Δα=+0.20)** single-stream wall (PROVISIONAL arithmetic, not a
  claim); the measured +8.3…+8.8 %/0.05α local slope predicts +17…+35 %.
- No regression: spec-canary PASS, c_d unchanged (~1.5 ms/draft token — same shapes, same quant),
  output-preservation spot checks pass (§f).

## (b) Tooling route (pinned)

**Route: `vllm-project/speculators` (Red Hat-authored; the "speculators 0.6.0 FastMTP recipe" of the
wave-3 research), MTP algorithm.** Pipeline = Convert → (regenerate-responses → prepare-data) → train
→ stitch-mtp. MTP fine-tuning shipped in v0.6.0 (2026-06-16, release notes: "MTP finetuning support:
Finetune native MTP heads"); our pin is the current release:

| Artifact | Pin | Evidence |
|---|---|---|
| speculators release | **v0.8.0** (PyPI, 2026-09-03) | installed in the D11 venv, `speculators --version` 0.8.0 |
| repo commit (reference tree) | `main` = `824125a36600b5af8fdd1b18286da51ed60f60a8` (2026-09-22) | shallow clone at `/mnt/<sata-disk>/d11-prep/speculators/` |
| key docs | `vllm-project/vllm: docs/user_guide/algorithms/mtp.md`, `vllm-project/vllm: docs/user_guide/tutorials/train.md`, `vllm-project/vllm: docs/user_guide/tutorials/response_regeneration.md` | in the clone |
| reference example | `examples/train/mtp_qwen3_5_9b_gsm8k_online.sh` — **same `qwen3_5` family as our 27B** | in the clone |
| upstream paper | FastMTP arXiv:2509.18362; code https://github.com/Tencent-BAC/FastMTP | wave-3 §2.1 |
| venv | `/mnt/<sata-disk>/d11-venv/` — Python 3.12.13 (uv cpython), torch 2.9.0+cpu / torchaudio 2.9.0+cpu / torchvision 0.24.0+cpu, transformers 5.16.1, datasets 5.0.1, speculators 0.8.0 | installed and `speculators --help` verified; nothing on `/` |

**Execution-time venv change:** the torch stack must be swapped to the **XPU wheel** (`torch>=2.9` XPU
builds exist for Battlemage) inside the same venv before the train smoke test; the CPU stack installed
today is for pipeline debugging only. Server-side prereq from the docs is `vllm>=0.19.1`; our serving
build 0.27.2rc1.dev77 is numerically above it — the real question is XPU kv-connector support (§h R1).

**Converter smoke test PASSED against our checkpoint (2026-09-23, CPU):**
`speculators convert <nogidx checkpoint> --algorithm mtp --algorithm-kwargs '{"num_speculative_steps": 4}'`
extracted exactly the 15 `mtp.*` tensors from `model-00005-of-00005.safetensors`, validated
(`MTPDraftModel.from_pretrained`), and wrote
`/mnt/<sata-disk>/d11-prep/mtp-converted-cpuidle/` with `speculators_config.algorithm=mtp`,
`speculative_tokens: 4`, `proposal_type: greedy`, verifier pinned to our path. The converter reads only
the `mtp.*` subtree — the 19 GB backbone is never loaded.

## (c) MTP-head inventory (from safetensors headers only — no weights loaded)

Source: `/mnt/<sata-disk>/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-nogidx/`
(local conversion of `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`). All 15 tensors live in
**`model-00005-of-00005.safetensors`**, all **BF16** (quantization_config `dynamic: {"-:.*mtp.*": {} }`
excludes MTP from GPTQ; `lm_head: false` keeps the verify head unquantized). Header-read 2026-09-23
via 8-byte length + JSON header parse.

| Key (all in shard 5) | Shape | Params |
|---|---:|---:|
| `mtp.fc.weight` | [5120, 10240] | 52,428,800 |
| `mtp.pre_fc_norm_hidden.weight` | [5120] | 5,120 |
| `mtp.pre_fc_norm_embedding.weight` | [5120] | 5,120 |
| `mtp.layers.0.input_layernorm.weight` | [5120] | 5,120 |
| `mtp.layers.0.self_attn.q_proj.weight` | [12288, 5120] | 62,914,560 |
| `mtp.layers.0.self_attn.k_proj.weight` | [1024, 5120] | 5,242,880 |
| `mtp.layers.0.self_attn.v_proj.weight` | [1024, 5120] | 5,242,880 |
| `mtp.layers.0.self_attn.o_proj.weight` | [5120, 6144] | 31,457,280 |
| `mtp.layers.0.self_attn.q_norm.weight` | [256] | 256 |
| `mtp.layers.0.self_attn.k_norm.weight` | [256] | 256 |
| `mtp.layers.0.post_attention_layernorm.weight` | [5120] | 5,120 |
| `mtp.layers.0.mlp.gate_proj.weight` | [17408, 5120] | 89,128,960 |
| `mtp.layers.0.mlp.up_proj.weight` | [17408, 5120] | 89,128,960 |
| `mtp.layers.0.mlp.down_proj.weight` | [5120, 17408] | 89,128,960 |
| `mtp.norm.weight` | [5120] | 5,120 |
| **Total** | | **424,699,392 (0.791 GiB BF16)** |

Structure = `Qwen3_5MultiTokenPredictor`: fc projects concat[last-hidden-state (5120) ‖ token embedding
(5120)] → 5120 (`pre_fc_norm_*` normalize each half), one full-attention decoder layer (gated q_proj:
12288 = 2×6144, `attn_output_gate`), then `mtp.norm` before the **shared** LM head. There is **no
dedicated draft lm_head or embedding in the checkpoint** (`mtp_use_dedicated_embeddings: false`) —
the draft shares the verifier's `lm_head.weight` [248320, 5120] (shard 2) and
`model.language_model.embed_tokens.weight` (shard 1); speculators mirrors this exactly
(`embed_tokens` and `lm_head` "loaded from verifier at runtime"; its training core deletes
`verifier_lm_head` and never trains the shared heads — verified in `src/speculators/models/mtp/core.py`).

## (d) Data recipe — self-distilled, on-policy

Per the speculators contract and FastMTP: **MTP training data must be generated by the target model
itself** ("the drafter learns to predict what the target actually generates" — response-regeneration
doc). We regenerate with the deployed champion stack at the model-card recommended sampling
(non-thinking temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5 — the published-run policy).

1. **Prompts (3–5 k conversations, mixed domains):**
   - On host: `results/2026/08/vllm-dense27-int4-mtp4-20260809T110844Z-343136/exact-prompts-200k.json`
     (captured serving prompts), `scripts/tmp/mia-prompts.json`.
   - Fetchable: gsm8k (the speculators example's domain), magpie / ultrachat via the repo's
     `scripts/response_regeneration/run_all.sh --dataset …`.
   - Bucket code / prose / chat (wave-3 probe 6): the fastest-decaying bucket gets the largest share.
2. **Regenerate responses** with `speculators regenerate-responses` against the champion vLLM XPU
   server (multi-turn regenerated turn-by-turn to keep history on-policy). Cost estimate at champion
   rates: ~4 M output tokens ≈ 8 h at B=4 (≈140 tok/s aggregate, S1) to 15 h single-stream.
3. **Prepare:** `speculators prepare-data --render-endpoint <server>` (chat template + tokenize +
   loss mask; measured 31 s for 5 k samples in the reference example).
4. **Position-shared-weight trick (the FastMTP fix):** one shared head drafts all positions
   recursively — at step k it consumes concat[verifier last hidden state at t; ground-truth embedding
   at t+k] and its own output feeds step k+1. Training unrolls `num_speculative_steps=4` with the same
   weights and applies normalized exponential per-step loss weights (FastMTP Eq. 2; `--step-weight-beta 0.6`)
   = **[0.460, 0.276, 0.165, 0.099]** for k=1..4 — the head is optimized for its recursive inference
   pattern, which is exactly what vanilla MTP training never did.
5. **Hidden states:** training inputs are verifier last-layer (layer 63 of 64; speculators pins MTP to
   "the last hidden layer" — `MTPDraftModel.target_layer_ids`) hidden states, extracted online by the
   vLLM hidden-states kv-connector (`ExampleHiddenStatesConnector` via `hs_connectors`), or offline via
   `speculators generate-offline-data`. **Online is mandatory at full volume**: offline caching at our
   dims costs 391 GiB for 5000×8192 (one 80 MiB `hs_*.safetensors` per sample) vs 178 GiB free on
   `/mnt`; a reduced offline run (1500×4096 ≈ 59 GiB) is the disk-safe fallback.

## (e) Training config (and the honest hardware-fit verdict)

One-line config (mirrors the passing reference example, sized to our k=4 champion):

```text
python -m speculators.train --verifier-name-or-path <stitched-or-nogidx ckpt> \
  --data-path <prepared> --vllm-endpoint http://localhost:<port>/v1 \
  --save-path <ckpt-dir> --speculator-type mtp --num-speculative-steps 4 \
  --target-layer-ids 64 --step-weight-beta 0.6 --epochs 3 --lr 1e-4 \
  --total-seq-len 8192 --loss-implementation fused --optimizer adamw
```

- **Frozen:** `embed_tokens`, `lm_head` (shared with verifier — the losslessness boundary); **trainable:
  `mtp.*` only (424.7 M params)**.
- optimizer adamw (schema default), lr 1e-4 (reference-example value; schema default 1e-3 is not what
  the shipped MTP example uses), 3 epochs, cosine schedule + ~3 % warmup, bf16, sdpa draft attention
  (`draft_attn_impl: sdpa` is the code default; eager attention OOMs long sequences per repo #886),
  `checkpoint_best` selected by acceptance eval.
- **Hardware fit on one B70 (32,656 MiB visible), computed 2026-09-23:**
  core state = params 0.79 + grads 0.79 + AdamW fp32 3.16 + frozen embed+lm_head 4.74 ≈ **8.7 GiB**;
  draft logits over vocab 248,320 × 4 steps dominate the rest: bf16 1.89 GiB (T=1024) → 15.2 GiB
  (T=8192), with fp32 CE intermediates ×2 on top in the eager loss.
  - **fused Triton loss:** T=8192 ≈ **24.8 GiB — fits**, tight (~6 GiB margin); T=4096 ≈ 17.3 GiB comfortable.
  - **eager loss (the XPU-safe fallback):** T=1024 ≈ 19.2 GiB fits; T=2048 ≈ 28.6 GiB borderline vs
    31,000 MiB gate; **T≥4096 does not fit.**
- **Hardware Verdict: EMPIRICALLY PROVEN ON B70 (2026-09-24).** The XPU train smoke test (`scripts/tmp/test_mtp_xpu_training_smoke.py`) passed cleanly on XPU:0 using `torch 2.13.0+xpu`. SDPA draft attention, forward pass, backward pass, finite non-NaN gradients across all 15 `mtp.*` tensors, AdamW parameter update, and multi-step loss convergence (64.88 -> 0.13) were verified. Measured VRAM scaling: T=64 (8.80 GiB), T=512 (10.84 GiB), T=1024 (14.71 GiB), T=2048 (20.51 GiB), confirming that sequences up to T=2048 train comfortably on a single B70 card with >11 GiB margin.

## (f) Acceptance-campaign validation plan (protocol-shaped)

1. **Pre-measure (business case, request-only):** per-step α at k=1..4 from spec-decode counter diffs
   on 5 fixed prompts × domains on the stock champion. This is the stock decay curve D11 attacks.
2. **Train + stitch** in an owned window; produce `stitched` checkpoint (§g).
3. **Screen (4 windows):** stock vs stitched, one variable = checkpoint dir; champion recipe identical
   (MTP4, draft-INT4 pair env, fp8 KV, greedy drafting — D12 rejected probabilistic; presence_penalty
   1.5 — D13); C1 p512/g128 and p8192/g128; counter-bracketed α; spec-canary each arm.
4. **Fresh n=5 finalist confirmation**, counterbalanced, both canaries, greedy byte-identity +
   30-draw × 6-prompt modal-class agreement (the D2 quality gate).
5. **Gates (declared before measuring):** ADOPT iff aggregate **Δα ≥ +0.08** AND throughput **≥ +3 %**
   AND canaries PASS AND output-preservation spot checks pass; else do-not-adopt and keep the stock
   head (the retrained artifact is retained as evidence either way).
6. **α resolution rule (RL-048, verbatim):** "single-run alpha deltas <= 0.05 are unresolvable at n=5
   given the observed alpha spread [0.48..0.70] — alpha claims need either many more windows or a
   retrained-head effect size (D11)… never promote an alpha lever from an n=5 window when the fleet's
   alpha spread is 0.48..0.70 (needs >= 0.10 effect or many more windows)." The D12 screen→finalist
   flip (screen +0.0505 α → finalist −0.0037) is the precedent; that is why the gate is +0.08 and the
   expected effect must be +0.10…+0.20 — exactly the resolvable class. Per-step curves are reported as
   diagnostics, not adoption proof.

## (g) Losslessness boundary + draft-INT4 re-quant

- **What changes:** only the 15 `mtp.*` tensors (draft-side). **What never changes:** the 27B target,
  its BF16 `lm_head.weight` (verification logits — the D8 boundary: the verify head is never
  quantized), `embed_tokens`, tokenizer, config. Serving stays **speculative with full-vocab target
  verification**, which is the published output-preservation property: greedy outputs byte-identical,
  sampled outputs from the target distribution (D1/D2 precedent). The draft head replacement does not
  touch this guarantee — rejected tokens are resampled from untouched target logits
  (losslessness ref: wave-3 §2.2).
- **Swap route:** `speculators stitch-mtp <ckpt_best> <verifier-ckpt> --output-path <new-dir>`
  (writes a new full checkpoint; source artifact untouched — copied, never written through). Then
  byte-verify: stitched `config.json` identical to source except nothing quantization-related
  (`quantization_config`, `dynamic -:.*mtp.*`, `b70_g_idx_stripped` unchanged; `tie_word_embeddings:
  false` unchanged); safetensors headers diff shows only `mtp.*` values changed; `lm_head.weight`
  content hash equal to source.
- **Re-quant:** the checkpoint keeps the head BF16 (the `-:.*mtp.*` exclusion is config, not weights),
  so the existing champion load-time machinery re-quantizes the new head exactly as it does today:
  `B70_MTP_BF16_DRAFT=1 B70_DRAFT_LMHEAD_INT4=1 B70_DRAFT_MTP_INT4=1` +
  `patch_mtp_nightly.py` / `patch_mtp_boundary.py` / `patch_gdn_fuse.py` (harness:
  `scripts/tmp/d12-finalist-20260923.sh`). `lm_head` is frozen during training, so the draft-LM-head
  W4A16 g128 quantization is bit-identical to the champion's input; only the MTP body weights differ.
  Do not quantize below the proven draft pair (ai-muninn counterexample; LEVERS do-not-attempt #6).

## (h) Risks + stop rules

| # | Risk | Likelihood | Mitigation / stop rule |
|---|---|---|---|
| R1 | vLLM XPU image lacks the hidden-states kv-connector (`ExampleHiddenStatesConnector`) for online extraction | Unknown — must grep the image first | Bounded check (≤1 session): grep image, try offline path. Fallback to a direct-extraction script on GPU 1. **If no extraction path lands within the declared budget: D11 = BLOCKED on tooling; stop; zero GPU burn.** |
| R2 | Fused Triton CE kernel fails on Intel Triton | Medium | `--loss-implementation eager` + seq ≤ 2048 (fits per §e arithmetic). If neither fits the training card: fallback (CUDA host), do not squeeze. |
| R3 | torch-XPU train smoke fails (untested upstream) | Medium | Bounded smoke (tiny subset, ≤1 window). CPU = debug only (§e). Then CUDA-host fallback or park D11 BLOCKED. |
| R4 | Trained head underdelivers (Δα < +0.08 at finalist) | Possible — external numbers are as-published, not reproduced here | Do-not-adopt per §f gates; keep stock head; record as a measured LOSS like D12/D14. |
| R5 | Third-party-style head/target mismatch (GPTOSS <10 % acceptance cautionary tale) | Low (we train the matched native head) | Acceptance-validate the stitched head on OUR target before any adoption; spec-canary + 5-prompt α first contact. |
| R6 | GPU contention with the owning campaign | Certain today | This program runs ONLY in an owned window: empty-GPU gate (§4 of contract), one engine, 150 W restore, cleanup evidence. Nothing GPU has been run during this prep. |
| R7 | Disk pressure on `/` | None | All artifacts under `/mnt/<sata-disk>/` (venv, clone, converted head, stitched ckpt, corpora, checkpoints). `/` stayed flat during prep. |
| R8 | Credential leakage | None | HF auth only via env/pre-existing key files; never written into any file here. |

**Standing constraints honored:** no commits, no pushes, no external publication; all numbers here are
PROVISIONAL or as-published-by-source; adoption requires the §f gates plus the claims-standard skill
audit before anything leaves the host.

## Deliverables / artifacts created during prep

- `/mnt/<sata-disk>/d11-venv/` — speculators 0.8.0 venv (CPU torch stack; swap to XPU wheel at execution)
- `/mnt/<sata-disk>/d11-prep/speculators/` — pinned reference clone (`main@824125a`)
- `/mnt/<sata-disk>/d11-prep/mtp-converted-cpuidle/` — converter smoke-test output (proves Convert stage against our checkpoint)
- This spec + `D11-fastmtp-readiness-checklist.md`
