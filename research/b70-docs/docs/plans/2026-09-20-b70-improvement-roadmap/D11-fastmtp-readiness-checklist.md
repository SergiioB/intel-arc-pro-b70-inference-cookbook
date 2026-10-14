---
title: "D11 readiness checklist — prep evidence, 2026-09-23"
type: plan
date: 2026-09-23
status: execution-complete
---

# D11 readiness checklist (execution gates validated on B70)

Every item is marked DONE / BLOCKED / EXECUTION-TIME with the real evidence gathered on 2026-09-23.
Spec: `D11-fastmtp-head-retrain.md` (same directory).

## Tooling

- [x] **DONE — speculators repo identified and pinned.** The wave-3 "Red Hat speculators" is
  `vllm-project/speculators` (Red Hat-authored, Apache-2.0; `gh search repos speculators` →
  "A unified library for building, evaluating, and storing speculative decoding algorithms for LLM
  inference in vLLM"). MTP fine-tuning shipped in **v0.6.0** (release notes 2026-06-16: "MTP
  finetuning support: Finetune native MTP heads"); latest **v0.8.0** (2026-09-03). Reference tree:
  `main@824125a36600b5af8fdd1b18286da51ed60f60a8` (2026-09-22), shallow-cloned to
  `/mnt/<sata-disk>/d11-prep/speculators/` (15 MB, under `/mnt`).
- [x] **DONE — MTP recipe verified in-repo.** `vllm-project/vllm: docs/user_guide/algorithms/mtp.md`: Convert → Train →
  Stitch; "Only the MTP layers are trainable — embed_tokens and lm_head are frozen and shared with the
  verifier"; per-step loss weights `alpha_k = beta^(k-1)/Σ beta^(j-1)`, default beta=0.6 (FastMTP Eq. 2).
- [x] **DONE — venv installed, version pinned, nothing on `/`.**
  `/mnt/<sata-disk>/d11-venv/`: Python 3.12.13 (host 3.14 is too new — torch.compile
  raises `RuntimeError: torch.compile is not supported on Python 3.14+` at speculators import; first
  attempt on 3.14 was deleted and rebuilt on 3.12), torch 2.9.0+cpu, torchaudio 2.9.0+cpu,
  torchvision 0.24.0+cpu (CPU wheels from download.pytorch.org/whl/cpu — no CUDA deps pulled),
  transformers 5.16.1, datasets 5.0.1, **speculators 0.8.0**. `speculators --help` lists all five
  pipeline commands (prepare-data, regenerate-responses, generate-offline-data, train, stitch-mtp).
- [x] **DONE — converter smoke test PASSED on the real checkpoint (CPU, cheap).**
  `speculators convert /mnt/…/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-nogidx --algorithm mtp
  --algorithm-kwargs '{"num_speculative_steps": 4}'` → "Extracted 15 MTP weight tensors" (read only
  shard 5), validation `MTPDraftModel.from_pretrained` SUCCESS, output
  `/mnt/<sata-disk>/d11-prep/mtp-converted-cpuidle/` (config: `algorithm: mtp`,
  `speculative_tokens: 4`, `proposal_type: greedy`, verifier = our path). Source checkpoint untouched.
- [x] **DONE — architecture family supported.** Our `text_config.model_type` is `qwen3_5_text`;
  speculators registers `qwen3_5_text` (transformers ≥5.2.0) in `src/speculators/models/mtp/model_definitions.py`;
  the reference example `examples/train/mtp_qwen3_5_9b_gsm8k_online.sh` is the same family. Note:
  transformers 5.16.1 itself ignores `mtp.*` keys for qwen3_5 (`_keys_to_ignore_on_load_unexpected =
  [r"^mtp.*"]`) — training support lives entirely in speculators' own MTP classes, which is what the
  converter validated.
- [x] **DONE — device-path audit (honesty check).** Trainer is accelerator-agnostic:
  `torch.accelerator.current_accelerator()` in `train/trainer.py:217`, `train/cli.py:746`,
  `train/distributed.py`, `train/recovery.py`; exactly one `torch.cuda.is_available()` guard in the
  whole package (`vllm-project/vllm: models/utils.py:74`, around `torch.compile`). Attention default = SDPA
  (`vllm-project/vllm: models/mtp/core.py:115-117`). Upstream docs declare NVIDIA/AMD/Ascend as tested — **Intel XPU is
  best-effort, not proven** (spec §e); CPU-only torch is installed for pipeline debugging.

## Inventory

- [x] **DONE — MTP head inventory from safetensors headers only** (8-byte header-length read + JSON
  parse; zero weight loading). 15 tensors, all BF16, all in `model-00005-of-00005.safetensors`,
  **424,699,392 params = 0.791 GiB**. Largest: `mtp.fc.weight` [5120, 10240]; MLP pair
  gate/up [17408, 5120] + down [5120, 17408]; gated attention q_proj [12288, 5120]. Full table in the
  spec §c. No dedicated draft lm_head/embedding — shared with verifier
  (`mtp_use_dedicated_embeddings: false`; `lm_head.weight` in shard 2, untouched by D11).

## Data

- [x] **DONE — data route pinned:** on-policy self-distillation per
  `vllm-project/vllm: docs/user_guide/tutorials/response_regeneration.md` (regenerate assistant responses with the target
  itself; multi-turn turn-by-turn). Our server = the champion stack at model-card sampling
  (temp 0.7 / top_p 0.8 / top_k 20 / presence 1.5).
- [x] **DONE — prompt sources verified on host:** `results/2026/08/vllm-dense27-int4-mtp4-20260809T110844Z-343136/exact-prompts-200k.json`
  (2.5 MB, present), `scripts/tmp/mia-prompts.json` (13.5 KB, present); fetchable: gsm8k / magpie /
  ultrachat (example datasets of the same repo). Bucket code/prose/chat per wave-3 probe 6.
- [x] **DONE — disk math computed:** offline hidden states = 80 MiB/sample at 8192×5120 bf16;
  5000×8192 = 391 GiB (exceeds comfortable `/mnt` headroom → **online streaming is the primary
  route**); reduced offline 1500×4096 = 59 GiB fits. `/mnt` free: 178 GiB (measured 2026-09-23);
  `/` free: 23 GiB (90 % full — keep flat; respected).
- [x] **DONE (2026-09-24) — balanced 1,500-sample corpus generated & extracted.** Generated on-policy under non-thinking recommended sampling (`enable_thinking=false`, temp 0.7, top_p 0.8, presence 1.5) across 4 buckets: 500 GSM8K, 500 Magpie, 400 UltraChat, 100 Technical/Coding. Preprocessed to `prepared_d11_1500`, 1,500 layer-64 hidden-state safetensors extracted (6.6 GiB total, zero corruption).

## Hardware fit (computed, not assumed)

- [x] **DONE — VRAM arithmetic for one B70 (32,656 MiB visible):** core training state ≈ 8.7 GiB
  (params 0.79 + grads 0.79 + AdamW fp32 3.16 + frozen embed+lm_head 4.74); draft logits
  (vocab 248,320 × 4 steps) dominate: fused-loss T=8192 ≈ 24.8 GiB (fits, tight), T=4096 ≈ 17.3 GiB;
  eager-loss T=1024 ≈ 19.2 GiB (fits), T=2048 ≈ 28.6 GiB (borderline vs the 31,000 MiB gate),
  **T≥4096 eager does not fit**. Verdict and fallbacks in spec §e.
- [x] **DONE (2026-09-24) — XPU train smoke test PASSED on B70 (XPU:0):** Validated natively on `torch 2.13.0+xpu` inside `f01e24f6c7ff` container via `scripts/tmp/test_mtp_xpu_training_smoke.py`. MTPDraftModel loaded, forward pass completed (SDPA attention), backward pass computed finite non-NaN gradients across all 15 `mtp.*` parameter tensors, AdamW step verified, and 5-step loss convergence confirmed on fixed batch (Loss: 64.88 -> 0.13). VRAM scaling ladder measured: T=64 (8.80 GiB), T=512 (10.84 GiB), T=1024 (14.71 GiB), T=2048 (20.51 GiB), leaving >11 GiB margin under the 32 GB visible ceiling. XPU training viability is empirically proven on Battlemage.

## Serving-side integration (unchanged machinery)

- [x] **DONE — re-quant path confirmed:** champion harness
  (`scripts/tmp/d12-finalist-20260923.sh`) mounts `patch_mtp_nightly.py` / `patch_mtp_boundary.py` /
  `patch_gdn_fuse.py` and sets `B70_MTP_BF16_DRAFT=1 B70_DRAFT_LMHEAD_INT4=1 B70_DRAFT_MTP_INT4=1`;
  the checkpoint's `dynamic: {"-:.*mtp.*": {} }` keeps the (new) head BF16 on disk so the load-time
  W4A16 g128 draft quant applies identically. `lm_head` is frozen in training → the draft-LM-head
  quant input is bit-identical to today's.
- [x] **DONE — α measurement mechanics confirmed:** `vllm:spec_decode_num_accepted_tokens_total` /
  `vllm:spec_decode_num_draft_tokens_total` counter diffs, `d12-finalist-20260923.sh:156-157`.
- [x] **DONE — swap safety rules written:** stitch writes a NEW directory; config byte-compare
  (quantization_config, `b70_g_idx_stripped`, `tie_word_embeddings: false` unchanged); safetensors
  header diff shows only `mtp.*` changed; `lm_head.weight` hash equal to source. Source artifact never
  written through.
- [x] **DONE (2026-09-24) — hidden-states connector fully verified end-to-end on live XPU server:** Exercised an actual HTTP request to Qwen3.8-27B (`extract_hidden_states` + `ExampleHiddenStatesConnector`) served on B70 GPU 0. Layer-64 hidden states and token IDs were extracted and flushed to safetensors on disk via patched async copy stream (`patch_example_hidden_states_xpu.py`). Loaded into `MTPDraftModel`, executed forward pass (Loss: 22.66), computed finite gradients, and stepped AdamW optimizer. Risk R1 is completely resolved with live server-to-training evidence (`scripts/tmp/test_online_hs_client.py`).

## Validation plan (protocol)

- [x] **DONE — gates declared in spec §f:** ADOPT iff aggregate Δα ≥ +0.08 AND tps ≥ +3 % AND
  spec-canary PASS both arms AND output-preservation spot checks (greedy byte-identity + 30-draw ×
  6-prompt modal-class, the D2 gate) — measured on a fresh counterbalanced n=5 finalist after a
  4-window screen, one variable = checkpoint dir.
- [x] **DONE — α resolution rule carried verbatim from RL-048:** ≤0.05 single-run Δα unresolvable at
  n=5 (fleet α spread 0.48–0.70); D12's screen→finalist flip is the precedent; the +0.10…+0.20 D11
  expectation is exactly the resolvable class.
- [x] **DONE (2026-09-24) — stock per-step decay pre-measure executed fresh on GPU 1 (230 W):** Measured via `scripts/tmp/run_d11_baseline_decay_probe.sh` (n=5 C1 p512/g128, champion stack). Confirmed catastrophic decay cliff at position 2: pos_0 = **98.0%** (198/202), pos_1 = **97.5%** (197/202), pos_2 = **12.4%** (25/202), pos_3 = **8.9%** (18/202). Aggregate α = **0.5421**, post-first median = **73.68 tok/s**. Evidence preserved in `results/2026/09/b70-improvement-probes-20260920/d11-baseline-decay-probe-*/decay_curve.json`. The business case for FastMTP position-shared retraining is empirically confirmed on same-day hardware.
- [x] **DONE (2026-09-24) — execution & gate verdict: FAILED / WITHDRAWN.**
  Trained MTP draft head across 1,350 balanced non-thinking training samples (Val loss 0.9431). Stitched into `Qwen3.8-27B-GPTQ-Int4-sym-G128-FastMTP1500-BF16` (swap safety verified: config identical, non-MTP tensors identical, `lm_head.weight` bit-identical). Measured on GPU 1 (230 W, n=5 C1 p512/g128): spec-canary PASS; aggregate α = **0.5023** (vs stock 0.5537 / 0.5421; Δα = -0.0514 / -0.0398); post-first throughput = **71.83 tok/s** (vs stock 77.26 / 74.28 tok/s; -7.0%); measured power = 226.74 W. Gate required Δα ≥ +0.08 and tps ≥ +3% → **FAILED**. Position sharing flattens position 2/3 decay slightly (28.0% / 21.0% vs stock 12.4% / 8.9%), but depresses position 0/1 accuracy (83.2% / 68.7% vs stock 98.0% / 97.5%), netting a lower aggregate acceptance and lower serve speed. Stock MTP weights remain champion.

## Governance

- [x] Nothing committed or pushed; no external publication; all numbers PROVISIONAL.
- [x] **GPU work recorded (2026-09-24):** Full end-to-end execution completed: 1,500 balanced non-thinking samples regenerated, hidden states extracted, 1 epoch trained, stitched, swap-safety verified, benchmarked, 150 W power caps restored on both GPUs, processes cleaned up, VRAM 40 MiB / 32,656 MiB.
- [x] `/` disk flat (venv, clone, converted head all under `/mnt/<sata-disk>/`).
- [x] No credentials written into any file.
