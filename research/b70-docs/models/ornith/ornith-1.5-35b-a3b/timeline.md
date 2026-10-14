---
status: living
title: "Timeline"
type: model-dossier
date: 2026-08-19
---

---
schema_version: 1
model_id: ornith-ai/Ornith-1.5-35B-A3B
model_slug: ornith-1.5-35b-a3b
document: timeline
as_of: 2026-08-20
status: provisional
truth_verdict: CONFIRMED
evidence_tier: E2
primary_evidence:
  - ../../../research/2026/ornith15-35b-a3b-adoption-20260819.md
  - ../../../docs/benchmark-history.md#run-50--ornith-15-35b-a3b-day-0-gptq--mtp-screen-2026-08-1920
  - ../../../results/2026/08/ornith15-35b-20260819T1630Z/supervise.log
---

# Timeline

| Date | Event | Evidence state |
|---|---|---|
| 2026-08-19 | Official release audited; topology matched to Qwen3.6-35B-A3B and no official GPTQ/AWQ was found in the recorded audit | Repository research note |
| 2026-08-19/20 | BF16 source downloaded and locally quantized with disk offload | Raw download and quantizer logs |
| 2026-08-20 | GPTQ contract verified: 30,720 qweights, 785 BF16 MTP tensors, zero MTP leaks | Quantization/verification log |
| 2026-08-20 | Exact post-run log audit counted 7,605 RTN fallbacks, 2,535 per expert projection; layer 39 had 297/768 | `supervise.stdout` |
| 2026-08-20 | vLLM XPU no-spec and MTP screens completed; MTP1 selected provisionally | C1 and acceptance JSON |
| 2026-08-20 | MixedCal-v2 attempt reached layer 35/39 but crossed the old `/mnt/<models-disk>` reserve; incomplete scratch was removed after evidence capture | MixedCal attempt-3 logs |
| 2026-08-20 | Dedicated 80 GiB ext4 workspace and fail-closed one-shot conversion workflow validated | `mixedcal-v2-conversion.md` and prevention-test log |
| 2026-08-20 | Safe `r3` conversion startup verified, then stopped at user request for a fresh night rerun; no candidate artifact published | `/mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/mixedcal-v2-20260820T165356Z-r3/` |
| 2026-08-20 | Authenticated LocalMaxxing eval-suite discovery retried; API route still returned HTTP 404, so no server, questions, score, or submission followed | `results/2026/08/ornith15-lmx-eval-20260820T175635Z/` |
| 2026-08-20 | Supported LocalMaxxing remote speed-test completed on the original GPTQ MTP1 stack: 85.8 tok/s median at 32-token prompt/g256, n=3; local validation passed; no submission | `results/2026/08/ornith15-lmx-speed-20260820T180754Z/` |
| 2026-08-20/21 | MixedCal-v2 r4 conversion completed (~7.7 h), atomic publish, contract OK, RTN 10.37% | `/mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/mixedcal-v2-20260820T204436Z-r4/` |
| 2026-08-21 | Quality A/B + n=3 screens + n=5 no-spec parity + MTP1 n=5 + exact-token 65K/131K/262K capacity | `results/ornith15-n5-maxkv-*`, `results/2026/08/ornith15-boundary-fixup-20260821T072410Z/` |
| 2026-08-21 | MTP2/MTP4 n=5: MixedCal MTP2 84.16 / MTP4 66.27 p512; MTP4 slower than no-spec | `results/2026/08/ornith15-mtpdepth-power-20260821T075858Z/` |
| 2026-08-21 | Paired 150↔230 W cold-input A/B on MixedCal: ~7.1k vs ~9.7k p2048/g1 | same power-ab/ |
| 2026-08-21 | DraftINT4 S+M1 n=5: MTP1 106.27 / 97.16, accept 81.9% vs S8 96.43 / 89.85, 77.0% | `results/2026/08/ornith15-draftint4-20260821T084132Z/` |
| 2026-08-21 | MixedCal-v2 uploaded to `SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2` (sha `ed26281b`, 6 shards size-matched) | Hugging Face public model |
| 2026-08-21 | LMX MixedCal MTP1 @150 W submitted: `cmt2sl6eg0hdcmv01gre5o3ub` APPROVED (`tokSOut` 94.1, p32/g256). Platform stored backend=cuda / specDecoding=false | `results/2026/08/ornith15-lmx-mixedcal-20260821T090450Z/` |
| 2026-08-21 | LMX + harness MixedCal no-spec 32K @230 W g=1: `tokSPrefill` 9556.4 (2899 tokens); **not submitted** (`tokSOut` 19k invalid) | `results/2026/08/ornith15-lmx-prefill230-20260821T095259Z/` |
| 2026-08-21 | LMX MixedCal no-spec 32K @230 W g128 submitted: `cmt2sr6gq0himmv01ogieh0c8` APPROVED (`tokSPrefill` 9780 / `tokSOut` 69.9, 2906+128 tokens) | `results/2026/08/ornith15-lmx-prefill230-g128-20260821T101300Z/` |
| 2026-08-21 | Cookbook Ornith section updated; publication gate dropped from PROVISIONAL banner to self-reported E2 with limits | cookbook `ornith15-35a3/` |
| 2026-08-21/22 | AutoRound W4A16 smoke + full conversion ar-r1 (3 h 26 m): contract 30720/0/785 identical, **zero RTN fallback**; host OOM reboot root-caused (qwen3_5_moe missing from auto_round MODEL_CONFIG → offloader silently disabled) and fixed via forced `is_model_patched` + streaming `quantize_and_save` | `/mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/ar-smoke-SUCCESS`, `/mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/ar-r1`, `research/2026/ornith15-autoround-vs-gptq-20260822.md` |
| 2026-08-22 | AutoRound-vs-GPTQ head-to-head (champion pattern, n=3): speed/KV parity (70.1 vs 69.0 p512/g128; MTP1 92.4–96.3, accept 78.5% vs 80.3%); 8/8 coherent probes both artifacts; AR recorded as quality-preferred candidate, not serving default | `results/2026/08/ornith15-ar-headtohead-20260822T0535Z/`, ADR-0007 |
| 2026-08-22 | **Reference logprob parity gate run** (streaming-BF16 sequential forward + vLLM prompt_logprobs, 315 positions): AR equal-or-best on all three primary metrics (mean Δ 0.2006 vs 0.2025, top-1 91.4% vs 90.5%, KL 0.277 vs 0.319); n=5 confirmation is the remaining pre-swap gate | `results/2026/08/ornith15-ar-parity-20260822/`, ADR-0007 v1.2 |
| 2026-08-22 | Intel vLLM-XPU skills cross-checked against all recorded serves: WNA16 MoE backend everywhere (no int4_gemm_w4a16 fall-through), fp8-KV+TRITON_ATTN pairing and bf16-activation A/B recorded as open cells | `research/2026/intel-vllm-xpu-skills-vs-b70-observations-20260822.md` |
| 2026-08-22 | Investigation published to the public cookbook: new cookbook doc `ornith15-35a3/AUTOROUND-VS-GPTQ.md` (head-to-head + streaming-BF16 parity gate + portable converter recipe) + family hub row; hygiene-audited, committed `2062b99`, not pushed | `intel-arc-pro-b70-inference-cookbook` repo |
| 2026-08-22 | **Local GPTQ copies removed at operator direction** (space recovery): `/mnt/<models-disk>` working copy (24.45 GB) + the entire 80 GiB `ornith-mixedcal-v2-ext4.img` (contained a second copy + residue). `/mnt/<models-disk>` 12→35 GB free; NTFS 137→217 GB free. HF-public artifact untouched (recovery = re-download, API 200 verified at deletion). AR artifact is now the only local Ornith INT4 artifact | `results/2026/08/ornith15-ar-parity-20260822/*-deletion-manifest.txt` |
| 2026-08-22 | Host-wide storage cleanup (operator-directed): Qwen3.8 Q4_K_M GGUF (16 GB), palmfuture Qwen3.6-35 GPTQ dead-end artifact (23 GB), vxk build trees + runtime caches (~5 GB). `/mnt/<models-disk>` 26→42 GB free; `/mnt/<models-disk>` 35→62 GB free. Nothing profile-referenced touched | `results/2026/08/ornith15-ar-parity-20260822/storage-cleanup-20260822-manifest.txt` |
| 2026-08-22 | **Launcher set correctly + Ornith unit added**: 8 dead GGUF profiles removed from `profiles.json` (backup kept), `defaultProfile` realigned to the Nemotron active pointer, new `ornith15-vllm-mtp1-16k` profile (AR artifact, `f01e24f6` digest override, MTP1 patches, model-card sampling from `generation_config.json`); `run-profile-vllm` gained per-profile image/patch overrides with Qwen3.6 defaults preserved. **Live-validated** through `vllm-mtp-profile.service`: healthy ~350 s, WNA16+FA kernels, coherent completion; state restored | `docs/operations/launcher-guide.md` § Catalog state |
| Open | LMX submit (not authorized); BF16 KL / task suite (`lmx eval suite list` still HTTP 404); native int4moe backport; DFlash2 | not measured |
