---
title: "Ornith-1.5-35B-A3B B70 adoption — 2026-08-19/20 (COMPLETE)"
type: research
date: 2026-08-19
status: final
---

# Ornith-1.5-35B-A3B B70 adoption — 2026-08-19/20 (COMPLETE)

Pre-authorized 18:30 job. **PROVISIONAL n=3 — NOT FOR PUBLIC HEADLINE.**
Evidence: `results/2026/08/ornith15-35b-20260819T1630Z/`

## Outcome

| Gate | Status |
|---|---|
| Release | Complete BF16 + FP8 + NVFP4 + GGUF + MLX. **No official GPTQ/INT4/AWQ.** |
| Architecture vs Qwen3.6-35B-A3B | **Match** (`qwen3_5_moe` / `Qwen3_5MoeForConditionalGeneration`) |
| GPTQ conversion | ✅ 24.42 GB, contract verified (30,720 qweight; 785 MTP BF16, 0 leaks) |
| vLLM XPU serve | ✅ champion `f01e24f6`, 32K fp16 KV U=0.85, KV 169,622 tok, 6,024 MiB free |
| Speculative decoding | ✅ MTP works (nightly+boundary patches); **MTP1 optimal: 94.3 t/s (+37%)** |
| Day-0 hypothesis (arch == Qwen3.6-35B) | **CONFIRMED** — same recipe transferred end-to-end |

## Headline numbers (PROVISIONAL n=3, C1, cache-off, greedy diagnostic, 230 W cap)

| Mode | p512/g128 post-first median | p~2.9K/g1 cold input median | Acceptance |
|---|---:|---:|---|
| no-spec 32K | **69.0 t/s** (68.9-69.3) | **9,537 t/s** (8,572-9,616) | — |
| MTP4 16K | 65.2 t/s filler / 69.8 real | 9,330 t/s | 24.8% overall; per-pos 81/15/2.5/0.5% |
| **MTP1 16K** | **94.3 t/s** filler; 94.3 real (90.0-98.4) | — | **80.3% pos0** |

Real-prompt vs filler (Run 46 lesson holds, smaller effect): MTP4 real +6% over filler;
MTP1 flat. The MTP4 regression on filler is the degenerate-prompt acceptance trap.

## 7. Why MTP1 (and not MTP4) for this head

Single-layer MTP head (same as Qwen3.6/3.8): per-position acceptance decays too steeply
for depth-4 (81% → 15% → 2.5% → 0.5%). Depth-1 keeps only the strong pos0 (80.3%) and
halves verify overhead → 94.3 t/s. This inverts the Qwen3.8 dense result (MTP4 optimal,
per-pos 92/84/73/69) — Ornith's head is much weaker past pos0 on this artifact.
Serving recommendation: `--speculative-config {"method":"mtp","num_speculative_tokens":1}`.

## 8. Failure ledger (this run)

1. **BLOCK 1**: `--device /dev/accel` doesn't exist on this host → docker 127. Use
   `--device /dev/dri --group-add render` only.
2. **BLOCK 2 (vLLM attempt 1)**: launch without `--language-model-only`,
   `--dtype float16`, `--block-size 64`, ZE envs and port mapping never became healthy
   in 15 min; `set -e` + trap destroyed the container before `docker logs` was captured.
   Fix: copy the exact champion pattern from `b70-qw38-speed-launch.sh`; capture
   `docker logs -f` from launch. Attempt 2 was healthy in 275 s.
3. **Swapfile-on-models trap**: 16G swapfile left models at 6.8G free — removed;
   offload + zram instead (architect guidance).
4. **GPTQ pace**: ~11-14 min/layer (subset-boundary calibration forwards dominate);
   6.4 h total for 40 layers at 128-text calibration. **Correction (2026-08-19):**
   the retained `quant_log.csv` records RTN fallback for **7,605 / 30,720 expert
   projection matrices (24.75%)**, split evenly across gate/up/down (2,535 each),
   rising to **297 / 768 (38.7%)** in layer 39. RTN here is GPTQModel's configured
   low-observation calibration fallback, not an XPU runtime fallback or failed INT4
   packing. The earlier ~6–9% estimate is superseded.
5. **Watcher bug**: failure detection keyed on "BLOCK" strings that only existed in
   Telegram subjects, not logs; fixed to watch the supervisor PID.

## 9. Reproduction (exact)

```bash
# serve (MTP1 recommended):
docker run -d --name ornith15 -p 8000:8000 --device /dev/dri --group-add $(getent group render | cut -d: -f3) \
  -v /dev/dri:/dev/dri:ro -v /mnt/<models-disk>/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro \
  -v spec-mtp1.json:/spec.json:ro -v patch_mtp_nightly.py:/patch_mtp.py:ro -v patch_mtp_boundary.py:/patch_boundary.py:ro \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash vllm/vllm-openai-xpu@sha256:f01e24f6... -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; SPEC=$(cat /spec.json); exec vllm serve /model \
   --quantization gptq --dtype float16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
   --kv-cache-dtype auto --port 8000 --host 0.0.0.0 --max-num-seqs 8 --max-num-batched-tokens 8192 \
   --block-size 64 --no-enable-prefix-caching --served-model-name ornith15 --language-model-only \
   --trust-remote-code --speculative-config "$SPEC"'
```

## Open gates (not measured)

- 64K/128K context ladder (KV headroom exists: 169,622 tokens at 32K config)
- Task-quality/reference parity vs BF16 (RTN fallback rate; speed ≠ quality)
- Cn concurrency, prefix-on agentic cells, power A/B, n=5 confirmation set
- Full-calibration re-quant only if quality gate demands it

## Telegram

START, AUDIT, 2× PROGRESS, GPTQ-OK, n=3 no-spec, FINAL sent to Hermes home channel.

## 1. Release (UTC 2026-08-19T16:30Z)

- `ornith-ai/Ornith-1.5-35B-A3B` sha `fbb995a79eedd569a5edc5f2af9644c0fa1124fc`
- BF16 tree 71.929 GB (16 shards; index `total_size` 71903645408)
- FP8 39.4 GB, NVFP4 23.5 GB (not a proven B70 XPU path)
- GGUF Q4_K_M 21.7 GB (llama.cpp, out of this vLLM-first job unless vLLM blocks)
- Collection API 404; org search listed the siblings above. No `*GPTQ*` repo.

## 2. Architecture diff vs `/mnt/<models-disk>/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4`

| field | Qwen3.6-35B-A3B | Ornith-1.5-35B-A3B | match |
|---|---|---|---|
| architectures | Qwen3_5MoeForConditionalGeneration | same | yes |
| model_type | qwen3_5_moe | same | yes |
| text_config.model_type | qwen3_5_moe_text | same | yes |
| layers / hidden / experts | 40 / 2048 / 256×8 | same | yes |
| hybrid GDN | 3× linear_attn + 1× full_attn, interval 4 | same | yes |
| head_dim / kv | 256 / 2 | same | yes |
| linear_attn geom | conv 4, k 16×128, v 32×128 | same | yes |
| MTP | mtp_num_hidden_layers 1 | same; 785 `mtp.*` tensors | yes |
| rope | θ=1e7, partial 0.25, mrope [11,11,10] | same | yes |
| vocab | 248320 | same | yes |
| vision | qwen3_5_moe_vision depth 27 | same | yes |
| router_aux_loss_coef | 0.001 | 0.0 | **no** (training leftover) |
| text_config.use_cache | true | false | cosmetic |
| transformers | 5.8.0 | 5.8.1 | cosmetic |
| expert packing (BF16) | parent GPTQ packed fused + per-expert GPTQ shards | backbone packed `experts.gate_up_proj/down_proj`; MTP **unpacked** `experts.{i}.gate_proj/up_proj` | packing delta, not topology |

b70-architect (gate 1): still MoE-A3B bandwidth-bound decode (~0.7 GB/token INT4 experts). Prefill compute-bound on Xe2. **fp8 KV not required** (only 10 full-attn layers; ~1.3 GB fp16 KV at 128K). GDN mixed-split is **not** assumed transferable from Qwen3.8 dense.

## 3. Patch audit vs images (in-container, not memory)

| Patch | pinned `2c427ef` vLLM 0.26.1rc1 / kernels 0.1.12 | champion `f01e24f6` vLLM 0.27.2rc1.dev77+gac7509e2b / kernels 0.1.12.3 | class |
|---|---|---|---|
| `qwen3_5_moe` + MTP module | present | present | start on **champion** (architect) |
| `patch_mtp_nightly.py` env `B70_MTP_BF16_DRAFT` | nightly already nulls quant when `dynamic` has `-:`+`mtp` | **same upstream gate** | `already_in_image` if we write `dynamic`; env overlay not needed |
| `patch_mtp_boundary.py` PARTIAL_FINAL_GROUP | not present | **not present** (`min(expected, actual)` still) | `still_needed` for exact-max-len MTP; not for 16K C1 screen |
| `patch_xpu_grouped_topk_native_v2.py` | `@torch.compile` still unconditional | **still unconditional** | `still_needed` if compiled grouped_topk asserts (Nemotron class). GitHub PR vllm#52159 still **open** |
| grouped-GEMM `at::zeros` | not inspected in .so this session | not inspected | `still_needed` only if graph MoE is nondeterministic |
| `patch_gdn_mixed_split_v5.py` | fused exclusive launch | **still single fused `gdn_attention`**, no mixed split | `not_applicable` for C1 homogeneous; candidate only if Cn mixed dies |
| draft-INT4 S+M1 / W4A8 / Nemotron prenorm | — | — | `do_not_use` |

Serve image: **champion `f01e24f6`**. Sanity parent GPTQ on it before Ornith if time allows.

## 4. Storage + download (COMPLETE)

Live at job start: `/` 9.2G, `/mnt/<models-disk>` 69G, `/mnt/<models-disk>` 50G. Empty GPU 31978 MiB `visible_avail`.

BF16 split (DONE 18:36Z, verified 71,903,871,064 B vs index 71,903,645,408):
- shards 01–11 → `/mnt/<models-disk>/Ornith-1.5-35B-A3B-BF16/`
- shards 12–16 → `/mnt/<models-disk>/Ornith-1.5-35B-A3B-shards/` + symlinks into the model dir

GPTQ output: `/mnt/<models-disk>/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16`
Offload: `/mnt/<models-disk>/gptqmodel_offload` (NOT models2, NOT /tmp). Swap: zram 16G + sda5.

## 5. GPTQ plan (retrying after BLOCK)

gptqmodel 7.3.2 `--no-deps` inside ephemeral champion container (do not `docker commit`; `/` is 96%).
`QuantizeConfig` as models2 35B parent dynamic excludes. Calibration: wikitext-2 128 texts.
Script: `/mnt/<models-disk>/ornith15-quant/quantize.py`.

**b70-architect (gate 2, post-download):** `offload_to_disk=False` would materialize 72G on 30G RAM → host reboot. Must `offload_to_disk=True` with path **`/mnt/<models-disk>/gptqmodel_offload`** (never `/tmp` on 9G root, never models2 next to the 23G save). Expected artifact **~23G**. First serve: 32K fp16 U=0.85 **seqs=8 no-spec**; 128K first-load is abort; 32K+MTP is abort-zone. Do **not** apply grouped_topk (Qwen3.5 uses plain top-8) or GDN mixed-split on C1.

**BLOCK 1 (2026-08-19T18:37Z):** `docker run --device /dev/accel` → `no such file or directory`, exit 127. BF16 verified (`71903871064` bytes). Swapfile 16G on models left 6.8G free — removed so offload can use that space; zram 16G + `/dev/sda5` for RAM. Retry: `--device /dev/dri --group-add render` only.

**GPTQ RUN 2 (historical state at 21:23Z; later completed):** LazyTurtle shell loaded in 3s (no 72G materialization; MemAvailable stayed 16-20G). Pace measured: **~13 min/layer × 40 layers ≈ 8-9 h** (subset-boundary calibration forwards dominate; solve partially on XPU, draw ~72W, CPU 100% single-core from pack). The contemporaneous “~6% / 123 by layer 2” estimate came from progress output and is **not** the canonical denominator. Final `quant_log.csv`: **7,605 / 30,720 projection matrices (24.75%)** used RTN fallback; layer 39 reached **297 / 768 (38.7%)**. Artifact remains PROVISIONAL pending reference/task-quality comparison.

ETA: quant done ~06:30Z Aug 20 → save ~30 min → serve + n=3 + MTP4 → FINAL ~07:30Z.

## 6. Serving plan (after contract verify)

C1, cache off (`--no-enable-prefix-caching`), fp16 KV, U=0.85, 32K, scheduler 8192, **seqs=8** (first load; 64 only if KV line shows ≥3×32K).
No-spec first; MTP4 only at 16K (architect: 32K+MTP fp16 U=0.85 is abort-zone).
No GDN mixed-split, no grouped_topk patch on this C1 screen.
Harness: `/mnt/<models-disk>/ornith15-quant/c1-screen.py` — client SSE TTFT + post-first, unique entropy-first prefixes, `include_usage` token counts, greedy diagnostic sampling.
Model-card sampling (server `--generation-config auto`): temp 1.0 / top_k 20 / top_p 0.95, chat template `enable_thinking` default false.

## Telegram

START and AUDIT sent to Hermes home channel (`telegram` / chat -1003582429844).
