---
title: "Intel vLLM-XPU skills vs B70 measured observations"
status: final
type: research
date: 2026-08-22
---

---
id: b70-research-intel-skills-vs-observations
title: Intel vLLM-XPU skills vs B70 measured observations — 2026-08-22
document_type: research
status: complete
authority: docs/methods/benchmark-protocol.md
scope: Cross-check of the new Intel GPU AI skills (vllm-xpu-run, model-config-recommend) against recorded B70 Ornith/Qwen campaign evidence
---

# Intel vLLM-XPU skills vs B70 measured observations

Sources read: `.agents/skills/vllm-xpu-run/SKILL.md` (+ `references/quantization.md`)
and `.agents/skills/model-config-recommend/SKILL.md`. This note stores what the
skills prescribe, what this host has actually measured, where they agree,
where we deliberately deviate, and what becomes actionable.

## 1. Point-by-point cross-check

### 1.1 Unquantized dtype: skill says `--dtype bfloat16`, never `auto`

Skill: "Battlemage runs bf16 better than fp16; `auto` may pick fp16 from the
checkpoint config. Set for unquantized serving only."

Observed: all B70 quantized-serving campaigns (Ornith GPTQ lineage, AR
head-to-head) run `--dtype float16` **with** `--quantization gptq` — the
skill's bf16 rule targets the unquantized case, so this is not a violation,
but it is an **untested alternative**: no campaign has measured
`--dtype bfloat16` activations for these W4A16 artifacts on the pinned
nightly (`f01e24f6…`, engine `v0.27.2rc1.dev77+gac7509e2b`). Open A/B cell.

### 1.2 Quantized pairing rule: fp8 KV + `TRITON_ATTN` always

Skill pairing table: AWQ/GPTQ/AutoRound → `--kv-cache-dtype fp8` +
`--attention-backend TRITON_ATTN` (FA-XPU does not implement fp8 KV;
`NotImplementedError` at engine init).

Observed: every recorded serve (control Aug-19, AR Aug-22, no-spec and MTP
phases) ran the **opposite pairing**: `--kv-cache-dtype auto` (bf16 KV) +
default **Flash Attention** backend (`xpu.py:198 Using Flash Attention
backend`), serving at 69–70 t/s p512/g128 no-spec and 169,622-token KV
capacity at 32K. This is a deliberate legacy-champion deviation that
predates the skills. Consequences:

- We have **no measurement** of the Intel-recommended pairing on this model.
  fp8 KV would roughly double KV capacity (or halve KV footprint at equal
  context) — relevant to the open 64K/128K context ladder gates.
- The skill's rationale is about what FA-XPU *cannot* do (fp8 KV). Our
  bf16-KV + FA path is legal per their own table (BF16 row: `auto` KV +
  FLASH_ATTN). So the deviation is sanctioned for bf16 KV; it just leaves
  fp8-KV capacity on the table.

**Actionable**: one screen cell — same artifact, `--kv-cache-dtype fp8
--attention-backend TRITON_ATTN` — measure decode t/s, TTFT, KV capacity,
and coherence. Only adopt if quality/capacity/speed all hold.

### 1.3 Recommender preference order ranks AutoRound above GPTQ

Skill: preference order bf16 → fp8 → **auto-round** → awq → **gptq** → inc
→ mxfp4 (pre-quantized formats only suggested when the checkpoint exists).

Observed: independent of the skill, this host completed the AutoRound-vs-GPTQ
head-to-head on 2026-08-22 and recorded ADR-0007 (AutoRound = default
converter) **before** reading the skill. The Intel recommender order is
therefore external corroboration of ADR-0007's direction.

Format nuance from the skill worth keeping: "AutoRound is an algorithm, not a
`--quantization` value — vLLM auto-detects from
`quantization_config.quant_method=auto-round` and routes through the gptq or
awq loader." Our AR artifact was saved with `format="auto_gptq"`, so its
config says `quant_method=gptq` and we serve it with explicit
`--quantization gptq` — identical loader, byte-format parity with the control
(that was the campaign design). If we ever want the auto-detected spelling,
save a copy with AutoRound's native format. Not required.

### 1.4 INT4 budgeted at 0.55 B/param

Skill roofline: int4 paths (AWQ/GPTQ/AutoRound/MXFP4) = 0.55 bytes/param.

Observed: Ornith AR artifact = 24.45 GB for 35.2 B params ≈ **0.695 B/param**
— above the 0.55 budget **by design**: the experts-only scope keeps GDN
linear-attention, self-attn, shared experts, router, vision, embeddings, and
the full MTP layer in BF16 (~4.8 GB). The 0.55 figure models an
all-linear-quantized checkpoint. Fitting math that uses 0.55 for our
mixed-scope artifacts would under-predict VRAM by ~5 GB — use the measured
artifact size for capacity planning.

### 1.5 GPTQ regression in vLLM 0.19.0 (#39474) and the int4_gemm_w4a16 fall-through (#38064)

Skill warnings: GPTQ regressed in vLLM 0.19.0 (use ≤0.18.x or an
AutoRound-AWQ checkpoint); check logs for silent fall-through to
`int4_gemm_w4a16`.

Observed:

- Engine line in **every** retained server log:
  `V1 LLM engine (v0.27.2rc1.dev77+gac7509e2b)` — the pinned nightly's own
  version scheme, digest-locked (`f01e24f6…`). It is not on the upstream
  0.19.0 release train, so the specific 0.19.0 regression does not map
  directly. **Rebase caution**: any future image bump must re-check #39474
  against the new engine line before comparing numbers across images.
- Fall-through check on all four retained Ornith serves (control ×2 phases,
  AR ×2 phases): logs show `int_wna16.py:297 Using 'XPU' WNA16 MoE backend`
  and **no** `int4_gemm_w4a16` anywhere. Both artifacts dispatch the WNA16
  (weight-native-int4, activation-16) MoE path. The silent-fall-through
  failure mode is **not present** in the recorded evidence.
- The skills' "validate output content — quant kernels can produce
  non-language output without raising" matches this repo's discipline
  (coherence probes in every campaign).

### 1.6 Other skill guidance vs practice

- `--enforce-eager` as conservative default: our champion pattern runs
  graphs ON (`VLLM_XPU_ENABLE_XPU_GRAPH=1`, cudagraph capture sizes logged)
  and is validated on this nightly — keep, per skill's own "drop it and
  re-bench" logic, already done historically.
- `--block-size=64`: matches our champion flag exactly.
- `--gpu-memory-utilization 0.85`: matches (GUI workstation reasoning aligns
  with this host).
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` / `--ipc=host` / `/dev/dri/by-path`
  mount / oneCCL gotchas: our single-GPU champion pattern doesn't set these;
  no init hangs recorded on this nightly. Note for any future multi-GPU or
  `intel/vllm:*-xpu` upstream-image work — those images' requirements differ
  from our pinned vllm/vllm-openai-xpu nightly.
- Roofline INT8 367 TOPS / FP16 45.88 TF on B70 and the "W4A8 decode win is
  workload-dependent" caveat are consistent with what we see: W4A16 (not
  W4A8) artifacts at 69–70 t/s decode — the INT4 weight-streaming benefit is
  realized through WNA16, not the W4A8 XMX tier.

## 2. Where this leaves the fleet decisions

| Topic | Skill position | B70 recorded state | Disposition |
|---|---|---|---|
| Converter | auto-round > gptq | ADR-0007: AutoRound default | Aligned (independently reached) |
| Quant artifact format | gptq loader either way | auto_gptq packing, explicit `--quantization gptq` | Aligned |
| KV/attention pairing (quantized) | fp8 KV + TRITON_ATTN | bf16 KV + FLASH_ATTN measured | **Deliberate deviation; A/B open** |
| dtype for quantized | (rule is for unquantized) | fp16 activations | Untested alternative (bf16 A/B open) |
| int4_gemm_w4a16 fall-through | check logs | WNA16 everywhere, no fall-through | Cleared on current image |
| 0.55 B/param int4 budget | planning number | 0.695 B/param measured (mixed scope) | Use measured size for capacity |
| GPTQ 0.19.0 regression | avoid 0.19.0 | digest-pinned nightly, own version line | Re-check on any rebase |

## 3. New open cells (added to campaign open gates)

1. fp8 KV + TRITON_ATTN screen on the MixedCal-v2 reference artifact
   (capacity/decode/TTFT/coherence vs current champion pairing).
2. `--dtype bfloat16` activations A/B on the same artifact.
3. On any image rebase: re-grep for `int4_gemm_w4a16` and re-check #39474
   against the new engine line before cross-image claims.

Evidence: `results/2026/08/ornith15-ar-headtohead-20260822T0535Z/server{1,2}.log`,
`results/2026/08/ornith15-35b-20260819T1630Z/server{1,2}.log`,
`results/2026/08/ornith15-ar-parity-20260822/` (parity campaign, this date).
