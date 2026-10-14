---
title: "Research plan — Muse-Glimmer-30B on vLLM XPU (B70)"
type: research
date: 2026-08-10
status: final
---

# Research plan — Muse-Glimmer-30B on vLLM XPU (B70)

**Status (2026-08-12):** Gates 1-3 have now passed as experimental correctness
smokes. Derived image: `vllm-openai-xpu:muse-51655-99a10304`, base digest
`2c427ef…`, PR head `99a10304dce8945119bd0b1a072297803c52a749`.
The overlay must be activated with `PYTHONPATH=/workspace/vllm`.

- Architecture/config: CPU compile/import passed; both Muse registry aliases and
  the real compressed-tensors config resolve.
- Text-only XPU: a narrow `has_vision=false` config plus text-only checkpoint
  vision-prefix filter loaded all five shards and selected the native
  compressed-tensors `XPUwNa16LinearKernel`. Raw `/v1/completions` deterministically
  produced correct continuations (`The capital of France is` → `Paris`; `2 + 2 =`
  → `4`). Chat output is semantically coherent but currently exposes the model's
  internal reasoning/channel markers in `content`, so raw completions are the clean
  text correctness control until parser/template integration is fixed.
- Full vision XPU: the unmodified complete checkpoint loaded successfully. A
  generated image containing a red square, blue circle, and the text `B70` was
  described correctly. The chat response again included reasoning/channel text
  before the correct final description.

Evidence root:
`results/2026/08/muse-compressed-tensors-vllm-smoke-20260812/`, including
`raw-completions-coherence.json`, `server-vision-full.log`, `vision-test.png`, and
`vision-smoke-response.json`.

A constrained C1 screen has also completed at a 150 W configured cap, text-only,
16,384-token context/scheduler budget, prefix cache disabled, one same-shape
warmup discarded and n=3 exact g128 requests per cell:

| Cell | Client post-first median | Range | TTFT median | E2E median |
|---|---:|---:|---:|---:|
| p512/g128 | 21.34 tok/s | 20.59–25.97 | 0.339 s | 6.289 s |
| p8192/g128 | 24.60 tok/s | 24.59–24.60 | 7.159 s | 12.321 s |

These are request-side client post-first rates from an internal text-only n=3 C1
screen, not engine-native decode and not a multimodal or production-serving
result. They are **PROVISIONAL — NOT FOR PUBLIC HEADLINE**. The local PR-head
overlay is not a pullable production image, and the screen does not establish
task quality, reference parity, deployment readiness, or a recommended
configuration. Post-load free VRAM was 5,929 MiB.

Synchronized `energy1_input` accounting over the three measured requests and
intervening request/metrics overhead yielded 134.840 W average card draw for the
19.500-second p512 cell window and 147.419 W for the 38.000-second p8192 cell
window, at a 150 W configured cap. The verified `pkg` sensor recorded
start/end/max temperatures of 55/64/66°C for p512 and 55/67/68°C for p8192. Raw
0.5-second interval values reached 193.536 W and 206.701 W respectively; because
these exceed the configured cap, they are retained as energy-counter interval
anomalies and are not credible peak-draw claims.

Evidence: `results/2026/08/muse-glimmer-int4-screen-20260812T113031Z-372721/`.

## Objective

Run Meta Muse-Glimmer-30B on the B70 via **vLLM XPU** with the
compressed-tensors INT4 checkpoint, targeting the dense-27B-class vLLM decode
(~50-70 t/s vs the llama.cpp GGUF path's 26.8 t/s DFlash).

## Checkpoint (ready)

- Repo: `cyankiwi/Muse-Glimmer-30B-AWQ-INT4` (compressed-tensors INT4 w4a16,
  24.1 GB, 5 shards, vision encoder included, 52 text layers, model_type
  `muse_glimmer`, arch `MuseGlimmerForConditionalGeneration`)
- Local: `/mnt/<models-disk>/muse-glimmer-awq-int4/` (models2 had 49 GB free — no
  cleanup needed)
- Note: the name says AWQ but the config declares `quant_method:
  compressed-tensors` — treat as compressed-tensors INT4 until proven otherwise.

## The three gates

1. **Gate 1 — vLLM arch support (EXPERIMENTALLY PASSED).** PR
   [vllm-project/vllm#51655](https://github.com/vllm-project/vllm/pull/51655)
   is still OPEN (reviews + CI only). The only Muse-capable image is
   `vllm/vllm-openai:muse-glimmer` — built from the PR **branch** (CUDA, not
   XPU). The pinned B70 nightly (`vllm/vllm-openai-xpu@sha256:2c427ef…`,
   vLLM 0.26.1rc1.dev457) predates the PR and cannot parse the checkpoint.
   **Trigger:** PR merge → vLLM 0.27.0+ release → XPU nightly rebuild.
2. **Gate 2 — compressed-tensors INT4 on XPU (PASSED FOR THIS ARTIFACT).** The proven B70
   vLLM path is GPTQ-INT4 via `XPUwNa16LinearKernel` (dense 27B, Run 29/31).
   Compressed-tensors INT4 (w4a16) is the same kernel family but a different
   packing/scales dispatch — may work or hit a gap like the FP8 `KeyError:
   PlatformEnum.XPU` (DENSE-FP8-GAP.md). Needs a real load test.
3. **Gate 3 — vision on XPU (SMOKE PASSED).** The checkpoint includes the vision
   tower (`has_vision`); `MMEncoderAttention` on XPU is untested. Plan: load
   text-only first; if the vision load fails, investigate stripping the tower
   or an env-gated skip (the vLLM model's `_muse_glimmer_has_vision` path).

## Test procedure (when gates 1-2 open)

1. Pull the post-merge XPU nightly (`vllm/vllm-openai-xpu:nightly` or the
   pinned digest once re-pinned). Verify in-container: vLLM version ≥ 0.27.0,
   `MuseGlimmerForCausalLM` present in the registry, `vllm-xpu-kernels`
   version, and the Muse PR's patches (`patch_mtp_nightly.py` etc.) do NOT
   apply to this arch (revalidate — §16.1 intake: pin digest + versions).
2. §8.1 gates (empty GPU, 150 W restore, VRAM check) before every load.
3. Serve text-only first:
   `vllm serve /mnt/<models-disk>/muse-glimmer-awq-int4 --quantization compressed-tensors
   --max-model-len 131072 --kv-cache-dtype fp8 --gpu-memory-utilization 0.88
   --max-num-seqs 64 --max-num-batched-tokens 8192` (+ XPU env §9.7).
   Note: `--language-model-only` is Qwen-specific — do not assume it exists
   for Muse.
4. Smoke: warmup (discard), then coherence check (reasoning model — expect a
   `reasoning_content` channel; use max_tokens ≥ 512).
5. If text loads: claims-standard lane (p2048/g1 + p8192/g1 cold input rate,
   p512/g128 + p8192/g128 decode, n=5, unique entropy prefixes, zero
   cache-hit delta, C1 vs Cn separate). Then the 128K exact cell.
6. Compare vs llama.cpp DFlash (26.8/22.9/21.1 t/s at p512/p8192/p32768,
   128K, 230 W — Run 35). vLLM INT4 dense-27B-class expectation: ~50-70 t/s.
7. Vision second (image → description via the OpenAI-compatible endpoint).
8. Evidence: new timestamped run root under `results/`, manifest per
   benchmark-protocol §16, append benchmark-history Run 36+.

## Alternatives

- **Thin PR-head source overlay on the pinned XPU image (preferred pre-merge
  architecture test).** PR #51655 head `99a10304dce8945119bd0b1a072297803c52a749`
  changes Python/Jinja/tests only—no C++/SYCL/CMake/kernel files. Build a small
  derived image from pinned digest `2c427ef…` that copies only the PR's 15 runtime
  Python files into `/workspace/vllm/vllm/`. Record PR/base SHAs and file hashes,
  then run CPU-only compile/import/config/registry gates. The PR base is newer
  than the pinned image, so API compatibility is unproven and broad ad-hoc
  backporting is not allowed. This avoids an unsafe full XPU rebuild while `/`
  has limited free space.
- **Full PR-head XPU wheel/image build (last resort).** Use only if the thin
  overlay proves it needs newer native interfaces. It is multi-hour and
  disk/network heavy; free or relocate Docker storage first. §15.9 has the
  kernel-wheel build recipe.
- **Fallback:** llama.cpp GGUF remains the production path (26.8 t/s DFlash,
  vision + reasoning verified) — nothing regresses while waiting.

## Watch

- PR #51655 merge state (checked 2026-08-10: open). Re-check daily or set a
  scheduled check. The merge is the sole trigger for Gate 1.
- HF for GPTQ-INT4 variants (none exist as of 2026-08-10 — a GPTQ would skip
  Gate 2 uncertainty since GPTQ-INT4 is the proven XPU path).

## Evidence trail

- llama.cpp baseline: `archive/legacy-docs-20260915/benchmark-history.md` Run 33/34/35
- Raw sweep: `results/2026/08/muse-dflash-sweep-manual-20260810T164735/`
- Adoption runbook: `.agents/skills/muse-glimmer-adoption/SKILL.md`
- Cookbook Muse page: `intel-arc-pro-b70-inference-cookbook/docs/muse-glimmer/MUSE-GLIMMER-B70.md`
- LocalMaxxing: engine `cmsnly2su00goo001wn6c98ly`, run `cmsnly2sy00gqo001ui1k5l67`

## Text-only load UNBLOCKED (2026-08-12)
- Built the missing XPU paged-decode kernel tuple `16,128,64,false,true,false` (q16_h128_p64_ftf): full vllm-xpu-kernels 0.1.12.3 rebuild (229/229 actions from de8eae16) + wheel + derived image `vllm-muse-text-pr51655:99a10304-xpu-1da0a954-pdd0123`.
- Muse text-only (language-model-only, compressed-tensors W4A16 G32) now LOADS and SERVES on the latest base: p512/g128 11.35 t/s and p8192/g128 11.21 t/s client post-first (n=3, C1, eager, cache off, 150 W); TTFT 0.30 s / 7.83 s (p512 ~1700, p8192 ~1046 input tok/s). Evidence: results/2026/08/muse-glimmer-int4-screen-20260812T192845Z-835270/.
- Chat-mode output echoes prompt content (known Muse-PR channel-presentation quirk; raw completions cleaner). Decode is eager-limited (same launch-overhead class as Nemotron) — graphs/compile and DFlash are the next levers.
