---
title: "Runtime patch preparation — vLLM XPU champion stack (PRs #53997/#53989/#53990/#53996, kernels 0.1.15)"
type: plan
date: 2026-09-23
status: assessment
---

# Runtime patches assessment — 2026-09-23

> **NOT-VALIDATED-YET.** Everything below is patch preparation and static assessment only.
> No GPU run, no model load, no container, no import test was performed (the benchmark
> campaign owns the GPUs). `py_compile` is a syntax check, **not** validation: it proves
> nothing about runtime behavior, op availability, Triton-XPU compilation, or performance.
> Every claim of effect is PROVISIONAL until the A/B plan in §5 runs under the benchmark
> protocol. NOT FOR PUBLIC HEADLINE.

Target generation: vLLM `0.27.2rc1.dev77+gac7509e2b`, image
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
`vllm_xpu_kernels==0.1.12.3`, torch `2.13.0+xpu`, triton `3.7.2+xpu`, Python 3.12
(`/opt/venv`) — pins read from `requirements/xpu.txt` at the tree base (`066ae99e`, one
commit past deployed base `ac7509e2`) and the image env. Champion route:
Qwen3.8-27B GPTQ-INT4 G128 (`model_type qwen3_5` → `Qwen3_5ForCausalLM`) + BF16 MTP4,
XPU graphs ON, C1 (`models/qwen/qwen3.8-27b/serving.md`).

Source: overnight research
`RESEARCH-online-novel-levers-20260923.md` (§1 rows 3–4, §2 point 1, §3 ready-to-patch list).

## 1. PR classifications

All four PRs are **open / unmerged** upstream (checked 2026-09-23 via `gh`), so none
qualify as `upstream-merged-use-wheel`.

| PR | Title (state) | Classification | Why (one line) |
|---|---|---|---|
| #53997 | Keep grammar-bitmask copies on the current stream under XPU graphs (open, +196/−4, 2 files) | **apply-to-tree** (applied, adapted) | Fixes the closed #53993 crash family (`torch/xpu/graphs.py:192` synchronize-inside-recording) on a file our generation has; protective today (grammar path only engages with structured-output requests, which the champion campaign does not serve) but it is crash insurance that unblocks any structured-output serving under graphs. |
| #53989 | Enable fused QK-norm+RoPE+gate Triton kernel on XPU (open, +5/−1, 1 file) | **apply-to-tree** (applied, adapted) | Qwen3.8-27B rides `qwen3_5.py`, whose `Qwen3_5Model` extends `Qwen3NextModel` and instantiates `Qwen3NextAttention`, so the gated flag IS on our exercised path; the kernel itself (`fused_qk_norm_rope.py`) is plain `vllm.triton_utils` Triton with no platform guard — enabling is a flag-only change; mRoPE stays excluded (`text_only` only), which matches our text-only 27B. Runtime question: Triton-XPU compile quality on BMG — A/B decides. |
| #53990 | Fall back to torch query when getMemoryInfo reports zero free memory (open, +5/−0, 1 file) | **apply-to-tree** (applied, verbatim) | B70 is exactly the "Arc B-series/Battlemage reports 0 free" class named in the PR; the fallback only fires when the C++ query returns `free == 0 and total > 0`, so it cannot change behavior on hosts where the current query already works. |
| #53996 | Skip absent fused-layer shards in the GPTQ per-layer check (Gemma 4 k_eq_v) (open, +98/−1, 3 files) | **not-applicable-here** (not applied) | Guards checkpoints where a fused qkv shard is absent entirely (Gemma-4 k_eq_v layers); the Qwen3.8-27B GPTQ artifact has all q/k/v shards present, so the patch would be dead code on our route — contexts do match locally, so it can be added later verbatim if we ever serve such checkpoints; skipped to keep the runtime delta minimal. |
| #58082 | Use torch's device-agnostic streams and events in the V2 model runner (open, +46/−42, 6 files, 2026-09-22) | **assess only — not applied** | Same crash family as #53997 but as a broad V2-runner streams/events refactor riding brand-new torch APIs; too much churn to hand-port at dev77 while #53997 already fixes our one concrete symptom — re-check after it merges. |

## 2. Patch application state of the tree

Tree: `~/vllm-src/vllm` @ base `066ae99e` (descendant of deployed base
`ac7509e2`). Rule compliance: **no state-changing git commands used** (status/diff only,
`git stash list` = 0); pre-existing uncommitted campaign work verified intact and
untouched — `vllm/v1/spec_decode/llm_base_proposer.py` (modified) and
`vllm/model_executor/model_loader/draft_int4.py` (untracked).

Files touched by this work (all edits via Edit/Write tools; no `sed -i`, no Bash writes):

| File | Change | py_compile |
|---|---|---|
| `vllm/v1/worker/gpu/structured_outputs.py` | #53997: `use_copy_stream = not current_platform.is_xpu()`, `copy_ctx` nullcontext pattern on both H2D copies, both `wait_stream` joins gated | **PASS** |
| `tests/v1/worker/test_structured_outputs_xpu_stream.py` | #53997 test, adapted (see §2.1); CPU-only fakes, no accelerator needed | **PASS** |
| `vllm/platforms/xpu.py` | #53990: zero-free-memory fallback to `torch.xpu.mem_get_info` | **PASS** |
| `vllm/model_executor/models/qwen3_next.py` | #53989: XPU enablement of `use_fused_qk_norm_rope_gate` (text-only) | **PASS** |

`python3 -m py_compile` over all four files: **PASS** (single invocation, exit 0).
Again: syntax-clean ≠ validated. Nothing has been imported, and the runtime image has
not been rebuilt — see §4.

### 2.1 Adaptations vs upstream hunks (record of every deviation)

- **#53997 source** — upstream hunk context carries `self.num_bonus_tokens` /
  `num_bonus_tokens` ctor arg and a mapping loop aware of `num_draft_tokens_per_req`
  (adaptive-verification offsets); our dev77 file has neither. Adapted by applying only
  the stream-selection delta onto the local file: import `contextlib` +
  `current_platform`, the `use_copy_stream` flag, `copy_ctx` around both copy blocks,
  and gating of both `wait_stream` calls. No local logic was rewritten.
- **#53997 test** — dropped the `num_bonus_tokens=0` ctor kwarg (absent in this
  generation); docstring records the adaptation. Everything else verbatim.
- **#53989** — upstream hunk context has trailing `and supports_dtype` /
  `and (text_only or supports_mrope)` conditions that do not exist at dev77; the same
  `is_cuda() or (is_xpu() and text_only)` disjunction was folded into the local
  4-condition form. Semantics identical for our text-only model.
- **#53990** — verbatim (exact context match at `vllm/platforms/xpu.py:94-97`).

Applied-diff evidence (combined `git diff`, one file per PR) +
upstream diffs, with sha256:

```
results/2026/09/b70-improvement-probes-20260920/runtime-patches-20260923/
  pr-53997.diff          ca99e6a6c5a9d48336eab941ea0cf324414e4d69efcb6a3df0da4b16953d8390
  pr-53989.diff          d808925936e7da4434d4da44888e93fc0692354feda6ce75656b8f5b6b2c56af
  pr-53990.diff          fc19cc9c3d8fe61e3dbe3a10aa6d119354e6d7a177cb0aedbd0c0d13f7420487
  pr-53997-upstream.diff 3422d3f3d923b69c857c134122468738f6bcfa4a170aa66b4b9ec777a4991834
  pr-53989-upstream.diff b22f505a36ddbfdc06409842d4d481f6d9e715bec6693cbc6fbfadfd40735f6d
  pr-53990-upstream.diff 6dad6fe17b65a393fe1999478bdfc8c8c9a0bec0eb441c8c81b16ed6b909ab5b
  pr-53996-upstream.diff 3e8c6d705694efc0d29c821cc50bc0e8c78002e9049e4921911a11f1d5ac1d76
```

(`pr-<n>.diff` = as-applied to our tree, evidence only — not re-apply recipes;
`pr-53996-upstream.diff` kept although not applied.)

## 3. vllm-xpu-kernels wheel provenance

- GitHub release **v0.1.15** (published 2026-09-22T11:14Z) ships **no release assets**;
  wheels live on PyPI, where the 0.1.15 line exists as patch builds
  **0.1.15.1 / 0.1.15.3 / 0.1.15.4** (no separate git tags for the .x patches; tag list
  jumps `v0.1.14` → `v0.1.15`).
- **ABI match:** the image generation pins `torch==2.13.0` from
  `download.pytorch.org/whl/xpu` (= `2.13.0+xpu`), and `pyproject.toml` at tag v0.1.15
  pins exactly `torch == 2.13.0+xpu` — same torch C++ ABI. Image Python is 3.12
  (`/opt/venv`, Ubuntu 24.04 base).
- **Fetched wheel** (via `pip download`, host pip; no curl/urllib; mount had 178 GB free,
  need ≈0.3 GB at 2× headroom for a 141 MB wheel):

```
/mnt/<sata-disk>/wheels/vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl
size    : 141.4 MB (135 MiB)
sha256  : 4ec262f7afdd07c62defc8286dc9f357569627b996c09b410f7972b653696fe7
ABI     : cp38-abi3 (stable ABI, satisfies the image's CPython 3.12), manylinux_2_28 x86_64
built   : 2026-09-22 (dist-info timestamps), v0.1.15 release line
requires: python >=3.9,<3.15 ; **no Requires-Dist on torch** — install cannot churn the pinned torch
```

### 3.1 Derived-image recipe (to build LATER — not built now; GPU-adjacent step)

```dockerfile
# FROM the champion image by digest, then:
COPY wheels/vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl /tmp/
RUN /opt/venv/bin/pip install --no-deps \
      /tmp/vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl \
 && rm /tmp/vllm_xpu_kernels-*.whl
```

`--no-deps` is deliberate: the wheel declares no deps, and this makes it impossible for
pip to touch the image's pinned `torch==2.13.0+xpu` / `triton==3.7.2+xpu`.

Post-install verification inside the derived container (before any serving):

```bash
/opt/venv/bin/python -c "import vllm_xpu_kernels as k; print(k.__version__)"        # expect: 0.1.15.4
/opt/venv/bin/pip show vllm-xpu-kernels | grep Version                              # expect: 0.1.15.4
/opt/venv/bin/python -c "import torch; print(torch.__version__)"                    # must stay: 2.13.0+xpu
/opt/venv/bin/python -c "import torch; print(torch.ops._xpu_C.compact_sampling_mask)"   # new v0.1.15 op must resolve
/opt/venv/bin/python -c "import torch; print(torch.ops._C.layer_norm)"                  # fused LayerNorm op must resolve
```

Expected kernel version string: **`0.1.15.4`** (the `torch.ops._xpu_C.*` /
`torch.ops._C.*` registrations come from that wheel). Any `NotImplementedError` /
symbol-missing at op resolution = ABI mismatch → stop, do not serve.

## 4. What this is NOT

- Not a benchmark result. No number in this file was measured here.
- Not an import/runtime test: the edits were never imported (importing vllm from the
  tree would require the container runtime path, which is a GPU-adjacent step the
  campaign currently forbids).
- #53996 was deliberately **not** applied (§1) — its absence is a decision, not an
  oversight.
- The three applied patches ride in the source tree only; they reach the runtime only
  through the derived-image build (mount the patched `vllm/` package or install from the
  tree), which is the same gating step as the wheel bump.

## 5. A/B verification plan (when the campaign frees the GPUs)

Baseline arm: current champion stack unchanged (image `f01e24f6…`, kernels `0.1.12.3`,
no source patches). Treatment arm: derived image = champion + kernels `0.1.15.4` +
patched `vllm/` tree (#53997 + #53990 + #53989). Everything else identical (GPTQ
checkpoint, BF16 MTP4 draft overlay, `VLLM_XPU_ENABLE_XPU_GRAPH=1`, recommended
non-thinking sampling, 230 W cap for the campaign, restore 150 W after).

Because the sampler itself changes in 0.1.15 (improved top-k/top-p, per-row RNG), both
arms **must** be measured in the same session, ABBA interleaved (A=base, B=patched,
B, A), n=5 finals per protocol §8; single-client gate enforced (one client at a time,
prefix-cache + spec-decode counter snapshots bracketing every run).

Cells (matched to the champion records in `GAINS-VS-CURRENT-DATA.md` G1 + Lane-1 card):

1. **p512/g128 decode, n=3** (screen) — then fresh **n=5 finalist** on the winner.
2. **p8192/g128 decode, calibrated cold input (8,189-token prompt), n=5** — the
   champion's headline cell.
3. **p2048/g1 and p8192/g1 cold-input prefill** — guards #53989 (the fused kernel also
   runs at prefill shapes; per-token grid must not regress prefill).
4. One representative real workload (same prompt class as the campaign's).

Counters that must stay unchanged / be explained:

- **Prefix cache:** zero cache-hit delta on cold-input cells (protocol §8 HTTP cache
  guards: 64-char unique entropy, snapshot queries/hits before+after).
- **Spec-decode counters:** MTP4 acceptance-rate distribution within run-to-run
  variance of the baseline arm; accepted/drafted token counts snapshotted per cell.
  A material acceptance shift after the sampler bump means the RNG/sampler change
  altered the sampled-stream class → re-label per protocol §8 (sampling policy),
  do not mix arms.
- **Engine identity:** record observed in-container vLLM / vllm-xpu-kernels /
  torch versions from server logs for both arms (§7 discipline).

Pass criteria (declared before running):

- **Functional gate:** server boots with XPU graphs ON + MTP4 on the patched stack; all
  `torch.ops._xpu_C` / `torch.ops._C` ops resolve; a structured-output smoke request
  (the #53997 path) completes without the `torch/xpu/graphs.py` recording crash.
- **Performance gate:** decode cells not-worse than baseline at the protocol's ABBA
  decision rule; a win is only claimed from fresh n=5 finals of the full card.
- **Quality gate:** greedy-synthetic spot parity vs baseline arm (labeled diagnostic);
  acceptance-rate class unchanged or re-labeled explicitly.
- **Stop rules:** any graph-capture crash, op-resolution failure, or acceptance collapse
  → abort the arm, keep the baseline as champion, record the failure in
  `LEVERS.md`/`OUTCOMES.md`.

## 6. Open items

1. Build the derived image and run the §3.1 op-resolution checks (blocked on GPU-free
   window; image build itself is CPU work but belongs with the serving step).
2. Decide whether Triton 3.7.2+xpu in the image compiles `fused_qk_rmsnorm_rope_gate`
  acceptably on BMG — first boot of the treatment arm answers this; if the kernel
  fails to compile, #53989 reverts to the eager path by flipping the flag back (single
  `git diff` line, evidence in §2 files).
3. Track #53997/#53989/#53990/#53996/#58082 merge state before any rebuild — a merged
   PR converts to `upstream-merged-use-wheel` at the next image bump.
4. The `StructuredOutputsWorker` ctor in this generation lacks `num_bonus_tokens`; when
   rebasing onto newer upstream, the adapted test will need that kwarg restored.

## 5. Derived-image build + verification (2026-09-23, orchestrator)

**Built and VERIFIED:** `b70-vllm-kernels-0.1.15.4:f01e24f6` (Dockerfile `scripts/tmp/Dockerfile.kernels-0154`,
build context = the wheels dir on the big mount; build log `results/2026/09/b70-improvement-probes-20260920/kernels-0154-build.log`).

Control test (single process, `import vllm` first — see the methodology note below):

| check | champion (0.1.12.3) | derived (0.1.15.4) |
|---|---|---|
| `pip show` version | 0.1.12.3 | **0.1.15.4** |
| `torch.__version__` | 2.13.0+xpu | **2.13.0+xpu** (pinned untouched) |
| `_xpu_C.int4_gemm_w4a16` | RESOLVED | RESOLVED |
| `_xpu_C.compact_sampling_mask` (new sampler op) | MISSING | **RESOLVED** |
| `_C.layer_norm` (fused LayerNorm) | MISSING | **RESOLVED** |

**Methodology correction to §3.1's verification steps:** the `torch.ops._xpu_C.*` ops register when
`vllm` (→ `vllm._xpu_ops`) is imported, NOT when `vllm_xpu_kernels` alone is imported. A first check
that imported only the wheel reported every op MISSING — including `int4_gemm_w4a16`, which the
draft-INT4 patches demonstrably exercised tonight. Check ops **after `import vllm`**, and always run
the champion image as the control: a missing op in both images is a methodology error, not an ABI
mismatch. Do not serve on an image whose control fails this shape.

### 7. Later #53989 screen outcome (2026-09-23)

The treatment screen was later run under `results/2026/09/qkrope53989-abba-20260923/screen/`.
Use `comparison-authoritative.json`, not the original `SUMMARY.txt` (its built-in parser counted
warmups). Corrected n=6/arm medians were p512 82.58→73.52 (-10.97%) and p8192
70.54→78.78 (+11.68%), with overlapping ranges at both cells; cold-input p2048/g1 was
-0.73%. Acceptance changed -0.0663/+0.102 beyond the predeclared ±0.03 guard. The greedy
canaries passed, but the campaign ran at 69–78 °C and its cooldown helper accepted stable
above-threshold temperatures. The fused kernel dispatch itself was not traced. Verdict: no
promotion and no performance claim from this screen.

Upstream archaeology later confirmed the equivalent XPU dispatch merged as vLLM #56096 on
2026-09-16. No duplicate #53989 issue/PR should be filed. The local screen is only diagnostic
for the older f01e24f6 image; any further performance test needs a specific reason to reopen,
a dispatch trace, and a valid <=55 °C cooldown.
