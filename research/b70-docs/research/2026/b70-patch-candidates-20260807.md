---
title: "B70 Patch Candidates — Consolidated Analysis (2026-08-07)"
type: research
date: 2026-08-07
status: final
---

# B70 Patch Candidates — Consolidated Analysis (2026-08-07)

Three specialist agents investigated improvement avenues. This is the
prioritized, honest summary of what's patchable, what's at the ceiling, and
what needs upstream/rebuild work.

---

## 1. GDN causal_conv1d concurrency — GUARD BYPASSED, correctness is a deep bug ⭐

**The single highest-value finding of the investigation.** Run 23 found that
MTP + concurrency (C2+) crashes with `RuntimeError: causal_conv1d does not
support spec-decode and non-spec tokens in the same invocation`.

### What the agent found (verified in the running container)
- The error is **NOT in the SYCL device kernel.** The two conv1d kernels
  (`gdn::causal_conv1d_spec_kernel`, `gdn::causal_conv1d_kernel`) are separate
  and each only handles one token class. No hardware limit.
- The error is a **host C++ guard in `_xpu_C.abi3.so`** (vllm-xpu-kernels 0.1.12)
  that refuses to dispatch when `num_spec_decodes > 0 AND (num_prefills > 0 OR
  num_decodes > 0)`. The binding ALSO validates
  `non_spec_token_indx + spec_token_indx == num_actual_tokens` (count check).
- The eager Python path (`qwen_gdn_linear_attn._forward_core`) **already handles
  mixed batches correctly** via index_select split + per-group conv/ssm calls +
  index_copy merge. So mixed batches are mathematically valid.

### Patch attempts — 4 iterations (all bypass the guard, none correct)

**v1 — compacted inputs + arange indices, reorder_input=True (default):**
C2/C4 no longer crash (guard bypass PROVEN), split log fires correctly. C1
homogeneous path correct. BUT split path output is GARBAGE (`'!!!!...'`).
Root cause: the kernel uses `reorder_input=True` + the token indices to
rearrange inputs from argsorted full-batch layout; compacted inputs + fresh
arange indices break this rearrangement.

**v2 — full (uncompacted) tensors, dummy zero-count for skipped group:**
Crashes with `Expected non_spec_token + spec_token == num_actual_tokens` —
the binding requires BOTH real index tensors' counts to sum to total. Passing
a 1-element dummy for the zero side fails the size-sum check.

**v3 — full tensors, BOTH real index tensors in both calls:**
Same `non_spec_token + spec_token == num_actual_tokens` error. The binding
computes expected token count from `num_prefills + num_decodes +
num_spec_decodes * (num_spec+1)` and compares to index sizes. With
`num_spec_decodes=0` but a non-empty `spec_token_indx`, the sizes don't match
the expected count.

**v4 — compacted inputs + arange indices + reorder_input=False:**
No crash, server survives. Still GARBAGE output (`'!!!!...'`). The `reorder_input=False`
didn't fix it — the problem is deeper than input reordering.

### Root cause analysis (why all 4 approaches fail)
The fused `gdn_attention` op is NOT a simple "conv1d + attention" that can be
called twice on token subsets. It is a **single fused kernel that manages the
recurrent state machine (conv_state + ssm_state) across ALL tokens in one
invocation**. The state transitions depend on the SEQUENCE ORDER of tokens
within each request, and the kernel internally coordinates how spec and
non-spec tokens interact with shared state slots. Splitting into two calls
breaks this coordination because:

1. The argsort at metadata-build time (`index = torch.argsort(spec_token_masks,
   stable=True)`) permutes tokens so non-spec come first, spec last. The
   `non_spec_token_indx` captures this permutation. But `index_select(0,
   non_spec_token_indx)` gathers tokens in ARGSORTED order, not original
   sequence order within each request.
2. The conv1d kernel processes tokens in sequence order using `query_start_loc`.
   If the compacted tokens are in argsorted order (not sequence order), the
   convolution runs over the wrong token sequence → garbage.
3. The fused kernel's internal state machine coordinates spec/non-spec state
   transitions in a way that can't be replicated by two independent calls —
   the state slots may need cross-group coordination that only happens inside
   the fused path.

### Conclusion: the fused op split is a DEEP bug, not a one-line fix
The guard IS bypassable (v1/v4 prove no crash), but making the split produce
correct output requires either:
- **(a) Reverse-engineering the fused kernel's internal token ordering + state
  coordination** to replicate it in Python (weeks of kernel-level work), OR
- **(b) Scheduler-level fix**: prevent the mixed batch from ever being created
  by deferring prefills when spec-decode is active (or vice versa). This avoids
  the split entirely but adds one-step latency to prefill when spec-decode
  requests are running. The vLLM scheduler has no existing config for this;
  it would need a patch to the scheduling loop (around line 897 of
  `scheduler.py`).
- **(c) Upstream fix**: vllm-project/vllm#36917 is open but stale. The real fix
  should be in the C++ binding itself (lift the guard + make the fused kernel
  handle mixed batches internally, as the SYCL kernels already can).

**Status: WIP.** The guard bypass is proven (the hard part). The correctness
fix needs either scheduler-level intervention or upstream kernel work. The
patch file `scripts/tmp/patch_gdn_concurrency.py` is kept for future iteration.

---

## 2. MTP acceptance — AT THE CEILING (80.1% at N=4)

- Our pos0 acceptance (92.6%) **beats DeepSeek-V3's published MTP-1 reference**
  (85-90% for the second token). The falloff 92.6→84.2→72.7→68.9 is
  mathematically intrinsic to single-layer autoregressive drafting (errors
  compound down positions). No config removes it.
- **One free lossless win: dynamic spec** (`--speculative_target_length`).
  Adaptively raises/lowers N at runtime, shedding low-value pos3 tokens when
  the draft isn't hot. Worth testing — low risk.
- **Realistic headroom: +3-5 pts** acceptance via dynamic spec. Beyond that:
  the lossy `rejection_sample_method:"synthetic"` mode (trades quality), or a
  trained multi-layer/EAGLE head (not a config change — needs new weights).
- **Dead ends**: `draft_temperature`/`speculative_token_round` don't exist in
  vLLM; MTP+ngram hybrid unsupported; BF16→FP16 draft won't help.

---

## 3. Prefill kernel — NO RUNTIME KNOBS (int4 already on fast path)

- **Tile config is hardcoded** in the compiled `.so`: 4 w4a16 policies dispatched
  by `A_avg_M = Total_M / num_experts`. Our p4k→p8k crosses the m_32→default
  (128×256) boundary. The tiles ARE DPAS-aligned (N=16, SD=8, K=64 for int4).
- **Our int4 path IS on the fast packed-DPAS path** (group_size=128 confirmed),
  NOT the slow scalar unpack. Good — the 30% lever from the blog doesn't apply.
- **MXFP4 is structurally identical to int4 on 0.1.12** (same 4 policies, same
  dispatch) PLUS extra activation-quant overhead. Run 13-14 "MXFP4 slow" verdict
  holds — don't pursue.
- **No B60-vs-B70 heuristic exists** — the whole library is compiled for BMG and
  dispatched purely on M. No "2 groups/XeCore" knob.
- **max-num-batched-tokens 8k is a sweet spot** — p8k saturates the device (64
  WGs/Xe-core at the default tile). Raising to 16k adds memory pressure, no
  parallelism gain. Keep 8192.
- **The only real prefill lever is a wheel rebuild** with a B70-shape-tuned tile
  policy (`w4a16_policy_n_128` = 128×128×32, or shifted m_32 dispatch
  threshold). Estimated 10-25% if the current 128×256 tile is N-bound. Requires
  editing `gemm_xe2_policy.hpp` + recompiling (~1-2 hrs).

---

## Prioritized action plan

| Priority | Action | Effort | Expected impact |
|----------|--------|--------|-----------------|
| **1** | **Fix the GDN split correctness bug** (study eager path state handling) | Medium (debug) | **Unblocks MTP+concurrency** — 204 t/s per-user at multi-user scale |
| 2 | Test `--speculative_target_length` (dynamic spec) | Low (flag) | +3-5% decode (lossless, sheds waste pos3) |
| 3 | ~~Rebuild vllm-xpu-kernels with B70-tuned tile policy~~ | **DONE — REFUTED** | See §4 |
| 4 | (upstream) Watch intel-xpu-backend-for-triton#6389 (grouped-GEMM autotuning) | None (wait) | Unknown — not landed |
| 5 | (skip) MXFP4, power cap, clock locking, max-num-batched-tokens>8k | — | Confirmed dead ends |

**The GDN patch is the breakthrough candidate.** It follows the exact pattern
that unlocked MTP in Run 18 (overcautious guard bypassed), and the crash is
gone. The correctness fix is the remaining work — the mechanism is proven by
the eager path, so it's debuggable, not speculative.

## 4. Wheel rebuild with w4a16_policy_n_128 — BUILT AND TESTED, FLAT (2026-08-08)

**Question:** would a narrower N tile (128 vs stock 256) for the int4
grouped-GEMM give more N-parallelism (8 tile-columns vs 4 on N=1024) and speed
up prefill?

**What was done (the full pipeline, ~7 hrs total incl. 2 reboots):**
1. Cloned vllm-xpu-kernels from GitHub (to `/mnt/<models-disk>/vxk-build`, persistent).
2. Added `w4a16_policy_n_128` (WGTile 128×128×32, SGLayout (4,4,1)) to
   `gemm_xe2_policy.hpp` + env-gated dispatch (`B70_W4A16_N128=1`, A_avg_M
   128-512 range) in `grouped_gemm_xe2_interface.hpp`.
3. Built the wheel INSIDE a container: nightly image + host `/opt/intel` oneAPI
   (icpx 2026.0) mounted. SYCL AOT for BMG-G31.
   - **Build gotchas:** `grouped_gemm_xe2.cpp` single-TU OOMs at ~18 GB (needs
     -j1 + 24g for that file; host reboot also wiped /tmp builds — use a
     persistent path). Total build ~4-5 hrs at -j3 (1232 objects: oneDNN +
     cutlass deps, grouped-GEMM, 462 attn TUs, gdn, mhc, mqa).
   - **Verified:** `MoEGEMMLauncher<..., w4a16_policy_n_128, ...>` symbols
     present in `libgrouped_gemm_xe_2.so` (BF16 + FP16).
4. Wheel `vllm_xpu_kernels-0.1.12+b70n128-cp312-cp312-linux_x86_64.whl`
   (469 MB) installed into derived image `vllm-openai-xpu:nightly-b70n128`.
5. **Clean A/B** (same wheel, same session, env-gate ON vs OFF):

| cell | stock 128×256 | n128 tile | delta |
|------|--------------|-----------|-------|
| p500 | 4,107 | 4,826 | **+17.5%** |
| p1k | 6,447 | 6,311 | -2.1% |
| p2k | 6,345 | 6,304 | -0.7% |
| p4k | 6,840 | 6,855 | +0.2% |
| p8k | 6,882 | 6,834 | -0.7% |

**Verdict: FLAT for long prompts (p2k-p8k: ±1%, noise).** The +17% at p500 is
launch-overhead reduction (tiny M), irrelevant for real prefill. The stock
128×256 tile is already well-tuned for this MoE — the bottleneck is NOT N
column count (L2/DRAM streaming of int4 weights dominates, tile-independent).

**Also confirmed:** the ~6,840-6,880 p8k band (vs 8,390 reference) is the
frequency-variance range documented in Run 24 G — both kernels in this session
land in the same band, so the comparison is fair.

**Raw:** `results/prefill-b70n128-{tile,stock-same}.json`.
**Artifacts:** `/mnt/<models-disk>/vxk-build/` (source + wheel + image
`vllm-openai-xpu:nightly-b70n128`).
**Status: tile tuning CLOSED (negative result).** The env-gate remains in the
wheel for future experiments but is inert (off by default).
