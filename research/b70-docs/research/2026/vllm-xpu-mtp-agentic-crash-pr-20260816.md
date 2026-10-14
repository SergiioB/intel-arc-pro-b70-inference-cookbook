---
title: "XPU GDN mixed spec/non-spec crash — PR-ready writeup (2026-08-16)"
type: research
date: 2026-08-16
status: final
---

# XPU GDN mixed spec/non-spec crash — PR-ready writeup (2026-08-16)

Do **not** open a new GitHub issue. This is already
[`vllm-project/vllm-xpu-kernels#510`](https://github.com/vllm-project/vllm-xpu-kernels/issues/510)
(open, 2026-08-07). Next action is a **comment on #510 + a PR**. It is **not**
on the 2026-08-13 submit list
(`~/B70-DOCS/research/vllm-xpu-pr-readiness-20260813.md`).

## 1. Title + summary

**Issue title (for the #510 comment / PR, not a duplicate):**
`[Bug] XPU GDN causal_conv1d host check still rejects mixed spec + non-spec batches (MTP + tool-call/prefill kills EngineCore)`

**Repo:** primary code change in **`vllm-project/vllm`**
(`vllm/_xpu_ops.py`). Optional follow-up in **`vllm-project/vllm-xpu-kernels`**
(`csrc/xpu/gdn_attn/gdn_attn_interface.cpp`) only if the fused C++ op should
accept mixed batches natively.

**Why vLLM first:** vLLM's metadata builder already *emits* mixed
`num_prefills>0 && num_spec_decodes>0` on purpose. CUDA already splits that
batch in Python (`QwenGDNLinearAttention._forward_core`). XPU does not: it
sends the mixed metadata into one fused `torch.ops._xpu_C.gdn_attention`. The
host `TORCH_CHECK` in kernels is the tripwire, but the production call site
and the CUDA precedent live in vLLM. #510 is the right tracker; a kernels-only
issue without a vLLM PR will not unblock serving.

**One paragraph:** On Intel Arc Pro B70, hybrid-GDN Qwen3.5/3.6/3.8 + MTP
dies as soon as one in-flight spec-decode shares a step with any non-spec
prefill (new chat turn, tool-result, chunked long document). The SYCL kernels
are not missing; they are two exclusive launchers. The fused XPU op refuses
to enter both. Pure all-spec decode and no-spec serving are fine. Production
workaround: drop `--speculative-config` (C1 p512/g128 **32.9** tok/s vs MTP4
**83.7** on the same 2026-08-15 image).

## 2. Environment

| Item | Value |
|---|---|
| Device | Intel Arc Pro B70 (BMG-31 / G31), xe driver, Ubuntu 26.04 |
| Image (current nightly) | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` |
| vLLM | `0.27.2rc1.dev77+gac7509e2b` |
| Kernels | `vllm-xpu-kernels 0.1.12.3` — `_xpu_C.abi3.so` still contains the exclusive-path string; `libgdn_attn_kernels_xe_2.so` does **not** (host binding, not device kernel) |
| Local kernel tree | `/mnt/<models-disk>/vxk-build` @ `de8eae16` (dirty; do not PR from it) |
| Local vLLM tree | `~/vllm-src/vllm` |
| Model (target) | `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, arch `qwen3_5_text` / `Qwen3_5ForConditionalGeneration`, hybrid GDN (28 layers, 3 linear + 1 full per group of 4), 1-layer BF16 MTP |
| Flags | XPU graphs on, `--kv-cache-dtype fp8`, `--no-enable-prefix-caching`, `--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`, `--max-num-seqs` > 1 |
| Historical crash image | `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97` (vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`, kernels 0.1.12) — quoted dumps below |

**Minimal repro (any hybrid-GDN Qwen3.5 family):**

1. Serve with MTP N≥1 and `max_num_seqs ≥ 2`. Prefix cache on or off does not matter.
2. Keep one request in speculative decode (any prompt that has already produced ≥1 output token).
3. Submit a second request (new prompt, tool-call result, or chunked prefill).
4. The next EngineCore step that packs both into one `gdn_attention` dies.

Documented mixes that already did this:

- Long + short: 1216-token prefill + two MTP4 seqs (5 tokens each) — mixed-load 2026-08-08.
- Tiny + one spec: 18-token new request + one MTP4 seq (5 tokens) — dense 27B 2026-08-09.
- Third-party: 32 534-token prefill + 3-token MTP2 spec at `num_running_reqs=2` (#510, greglechin).

`--max-num-seqs 1` avoids it (one sequence cannot be in both groups). No-spec
avoids it. Isolated C1 MTP (no concurrent prefill) is fine.

## 3. Exact error

```
RuntimeError: causal_conv1d does not support spec-decode and non-spec (prefill + decode) tokens in the same invocation; the spec path and the non-spec path are mutually exclusive
```

Then `vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.`

**Stack (load-bearing, mixed-load 2026-08-08, Qwen3.6-35B-A3B MoE):**
`~/B70-DOCS/results/vllm-pi-mixed-load-20260808T211221Z-60972/server.log`

```
gpu_model_runner.py  _model_forward
  qwen3_5.py  forward
    compiled qwen3_next.py
      vllm/_xpu_ops.py:170  _gdn_attention_core_xpu_impl
        torch.ops._xpu_C.gdn_attention(...)
          RuntimeError: causal_conv1d does not support ...
```

Same string, same `_xpu_ops.py` call, dense Qwen3.6-27B:
`~/B70-DOCS/results/vllm-dense27-int4-mtp4-20260809T120403Z-362107/server.log`.

**Scheduler dump that proves the mix (08-08):**

```
num_scheduled_tokens={...: 5, ...: 5, ...: 1216}
total_num_scheduled_tokens=1226
scheduled_spec_decode_tokens={reqA: [-1,-1,-1,-1], reqB: [-1,-1,-1,-1]}
num_running_reqs=3
```

**Smallest local mix (08-09 dense):**

```
num_scheduled_tokens={new: 18, cached: 5}
scheduled_spec_decode_tokens={cached: [-1,-1,-1,-1]}
num_running_reqs=2
```

Cookbook: `~/intel-arc-pro-b70-inference-cookbook/docs/REAL-WORLD-PI-BENCHMARKS.md`
§8.1 — MTP4 mixed-load **0/21** completed. No-spec mixed succeeded at **74.46**
aggregate tok/s (wall-clock includes the 64K prefill; not a C1 decode number).
`AGENTS.md` §15.10: “MTP4 still crashes when a long prefill and speculative
decode share the XPU `causal_conv1d` invocation.”

Current nightly still has the check (image `.so` strings + `_xpu_ops.py` still
one fused call). #510 reproductions: official `vllm/vllm-openai-xpu:v0.27.1`
(seijikun, MTP and DFlash) and vLLM `main` + kernels `main` (greglechin,
Qwen3.8 dense).

## 4. Root cause

Not a missing SYCL kernel for one token class. Three stacked constraints:

### 4.1 Host exclusive check (tripwire)

`/mnt/<models-disk>/vxk-build/csrc/xpu/gdn_attn/gdn_attn_interface.cpp`
`causal_conv1d()`:

```cpp
// A single invocation handles either the spec-decode path OR the non-spec
// (prefill + decode) path, never both at once.
TORCH_CHECK(
    !(num_spec_decodes > 0 && num_prefills + num_decodes > 0),
    "causal_conv1d does not support spec-decode and non-spec (prefill + "
    "decode) tokens in the same invocation; ...");
// later:
TORCH_CHECK(spec_token == num_spec_decodes * (num_speculative_tokens + 1));
TORCH_CHECK(non_spec_token + spec_token == num_actual_tokens);
auto z_active = z.narrow(0, 0, num_actual_tokens);
```

This is a **host** `TORCH_CHECK`, not a device assert. Same string is in the
image `_xpu_C.abi3.so`.

### 4.2 Two mutually exclusive launchers

`gdn::kernel_launcher` in `causal_conv1d.hpp`:

- `num_accepted_tokens != nullptr` → `causal_conv1d_spec_kernel`, grid
  `get_nd_range(num_spec_decodes, qkvz_elems)`, then **return**.
- else → `causal_conv1d_kernel`, grid `get_nd_range(num_virtual_tokens, qkvz_elems)`,
  plus `update_states_kernel` when `num_prefills>0`.

Spec kernel (comment in that file): each spec seq has exactly
`num_spec_tokens == num_spec+1` contiguous **local** tokens; **global** rows
via `token_indx`; sliding window from
`conv_states[cache_indices[batch, num_accepted_tokens[batch]-1]]`; writeback
**only** last slot `cache_indices[batch, num_spec]`.

Non-spec uses `query_start_loc` + variable-length sequences. On Xe2, when
`num_prefills>0`, `causal_conv1d()` does **not** use the native launcher: it
calls `gdn::chunk_causal_conv1d_tiled_xe2` / `gdn::chunk_causal_conv1d_xe2`
and allocates **chunk-padded** `{q,k,v,b,a}` (`padding_size = batch_size * (chunk_size_xe2 - 1)`;
`b`/`a` are also transposed FP32). Spec intermediates are unpadded native layout.
They cannot be concatenated.

### 4.3 Fused op can consume only one 5-tensor group

Legacy `gdn_attention()` chains `causal_conv1d` then:

```cpp
TORCH_CHECK(intermediates.size() == 5, "expected 5 intermediate tensors...");
gated_delta_rule(..., intermediates[0..4], num_prefills, num_decodes,
                 num_spec_decodes, /* both index sets */, num_actual_tokens);
```

Even if the exclusive check were lifted and both launch arms ran, the fused
wrapper would still be wrong: two 5-groups (10 tensors) or mixed
chunk-padded / native layouts into one `gated_delta_rule`.

### 4.4 Why CUDA “handles it”

`QwenGDNLinearAttention` (`qwen_gdn_linear_attn.py`):
`current_platform.is_xpu()` → `forward_xpu` → one
`torch.ops.vllm.gdn_attention_core_xpu`. CUDA `_forward_core` already
`index_select`s spec vs non-spec and calls `causal_conv1d_update` (spec) and
`causal_conv1d_fn` (non-spec) separately, then stitches. vLLM #44700 (merged)
is a *different* CUDA split (prefill vs decode inside the non-spec group for
CuteDSL vs recurrent). Related merged CUDA fix: #34871 (mixed decode /
spec-decode). XPU MTP land: #43565. Unmerged XPU dispatcher: #42382 (closed).

XPU wrapper (`_gdn_attention_core_xpu_impl`) still makes **one**
`torch.ops._xpu_C.gdn_attention` with the mixed counts. Confirmed in the
current image and in `~/vllm-src/vllm/vllm/_xpu_ops.py`.

### 4.5 Why the scheduler produces the mix (not a scheduler bug)

`GDNAttentionMetadataBuilder` (`vllm/v1/attention/backends/gdn_attn.py`):

```python
# num_decodes and num_spec_decodes are mutually exclusive.
# Reclassify non-spec decodes as prefills when spec decodes exist
if num_decodes > 0 and num_spec_decodes > 0:
    num_prefills += num_decodes
    ...
    num_decodes = 0
```

It never folds prefills into the spec group. Mixed `num_prefills>0 &&
num_spec_decodes>0` is a **supported vLLM state**. Token permutation is
`torch.argsort(spec_token_masks, stable=True)` → `non_spec_token_indx` then
`spec_token_indx` (absolute positions in the full batch).

V1 scheduler (`vllm/v1/core/sched/scheduler.py`) has no “don’t mix spec with
prefill” flag. RUNNING loop schedules in-flight spec tokens into
`scheduled_spec_decode_tokens`; WAITING loop then admits new/resumed prefills
into leftover `token_budget`. Agentic trigger: MTP decode in flight + tool
result / new turn waiting. `pad_spec_decode` only pads a 1-token *new* decode
when the running batch is decode-only — not this crash path.

### 4.6 Why naive two-call splits fail

`spec_token_indx` / `non_spec_token_indx` are **global** indices into
`[num_actual_tokens]`. A second call with `num_actual_tokens = group_count`
but those global indices fails the scatter (`z.narrow(0, 0, num_actual_tokens)`).
A call with the full count and one group empty fails
`non_spec_token + spec_token == num_actual_tokens` (empty dummy tensors also
fail this). Local WIP patches in
`~/B70-DOCS/scripts/tmp/patch_gdn_concurrency.py` and
`scripts/tmp/vllm-xpu-causalconv-patch/patch_gdn_mixed_split.py` hit exactly
that (v1–v4; compacted+arange without compacting *inputs* produced `!!!!`).
**Do not treat those as a proven fix.**

Tests (`/mnt/<models-disk>/vxk-build/tests/gdn_attn/test_causal_conv1d.py`):
`test_causal_conv1d` forces `num_spec_decodes=0`; `test_causal_conv1d_mtp`
forces `num_prefills=0, num_decodes=0`. **No mixed-batch test exists.**

## 5. Fix design (preference order as requested)

### a. Scheduler: never batch spec with non-spec

- **Where:** `vllm/v1/core/sched/scheduler.py`, WAITING loop (starts
  “Next, schedule the WAITING requests.”). Condition: if
  `scheduled_spec_decode_tokens` is nonempty, do not admit prefills this
  step (or, conversely, hold spec tokens when a prefill is already
  scheduled). RUNNING already ran first, so the cheap form is “don’t fill
  leftover budget with WAITING prefills when any spec decode is in this
  step.”
- **Pros:** no kernel change; all backends; one-line policy; unblocks
  EngineCore immediately.
- **Cons:** extra step of latency on every tool-call / new-turn arrival;
  does **not** unlock mixed-batch throughput; agentic loops (the reason we
  want MTP) become “spec step, then prefill step, then spec step.”
- **Effort:** S. Risk: throughput/latency regression on mixed queues;
  interaction with chunked prefill, `pad_spec_decode`, DP prefill throttle.

### b. Kernel: accept mixed batch inside C++

- **Where:** `gdn_attn_interface.cpp` `causal_conv1d` + `gdn_attention` +
  `gated_delta_rule`. Launchers already exist
  (`causal_conv1d_spec_kernel` vs native / `chunk_causal_conv1d_*_xe2`).
- **What:** lift the exclusive `TORCH_CHECK`; run **both** existing arms;
  do **not** concatenate intermediates (Xe2 non-spec is chunk-padded, spec
  is not). Change `gdn_attention` to call `gated_delta_rule` **twice**
  (spec group, then non-spec group). `gated_delta_rule.hpp` already
  scatters via `token_indx`, so two calls with the matching index set and
  a group-local `num_actual_tokens` is the coherent C++ shape — but only
  after the size-sum / `narrow` contract is made per-group, not global.
- **Pros:** fused op matches the metadata vLLM already builds; no Python
  `index_select` on long prefills.
- **Cons:** layout + contract change; fused API semantics; needs a new
  mixed test in `tests/gdn_attn/test_causal_conv1d.py`.
- **Effort:** L. Risk: SSM/conv rollback slots, Xe2 padding, graph replay.

### c. XPU dispatch: compact + two fused calls (preferred landing)

- **Where:** `vllm/_xpu_ops.py` `_gdn_attention_core_xpu_impl`.
  `gdn_attention_core_xpu` is already a compile splitting op.
- **What (Hoborific sketch on #510):** if
  `num_spec_decodes>0 && num_prefills+num_decodes>0`:
  1. `index_select` `projected_states_qkvz/ba` on `non_spec_token_indx`.
  2. Call fused op with `num_spec_decodes=0`, `num_actual_tokens=n_non_spec`,
     `non_spec_token_indx=None` (already compact), keep non-spec metadata.
  3. `index_copy_` `core_attn_out` **and** `z` back.
  4. Same for spec: compact inputs, `num_prefills=num_decodes=0`,
     `spec_token_indx=arange(n_spec)`, keep `spec_query_start_loc` /
     `spec_state_indices_tensor` / `num_accepted_tokens`.
  5. `index_copy_` outputs.
- This is the CUDA split, expressed at the XPU fused-op boundary. It
  satisfies size-sum + `narrow` because each invocation sees a compact
  `[0, group_count)` buffer.
- **Fallback-to-eager-non-spec for the whole mixed batch** would also
  survive the check but would drop MTP on every agentic step — almost
  never spec in a tool loop. Do not ship that as the fix.
- **Effort:** S/M (~80–150 lines + a repro). Risks: XPU-graph capture of
  the extra `index_select`/`index_copy_`/`arange`; long-context gather
  cost (greglechin: flat on pp4096, **untested at 64K–128K**); must
  scatter `z` (Hoborific does; some local patches did not).
- **Third-party status:** greglechin (2026-08-15, #510) reports the
  compacting split works through c=8 on Qwen3.8-27B MTP2 (llama-benchy,
  acceptance unchanged, c=1 matches baseline). **This host has not
  revalidated that patch.**

**Recommendation:** land **(c)** in `vllm-project/vllm` as the PR; mention
**(a)** only as an emergency server-side workaround (`max_num_seqs=1` or a
local scheduler guard) until (c) merges; keep **(b)** as a kernels follow-up
if Intel wants the fused op to stop lying about mixed batches.

## 6. PR feasibility

| | |
|---|---|
| Fork | `SergiioB/vllm` (wrapper) and `SergiioB/vllm-xpu-kernels` (C++). Clean branch off upstream. **Do not** PR from dirty `/mnt/<models-disk>/vxk-build` (local GEMM/GDN experiments). |
| Existing PRs (unrelated) | kernels [#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524) grouped-GEMM `at::zeros`; vLLM [#52159](https://github.com/vllm-project/vllm/pull/52159) XPU `grouped_topk`. Do not pile this crash onto those. |
| Minimal diff | `_gdn_attention_core_xpu_impl`: early-return the current single call when not mixed; else compact / two `_invoke` / scatter. No C++ rebuild. Hot-mount works because the op is `direct_register_custom_op`. |
| Tests | (1) unit: mixed metadata → two fake/real calls, outputs match isolated groups. (2) serve repro: MTP N=2/4, `max_num_seqs≥2`, overlapping prefill+decode. (3) greedy A/B vs `--max-num-seqs 1`. (4) XPU-graph on/off. Add a kernels mixed test only if (b) is pursued. |
| Effort | **S** wrapper if we treat greglechin as direction and revalidate on this B70 image; **M** with graph + 64K–128K gather; **L** for true kernel mixed path. |
| Risks | No public XPU CI for mixed MTP; graph replay; GDN conv/SSM rollback slots; `reorder_input` / GQA layout on compacted slices (Hoborific keeps `reorder_input=not gqa_interleaved_layout` on both calls). |
| Process | Comment on #510 with this writeup + “PR incoming against vllm `_xpu_ops.py`”. Do not file a second issue. |

## 7. Unverified — do not claim

- Qwen3.8 `pi-games` / 2026-08-16 EngineDead is **this** `TORCH_CHECK`. Same
  image, same hybrid GDN, same `_xpu_ops.py`, and live
  `pi-agent-serve` is already running **no-spec**
  (`speculative_config=None`, `--max-num-seqs 64`) as the production
  workaround (`~/B70-DOCS/results/qwen38-speed-push-20260815/pi-agent-serve/server.log`).
  Matching Qwen3.8 server.log with this stack was **not** recovered.
  Recipe in `~/B70-DOCS/research/qwen38-pi-agent-backend-20260816.md`
  still documents `SPEC_N=4`.
- Hoborific / greglechin compacting split on **this** host / this digest /
  MTP4 / 64K–128K / XPU graphs / TP>1.
- That lifting the C++ exclusive check alone is sufficient (fused 5-tensor
  assumption and Xe2 padding say it is not).
- Kernel-level single-grid mixed path (not implemented).
- Scheduler isolation patch (not coded here).
- Any new tok/s number. Cited C1 p512/g128 **32.9 / 83.7** is from
  `~/B70-DOCS/results/vllm-qwen38-4mode-230w-20260815T170840Z-581426/tables.md`
  (same image; isolated decode, not agentic mixed). Historical 74.46 is
  no-spec mixed aggregate, not MTP.

## Sources (read for this note)

- Kernels: `/mnt/<models-disk>/vxk-build/csrc/xpu/gdn_attn/gdn_attn_interface.cpp`,
  `causal_conv1d.hpp`, `tests/gdn_attn/test_causal_conv1d.py`
- vLLM: `~/vllm-src/vllm/vllm/_xpu_ops.py`,
  `v1/attention/backends/gdn_attn.py`,
  `v1/core/sched/scheduler.py`,
  `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
- Logs: `results/2026/08/vllm-pi-mixed-load-20260808T211221Z-60972/`,
  `results/2026/08/vllm-dense27-int4-mtp4-20260809T120403Z-362107/`
- Tracker: https://github.com/vllm-project/vllm-xpu-kernels/issues/510
