# PR 1 — `vllm-xpu-kernels`: zero the grouped-GEMM scheduler counter; add Muse local-attn decode tuple

**Target:** `vllm-project/vllm-xpu-kernels`  
**Against:** `de8eae16cdf2d864c86a203d9431b9bcdb42fb3c` (v0.1.12.3 era)  
**Type:** two independent one-line kernel-config / init fixes in the same tree.  
**Do not include:** local image tags, Nemotron converters, MTP patches, SSU JSON, W4A8.

This is the kernels half of the B70 XPU graph path. The router change is a
separate vLLM PR.

---

## Issue A — uninitialized grouped-GEMM atomic counter

### Symptom

XPU graph replay of grouped-topk MoE (NemotronH / Nemotron-3.5 Lightning GPTQ
WNA16, and any other path that uses `libgrouped_gemm_xe_2.so` with the
persistent-block scheduler) produces **non-deterministic** outputs at
`temperature=0` / fixed seed. Continuations stay coherent; they diverge at
contested near-tie tokens.

Eager mode is intermittent (allocator reuse). Graph mode is systematic: the
captured buffer address is reused every replay.

### Root cause

`csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp` allocates the
persistent-block scheduler counter with `at::empty`. The kernel assumes
`group_id == 0` stores `0` before other groups `atomicAdd`. SYCL does **not**
guarantee that group 0 runs first. A dirty leftover integer becomes the
starting tile index.

### Fix

```diff
   at::Tensor atomic_buffer =
-      at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));
+      at::zeros({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));
```

`at::zeros` queues a `fill_(0)` that happens-before the GEMM. Graph capture
includes that fill, so every replay starts at 0. The GEMM reduction itself is
tile-disjoint (no Split-K atomics); this is scheduler-counter init only.

### Reproduce (issue)

1. Build / install current `vllm-xpu-kernels` **without** this change.
2. Serve any grouped-topk MoE that hits Xe2 grouped-GEMM, with
   `VLLM_XPU_ENABLE_XPU_GRAPH=1` (compiled / graph path).
3. Send 8 identical `/v1/completions` requests:
   `temperature=0`, same `seed`, same prompt, `max_tokens>=32`.
4. Observe 2–4 distinct texts.

On Arc Pro B70, Nemotron-3.5 Lightning GPTQ-INT4 G64-sym, public image
`vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57`,
this was 8/8 mismatch before the change.

### Reproduce (fix)

Same 8 requests after the one-word change: 8/8 byte-identical. Throughput
unchanged within 0.4% (93.00 vs 92.65 tok/s on the no-spec graph card).

A unit-level check that does not need a 30B model: after constructing
`atomic_buffer`, assert it is a single int32 zero before launch. Existing
`tests/fused_moe/test_grouped_gemm.py` should keep passing; add a host-side
zero check if maintainers want it in-tree.

### What this is not

- Not a clamp on expert IDs.
- Not a compiled-mode workaround.
- Not a claim that XPU `torch.compile` reductions are now globally
  deterministic (sampler / other compiled kernels can still race). This PR
  only fixes the grouped-GEMM scheduler counter.

---

## Issue B — missing Muse-Glimmer local paged-decode tuple

### Symptom

After Muse-Glimmer-30B weights load and WNA16 dispatch, the first paged-decode
warmup raises:

```
RuntimeError: Paged decode kernel tuple not compiled for this configuration.
16,128,64,false,true,false
```

`16,128,64,false,false,false` is already in
`csrc/xpu/attn/kernel_configs/paged_decode_default.conf` (GLM / Seed). Muse
needs the **local=true** sibling (`causal=false, local=true, sink=false`).

### Fix

In `csrc/xpu/attn/kernel_configs/paged_decode_default.conf`, under the GLM /
Seed qgroup=16 block:

```diff
 # --- GLM / Seed qgroup=16 decode heads --------------------------------------
 # zai-org/GLM-4-9B-chat, zai-org/glm-4-9b-hf
 # ByteDance-Seed/Seed-OSS-36B-Instruct
+# meta-models/Muse-Glimmer-30B (local attn, qgroup=16, head=128, page=64)
 16,128,64,false,false,false
+16,128,64,false,true,false
```

Format: `qgroup,headsize,pagesize,causal,local,sink`.

Rebuild compiles one extra template
(`paged_decode_kernel_template_q16_h128_p64_ftf.cpp`, ~30 s).

### Reproduce

1. Load Muse-Glimmer-30B (text-only is enough) on XPU with current kernels.
2. First decode / warmup → missing-tuple RuntimeError.
3. Rebuild with the added line → decode proceeds.

This is independent of Nemotron DFlash. It does **not** make Muse vLLM a
production recommendation on B70; it only unblocks the compiled decode
specialization.

---

## Suggested commit split (same PR or two)

1. `fix(grouped_gemm): zero Xe2 persistent-block atomic counter`
2. `feat(attn): add paged-decode tuple 16,128,64,false,true,false for Muse local attn`

If maintainers prefer one file per PR, ship A first (correctness), B second
(model coverage).

## Out of scope

- SSU B8/W4 JSON (device-specific, optional later)
- `grouped_topk` Python router (vLLM, not this repo)
- Any local Docker tag (`1da0a954-det0123` is not pullable)
