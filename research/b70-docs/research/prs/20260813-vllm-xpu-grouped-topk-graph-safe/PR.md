# PR 2 — `vllm`: graph-safe grouped_topk on XPU

**Target:** `vllm-project/vllm`  
**File:** `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`  
**Type:** platform guard + native XPU body. CUDA path unchanged.  
**Depends on (recommended):** kernels PR that zeros the Xe2 grouped-GEMM
atomic counter. Without that, graph-mode MoE can still be non-deterministic
even if this router is correct.

Do not bundle MTP prenorm, V1-bypass, W4A8, converters, or expert-id clamps.

---

## Issue

Unconditional `@torch.compile` on `grouped_topk` emits an XPU Inductor kernel
that gathers `scores[topk_ids]` with **out-of-range expert ids**:

```
Assertion `index out of bounds: 0 <= tmp0 < 128` failed.
```

This fires at engine-init warmup for any path that uses the torch grouped-topk
router on XPU (NemotronH MTP draft MoE, and any fallback that is not fused
inside `XPUExpertsWNA16`). It also prevents clean XPU graph capture of
grouped-topk MoE.

The fused base WNA16 MoE kernel does **not** execute this Python router, which
is why the backbone can load while the draft / compiled router dies.

This is the same class of compiled-reduction bug vLLM already documents for
other backends (see the PowerPC argmax note in `topk_topp_sampler.py`).

---

## Fix (two parts, one file)

### 1. Do not compile the router on XPU

```python
@(
    torch.compiler.disable
    if current_platform.is_xpu()
    else torch.compile(
        dynamic=True,
        backend=current_platform.simple_compile_backend,
        options=maybe_disable_graph_partition(
            current_platform.simple_compile_backend
        ),
    )
)
def grouped_topk(
```

`torch.compiler.disable` is a graph break at the router only. The rest of the
model can stay compiled / XPU-graph-captured.

### 2. Native XPU body (argmax + masks, on-device)

Insert **before** `if envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK:` a
`current_platform.is_xpu()` branch that:

- scores with softmax or sigmoid (same as reference);
- adds `e_score_correction_bias` when present;
- computes per-group top-2 via `max` + `masked_fill(-inf)` (matches
  reference `topk(2).sum()`);
- selects `topk_group` groups by repeated `argmax`;
- selects `topk` experts inside the mask by repeated `argmax`;
- gathers unbiased weights from `original_scores`;
- renormalizes and applies `routed_scaling_factor`;
- returns `float32` weights and `int32` ids.

Reference implementation used on B70:
`intel-arc-pro-b70-inference-cookbook/patches/patch_xpu_grouped_topk_native_v2.py`

Keep CUDA on `@torch.compile` + existing fused / fallback path.

---

## Reproduce (issue)

On Intel XPU, current vLLM, NemotronH / Nemotron-3.5 Lightning with a draft
that hits Python grouped_topk (native MTP draft MoE is sufficient):

```
vllm serve <nemotronh-checkpoint> --quantization gptq ...
```

EngineCore dies during warmup with `0 <= tmp0 < 128`.

A smaller isolated repro: call `grouped_topk` on XPU under
`torch.compile` with `num_experts=128`, `num_expert_group=8` (or the
checkpoint’s group count), random gating logits, and assert every id is in
`[0, num_experts)`. Compiled path fails; eager / this PR passes.

---

## Reproduce (fix)

1. Apply this PR (or the cookbook patch) inside the XPU image.
2. Same serve command no longer asserts in the router.
3. With `VLLM_XPU_ENABLE_XPU_GRAPH=1` and the kernels `at::zeros` fix,
   PIECEWISE+FULL decode graphs capture.
4. Numerical check: XPU native ids/weights match CPU reference
   `topk(2).sum()` group scores + unbiased gather, for softmax and sigmoid,
   with and without `e_score_correction_bias`.

On Arc Pro B70 this is what unblocked the no-spec graph card (93.00 / 87.25
tok/s n=5 at p512/p8192 g128) and the later DFlash card. Those throughputs
are **not** this PR’s claim; they need graphs + kernels init + DFlash.

Full-eager + this router is deterministic at temp-0. Compiled/graph mode can
still diverge in *other* XPU compiled kernels (sampler / GEMM without
`at::zeros`). Do not advertise this PR as “XPU is now deterministic.”

---

## Tests to add in-tree

Device test (skip if not XPU):

- `grouped_topk` vs reference for
  `num_experts=128`, `num_expert_group=8`, `topk=8`, `topk_group=4`
  (DeepSeek-style) and the NemotronH group layout if a tiny fixture exists.
- Assert `topk_ids.min() >= 0` and `topk_ids.max() < num_experts`.
- Assert weights match reference within atol/rtol used by existing MoE tests.
- Run once under `torch.compile` (must not device-assert) and once with the
  XPU native branch.

Do **not** add a B70 campaign script to CI.

---

## Suggested commit

`fix(xpu): disable compiled grouped_topk and use native argmax/mask router`

## Out of scope

- Expert-id clamps (they hide the assert)
- Forcing `--enforce-eager` as the product answer
- MTP prenorm / concat patches (still 0% draft acceptance)
- Shipping a local Docker image as a prerequisite
