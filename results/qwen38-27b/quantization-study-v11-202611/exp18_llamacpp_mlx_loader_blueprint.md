# BiasCorr sidecar loader blueprint: llama.cpp (GGUF) and mlx-lm

Date: 2026-11-23. Scope: strictly CPU, read-only source inspection. No GPU, no Docker, no builds were run. No external repo was modified.

## 0. Source verification basis

llama.cpp facts below were verified against the local checkout:

- Tree: `data-root/sata2/inkling2/llama.cpp`
- Commit: `0ce7f213b0749fa4dee4e87d8ae7bf63e621c1ad` (2026-11-24 era tree, newest full checkout on disk)
- Note: the path in the task brief, `data-root/nvme/engines/llama.cpp-latest`, does not exist on this host at inspection time. Other checkouts (`~/llama.cpp` 2026-08-14, `~/llama.cpp-sycl-latest` 2026-08-11) are older. All `file:line` citations below refer to the inkling2 tree.

MLX facts were verified against upstream primary source, because neither `mlx` nor `mlx-lm` is importable on this host (`ModuleNotFoundError` for both) and no full checkout exists on disk:

- `ml-explore/mlx` HEAD `5996d291ee88db9535b33eec458ca241428c0d2b`, file `python/mlx/nn/layers/quantized.py` and `python/mlx/nn/layers/base.py`
- `ml-explore/mlx-lm` HEAD `6fe0ce48c90cec11f2c305830ceba461051c33b2`, file `mlx_lm/utils.py`

BiasCorr artifact inspected:

- `study-root/qwen38_mlx4_biascorr_496L.safetensors`: 15,678,408 bytes, 496 tensors, all float32 1D vectors, key suffix `.bias_corr`
- Companion meta: `size_mib` 14.952, `mean_l2` 2.82117, `max_l2` 16.30087
- IMPORTANT naming caveat: this artifact is for a hybrid linear-attention model. Keys look like `language_model.model.layers.{i}.linear_attn.in_proj_qkv.bias_corr`, `in_proj_a`, `in_proj_z`, `out_proj`, plus MLP `gate_proj`, `up_proj`, `down_proj`. These are NOT the standard `self_attn.q_proj` schema. Section 6 covers the mapping consequence.

## 1. Executive summary

1. llama.cpp already loads optional Q/K/V bias tensors from GGUF for every architecture that uses the shared helper `create_tensor_qkv` (`src/llama-model.cpp:4280-4316`). The HF config flag `bias=false` does not control this; presence of `blk.{i}.attn_q.bias` / `attn_k.bias` / `attn_v.bias` / fused `attn_qkv.bias` in the GGUF file is sufficient. Loading is `TENSOR_NOT_REQUIRED`, so old GGUFs without biases keep working.
2. The runtime graph plumbing for applying those biases already exists and is generic: fused and separate QKV paths in `llama-graph.cpp:2042-2119`, attention output bias `wo_b` at `llama-graph.cpp` build_attn, dense FFN `up_b`/`gate_b`/`down_b` at `llama-graph.cpp:2141-2315`, and full MoE expert bias support at `llama-graph.cpp:2362-2721`. No new ggml math kernels are needed.
3. The gap is per-architecture wiring: which tensor slots each `src/models/*.cpp` loader actually creates and passes. Llama-style archs pass everything already; Qwen3 loads QKV biases for free but never creates `wo_b`/`ffn_*_b`; Gemma4 passes hardcoded `nullptr` everywhere. Costs: 0 LOC for llama-arch dense models, about 7 lines for qwen3/qwen2 dense, about 8-12 lines for qwen3moe, 25-40 lines for gemma4.
4. Backend support for the ops involved (`GGML_OP_ADD`, `GGML_OP_ADD_ID`) exists on CPU, CUDA, SYCL and Metal already. CUDA additionally has a fusion pass that absorbs `MUL_MAT + ADD` (and `MUL_MAT_ID + ADD_ID`) pairs feeding GLU into one kernel sequence; everything else runs as one extra elementwise pass.
5. Bias tensors are 1D (or small 2D for MoE experts) and `.bias` suffixed, so llama.cpp quantization never touches them (`src/llama-quant.cpp:287-297` only quantizes tensors with >= 2 dimensions whose names end in `weight`). An FP32 BiasCorr survives embedding into any GGUF losslessly.
6. The existing `--lora` mechanism cannot carry BiasCorr: the adapter loader rejects any tensor suffix other than `.lora_a`/`.lora_b` (`src/llama-adapter.cpp:274-296`). A sidecar needs either direct GGUF embedding (preferred for v1) or a small new adapter-type loader.
7. MLX: `nn.QuantizedLinear` already supports bias natively. Its `__call__` runs `mx.quantized_matmul` then `x = x + self["bias"]` (upstream `quantized.py:272-285`). Bias application has no explicit cast, so an FP32 bias promotes the sum to FP32 under MLX/JAX-style type promotion. Applying a 15 MiB sidecar post-load is genuinely ~20 lines of Python, no Metal kernel work.
8. Quant applicability: bias correction is orthogonal to weight quantization format. `Q4_K_M`, `Q3_K_M`, `IQ3_XS`, `IQ2_XXS`, `TQ1_0` all benefit immediately through the same path; expected gain is largest for the aggressive formats (IQ2_XXS, IQ3_XS, TQ1_0) where quantization error is largest.

## 2. llama.cpp: exact wiring facts

### 2.1 Tensor creation (what gets loaded from GGUF)

`llama_model_base::create_tensor_qkv` at `src/llama-model.cpp:4280-4316`:

- When merged `blk.{i}.attn_qkv.weight` exists, it tries `blk.{i}.attn_qkv.bias` with `TENSOR_NOT_REQUIRED`.
- Otherwise it tries separate `blk.{i}.attn_q.bias`, `attn_k.bias`, `attn_v.bias`, all `TENSOR_NOT_REQUIRED`.
- Net effect: for any arch using this helper, QKV biases are opt-in purely by file content.

`llama_model_loader::create_tensor` at `src/llama-model-loader.cpp:1211-1485` returns null for absent optional tensors. `done_getting_tensors` at `src/llama-model-loader.cpp:1487-1497` validates created-vs-file tensor counts and can throw on mismatch. There is unused-tensor skip logic with the warning `model has unused tensor ... -- ignoring` near `src/llama-model-loader.cpp:1272-1275`, but its interaction with the count check is fragile. Design rule: every injected bias tensor must be explicitly requested by the architecture loader. Never rely on "extra tensor gets ignored".

### 2.2 Graph application (math already exists)

- Fused QKV path `src/llama-graph.cpp:2042-2054`: `ggml_add` of `wqkv_b` after the fused matmul, or concatenation of the three separate bias vectors.
- Separate Q/K/V path `src/llama-graph.cpp:2082-2119`: `ggml_add` of `wq_b`/`wk_b`/`wv_b` after each matmul, before clamp/rope/reshape. This is the correct position for a projection-output correction: bias is applied in pre-norm, pre-rope projection space.
- `build_attn` applies the output-projection bias `wo_b` after the `wo` matmul (Qwen3 already passes `layer.wo_b` at `src/models/qwen3.cpp:116-118`; the slot is simply never populated by Qwen3's loader).
- Dense FFN: signature accepts `up_b`/`gate_b`/`down_b` at `src/llama-graph.cpp:2141-2155`; `up_b` applied at 2161-2164, `gate_b` at 2180-2183 (both before the activation gate multiply, correct position), `down_b` at 2313-2315.
- MoE: full overload accepts router/up/gate/down expert biases at `src/llama-graph.cpp:2362-2385`; expert biases applied with `ggml_add_id` at 2570-2573 and 2585-2599; down-expert bias at 2718-2721. The simple overload used by qwen3moe passes nulls (`src/llama-graph.cpp:2339-2358`).
- `ggml_add` is a real distinct graph op (`ggml/src/ggml.c:2089-2110`), not fused into the preceding matmul by default. `ggml_add_id` at `ggml/src/ggml.c:2152-2171`.

Naming caution: in `struct llama_layer` (`src/llama-model.h:315-324`) the members `wq_b`/`wk_b`/`wv_b` mean bias vectors in conventional attention, but MLA-style architectures reuse the same members as projection matrices. Do not key any logic off member names alone; key off the GGUF tensor names.

### 2.3 Per-architecture matrix (verified)

| Arch file | QKV bias loaded? | wo_b created? | ffn bias created? | MoE expert bias? | Graph passes |
|---|---|---|---|---|---|
| `src/models/llama.cpp` | yes (helper) | yes | yes | n/a dense | all wired, 0 LOC |
| `src/models/qwen3.cpp` | yes (helper, line 40) | no (only `attn_output.weight` line 41) | no (lines 47-49) | n/a | `wo_b` already passed (116-118); FFN passes NULL (133-138) |
| `src/models/qwen3moe.cpp` | yes (helper, line 32) | no (line 33) | n/a | no (lines 52-54 weights only) | `wo_b` passed (112); MoE uses simple overload with nulls (128-139) |
| `src/models/gemma4.cpp` | no (manual QKV, lines 285-325) | no, hardcoded nullptr (342-349) | no, hardcoded nullptr (374-379, 423-428) | no, hardcoded nulls (398-409) | needs both loader and call-site work |
| `src/models/qwen2.cpp` | yes (helper) | no | no | n/a | same shape as qwen3 |

Positive control templates already in tree: `starcoder2.cpp` (uses `wo_b`, `ffn_down_b`, `ffn_up_b`), `refact.cpp` (TENSOR_NOT_REQUIRED bias slots), `smollm3.cpp`, `qwen3vl.cpp`, `qwen3vlmoe.cpp`, `smallthinker.cpp`. For MoE expert biases the in-tree proof of life is `src/models/openai-moe.cpp:52-54` loading `FFN_GATE_EXPS`/`FFN_DOWN_EXPS`/`FFN_UP_EXPS` `.bias` with shapes `{n_ff_exp, n_expert}` / `{n_embd, n_expert}` and passing them at lines 122-124. Copy this pattern exactly.

### 2.4 GGUF naming and conversion facts

GGUF tensor name constants: `blk.{bid}.attn_qkv`, `blk.{bid}.attn_q`, `blk.{bid}.ffn_down` etc. in `gguf-py/gguf/constants.py` (~1847-1898); backend name table also at `src/llama-arch.cpp:549-565` and op classification at `src/llama-arch.cpp:1001-1246` (`ATTN_Q`/`ATTN_OUT` are `GGML_OP_MUL_MAT`, `FFN_*_EXPS` are `GGML_OP_MUL_MAT_ID`).

HF to GGUF key mapping for BiasCorr (strip the `.bias_corr` suffix, strip optional `language_model.`/`model.` prefix per converter prefix filtering, then map):

| BiasCorr key (HF style) | GGUF tensor | Shape |
|---|---|---|
| `model.layers.{i}.self_attn.q_proj.bias_corr` | `blk.{i}.attn_q.bias` | `[n_head*n_embd_head_k]` |
| `...k_proj.bias_corr` | `blk.{i}.attn_k.bias` | `[n_head_kv*n_embd_head_k]` |
| `...v_proj.bias_corr` | `blk.{i}.attn_v.bias` | `[n_head_kv*n_embd_head_v]` |
| `...o_proj.bias_corr` | `blk.{i}.attn_output.bias` | `[n_embd]` |
| `...mlp.gate_proj.bias_corr` | `blk.{i}.ffn_gate.bias` | `[n_ff]` |
| `...mlp.up_proj.bias_corr` | `blk.{i}.ffn_up.bias` | `[n_ff]` |
| `...mlp.down_proj.bias_corr` | `blk.{i}.ffn_down.bias` | `[n_embd]` |
| per-expert `...experts.{e}.gate_proj.bias_corr` | `blk.{i}.ffn_gate_exps.bias` | `[n_ff_exp, n_expert]` stacked over e (ggml dim order `{n_ff_exp, n_expert}` per `openai-moe.cpp:52`) |
| per-expert up/down | `blk.{i}.ffn_up_exps.bias` / `blk.{i}.ffn_down_exps.bias` | same stacking, down is `{n_embd, n_expert}` |

Converter mechanics: `conversion/base.py:868-872` maps names with default suffixes `(".weight", ".bias")`; QKV fusion detection at `conversion/base.py:894-937` concatenates q/k/v weights AND biases into `attn_qkv.weight`/`attn_qkv.bias` when fusion is enabled (`conversion/base.py:1145-1179`). Practical rule: emit whichever layout the base GGUF uses. If the model has fused `attn_qkv.weight`, emit concatenated `attn_qkv.bias` in q,k,v order; otherwise emit three separate vectors. Both loader branches exist.

Quantization safety: `src/llama-quant.cpp:287-297` quantizes only tensors with at least 2 dimensions and names ending in `weight`. 1D `.bias` vectors are never quantized. Note the 2D expert biases are also safe because the gate is the `weight` suffix, not dimensionality alone.

### 2.5 Why `--lora` cannot carry it

`src/llama-adapter.cpp:274-296`: the LoRA loader accepts only `.lora_a`/`.lora_b` pairs, ignores `_norm.weight`, and raises `LoRA tensor ... has unexpected suffix` for anything else. A `.bias`-only GGUF therefore fails. Options: (a) embed biases in the base GGUF, zero new loader code; (b) add a small `biascorr` adapter type reusing the GGUF parser. For a 15 MiB research artifact, (a) is the correct v1.

### 2.6 Backend support and overhead

Verified op support:

- CPU: `GGML_OP_ADD` and `GGML_OP_ADD_ID` dispatch in `ggml/src/ggml-cpu/ggml-cpu.c` (~1802-1811, feature support ~2304-2314).
- CUDA: dispatch at `ggml/src/ggml-cuda/ggml-cuda.cu` (~2847-2853); support table (~6726-6748) with type constraints (F32/BF16 paths).
- SYCL: `ggml_sycl_add` / `ggml_sycl_add_id` dispatch in `ggml/src/ggml-sycl/ggml-sycl.cpp` (~6164-6171), support listing (~7608-7618). Kernels live in `ggml/src/ggml-sycl/element_wise.cpp` and `add-id.cpp`.
- Metal: support predicate in `ggml/src/ggml-metal/ggml-metal-device.m` (~1665-1671) requires contiguous rows, F32 or F16, and `src[0]` type == `src[1]` type; execution in `ggml-metal-ops.cpp` (~290-302). Since matmul outputs in these graphs are F32, FP32 sidecar biases satisfy this. If a future build runs BF16 accumulation end-to-end, add a dtype check or `ggml_cast` in the graph builder; do not silently rely on mixed-type add.

CUDA fusion bonus: `ggml-cuda.cu:4217-4234` recognizes `MUL_MAT, ADD, MUL_MAT, ADD, GLU` (and the `_ID` variants) as one fusable subgraph, with bias-op validation at `ggml-cuda.cu:2256-2274`. In plain terms: on CUDA, adding gate/up biases costs nothing extra in the common SwiGLU case because the adds get absorbed into the fused GLU sequence. Q/K/V/o/down adds remain separate small elementwise passes on all backends.

Overhead estimate per add: the elementwise pass is `O(rows*tokens)` work plus one full activation read/write, versus `O(rows*cols*tokens)` for the matmul. Arithmetic ratio is about `1/n_embd` for input-side projections (q/k/v/gate/up) and `1/n_ff` for down. For a 27B-class dense model (n_embd ~5120) that is under 0.05% FLOPs; in memory-bound decode the real cost is the extra activation round-trip, on the order of a few tenths of a percent of end-to-end decode per bias, unmeasurable to ~1-2% total depending on how many of the 7 slots per layer are populated. Not worth writing fused epilogue kernels for v1. Measure with a same-binary A/B before optimizing.

## 3. llama.cpp implementation plan (two phases)

### Phase 1: embed into GGUF, minimal arch wiring (recommended v1)

No new CLI, no new loader. A small Python tool reads the `.safetensors` sidecar, remaps keys per section 2.4, and writes them into the GGUF during or after conversion (GGUF is append/rewrite friendly via `gguf-py`; the converter already handles `.bias` suffixes and QKV bias fusion). Then close the per-arch gaps:

- qwen3.cpp (dense): add `wo_b`, `ffn_gate_b`, `ffn_up_b`, `ffn_down_b` create_tensor calls with `TENSOR_NOT_REQUIRED` (4 lines), replace the three NULLs at 133-138 with the layer members (3 lines). `wo_b` is already passed at 116-118. About 7 LOC.
- qwen2.cpp: identical pattern, about 7 LOC.
- llama.cpp (dense arch): 0 LOC, everything already wired.
- qwen3moe.cpp: add 3 expert-bias create_tensor calls with shapes copied from `openai-moe.cpp:52-54`, switch the `build_moe_ffn` call at 128-139 to the full overload passing the bias members. About 8-12 LOC. Note the qwen3moe expert bias shapes are 2D `[rows, n_expert]`; the artifact must be stacked per layer, not per expert file.
- gemma4.cpp: manual QKV path means manual bias adds after the `build_lora_mm` calls at 295/321-324 (or after fused split at 290-293), plus `wo_b` in both `build_attn` calls (342-349) replacing the hardcoded nullptrs, plus FFN bias members in both dense call sites (374-379, 423-428) and the MoE call (398-409). 25-40 LOC, the most invasive of the set. Recommend deferring Gemma4 to phase 2 unless it is a target model.
- Hybrid/linear-attention archs (qwen35/qwen4exp style): see section 6 before touching them; the in_proj/out_proj tensor family is a different schema.

Tests: convert a small slice model with and without sidecar; assert `llama-model-loader` logs the bias tensors as loaded; assert deterministic logit diff vs baseline; assert a zero-filled sidecar produces bit-identical logits to no sidecar (strong correctness check, since add of exact zeros is a no-op in IEEE arithmetic).

### Phase 2: optional `--biascorr-file` runtime sidecar

Only if shipping pre-merged GGUFs is impractical. Design:

- Format: GGUF, not safetensors, because llama.cpp already has a GGUF parser and the tensor naming is already the GGUF namespace. Metadata: `general.type=adapter`, `adapter.type=biascorr`, `adapter.biascorr.schema_version=1`, plus architecture and layer count keys for validation.
- New loader modeled on the LoRA adapter but far simpler: no A/B pairing, no scaling, just name-to-tensor validation against the base model. Estimated 150-200 LOC in a new compilation unit plus ~30 LOC API/CLI plumbing (`common.cpp` loads adapters at lines 1335-1351 before context creation at 1391; a biascorr load must happen at the same point, before graph construction).
- The injection writes into the same `llama_layer` bias slots, so phase 1 arch wiring is a prerequisite either way. The sidecar does not avoid the per-arch work; it only avoids re-embedding the GGUF.
- Lifetime: the adapter object must outlive the context because graph buffers alias its tensors; mirror the LoRA ownership pattern (`common.cpp` keeps `lora` objects in the init result).

Do not reuse the LoRA file format by renaming tensors to `.lora_a`: besides being a lie semantically, rank-1 decomposition of a bias vector is wasteful and the loader validates pairs.

## 4. MLX / mlx-lm integration (Apple Silicon)

Verified upstream facts (`mlx` commit 5996d291, `mlx-lm` commit 6fe0ce48):

- `QuantizedLinear.__init__` accepts `bias: bool = True` and creates `self.bias = mx.zeros((output_dims,))` when true (`python/mlx/nn/layers/quantized.py:230-259`).
- `QuantizedLinear.__call__` at lines 272-285: runs `mx.quantized_matmul(x, weight, scales=..., biases=quant_biases, transpose=True, ...)` then `if "bias" in self: x = x + self["bias"]`. The `biases=` argument to quantized_matmul is the quantization affine offset table, NOT the layer bias; do not confuse them.
- `from_linear` copies an existing bias at lines 301-303.
- `Module.__setattr__` (`python/mlx/nn/layers/base.py:105-115`): assigning an `mx.array` attribute registers it in the module dict, so `layer.bias = mx.array(vec)` on a bias-less quantized layer is valid and becomes part of `parameters()`.
- `Module.load_weights` strict mode (`base.py:179-207`) compares key sets exactly: if you inject bias arrays BEFORE `load_weights` with `strict=True`, the load fails with "Missing N parameters" because the checkpoint has no bias keys. `mlx-lm` `load_model` calls `model.load_weights(..., strict=strict)` at `mlx_lm/utils.py:611-612` with strict defaulting True (`utils.py:422-429`).
- Consequence: apply BiasCorr AFTER `load_model` returns. Do not weaken strict loading globally.

Dtype answer: there is no explicit cast in `QuantizedLinear.__call__`. MLX follows JAX-style type promotion (`mlx/dtype.cpp` type_rules table); float16 or bfloat16 matmul output plus float32 bias promotes the sum to float32. So an FP32 sidecar is applied in FP32 precision, at the cost of widening the activation stream downstream. If end-to-end FP16/BF16 activation flow is preferred, cast once at injection: `bc.astype(model_dtype)`. Measure both; the FP32 path is the faithful reproduction of the sidecar, the cast path is faster and usually within noise for vectors of magnitude ~3.

Reference injection (~20 lines, post-load):

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

bc = mx.load("qwen38_mlx4_biascorr_496L.safetensors")  # keys end in .bias_corr

def apply_biascorr(model, bc, dtype=None, prefix_strip="language_model."):
    leaves = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    n = 0
    for key, vec in bc.items():
        path = key[:-len(".bias_corr")]
        if path.startswith(prefix_strip):
            path = path[len(prefix_strip):]
        # sidecar keys use HF names like model.layers.3.mlp.gate_proj
        # mlx-lm module paths match the same dotted layout for supported archs
        mod = leaves.get(path)
        if mod is None or not isinstance(mod, (nn.Linear, nn.QuantizedLinear)):
            raise KeyError(f"no linear module for {key}")
        mod.bias = vec.astype(dtype) if dtype else vec
        n += 1
    mx.eval(model.parameters())
    return n
```

Caveats to verify on first real run: exact dotted path format produced by `tree_flatten` for the specific arch (strip or keep the leading `model.` segment based on observation), and the hybrid linear-attention module types, which are not `nn.Linear` instances in some mlx-lm arch implementations. The injection loop above intentionally fails loudly rather than silently skipping.

Memory cost on Mac: 15 MiB FP32 as-is, half that if cast to FP16/BF16. Zero kernel work, zero graph surgery; the add is one extra Metal elementwise op per linear, same cost analysis as section 2.6.

## 5. Quant-format applicability matrix

| Format | BiasCorr interaction | Expected value |
|---|---|---|
| Q4_K_M | independent; weight error small but nonzero | modest perplexity/logit improvement, cheap insurance |
| Q3_K_M | independent | moderate |
| IQ3_XS | independent, importance-matrix quants leave larger residual error | significant |
| IQ2_XXS | independent, largest dense error of the set | largest expected gain |
| TQ1_0 (ternary) | independent, extreme weight error | largest expected gain; strong candidate for the 15 MiB correction budget trade |

None of these formats changes the add path: matmul output is dequantized into the same FP32 activation regardless of storage type, so one implementation covers all. KV cache quantization is irrelevant to this mechanism.

## 6. Hybrid linear-attention artifact warning (qwen3.8 specific)

The inspected 496-tensor artifact targets `linear_attn.in_proj_a` (48), `in_proj_qkv` (10240), `in_proj_z` (6144), `linear_attn.out_proj` (5120) plus MLP projections. In llama.cpp the qwen35/qwen4exp-family linear attention uses different GGUF tensor families (see the `ATTN_QKV`-adjacent entries in `src/llama-arch.cpp:649-1059` including gated qkv variants), not the standard `attn_q`/`attn_k`/`attn_v` schema, and the graph code is a different builder that this blueprint did not fully map. Before writing the conversion tool for this specific artifact:

1. Dump the target GGUF's tensor list (`gguf-py` reader) and identify the actual in_proj/out_proj GGUF names for the hybrid arch.
2. Check whether the linear-attention builder applies bias anywhere today; unlike `build_attn`, the linear-attention path was not verified to have bias slots, so it may need new `ggml_add` insertion in addition to loading.
3. The MLP keys (`gate_proj`/`up_proj`/`down_proj`) map cleanly per section 2.4 and are safe to implement first.

Do not assume a one-to-one q/k/v schema. Splitting `in_proj_qkv.bias_corr` (10240) into q/k/v parts requires the head layout from the model config, and `in_proj_a`/`in_proj_z` have no conventional-attention analog at all.

## 7. Recommended execution order

1. MLX path first: it is ~20 lines, zero risk, and directly matches the artifact's native format (the artifact was clearly produced for an MLX 4-bit model, per its filename). Validate quality lift there before investing in llama.cpp.
2. llama.cpp phase 1 for qwen3 dense + llama arch: ~15 total LOC across two model files, plus the GGUF embedding tool. This covers the standard-attention MLP portion of any artifact immediately.
3. Measure with a fixed-prompt logit KL / perplexity delta on Q4_K_M and IQ2_XXS, same binary, sidecar vs none. Only then decide whether qwen3moe, gemma4 or the hybrid linear-attention wiring deserves phase 2.

## 8. What was not verified (explicit honesty)

- SYCL `element_wise.cpp` / `add-id.cpp` kernel internals were not read line by line; dispatch and support-table presence were verified, which is sufficient for the no-new-kernels claim but not for a microsecond cost model.
- Exact HF-to-GGUF mapping lines in `gguf-py/gguf/tensor_mapping.py` for `q_proj.bias` were not re-captured line-number-perfectly; the mapping table in section 2.4 is derived from the GGUF name constants, the converter suffix logic at `conversion/base.py:868-872`, and the loader names in `src/llama-model.cpp:4280-4316`, all verified.
- `unused tensor` skip vs `done_getting_tensors` count validation interaction is unresolved; the blueprint deliberately avoids depending on it by requiring explicit optional tensor creation.
- MLX was verified from upstream source at the cited commits, not from a local install; line numbers refer to those commits. No MLX runtime test was possible on this Linux host.
