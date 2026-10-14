---
title: "Run-4 subagent report: xtq-runtime (2026-10-11)"
type: research
date: 2026-10-11
status: provisional
---

# XTQ runtime map — serving XTQ without new engines (lane E8DecodePrice/XTQ-runtime, 2026-10-11)

Read-only inspection of `ghcr.io/0xsero/exl3xpu:0.1.0` via the isolated daemon
(`sudo -n docker -H unix:///run/<exl3-docker>.sock run --rm --network none --entrypoint bash … -c`).
No image modified, no GPU touched. Sources extracted read-only to
`scripts/tmp/xtq_runtime_20261011/extract/` (csrc/exl3_esimd.h 938 ln, csrc/exl3_ops.sycl 804 ln,
exl3xpu/{vllm_plugin,vllm_patches,ops,ref,triton_kernels}.py, scripts/build_ext.sh, bench/linear_budget.py);
all `file:line` below refer to those copies (paths inside the image are `/opt/exl3xpu/…`).
EXL3 kernel edits remain operator-stopped (RL-122); XTQ is specced below as a **separate kernel module**.

## 1. The exl3xpu serving path (file:line)

### vLLM integration (plugin = quantization plugin, not an engine fork)
- Entry point: `vllm.general_plugins` → `register()` `exl3xpu/vllm_plugin.py:329` (runs in every vLLM
  process: API server, engine core, workers).
- Quant method: `@register_quantization_config("exl3")` `vllm_plugin.py:50`; `Exl3Config` `:51`
  (bits, head_bits, codebook, mtp_bits); `get_quant_method` `:136` → `Exl3LinearMethod` `:150` for
  `LinearBase` and `ParallelLMHead`; non-linear/visual modules fall through unquantized.
- Which modules are EXL3 is decided by the checkpoint, not the config: `maybe_update_config`
  `:85-108` — a module is EXL3 iff `model.safetensors.index.json` has `<module>.trellis`
  (and `quantization_config.json` `tensor_storage[key].quant_format == "exl3"` carries
  `bits_per_weight`).
- Weight tensors (`create_weights` `:157`, `reg(...)` `:177-180`):
  `trellis` int16 `[k/16, n/16, 16*K]` (16×16-weight tiles, K = bits/weight), `suh` fp16 `[S, k]`
  (input-side signs per fused shard), `svh` fp16 `[n]` (output-side signs), plus a scalar codebook
  selector (`3inst|mcg|mul1` → CB id `:28`). Constraints `:160-164`: **no TP sharding (DP only)**,
  all dims multiples of 128 (the Hadamard block). Fused vLLM modules map to checkpoint constituents
  via `FUSED` `:23-28` (gate_up_proj/qkv_proj/…), loaders `_load_trellis/_load_suh/_load_svh` `:190-221`.
- `process_weights_after_loading` `:223-244`: repacks `suh` to one row per shard, builds
  `shard_of_nb` (128-col block → shard id) and `exl3_bounds`, decides `exl3_cpp` **once, outside any
  traced code** (`:243-244`).

### Linear apply (the hot path)
- `Exl3LinearMethod.apply` `vllm_plugin.py:250-258` → `torch.ops.exl3xpu_C.linear(x, trellis, suh,
  svh, shard_of_nb, bounds, K, cb, SMALL_M_MAX, RECON_SLICE_N)`; Python fallback `ops.exl3_linear_impl`
  `exl3xpu/ops.py:72-124` (used only if the ESIMD lib lacks `linear`). Bias add outside `:258`.
- C++ `linear` op: `csrc/exl3_ops.sycl:644+` (registration `TORCH_LIBRARY(exl3xpu_C, m)` `:774-802`;
  ops: `linear`, `exl3_gemm_small`, `exl3_reconstruct`, `exl3_had_in_rm`, `exl3_had_out_h`,
  `exl3_gemm_raw`, `exl3_sdpa*`, `exl3_fa_fwd`, and host knobs `exl3_set_target_threads{,_mb16,_mb64}`,
  `exl3_set_vec_max_m`, `exl3_set_fused`, `exl3_set_local`, `exl3_set_max_mb`, `exl3_set_int8`).
- Dispatch in `exl3_gemm_small` `exl3_ops.sycl:255-316`:
  - `M <= SMALL_M_MAX (=128, `ops.py:12`)` → **fused decode GEMM**; `use_vec = M <= g_vec_max_m (=2)`
    `exl3_ops.sycl:172` → `GemvKernel` (M=1,2) else `DpasKernel` (M≥3). MR/MB ladders `:265-273`;
    column-strip width `nt_of(K,MR)` `:182-183` (K=4: NT=8 at M=1, 4 at M=2-4, 2 at M=8).
  - `M > 128` (prefill) → `had_in` → `exl3_reconstruct` fp16 slices (`RECON_SLICE_N=16384`,
    `ops.py:24`) → oneDNN `torch.matmul` (or fused W8A8 `INT8_PREFILL`, `ops.py:14-17`) → `had_out`.
  - Optional single-kernel fused path `fused_small` `exl3_ops.sycl:213-250` (had_in+GEMM+split-K+had_out
    in one kernel, M≤8) — **off by default**: `g_fused=false` "measured slower under XPU graphs
    (serialized split-K tail)" `:252`.
  - Split-K sizing is tuned and host-side-fixed: `g_target_threads=1024` (vector + DPAS MB=32),
    `1408` (MB=8/16), `2048` (MB=64) `:174-180`, with measured notes ("M=4 35.1→32.2 ms plateau,
    cliff at 1664"); 256-GRF DPAS kernels (`grf_size<256>` `:122-131`) hold ~1024 resident threads.

### Where the trellis code is evaluated (exl3_esimd.h)
- Tile geometry `Geo<K>` `:28-36`: K=4 → 1 word per period, V=8 values/period, G=32 periods per
  16×16 tile, `WORDS=8K=32` words = **128 B per tile** (=256 weights × 4 bpw).
- State: `tile_states` `:121-150` — 16-bit sliding window over the packed words (bit window
  `e=(u+1)K … e-16`, cross-word shift-or, `& 0xFFFF`); `load_words` `:154-170` (128-B aligned
  `block_load`; "prev" = words shifted one dword in registers, lane-0 patched; measured M=1
  32.5→29.2 ms — the second p-1 load is redundant).
- Code function: `decode_cb_h<CB,N>` `:41-80` — **bit-exact with exllamav3** (comment `:39-40`):
  `mul1` (CB=2) = `st*0x83DCD12D` → `dp4a(0x6400, x, 0x01010101)` (byte sum packed as fp16
  1024+bytesum) → one fp16 mad `*0.00676727… + (-10.3828125)`; `mcg` (CB=1)/`3inst` (CB=0) =
  mult + nibble mask + `__hadd` of two fp16 views. Affine folding trick `decode_mul1_raw` `:84-92` +
  `kMul1C1/C2` (the w = c1·h + c2 affine is applied to the **dot product**, saving one mad/value;
  DPAS FOLD path `:620-629`, `:836-847` uses an all-ones-B dpas on the idle XMX for Σa). Optional
  `EXL3_LUT` 65536×fp16 LUT replacing mul+dp4a+mad with one L1-cached gather `:98-115` (built
  bit-exact from `decode_cb_h` at first use, `exl3_ops.sycl:60-78`).

### suh/svh Hadamard sides (this is XTQ's "two-sided block-128 sign-Hadamard", verbatim)
- Semantics: `xh[g,m,128·b:]=fp16(H(x[m,128·b:]·suh[g,128·b:])/√128)` (comment `:238-239`,
  `HadInKernel` `:286-316`) and `out[m,128·nb:]=svh·H(Σ_p part[p,m,:])/√128`
  (`HadOutKernel` `:321-340`). `suh`/`svh` are per-shard/per-column **sign vectors** (fp16 ±1);
  H = the 128-point FWHT butterfly in registers (`fwht128` macro `:210-218`, add/sub pairs).
- Fused variants `fused_had_in` `:240-256` / `fused_had_out` `:258-281` (in-kernel, one 128-block
  per 8 tile-rows = one Hadamard block per 8 K-tile rows).
- The whole transform is per-128-block; hence the 128-multiple dimension constraint
  (`vllm_plugin.py:163`) and the draft-head 128-column slicing rule (`:269-271`).

### Tile/packing layout and the M=4 verify path
- Packing: `trellis[k/16, n/16, 16K] int16` viewed as uint32 words `[Kdim/16, N/16, 8K]` (128 B/tile);
  one **16×16 tile = one DPAS B operand** (K16×N16 VNNI: element (k,n) at `((k>>1)*16+n)*2+(k&1)`),
  decode-into-VNNI written by `build_k4` `:670-712` ("contiguous 32-lane writes and no scatter moves";
  generic `build<0>` `:630-658` for K≠4; a `EXL3_K4_HALVES` faster variant is recorded REJECTED —
  wrong B layout — `:657-666`; `EXL3_DPAS_PREFETCH` recorded REJECTED, register pressure `:765-770`).
- **M=4 verify runs on DPAS (XMX), not vector FMA**: dispatch `exl3_ops.sycl:265-273`
  (`g_vec_max_m=2` `:172` — comment "vector kernel for M<=2, DPAS (flat cost up to M=8) above");
  `xmx::dpas<8, RC, float, float, fp16, fp16>` `exl3_esimd.h:829-831` (RC=8, or RC=MB for MB=4).
  Verified against the **shipped** `_C.so` (JIT spir64, per_kernel split; `strings` instantiation
  census): `DpasKernel<4,2,8,4,FUSED=0/1>`, `DpasKernel<4,2,{48,64},2>`, `GemvKernel<4,2,{1,8|2,4|4,4|8,2},…>`
  — no `MB=4` instantiation, so this image serves M=4 as **MB=8 (4 real rows + 4 zero-padded)**;
  the Sept-30 forensics capture `DpasKernel<4,2,4,4>` was an `EXL3_DPAS_MB4` build. Either way the
  product is XMX; the measured wall is the decode issue stream feeding it (B_FAKE `:813-819`;
  A_ONCE `:793-796`; BV_ALL WAR/banking probe `:799-811` — all timing-only, wrong-by-construction).
- fp16 partial dots in the vector path fold to fp32 per tile-row (`GemvKernel` `:445-578`,
  `hacc`→`acc += convert<float>`); DPAS accumulates fp32 directly (`:738`).

### Graph-capture constraints (decode)
- `model.yaml` (image `/opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw/model.yaml`): vLLM 0.26.1 XPU,
  `compilation_config.cudagraph_mode: FULL_DECODE_ONLY`, `cudagraph_capture_sizes:
  [1,2,4,8,12,16,24,32,40,48,56,64]`, `VLLM_XPU_ENABLE_XPU_GRAPH=1`, MTP `qwen3_5_mtp` k=3.
- The linear is an **opaque custom op** precisely so capture treats it as one node; a symbolic-shape
  fake is registered (`ops.py:27-31`, "a C++ Meta kernel would specialise the token dim").
- Inside a capture everything is static: dispatch ladder, split-K target threads, MB choice; the
  `exl3_set_*` knobs are host-side and must be set before capture (they are, at library load,
  `ops.py:33-41`). Fused single-kernel path stays off under graphs (`exl3_ops.sycl:252`).
- MTP verify batches = k+1 = 4 rows/seq → the **M=4 cell is the hot verify shape**; capture sizes
  pad M up (M=4 exactly one capture size). `vllm_patches.apply_all` `vllm_patches.py:354-358`
  removes per-step host syncs (align sync skip `:323-349`, GDN mask index `:46-69`, fp8-KV prefill
  with `FP8KV_PREFILL_MIN_QUERY=64` so k+1 verify batches never take the prefill path `:74`,
  XPU block-size for FA2 varlen decode/verify attention `:219-259`).

## 2. Minimal XTQ integration (separate module; exl3xpu files untouched)

Checkpoint (mirror the existing shape — loaders stay byte-compatible):
- `<module>.xtq_codes` int16 `[k/16, n/16, 16·K]` (same tile packing/bitstream as `.trellis`;
  XTQ K=4 → identical tensor shape), `<module>.suh` fp16 `[S,k]`, `<module>.svh` fp16 `[n]`,
  optional `<module>.xtq_meta` (rate per tensor, code-id, version). `quantization_config.json`
  `tensor_storage[key] = {"quant_format": "xtq", "bits_per_weight": …}` so the module-detection
  logic (`vllm_plugin.py:85-108`) pattern-matches with one new suffix.
- vLLM side: `@register_quantization_config("xtq")` + `XtqConfig`/`XtqLinearMethod` = copy of
  `Exl3Config`/`Exl3LinearMethod` (`vllm_plugin.py:50-262`) with the new suffix and op namespace;
  `FUSED`, `_norm_key`, shard loaders, `process_weights_after_loading`, draft-head slicing all reuse
  unchanged (the 128-column Hadamard block rule carries over because suh/svh semantics are identical).
- Kernel side (new files, e.g. `csrc/xtq_ops.sycl` + `xtq_esimd.h`, `TORCH_LIBRARY(xtq_C, …)` with the
  same signatures as `exl3_ops.sycl:774-802`): reuse structure of `load_words`, `tile_states`,
  `build_k4`/`build`, `GemvKernel`/`DpasKernel` shells, `HadInKernel`/`HadOutKernel`,
  `ReconstructKernel`, split-K/partials + target_threads tuning, and the same `EXL3_*` GRF/LUT
  switch patterns. **The only new device code is the code function** (XTQ `f(state)` → fp16, in the
  `decode_cb_h` slot, decode-into-VNNI preserved) and per-tensor-rate handling (rates 3/4/5 change
  K and thus `Geo<K>` instantiation; K=3,5 already have Geo support patterns, `exl3_ops.sycl:303-307`).
  [UNVERIFIED license status of 0xsero/exl3xpu sources — check before source-level copying; the
  structural mapping above is written to allow clean-room reimplementation.]
- Entry points XTQ must expose to vLLM: `linear(x, codes, suh, svh, shard_of_nb, bounds, K, code_id,
  small_m_max, slice_n)`, `reconstruct`, `had_in_rm`, `had_out_h`, `supported(K, code_id)` + the
  `set_*` knobs — i.e. the exact surface at `exl3_ops.sycl:774-802`.
- MTP/verify flow reuses as-is: `qwen3_5_mtp` k=3 (`model.yaml`), pruned draft head
  (`vllm_plugin.py:285-327`, 128-column block slicing — same rule), FULL_DECODE_ONLY capture with the
  same sizes; XTQ adds no new flow. Prefill path needs an XTQ `ReconstructKernel` clone
  (`exl3_esimd.h:870-936`) or its own W8A8 prefill.

## 3. Runtime levers around the GEMV (decode, M=4)

**Q1 — vector FMA or DPAS at M=4?** DPAS. Vector GEMV only for M≤2 (`exl3_ops.sycl:172`); M=4 verify
runs `DpasKernel` with `xmx::dpas<8,RC,float,float,fp16,fp16>` (`exl3_esimd.h:829-831`), decode into
the VNNI B operand once per 16×16 tile and amortised over the batch block (`:586-591` comment:
"Decoding each tile once feeds MB/8 DPAS ops"). The product itself is not the wall — the decode issue
stream is (B_FAKE `:813-819`, RL-119: 381 → 636 logical GB/s with decode stubbed; the dpas+sends are
7 of the 336 issues/512 B). XTQ's design condition "vector units only decode, XMX does the product"
is therefore already the production shape; XTQ changes the decode chain, not the product.

**Q2 — issue budget per weight at 608 GB/s.** Hardware (local://xe2-hardware.md:36-48): Xe2-HPG
BMG-G31, **32 Xe-cores × 8 vector engines (XVE) = 256 XVE**, SIMD16; FP32 vector 256 ops/clk/Xe-core
(=32/clk/XVE); XMX 256 engines; clock **2.8 GHz max dynamic** (Intel datasheet 2800 MHz; Phoronix
2.8 max / 2.28 graphics clock). Live sysfs clock unreadable this session (no `gt/*/freq` entries
under `/sys/class/drm/card1` or `/sys/kernel/debug/dri/0000:0f:00.0/gt` via `sudo -n`; checked
2026-10-11) → datasheet values used, marked accordingly.

Ledger A — per-weight instruction budget (the operative one; EXL3's own accounting):
- 608 GB/s at 4.0 bpw = **1216 G weights/s** (pin; XTQ doc §2).
- EXL3 measured: 381 GB/s = 762 G weights/s at 336 issues/512 B = **0.328 issues/weight** →
  250 G issues/s sustained (8.6 G issues ÷ 34.46 ms; ~3.1 issues/cycle/Xe-core of the ~8/cycle
  SIMD8-class theoretical). RL-119 B_FAKE floor ≈ 7 issues/512 B (4 dpas + 3 sends).
- 495 GB/s (1.30× kernel) needs ≤ **0.253 issues/weight** (258/512 B) at the same issue rate.
- 550 GB/s (1.44× kernel → 1.30× route at f≈0.75) needs ≤ **0.227** (233/512 B).
- 608 GB/s (pin) needs ≤ **0.206** (211/512 B) — i.e. **1.6× fewer issues per weight than EXL3**.

Ledger B — hardware lane-ops (peak, sanity envelope): 256 XVE × 2.8 GHz × 16 lanes = **11.5 T
fp16 lane-ops/s** at ≥1 SIMD16 issue/clk/XVE (22.9 T at the 32-FP32-ops/clk/XVE rate with fp16
packing [UNVERIFIED fp16=2×FP32 on the vector path]); at pin that is **9.4–18.8 lane-ops/weight of
peak budget**. EXL3's own chain is ~5-8 lane-ops/weight (mixed-width asm: the 336 lines include
(1|M0) packed-state muls and (32|32) decode/movs); its measured issue efficiency (~20-35 % of peak
issue slots, from 3.1 of ~8/cycle) makes the **practical budget ≈ 3-6 lane-ops per weight** for
decode + dot. This is exactly the envelope XTQ §2 states ("a few Xe2 ALU ops per weight") and the
one QTIP-HYB estimates (~3-4 ALU/weight + gathers) sit inside [UNVERIFIED estimates, FormatReview].
Consequence for the XTQ code choice (K3): beating EXL3 by ≥1.4× means the code function plus its
operand moves must fit **≤ ~2 lane-ops/value** in the same VNNI write path — the current `mul1` is
~3 (mul + dp4a + 1 mad, affine already folded into the dot, `exl3_esimd.h:41-92`), so the K3 bar is
met only by codes cheaper than mul1 per value, with `EXL3_LUT`-style gathers priced as sends.

## 4. Provenance / commands

- `sudo -n docker -H unix:///run/<exl3-docker>.sock run --rm --network none --entrypoint bash
  ghcr.io/0xsero/exl3xpu:0.1.0 -c 'find /opt/exl3xpu -type f …'`, `tar cf - … | tar xf -` into
  `scripts/tmp/xtq_runtime_20261011/extract/` (read-only extraction), `strings …/_C.so | grep -oE
  "(Dpas|Gemv)KernelILi…"` for the instantiation census (output quoted §1).
- Checkpoint identity (image `models/qwen3.8-27b-exl3-4.00bpw/recipe.json`): turboderp/Qwen3.8-27B-exl3
  rev `113cf7ab958054860e43fb7f3063b1af19171095`, "validated (local-ai-registry)", 14.9 GiB weights
  + MTP head + vision; C1 decode 61-77 tok/s, C16 344-410 tok/s (registry claim, not re-measured here).
- No GPU work; no guard needed. Kernel-tooling coordination sent to XeTrellisCode (decode-state
  reuse finding, budget ledger, JIT-vs-AOT ISA capture note).
