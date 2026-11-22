# Intel Arc Pro B60/B70 (Battlemage Xe2) Theoretical Ceilings

Always compute the hardware roofline before running multi-hour sweeps or writing GPU kernels. Spec-sheet peaks overstate real throughput; use measured B70 baselines.

## 1. Measured Intel Arc Pro B70 Hardware Constants

| Quantity | Marketing / Spec | Measured Practical Baseline | Notes |
|---|---:|---:|---|
| VRAM (discrete GDDR6) | 32 GiB | ~32,615 MiB visible (`vram_mm`) | Keep >= 1,500–2,048 MiB free on C1; < 500 MiB abort |
| Memory bandwidth | 608 GB/s | **521–528 GB/s** sustained | Use **525 GB/s** for practical decode rooflines |
| BF16 / FP16 XMX compute | ~157 TFLOPS | **~75–85 TFLOPS** (~50% MFU) | Thermal/power limited at 230 W cap |
| Native FP4 / FP8 DPAS | **None** | **Unpacked to BF16/FP16** | NVFP4/MXFP8/INT4/EXL3 save bandwidth only (`W4A16`/`W8A16`) |
| Typical power cap | 230 W (board-dependent) | Read live `power1_cap` | Never alter caps to pass a benchmark |
| Target AOT ISA | Battlemage Xe2 (`bmg_g31`) | `icpx -Xs "-device bmg_g31 -options -ze-opt-large-register-file"` | Required for CUTLASS Xe2 register occupancy |

*(For Arc Pro B60 24 GiB, scale VRAM to 24 GiB and measure device-copy bandwidth live before pricing.)*

## 2. Decode Ceiling (Memory-Bandwidth Bound)

Bytes read from GDDR6 per generated token at context length `L`:

```text
B_token(L) = B_dense_visits
           + B_active_experts_visits
           + B_lm_head
           + B_kv_read(L)
           + B_recurrent_state
```

Watch three common counting errors:
1. **Looped architectures:** count *visits*, not just stored layers. Inkling 2 Mini stores 24 layers, but layers 6–17 run 3 passes -> **48 layer visits/token** (44 NVFP4 visits + 4 MXFP8 visits). Active bytes are ~5.4B effective parameters, not 3.4B physical.
2. **Large-vocabulary `lm_head`:** Inkling has `vocab=201,024`, `hidden=2,304` -> **0.926 GB** in BF16 (~17% of an entire decode step!). Casting it to FP32 per step reads/writes an extra ~1.72 GB + triggers bandwidth stall (why vLLM PR `#69186` gained **+39–42%** in 2 lines).
3. **Hybrid / SWA KV reads:** only full-attention visits grow linearly with `L`; SWA visits cap at `min(L, window_size)` (e.g. `1,024` tokens).

```text
decode_ceiling_tok_s(L) = BW_sustained_B_s / B_token(L)
```

### Speculative Decoding Ceiling (`DSpark` / `MTP` / `DFlash`)

With draft length `k` and mean accepted tokens per verify step `a` (`1 <= a <= k + 1`):

```text
t_verify = max(B_token(L) / BW_sustained,  (k + 1) * FLOPs_per_tok / Compute_practical)
t_step   = t_draft(k) + t_verify
tok_s    = a / t_step
```

Key consequence:
- Verification reads the target weights **once** for `k + 1` tokens, dividing weight-bandwidth cost by `a` — *as long as* the verify kernel (`M = k + 1`, e.g. `M = 8` for DSpark `k=7`) is compute-efficient rather than falling back to a padded or slow Triton loop!
- Replacing Triton `M=8` MoE verify with native NVFP4/MXFP8 CUTLASS Xe2 cut Inkling's 44-visit MoE verify time from ~93 ms to ~25 ms, driving realistic card-sampling decode from `57.2 -> 80.4 tok/s` (`+40.5%`) and greedy from `71.6 -> 95.1 tok/s`.

## 3. Cold Prefill Ceiling (Compute Bound)

```text
FLOPs_prefill(L) = 2 * active_params_per_tok * L
                 + 4 * n_full_attn_visits * hidden_attn * L^2   # causal factor 0.5 already folded into 2*2
                 + 4 * n_swa_visits * hidden_attn * L * min(L, window)
```

Divide `FLOPs_prefill(L)` by practical BF16 XMX throughput (`~78 TFLOPS`, ~50% MFU) to get a realistic TTFT lower bound, and `L / TTFT` for cold input rate. If a user asks for `10,000 tok/s` on a 5.4B-active looped model on 1x B70:
- `2 * 5.4e9 = 10.8 GFLOP/tok` -> `10,000 tok/s` requires **`108 TFLOPS` sustained** (~69% of raw 157 TFLOPS peak across *every* op including dequant, MoE routing, attention, and norms).
- On 1x B70, ~4,000–5,500 tok/s is the realistic ceiling after CUTLASS Xe2 fusion; reaching `10,000 tok/s` requires 2x B70 or a different model/precision. State this arithmetic early.

## 4. KV Cache & vLLM SWA `--max-num-batched-tokens` Ceiling

For a model with `N_full` full-attention visits, `N_swa` SWA visits (window `W`), `H_kv` KV heads, `D` head dim, and `b_kv` bytes/element (`2` for BF16, `1` for FP8):

```text
kv_bytes_per_full_token = 2 * N_full * H_kv * D * b_kv
```

**Critical vLLM SWA trap:** vLLM's `SlidingWindowManager` reserves in-flight SWA slots proportional to `max_num_batched_tokens` across **every** SWA layer visit (`vllm/config/vllm.py`, `single_type_kv_cache_manager.py`):
- On Inkling 2 Mini (40 SWA visits, 8 full-attention visits):
  - `mbt=4096` or `8192` explodes SWA KV reservation to `10.2–17.4 GiB`, collapsing max context from `200K` down to `~7K`.
  - `mbt=2048` fits `106K` (DSpark `u=0.93`) / `200K` (no-spec `u=0.95`).
  - `mbt=1024` (`u=0.95`) unlocks **`163K` (DSpark, `194K` KV)** and **`256K` (no-spec, `273.8K` KV)**.
  - `mbt=512` (`u=0.95`) expands the no-spec KV pool to **`308,696` tokens**.

Use `python3 skills/intel-xpu-inference-optimisation/scripts/b70_ceiling.py --spec examples/inkling2-mini-b70.json` to run these exact equations automatically.
