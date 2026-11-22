# Measurement Methodology & Guardrails (Intel Arc Pro B60/B70)

Most apparent speedups or regressions in local LLM tuning turn out to be observer bugs, warm-cache leakage, or mixed metric classes. Every publishable B70 measurement obeys these rules.

## 1. Metric Definitions (Never Mix Classes)

| Metric | Exact Definition | Never Confuse With |
|---|---|---|
| **Cold prefill rate (`cold_input_tok_s`)** | `actual_prompt_tokens / TTFT_s` on a unique entropy-first prompt with verified `prefix_cache_hits_total` delta `== 0.0` | Engine-native `pp` rate or warm/cache-hit TTFT |
| **Post-first-token decode (`decode_tok_s`)** | `(completion_tokens - 1) / (t_end - t_first_token)` from streaming client observer | SSE chunk rate (speculation emits multiple tokens/chunk) or `e2e_tok_s` |
| **Decode energy (`decode_j_per_tok`)** | `(energy_end_J - energy_start_J) / (completion_tokens - 1)` from hardware `energy1_input` counter | Configured TDP (`230 W`) divided by tok/s |
| **Concurrent throughput (`cN`)** | Total completion tokens across all `N` concurrent streams divided by wall time from first start to last finish | Sum of non-overlapping per-stream rates |

## 2. Sampling & Depth Discipline

- **Public headline speed:** Always use the **model card's realistic sampling** (`temperature`, `top_p`, default `reasoning_effort`, fixed output budget such as `2,048` tokens, `n >= 3` seeds) across all reachable standard depths (`0`, `8k`, `32k`, `64k`, `128k`, `max`).
- **Greedy diagnostics (`temperature=0`, `top_p=1`):** Used strictly for exactness, same-session A/B attribution, and speculation parity checks. Never headline greedy speed as realistic serving throughput.
- **Thinking models (`--reasoning-parser`):**
  - Capture **both** `delta.reasoning` and `delta.content` (plus `delta.reasoning_content` for older engines) to detect first token accurately.
  - Record `empty_answer` rate explicitly. If `effort=0.9` saturates the output cap inside `<think>`, pair it with a finished-answer cell (e.g. `effort=0.2`) so readers see both full-thinking speed and non-empty-answer speed.
- **Capacity probes vs speed ladders:** Changing `--max-num-batched-tokens` (e.g. `2048 -> 1024/512`) or completion budget (`2048 -> 128`) creates a separate scheduler profile. Publish it in its own capacity table, never merged into the `mbt=2048` speed ladder.

## 3. Noise Control & Attribution

1. **One variable per arm:** Isolate patch, kernel, quant, and drafter separately.
2. **Same-session controls:** Run interleaved `control -> candidate -> control` (at least `n=3` reps for a screen, `n=5` for tight differences). Discard one same-shape warmup after cold Triton/SYCL JIT compilation.
3. **Fixed thermal & power regime:** Record as-found `power1_cap` before and after; never change caps. Require package temp `<= 55 C` at admission and verify no competing GPU process exists.
4. **Live VRAM safety tiers (`vram_mm`):**
   - `< 500 MiB`: hard abort.
   - `500–1,023 MiB`: isolated capacity observation only.
   - `1,024–2,047 MiB`: staged C1 research / max-context profile.
   - `>= 2,048 MiB`: preferred C1 performance profile.
   - `>= 3,072 MiB`: multi-user concurrency.
5. **Post-run cleanliness proof:** Record `min_vram_mib`, `vram_after_mib` (must return to baseline, e.g. `32,615 MiB`), `caps_after_uw`, and `dmesg_ok` (zero `xe` GPU faults/CCS resets or PCIe AER errors).

## 4. Mandatory Quality & Retrieval Gate

Speed without correctness is not a result:
- **Kernel unit parity:** Compare custom GPU kernels against an exact FP32 reference across all active shapes (`M = 1, 8, 64, 512, 2048`) and edge-case scales (including E4M3 subnormals). Report actual `rel_l2` and `max_abs_err`; never call `rel_l2 ~ 5e-5` "bit-exact".
- **Deterministic scenario battery:** Math, structured JSON dispatch, code state-trace, boundary logic, multi-turn correction, and multi-depth needle-in-a-haystack retrieval (`1.4K`, `8.5K`, `33.8K`, and `>= 100K` when context allows).
- **Speculation parity:** Compare same-prompt greedy output SHA-256 hashes between no-spec and speculated runs. Only claim `drafted == serial` when hashes match directly; otherwise explicitly state that speculation alters greedy hashes.
