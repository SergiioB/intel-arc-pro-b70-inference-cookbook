# Ornith-1.5-35B-A3B: AutoRound vs GPTQ — Same Scope, Same Corpus, Measured

> A single-variable converter head-to-head on Intel Arc Pro B70: GPTQModel
> vs Intel AutoRound signSGD, identical tensor scope, identical calibration
> corpus, identical gptq packing — decided by reference logprob parity
> against the BF16 source. Self-reported, E2. Conversion/serve diagnostics
> n=3; parity gate single deterministic run per side.

## TL;DR

| Question | Answer |
|---|---|
| Which converter for INT4 on B60/B70? | **AutoRound (signSGD)** — now the default converter for new conversions |
| Why | **Zero RTN fallback** (every targeted tensor tuned) + **equal-or-best logprob parity vs BF16** on all three primary metrics |
| Speed impact | None — same gptq packing, same kernels (WNA16 MoE backend), decode/prefill within n=3 spread |
| Conversion cost | ~3.5 h wall for a 35B-A3B MoE at 200 iters / 128×1536 calibration (vs ~6.4–7.7 h for the GPTQ runs on the same host) |
| Catch | AutoRound 0.14.2 has host-memory traps on small-RAM machines — see the recipe section |

## The model and the two artifacts

Ornith-1.5-35B-A3B is a `qwen3_5_moe` hybrid GDN MoE (40 layers, 256 experts,
8 active, one MTP layer) — same measured topology as Qwen3.6-35B-A3B. Both
artifacts use the **identical scope and format**; only the converter differs:

| | GPTQ control (MixedCal-v2) | AutoRound candidate |
|---|---|---|
| Converter | GPTQModel (image default) | auto-round 0.14.2 (pinned), signSGD 200 iters |
| Calibration | MixedCal-v2 (128×1536, mixed-domain) | **same corpus, same seed, same sha** |
| Format | gptq: 4-bit, sym, group_size 128, desc_act false | **identical** (auto_gptq packing) |
| Scope | experts-only W4A16; GDN/attn/router/shared-expert/vision/lm_head/MTP stay BF16 | **identical** (30,720 qweight tensors each, 0 leaks) |
| Artifact size | 24.42 GB | 24.45 GB |
| **RTN fallback** | **10.37%** of expert projections (24.76% in the first WikiText-calibrated conversion) | **0%** — 30,720/30,720 tuned, deep-block losses cut 70–83% |

The RTN fallback ladder is the whole motivation: GPTQModel silently leaves
low-observation tensors with plain round-to-nearest weights (no error
compensation) — one in four experts in the worst layer of the original
conversion. AutoRound's gradient-guided rounding tunes every targeted tensor.

## Head-to-head serving results (n=3 diagnostics)

Same vLLM nightly digest, same flags (`--quantization gptq --dtype float16`,
32K, U=0.85, block 64, prefix-caching off), C1, 230 W cap:

| Cell | GPTQ | AutoRound |
|---|---:|---:|
| p512/g128 post-first median | 69.0 t/s | 70.1 t/s |
| p2048/g1 cold input (rows) | 8572–9616 t/s | 8271–9685 t/s |
| MTP1 speculative decode (real prompts) | 94.3 t/s | 96.3 t/s |
| MTP1 pos-0 acceptance | 80.3% | 78.5% |
| KV cache @ 32K | 169,622 tok | 169,622 tok |

Verdict: serving is a **drop-in** — speed class, KV footprint, and MTP
behavior all match. Kernel logs confirm both artifacts dispatch the same
`XPU WNA16 MoE backend` (watch for silent fall-through to `int4_gemm_w4a16`
— not present here; see the vLLM-XPU quantization pairing notes).

## Reference parity vs BF16 — the method and the numbers

A 72 GB BF16 checkpoint cannot be served whole on a 32 GB GPU with 30 GB of
host RAM. Standard option (perplexity/KL harnesses that load the full
reference) is therefore unavailable. The gate below runs on such hosts:

1. **Reference**: streaming sequential forward — instantiate the model on
   the meta device, materialize **one transformer block at a time** via
   forward hooks, reading weights as **file-backed memmaps** of the
   safetensors shards (never `safe_open.get_tensor`, which copies each
   tensor into anonymous RAM and will OOM a small host). One batched
   teacher-forced pass over fixed prompts; keep per-position logprobs of the
   actual next token plus the top-20.
2. **Quantized sides**: serve each artifact in vLLM, request
   `prompt_logprobs=20` on the **same token ids** (teacher-forced on both
   sides — apples to apples).
3. **Metrics**: mean/median |Δlogprob| on actual tokens (exact, no
   truncation), top-1 argmax agreement, KL over the top-20 union
   (renormalized; approximate — label it as such).

Results — 12 out-of-calibration prompts, 315 predicted positions:

| Metric vs BF16 | AutoRound | GPTQ | Winner |
|---|---:|---:|---|
| mean \|Δlogprob\| | **0.2006** | 0.2025 | AR (ratio 0.991) |
| median \|Δlogprob\| | **0.0793** | 0.1039 | AR (24% closer) |
| top-1 agreement | **91.4%** | 90.5% | AR |
| top-20-union KL (approx.) | **0.277** | 0.319 | AR (13% lower) |
| p95 \|Δlogprob\| | 0.702 | **0.663** | GPTQ (tail only) |

**AutoRound is equal-or-best on all three primary metrics.** Both artifacts
sit at ~0.20 mean |Δ| with ~91% top-1 agreement — normal 4-bit MoE noise,
neither degenerate. Note this is logprob parity, not a task suite; end-task
accuracy parity is not claimed.

## The portable converter recipe (and its traps)

What actually made AutoRound work on a 30 GB-RAM host — each item cost one
failed run to learn:

1. **Pin the version** (`auto-round==0.14.2` at time of writing). The API
   surface moves fast (`scheme` accepts a string like `"W4A16"`, not a dict;
   `ignore_layers` is a comma-separated **string**; calibration samples for
   multimodal-model code paths must carry `input_ids` as a batched tensor,
   not a list).
2. **`qwen3_5_moe` is absent from auto_round's `MODEL_CONFIG` registry.**
   Consequence: `is_model_patched` stays `False` and the compressor
   **silently** disables `low_cpu_mem_usage` — no per-block release, host
   RAM ratchets ~2 GiB/block (72 GiB model → OOM at ~block 17; on a
   workstation this took the whole host down via the kernel OOM cascade).
   Fix: set `ar.model_context.is_model_patched = True` after init — the
   flag's only consumer is the block-release gate, so the compute path is
   unchanged.
3. **Use `quantize_and_save(output_dir=…, format="auto_gptq", inplace=True)`**
   instead of `quantize()` + `save_quantized()`. It binds output/format
   before `post_init`, activating `is_immediate_saving`: each quantized
   block streams straight to disk shards during tuning instead of
   assembling the full packed model in RAM at save time (the assembly
   OOM'd a 24 GiB container twice before this switch). Peak host RAM with
   streaming: ~16 GiB flat.
4. **`AR_WORK_SPACE` defaults to a relative path** (`ar_work_space`) — set
   it to an absolute directory with ≥ 1.2× the model's BF16 size free (the
   offloader's disk check uses a 1.2× margin; a 72 GiB model needs ~87 GiB).
5. **Run the conversion container under a memory cgroup cap** (e.g. 24 GiB)
   plus a host-side watchdog that stops *that exact container* when
   MemAvailable+SwapFree drops below a floor. This turns converter memory
   bugs into a clean rc=137 instead of a host OOM reboot. Note
   `--memory-swap` is inert when kernel swap accounting is off — treat the
   RAM cap as hard.
6. **Fail-closed artifact contract before publishing**: verify the
   safetensors index covers exactly the intended tensors (30,720 expert
   `qweight`s, zero `qweight` in forbidden modules, MTP tensors present and
   unquantized — AutoRound drops MTP tensors into an indexed
   `model_extra_tensors.safetensors`, which loaders pick up via the weight
   map), then publish atomically with `os.replace`.

## What is still open

- n=5 confirmation set on the AutoRound artifact before it replaces the
  GPTQ serving reference (parity gate passed; throughput n=5 pending).
- Task-quality suite (logprob parity ≠ end-task accuracy; 12 prompts /
  315 positions only).
- A direct KLD/perplexity harness comparison on a host that can hold the
  BF16 reference — the streaming gate here is a small-host substitute.

## References

- Intel AutoRound: <https://github.com/intel/auto-round>
- GPTQModel: <https://github.com/modelcloud/gptqmodel>
- vLLM XPU W4A16 kernel fall-through check: vLLM issue #38064
- GPTQ regression in vLLM 0.19.0: vLLM issue #39474 (stay ≤ 0.18.x or use
  an AutoRound-AWQ checkpoint; re-verify kernel dispatch on any image bump)
- Sibling analysis (same architecture family, GGUF K-quants vs GPTQ):
  [qwen36-35a3/QUANTIZATION-QUALITY.md](../qwen36-35a3/QUANTIZATION-QUALITY.md)
