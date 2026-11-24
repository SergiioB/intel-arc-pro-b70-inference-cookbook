# Evidence: Inkling 2 Mini on vLLM XPU — Opt-in Exact-Tail Single-Slot Ring Prefix Cache

- **Date:** 2026-11-24
- **Hardware:** 1x Intel Arc Pro B70 (32 GB, non-display card, `xe` driver, as-found 230 W power cap)
- **Model:** `thinkingmachines/Inkling-2-Mini` @ `ed40e7da625f594fc9c4990f5f66be24873c22d7` (native NVFP4 L0–19 / MXFP8 L20–23 experts, BF16 dense/attention/`lm_head`, FP8 E4M3 KV cache)
- **Engine:** `vllm-0.34.0rc1.dev1133+g647a76866.xpu` (`647a76866b4559a74e27d1c6b55fbffe85c07ce4`) + `mgaruccio/vllm-inkling2-xpu` @ `2e9d28b04ff23f5d9996a221ffaa0ae3a440d8f9` + [`patches/inkling2-mini-vllm-xpu/`](../../../patches/inkling2-mini-vllm-xpu/APPLY.md), PyTorch `2.15.0+xpu`, Triton `3.9.0`
- **Launch shape:** `--dtype bfloat16 --max-model-len 162816 --max-num-batched-tokens 2048 --max-num-seqs 1 --block-size 64 --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 --enable-prefix-caching --enable-prompt-tokens-details --limit-mm-per-prompt '{"image":4}' -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY --reasoning-parser inkling --tool-call-parser inkling --speculative-config dspark k=7` + `INK2_CUTLASS_MOE=1 INK2_HYBRID_PREFIX_SDPA=1 INK2_ESIMD_DECODE_ATTN=1`; `INK2_RING_PREFIX_CACHE=1` vs `0` is the only difference between candidate and control
- **Observer:** client-side streaming OpenAI `/v1/chat/completions` observer with `return_token_ids` (greedy `temp=0`), endpoint `cached_tokens`/`prompt_tokens` usage, needle prompts with per-depth `KEY-…` canaries, server `INK2_RING_PREFIX_DEBUG` replay lines, `dmesg` watch

## What the mechanism is

vLLM's block-hash prefix reuse restores K/V pages but **not** the rest of this hybrid stack's recurrent state — the sliding-window short-conv caches, the sliding-window local-attention K/V rings, and the DSpark draft caches. A stock prefix hit therefore replays a prefix built on missing state and returns **different** output on an exact resend.

The opt-in adds two pieces (all hooks no-op when `INK2_RING_PREFIX_CACHE` is unset/`0`):

- **`RingPrefix` (scheduler side, `vllm/v1/core/kv_cache_manager.py`):** `limit()` admits only hit-aligned boundaries inside a live or saved frontier; `covers()` vetoes the coordinator's admitted hit otherwise; `scheduled()`/`finish()` track the frontier and block-hash chain; `reset_prefix_cache` clears it. Geometry on this stack: `hash_tokens=16` (GCD of KV-group block sizes {16,128,64}), `hit_tokens=128` (`cache_hit_alignment_tokens` = LCM scheduler block; partial-hash hits are disabled), so `slack = hit_tokens + min_snapshot_rows − 1 = 136`.
- **`RingPrefixWorker` + `PrefillTail` (worker side, `vllm_inkling2_xpu/`):** on every forward of ≥9 rows it snapshots the final-`slack` tail — token ids, hidden/aux rows, and per-cache byte rows read through each paged cache's `slot_mapping`. An exact-tail resend replays that tail byte-for-byte; a `+new-turn` hit gets the saved conv-state rows written into its prefix-region slots. Any hit that arrives outside both retained frontiers fails closed with `RuntimeError` — it never fired.

## Control: the bug it fixes is real and pre-existing

Flag-off control (identical args, `INK2_RING_PREFIX_CACHE=0`, `--enable-prefix-caching` kept on) on one repeated 2K prompt:

| phase | token-id SHA-256 |
|---|---|
| cold | `baf9a0e6` |
| resend-1 | `fea5d784` |
| resend-2 | `2a0d3a60` |

Three **distinct** digests, first divergence at token index 12–72 across trials — the upstream [vLLM #53912](https://github.com/vllm-project/vllm/issues/53912) class of hybrid-state prefix corruption. The same mismatch reproduces with `--speculative-config` removed, so DSpark is ruled out as the cause. The mechanism does not introduce the divergence; it eliminates it via tail replay.

## Gate receipts

### Initial port (`slack=32`), 2026-11-24 first session

| depth | prompt | cold TTFT | resend-1/2 (exact SHA) | cached / uncached | +new-turn | new-salt |
|---:|---:|---:|---:|---:|---|---|
| 18K | 18,075 | 5.25 s | **0.055 / 0.054 s** ✓ | 18,048 / 27 | `cached=0` designed miss (5.21 s), needle ✓ | `cached=0`, SHA ✓ |
| 75K | 75,021 | 25.45 s | **0.133 / 0.129 s** ✓ | 75,008 / 13 | `cached=0` designed miss (25.58 s), needle ✓ | `cached=0`, SHA ✓ |
| 128K | 128,013 | 49.55 s | **0.233 / 0.231 s** ✓ | 128,000 / 13 | `cached=0` designed miss (49.99 s), needle ✓ | `cached=0`, SHA ✓ |

Resends replayed only when the prompt end landed inside the 32-token ring below a frontier — the 27/13/13 uncached luck above; everything else got a clean designed miss.

### Full matrix (`hit_tokens=128`, `slack=136`), second session

| depth | rem (mod 64) | cold TTFT | resend-1/2 (exact SHA) | cached / uncached | +new-turn (needle ✓) | new-salt |
|---:|---:|---:|---:|---:|---|---|
| 18,098 | 50 | 5.35 s | **0.053 / 0.054 s** ✓ | 18,048 / 50 | `cached=18048`, **0.23 s** | `cached=0`, SHA ✓ |
| 18,112 | 0 | 5.23 s | **0.054 / 0.054 s** ✓ | 18,048 / 64 | `cached=18048`, **0.18 s** | `cached=0`, SHA ✓ |
| 18,116 | 4 | 5.21 s | **0.053 / 0.053 s** ✓ | 18,048 / 68 | `cached=18048`, **0.18 s** | `cached=0`, SHA ✓ |
| 75,060 | 52 | 25.39 s | **0.138 / 0.131 s** ✓ | 75,008 / 52 | `cached=75008`, **0.33 s** | `cached=0`, SHA ✓ |
| 128,044 | 44 | 49.47 s | **0.233 / 0.211 s** ✓ | 128,000 / 44 | `cached=128000`, **0.47 s** | `cached=0`, token-SHA ✗* |
| 128,060 | 60 | 49.59 s | **0.214 / 0.212 s** ✓ | 128,000 / 60 | `cached=128000`, **0.47 s** | `cached=0`, token-SHA ✗* |

\* see anomaly note — final answer text byte-identical in both cases.

All six remainders (44–68 past the boundary) sit **outside** the old 32-token window — every gate above was a designed miss under `slack=32`. 100% of prompt lengths now get verified hits.

### 5-turn growing session at 75K

| turn | input | cached | uncached | TTFT | decode | recall |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 75,036 | 0 | 75,036 | 74.74 s† | 499.7 tok/s | plants `KEY-B02A0052` |
| 1 | 75,066 | 75,008 | 58 | **0.35 s** | 84.0 | — |
| 2 | 75,096 | 75,008 | 88 | **0.36 s** | 81.9 | plants `KEY-281AED7A` |
| 3 | 75,119 | 75,008 | 111 | **0.35 s** | 113.0 | codeA ✓ |
| 4 | 75,148 | 75,008 | 140 | **0.38 s** | 77.2 | codeA + codeB ✓ |

Every growing turn hits the shared-prefix boundary `75008`; `uncached` grows only by the appended turn size, never the transcript. Turn 4 recalled both the turn-0 and turn-2-planted needles exactly.

† turn-0's 74.7 s includes one Xe driver job-timeout/reset (single `dmesg` event, `ccs` engine reset); the request completed correctly and other 75K colds ran ~25 s. The cold path performs zero ring operations — flagged as an infra event, not a mechanism artifact.

## Anomalies (documented honestly)

1. **128K `new-salt` token-id SHA diverges from cold (2/2 gates)** while the answer text is byte-identical (needle `KEY-…` correct, `finish=stop`). Diagnosis: two back-to-back *independent* full computes of the same 128K prompt also diverged once in reasoning-token ids — sporadic engine/kernel nondeterminism at 128K (likely spec-decode verify or MoE borderline logits), **not** mechanism state contamination: `cached=0` requests perform no ring reads/writes, resends remained bit-exact, and the fail-closed `RuntimeError` never fired. 18K/75K recomputes are bit-identical (5/5 controls).
2. **One Xe engine timeout** during the 75K session cold turn-0 (see †). `dmesg` clean otherwise; VRAM fully released post-teardown (~32,615 MiB free, 0 vllm processes, SIGTERM clean).

## Quality & safety

- `17*23=391` ✓, `20-9+3=14` ✓, Vision OCR (`data:image/png`) read `OCR-TEST-7734` exactly ✓ — multimodal forwards are never tail-captured (`IMAGE_PAD_ID` rows embed encoder features token ids cannot re-derive).
- `cached_tokens` usage reporting matched admitted hits in every replay case.
- CPU unit tests on the ported modules: `test_ring_prefix` 13/13 (all `r ∈ [0,128)` resend coverage, salt isolation, stripped-assistant & full-transcript multi-turn, small-append continuations, slack bounds), `test_prefix_tail` 2/2, `test_flag_gate` 4/4 — all pass (`unittest`, no GPU).

## Files

| File | Contents |
|---|---|
| `ring-prefix-cache-20261124.json` | Full sanitized receipt: launch shape, mechanism, flag-off control, slack=32 initial gate, slack=136 six-gate matrix, 5-turn session, anomalies, quality, safety |

## Reproduce

```bash
# apply (no GPU, no compiler needed; atomic dry-run preflight, idempotent)
VENV=./vllm-inkling-venv patches/inkling2-mini-vllm-xpu/install-ring-prefix-cache.sh

# serve with the ring prefix cache (exports INK2_RING_PREFIX_CACHE=1 + --enable-prefix-caching)
RING_PREFIX_CACHE=1 VISION=1 KV_DTYPE=fp8 DSPARK=1 \
  INK2_CUTLASS_MOE=1 INK2_HYBRID_PREFIX_SDPA=1 INK2_ESIMD_DECODE_ATTN=1 \
  CTX=163840 MAX_BATCHED=2048 GPU_MEM_UTIL=0.92 \
  VLLM_BIN=./vllm-inkling-venv/bin/vllm MODEL=thinkingmachines/Inkling-2-Mini \
  benchmarks/inkling2-mini/launch-inkling2-mini-vllm-xpu.sh
```
