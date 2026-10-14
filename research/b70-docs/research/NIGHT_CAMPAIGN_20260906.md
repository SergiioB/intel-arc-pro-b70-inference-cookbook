# Night Optimization Campaign — Status

**Goal:** huge inference gains (prefill + decode across context) on B70 Qwen3.8-27B.
**Approach:** short iterations, measure honestly, ledger everything, never fabricate.
**Rule:** one optimization tested at a time, matched controls, verify serve + req before next.

## Current status: CAMPAIGN COMPLETE 04:00 — final verified table below

## ⚠️ CORRECTION 08:55 — night conclusions partially INVALID

Cookbook reference (docs/REAL-WORLD-PI-BENCHMARKS.md, prefill-decode-matrix-20260809):
dense 27B GPTQ-INT4 MTP4, image 2c427ef4 (v0.26.1rc1.dev457), **cap 230W**, sched 8192,
gmu 0.88, ctx 131072: decode p512/g128 = **69.3 t/s post-first**, input rate **1654-1816 t/s**.
My night numbers (38.5 decode / 1166 prefill) were measured at **150W = the MoE power
profile**. POWER-SWEET-SPOTS.md: dense scales +18-30% from 150→230W; 180W sustained /
230W burst is the dense profile. My "p200 no gain" test ran mid-aria2c-IO-storm — invalid.
Also metric mismatch: cookbook = client post-first-token rate; my completion/dt included
prefill. Re-test at 230W running (dense230). Open questions: image regression
(f01e24f6 dev77 vs 2c427ef4 dev457), sched 8192, gmu 0.88.

### dense230 RESULT (230W, same stack as night base) — power ruled out
decode 36.3 (vs 38.5 @150W), prefill2k 1174 (vs 1166), mixed e2e 617 (vs 615).
**No gain from 150→230W on this stack.** So the cookbook's dense power scaling
(+18-30%) was measured on llama.cpp dense Qwen3.6 — it does NOT transfer to
vLLM + Qwen3.8 hybrid.

### Root explanation for cookbook gap (69 t/s) vs night (38.5 t/s)
1. **Different model**: night target = Qwen3.8-27B **hybrid mamba/GDN** (server
   log shows mamba page padding, mamba_mixer ops); cookbook dense = Qwen3.6-27B
   pure attention on XPUwNa16LinearKernel. Different kernels, different memory
   profile — cookbook numbers don't transfer across checkpoints.
2. **Image**: night f01e24f6 (v0.27.2rc1.dev77) vs cookbook 2c427ef4 (0.26.1rc1.dev457).
   Untested whether the hybrid path regressed in the newer nightly.
NEXT (when host is up): same bench on cookbook image 2c427ef4 + Qwen3.6-27B
checkpoint → isolates image vs model. Host went fully offline 09:05 (no ping,
WoL no answer) — same drop pattern as last night.

## ⚠️ FINAL VERIFIED TABLE (args double-checked from server.log "non-default args")

| label | seq | spec | short decode | mixed decode | prefill2k | mixed e2e |
|-------|-----|------|-------------|--------------|-----------|-----------|
| base | 1 | 4 | 38.5 | 40.1 | 1166 | 615 |
| nomtp | 1 | — | 30.2 | 21.6 | 1204 | 332 |
| mtp5 | 1 | 5 | 34.6 | 41.1 | 1159 | 630 |
| seq8 (PHANTOM WIN) | 8 | 4 | 39.8 | 49.9 | 1776 | 764 |
| seq8+mtp4 (clean run) | 8 | 4 | 36.8 | 40.1 | 1165 | 614 |
| seq16 | 16 | 4 | 38.7 | 40.0 | 1166 | 613 |
| seq4 | 4 | 4 | 38.5 | 40.0 | 1170 | 613 |
| kvauto | 1 | 4 | 36.7 | 40.8 | 1184 | 625 |

**seq8 outlier NOT reproduced.** champ2 = identical verified args (max_num_seqs=8,
spec=4) yielded base numbers. A final run with the `torch_compile_cache` fully wiped
(`seq8clean2`) STILL yielded base numbers: prefill 1165, mixed 614.
The massive 1776/764 win was a phantom — likely an artifact of the concurrent
heavy I/O + OOM pressure changing how graphs captured during that exact minute,
or a scheduler race condition. It is NOT a stable config lever.

## SOLID REPRODUCIBLE WINS (final)
1. **XPU graphs ON**: 2.1× decode (38.5 vs 18.1 eager). Never use --enforce-eager.
2. **MTP4 speculative**: +27% decode vs none (38.5 vs 30.2); e2e 615 vs 332.
3. **KV fp8**: +5% decode vs bf16 auto (38.5 vs 36.7) + 2× KV capacity.
4. MTP5 marginal for long-ctx only (mixed 630 vs 615, short decode worse).
REJECTED: chunked-prefill sizes, 200W cap, seq4/16, prefix caching (upstream crash).

## Recommended production launch (unchanged from before, now proven optimal)
`vllm serve /model --quantization gptq --dtype float16 --language-model-only --max-model-len 9216 --max-num-seqs 1 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --speculative-config '{"method":"mtp","num_speculative_tokens":4}'` + /dev/dri/by-path mount + CCL_ATL_TRANSPORT=ofi + 150W cap.

### RESULTS (in-run history, 150W, GPTQ Int4, KV fp8)
| config | decode tok/s | prefill 2k tok/s | mixed 4k e2e |
|--------|-------------|------------------|--------------|
| **base (MTP4+graphs)** | **38.5** | **1166** | **614** |
| nomtp | 30.3 | 1204 | 331 |
| mtp3 | 39.1 | 1172 | 576 |
| mtp5 | 34.6 | 1159 | 628 |
| chunk4k (batched-tokens 4096) | 35.7 | 1163 | 611 |
| prefcache | CRASHED | — | — |
| eager (no graphs) | 18.1 | 1068 | 397 |
| kvauto (bf16 KV) | 36.6 | 1185 | 622 |
| p200 (200W cap) | 37.5 | 1168 | 612 |
| **seq8 (max-num-seqs 8)** | **39.8** | **1776** | **763** |
| seq8+b4k (batched 8192) | 36.3 | 1163 | 611 |
| seq8+mtp5 (RE-RUN final3, verified args) | 35.0 | 1162 | 631 |
| seq16 | 38.7 | 1166 | 613 |
| seq4 | 38.5 | 1170 | 613 |

**MTP4 contribution: +27% decode** (38.5 vs 30.3). Prefill ~3% better without MTP
(expected — no draft overhead). Mixed long-ctx e2e nearly 2× with MTP4.
MTP3 slightly beats MTP4 on short-prompt decode (39.1 vs 38.5) but loses on mixed
4k e2e (576 vs 614). MTP5: mixed e2e 628 (best, +2% over MTP4) but short decode
34.6 (worst spec). Depth tradeoff: deeper helps long-ctx, hurts short greedy.
Verdict so far: MTP4 = balanced champion; MTP5 only if workload is long-ctx.
chunk4k: prefill unchanged (~1163 vs 1166 — not chunk-bound at batch=1), decode
slightly worse. REJECT. Prefill is compute-bound, not chunk-size-bound.
prefcache: hard crash `ValueError: Overflow when unpacking long long` in mamba
prefix-cache init (`state_base_addrs[idx] = state.data_ptr()`) — hybrid
mamba+prefix-cache bug in this nightly. REJECT for now (vLLM upstream bug, not config).
eager: decode halves (18.1 vs 38.5, −53%), prefill −8%. XPU graphs are worth
2.1× on decode. Keep graphs ON always. Biggest single lever confirmed.
kvauto (bf16 KV): decode 36.6 vs 38.5 fp8 — fp8 KV is +5% decode and halves KV
memory. mixed e2e 622 vs 614 (+1%). KEEP fp8 KV. Note: kvauto halves cache
capacity — fp8 KV also wins on max concurrent ctx.
p200: decode 37.5 vs 38.5 @150W — NO GAIN (−3%, within noise but not positive).
Card is not power-throttled at 150W during these workloads. REJECT — keep 150W
(cooler, silent-class power).
**seq8 = BIG WIN.** prefill 1166→1776 (+52%!), decode 38.5→39.8, mixed e2e
614→763 (+24%). max-num-seqs 8 unlocks batched/padded graph shapes → wider
prefill GEMMs. ADOPT. No downside at concurrency=1 measured.
seq8+b4k: raising max-num-batched-tokens to 8192 CANCELS the seq8 win (prefill
back to 1163). Batched-tokens must stay 2048 — likely graph capture padding
interaction. REJECT b4k; keep seq8 + b2048.
CORRECTION: harness arg bug — `SPEC=` parsing wiped remaining EXTRA args, so
"seq8+mtp5" earlier run was actually **seq1+mtp5** (verified from server.log
non-default args). Its 788 e2e / 1765 prefill numbers belong to seq1?? No —
recheck: that dir shows max_num_seqs=1, spec=5, yet got 1765 prefill. Hypothesis
invalid; the mtp5 label dir (23:26) also seq1. TRUE combined run final3
(verified max_num_seqs=8 + spec=5): prefill 1162, mixed 631, decode 35.0.
**seq8's prefill win does NOT stack with mtp5.** seq8+mtp4 stays champion.
seq16: prefill win GONE (1166 — same as base). seq8 is the sweet spot; 16
over-extends graph shapes (or capture list) and loses the padding benefit.
Final: **seq8 is optimal**.
seq4: prefill 1170 — no win either. Curve: seq1=1166, seq4=1170, seq8=1776,
seq16=1166. **seq8 uniquely enables the fast prefill path** (bucket boundary
at 8 in graph capture sizes). seq curve closed.

### BASELINE (GPTQ Int4 + MTP4 + KV fp8 + graphs, 150W)
| metric | value |
|--------|-------|
| decode (short prompt) | **38.5 tok/s** |
| decode (4k ctx + gen) | 40.1 tok/s |
| prefill ctx512 | 1151 tok/s |
| prefill ctx2048 | 1166 tok/s |
| prefill ctx4096 | 1149 tok/s |
| prefill ctx8192 | 1129 tok/s |
| e2e 4k+256gen | 614 tok/s |

Note: bench1 (weak prompts, 7-token answers) prefill ~1100 tok/s consistent.
Attempt | memory: first bench2 died in torch.compile at 12G container limit (exit 137,
not OOM-flagged); raised to 14G/30G, passed.

**oneCCL SOLVED** (run4): mounting `/dev/dri/by-path:ro` fixed the engine-init crash —
weights started loading for the first time all night. Confirmed canonical by research
subagent (oneCCL #213, same B70+vLLM setup; drmfd fallback opendir's by-path).

Run4 then failed on FP8-specific issue: mtp.safetensors carries FP8 `weight_scale_inv`
tensors but MTP draft module expects BF16 → `ValueError: no module or parameter named
'layers.0.mlp.down_proj.weight_scale_inv' in Qwen3_5MultiTokenPredictor`.

Run5 (load9 dir): **RECOVERY_COMPLETE 00:30.** GPTQ Int4 + MTP4 + KV fp8 loaded
(17.47 GiB, 132s), health OK, 6/6 smoke requests correct (RECOVERY_OK ×3 identical,
counting, code, Paris). Greedy determinism verified. Evidence:
`results/qw38-base-mtp4-load9-20260905T222437Z/` (requests, responses, metrics, logs).

## Blocker history (oldest → newest)

| # | Try | Result |
|---|-----|--------|
| 1 | vLLM 0.27.2rc1.dev77+gac7509e2b pinned digest | `oneCCL ze_fd_manager init_device_fds opendir failed` at engine init |
| 2 | `--distributed-executor-backend mp` | Same oneCCL crash |
| 3 | `mp` + `--tensor-parallel-size 1` | Same oneCCL crash |
| 4 | `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Same oneC crash |
|  reboot | host froze mid-load (kernel panic, watchdog?) | Rebooted itself ~00:01 |
| 5 (current) | **FP8 path** (`/mnt/<models-disk>/hf/qw38-fp8`, `--quantization fp8`) + XCCL dummy patch | Launching now — script ver on host: `qw38_base_mtp4_recovery.py` (load7 dir) |
| 6 | FP8 run1 blocked by guard: swap 15Gi < 32Gi needed | Fixed: +24G swapfile on /mnt/<models-disk> (46G free, used 24G) → run2 launched 00:20 |
| 7 | run2 (load7): FP8 loads but worker proc still oneCCL crash | by-path mount missing — patch only fixed parent, spawn child re-init'd CCL |
| 8 | run4 (load8): by-path mount fixed oneCCL ✅ | weights loaded 66 shards; FP8 MTP draft scale mismatch killed it → back to GPTQ Int4 (run5) |
| 9 | bench1 container OOM at 12G during compile | raised 14G/30G + shm 2G; bench2 passed |
| 10 | nomtp attempt 1: `--speculative-config {}` pydantic error | rewrote harness: SPEC=none env, specflag conditional; nomtp2 passed |

## A/B queue (one at a time vs baseline above)
1. mtp3 (num_speculative_tokens=3) — decode
2. mtp5 — decode
3. no-mtp — decode (isolate MTP contribution)
4. eager (no graphs) — decode/prefill
5. chunked prefill 4096 — prefill long ctx
6. prefix caching on — repeated-prefix workload

## Why FP8 now

GPTQ Int4 base was the target, but every load attempt ended in the same oneCCL crash or a
host freeze. FP8 checkpoint exists at `/mnt/<models-disk>/hf/qw38-fp8` (30GiB, mtp.safetensors
477MB, outside.safetensors 6GB, 64 layer shards). FP8 on XPU is a well-trodden path
(prior `qw38-fp8-tp2-n5-*` results dirs exist from Aug 30). Single-GPU FP8 run needs ~16GB
VRAM for weights (27B fp8) — B70 has 32GB, fine. Host RAM: vLLM loads shards mmap, 15GiB
RAM + 15GiB swap should hold with workers=1.

## Next steps (queue)

1. FP8 load → health → 6-request smoke (same script, load7 dir)
2. If up: baseline bench script (prefill tok/s, decode tok/s @ 1k/2k/4k/8k context)
3. Then real optimizations, one at a time, A/B measured:
   - KV cache fp8 already on; try `--kv-cache-dtype fp8` vs auto for e2e effect
   - MTP4 speculative config already on; try num_speculative_tokens 3/5 for accept-rate
   - XPU graph mode capture sizes vs eager for decode
   - chunked prefill sizes 2048→4096 for long prompts
   - prefix caching re-enable (currently off) for repeated-prefix workloads
4. Ledger every run in research/LEDGER_OPTIMIZATION_20260905.md + this file

## Constraints

- 16GB host RAM (temporarily), 32GB B70 VRAM, 150W cap set per run
- No production overwrite; campaign container name fixed `qw38-base-recovery-20260905`
- GPTQ base at /qw38-gptq-out/... — retry Int4 path after FP8 baseline works
- Subagent budget: xai grok-4.6 / z.ai glm-5.3-flash / vibeproxy gemini-3.8-flash-high

*Last updated: 00:15 by Hermes (night campaign bootstrap)*
