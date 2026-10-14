---
title: "Cascadia PP screen — Qwen3.8-27B 2-stage vs 1-stage on dual B70"
type: worklog
date: 2026-09-23
status: provisional-screen
---

# Cascadia PP screen — Qwen3.8-27B (2026-09-23)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Operator hypothesis: Cascadia PP may retain speed on this no-P2P
x8/x8 host because PP moves activations only at stage boundaries. This was a bounded smoke + n=3 rate screen,
not a publication-grade benchmark. DeepSeek/K216 was not used (operator retired it).

## Setup and safety

- Cascadia v0.2.3, `qwen35` engine, same source/checkpoint/quant tree: `shard-qwen38-27b-2s` (2 stages) vs
  `shard-qwen38-1s-64` (1 stage), both manifest source `Qwen3.8-27B-int4-ov`, arch `qwen3_5`, 64 layers.
- PP placement: stage 0, layers 0–31, embeddings, GPU.0; stage 1, layers 32–63, head, GPU.1; loopback relay/API.
  API was moved to `127.0.0.1:8031` because unrelated Python `http.server` owns port 8000.
- One-gpu control: 1-stage shard on GPU.0, same API port after PP teardown, prefix cache explicitly 0.
- Empty-GPU gate before PP: 31,708 / 32,615 MiB visible; post-load PP: 24,067 / 25,137 MiB. Power caps untouched: 180 W / 230 W. Final cleanup: workers stopped, ports 8031/9100/9101 closed, VRAM returned to 31,651 / 32,615 MiB.
- Cascadia’s API/relay is unauthenticated plaintext. We bound API and relay to loopback, but **Cascadia 0.2.3 hardcodes its probe listener to `0.0.0.0:9101`** despite loopback args (observed in `ss -ltnp` and rank0 log). Listener was closed at cleanup; do not expose this PP worker on an untrusted network without a firewall/auth proxy.

## Result

Same prompt bytes, greedy, n=3, g128, actual server prompt 1,692 tokens in both arms, one warmup discarded:

| Route | samples (tok/s, request wall time) | Median | Correctness |
|---|---|---:|---|
| Cascadia PP2 (GPU.0→GPU.1) | 4.249 / 4.267 / 4.286 (≈30.0 s each) | **4.267 tok/s** | Coherent “Jupiter … largest planet … 142,984 km” correctness smoke passed |
| Cascadia 1-stage (GPU.0) | 4.272 / 4.278 / 4.287 (≈30.0 s each) | **4.278 tok/s** | Same correctness response |

PP2 vs 1-stage: **−0.26 %** (essentially parity; no measured PP speedup). The old same-host Cascadia PP record was a different model (Qwen2.5-1.5B: 70.0 vs 85.8, −18%); it is not compared numerically here. The Qwen3.8 27B PP result validates the local PP topology and correctness, but it **does not demonstrate a speed win**.

### Classification / limits

- Metric: client-observed full-request wall rate (`tok_per_s`), NOT engine-native decode timing.
- Workload: synthetic greedy placeholder-list prompt, p1692/g128, C1, n=3; not recommended sampling and not a real chat/task-quality workload. Raw output begins `1. item 1 description…`; this is explicitly a synthetic rate probe.
- Sample count is screen-class only. The earlier objective's 5-sample finalist protocol has not been run here.
- Therefore: **PP speed retention on Qwen3.8-27B is not established**; this is a marginal screen with the observed PP2 point 0.26% below the 1-stage control. No LocalMaxxing/cookbook/public claim.

## Evidence

`results/2026/09/cascadia-pp-qwen38-20260923/`:
- `evidence-pp-correctness.json` — PP correctness response.
- `evidence-pp-rate.json` — PP2 n=3 (actual p1692/g128; 4.249, 4.267, 4.286).
- `evidence-1stage-correctness.json` — 1-stage correctness response.
- `evidence-1stage-rate-exclusive.json` — 1-stage n=3 exclusive run (4.272, 4.278, 4.287; same actual p1692/g128).
- `logs/worker-{rank0-gpu0,rank1-gpu1,1stage-gpu0}.log`, `logs/vram-*.txt`, `logs/temps-*.txt`, `logs/power-cap-pre.txt`.
- NOTE: a concurrent p517 single-stage request was stopped before use; the paired comparison uses only the later exclusive p1692/g128 control. Do not use `evidence-1s-rate-p517.json`.

## Verdict

**Cascadia PP2 is functional and coherent, but not faster than the same-shard 1-stage control at p1692/g128 (−0.26%, screen n=3).** The claim “Cascadia retains speed on PP” is not verified on Qwen3.8-27B yet. If the operator wants a production PP recipe, next test must use a real recommended-sampling workload with an exact prompt/output and n=5 finalist; otherwise retain independent single-card stacks as default.
