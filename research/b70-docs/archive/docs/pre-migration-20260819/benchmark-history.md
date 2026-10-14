# Benchmark History — Run Log

Every benchmark run, oldest to newest. Each entry: date, build, models, config, results.
This replaces the scattered dated result files — one growing log.
---

## Run 32 — Dense prefill ceiling confirmed + matched power + clock verdict (2026-08-10)

**Status:** E2 provisional self-report with raw evidence. Three questions
answered with matched measurements, not assumptions.

**What:** (1) scheduler-budget prefill A/B (8192 vs 16384) on MTP4 at 230 W;
(2) matched power A/B across all four MTP modes on the same mixed workload;
(3) GPU clock/cap investigation. W4A8 discarded by operator decision (KL
divergence + stacking errors) — the honest W4A16 prefill ceiling is what this
run nails down.

**Evidence roots:**
- Budget A/B: `results/dense27-prefill-budget-ab-20260809T222808Z/`
- Matched power A/B: `results/dense27-matched-power-20260809T214116Z/`

**1. Scheduler budget is NOT a dense prefill lever (FLAT).** MTP4, 230 W,
p2048/p4096/p8192 g1, n=5, fresh entropy prompts, zero cache-hit delta:

| Budget | p2048 | p4096 | p8192 |
|---|---:|---:|---:|
| 8,192 | 1,789 | 1,746 | 1,698 |
| 16,384 | 1,786 | 1,746 | 1,697 |
| Δ | -0.2% | 0.0% | -0.1% |

Contrast: the MoE gained +17.6% from the same budget change (Run 28 era).
Dense prefill is compute-bound at ~10% XMX peak; the budget lever changes
memory layout, not GEMM throughput. **The dense 230 W W4A16 prefill ceiling
is ~1,700-1,790 t/s (input tokens / TTFT).**

**KV capacity finding:** budget 16384 at max-model-len 131072 REQUIRES
`gpu-memory-utilization=0.90`. At 0.88 the activation reservation shrinks KV
cache to 4.97 GiB vs 5.23 GiB needed — engine aborts with estimated max
length 123,136. At 0.90 it loads (KV capacity 177,163 tokens).

**2. Matched power A/B — MTP depth is NOT a power lever.** Same mixed
workload (1× p2048/g1 prefill + 2× p2048/g128 decode), fresh entropy prompts,
true per-mode server, 230 W cap, monitor windowing only the active requests,
cooldown ≤55°C between modes:

| Mode | Mean (W) | Max 0.5s (W) | pkg max (°C) | vram max (°C) |
|---|---:|---:|---:|---:|
| No spec | 149.9 | 238.2 | 70 | 72 |
| MTP4 | 151.0 | 251.5 | 73 | 72 |
| MTP1 | 156.1 | 249.6 | 74 | 74 |
| MTP2 | 153.3 | 242.9 | 72 | 72 |

All four modes within a 6 W band. The Run 31 campaign-monitor numbers
(195/197/146/146 W) were window-coverage artifacts, now corrected with a note
in Run 31 (never rewrite the historical table). Max 0.5 s samples above the
230 W cap are short-burst overshoot before the cap controller engages —
recorded as measured; the card's advertised/stock TDP is 230 W (a ~300 W
write to `power1_cap` is rejected — 230 W is the ceiling).

**3. GPU clock verdict — there is no clock cap to un-cap.** Writing
300,000,000 µW to `power1_cap` is REJECTED by the driver (readback stays
230,000,000 — the stock max). The `xe` driver exposes no gt_min/gt_max/RPS
(xpu-smi and intel_gpu_frequency both fail, AGENTS §15.8). Frequency IS
measurable via perf PMU: idle actual ≈ 1,000-1,050 MHz / requested 583-633
MHz; under load at 230 W actual **4,783 MHz** (budget 8192) / **4,200 MHz**
(budget 16384) — the card boosts to ~4.2-4.8 GHz, so the power cap is the
only lever and it is already at its driver-enforced maximum.

**W4A8 status: DISCARDED (2026-08-10).** Run 30's W4A8 prefill numbers
(2,395-2,576 t/s) are a research kernel A/B, not a production path: per-token
int8 activation quantization diverges from W4A16 at word 15 (32% content
similarity at greedy sampling) and stacks error on long generations. Do not
use W4A8 in publications, submissions, or recommendations.

**Interpretation:** the dense 27B prefill story is now fully grounded — ~1.7K
t/s cold input at 230 W W4A16 (1,654-1,816 across modes/budgets), compute-bound
at ~10% XMX, no scheduler or power headroom left (cap already max), no clock
lever exists. 3K+ t/s prefill remains the MoE track's headline (7-8K t/s).

---

## Run 31 — Dense 27B 4-mode Lane 1 model card: no-spec/MTP1/MTP2/MTP4 + power draw (2026-08-09)

**Status:** E2 provisional self-report with raw evidence. Not independently
reproduced. Follows the `benchmark-model` Lane 1 five-cell card (p2048/g1,
p8192/g1, p512/g128, p8192/g128, one real workload; n=5) plus the 4-mode and
power-draw comparison the user requested for this model.

**What:** Dense Qwen3.6-27B GPTQ-INT4 MTP-Preserved on the pinned nightly
(vLLM 0.26.1rc1.dev457, vllm-xpu-kernels 0.1.12, fp8 KV, scheduler 8192,
context 131072, GPU util 0.88-0.90, prefix cache on with zero hit delta).
Four true per-mode servers at the 230 W cap: no-spec, MTP1, MTP2, MTP4.
Each mode runs the full C1 matrix (prefill p2048-p8192, decode curves
g32/g128/g256/g512 at p512+p8192, control p9445/g128, full-context
p130944/g128 + p130560/g512), the MTP4 track additionally ran the real Pi
workload flow (cold/warm/RAG/32K-doc/resident follow-up).

**Canonical evidence:** `results/vllm-dense27-4mode-230w-20260809T163138Z-74780/`
(`summary.json` 60 rows, `tables.md`, per-mode manifest.txt + monitor.jsonl,
per-cell results.json + raw SSE, `b70-dense27-4mode-dashboard.svg`, Pi flow
`mtp4/piflow/`). Naming note: the first three modes stored the p8192/g1 cell as
`decode-p8192-g1`; mtp4 used `prefill-p8192` — the compiler treats them as one
coordinate (`b70-compile-dense27-summary.py`).

**Cold input rate (actual endpoint tokens / client TTFT, g1, C1, zero cache
hits, median n=5, tok/s):**

| Mode | p2048 | p4096 | p6144 | p8192 |
|---|---:|---:|---:|---:|
| No spec | 1,781 | 1,813 | 1,782 | 1,742 |
| MTP1 | 1,816 | 1,776 | 1,747 | 1,713 |
| MTP2 | 1,812 | 1,767 | 1,744 | 1,711 |
| MTP4 | 1,755 | 1,693 | 1,683 | 1,654 |

**Decode (client post-first tok/s, median n=5):**

| Mode | p512/g32 | p512/g128 | p512/g256 | p512/g512 | p8192/g32 | p8192/g128 | p8192/g256 | p8192/g512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No spec | 32.90 | 32.85 | 32.78 | 31.54 | 31.48 | 31.46 | 31.45 | 31.42 |
| MTP1 | 50.00 | 50.47 | 50.19 | 48.88 | 48.08 | 46.90 | 47.97 | 47.33 |
| MTP2 | 62.15 | 63.59 | 61.45 | 59.95 | 63.98 | 60.73 | 59.62 | 57.10 |
| MTP4 | 72.78 | 69.30 | 64.06 | 64.13 | 67.44 | 64.11 | 65.87 | 57.79 |

**Matched control p9445/g128 (tok/s):** no-spec 31.35 · MTP1 48.41 · MTP2
60.12 · MTP4 67.25.

**Full-context (exact 131,072 total tokens, n=1):** p130944/g128: no-spec
23.14 · MTP1 36.77 · MTP2 42.67 · **MTP4 47.61**; p130560/g512: no-spec 23.05 ·
MTP1 37.21 · MTP2 36.18 · MTP4 42.56.

**MTP acceptance (p512/g128, totals-diff):** MTP1 95.1% · MTP2 94.8% · MTP4
83.8% — consistent with the known per-position acceptance decay (§15.8).

**Measured power draw (live `energy1_input`, 0.5 s interval avg, 230 W cap):**

| Mode | Mean (W) | Max 0.5s (W) | Coverage |
|---|---:|---:|---|
| No spec | 195 | 275 | full campaign 1462 s (n=2926) |
| MTP1 | 197 | 278 | full campaign 1427 s (n=2856) |
| MTP2 | 146 | 266 | partial window 223 s (n=448, campaign interrupted) |
| MTP4 | 146 | 246 | dedicated mixed prefill+decode observation (24 s) |

Max spikes above the 230 W cap on short prefill bursts — recorded as measured.

> **CORRECTION (2026-08-10, matched power A/B — supersedes the table above):**
> the campaign-monitor means above were window-coverage artifacts, not mode
> differences. A matched A/B (same 1× p2048/g1 prefill + 2× g128 decode
> workload, fresh entropy prompts, per-mode server, 230 W cap, monitor
> windowing only the active requests, cooldown ≤55°C between modes) measured
> all four modes within a 6 W band: **no-spec 149.9 W · MTP4 151.0 W · MTP1
> 156.1 W · MTP2 153.3 W mean** (max 0.5s 238-252 W, pkg 70-74°C). MTP depth
> is NOT a power lever on dense. Evidence:
> `results/dense27-matched-power-20260809T214116Z/`. The 195-197 W campaign
> numbers included the full-context 130K prefill cells inside the no-spec/mtp1
> monitor windows; mtp2's 146 W came from a 223 s window covering only five
> short decode cells. The historical table stays as recorded (never rewrite),
> but it must not be cited as a mode-vs-mode power comparison.

> **W4A8 DISCARDED (2026-08-10):** the Run 30 W4A8 prefill numbers
> (2,395-2,576 t/s) are a research kernel A/B, NOT a production path — per-token
> int8 activation quantization diverges from W4A16 at word 15 (32% content
> similarity) and stacks error on long generations. The honest dense prefill
> ceiling at 230 W is the W4A16 number plus the scheduler-budget lever. Do not
> use W4A8 in publications, submissions, or recommendations.

**Real Pi workload flow (MTP4, g128 outputs):** cold conversation 54.2 t/s
post-first (TTFT 0.424 s) · warm shared system 62.9 (0.423) · multi-turn 46.9
(0.477) · RAG append 55.4 (0.574) · cold 32K-doc 43.7 (TTFT 25.2 s = 1,295
input t/s) · resident-doc follow-up 52.3 (TTFT 2.95 s, 29,952/32,797 tokens
cached = 91.3% reuse, zero MTP loss at cache hit). Realistic short-turn decode
is 44-56 t/s, not the 73 t/s synthetic peak — same finding as Run 29 §15.11.

**Findings:**
- MTP4 wins every decode cell (p512/g128 69.3 vs 63.6/50.5/32.9; full-context
  p130944/g128 47.6 vs 42.7/36.8/23.1). MTP2 is the close second. MTP gains
  shrink at g512 (MTP4 57.8-64.1 vs MTP2 57.1-60.0) — draft-run overhead
  catches up on long outputs.
- Prefill is flat across MTP depth (1,654-1,816 t/s): the draft head costs at
  most a few percent at p2048-p8192; MTP4's 1,654 at p8192 is the worst cell
  but within ~5% of the best mode.
- Power: dense MTP decode draws ~146 W mean (MTP2/MTP4) while no-spec/MTP1
  campaigns averaged 195-197 W — the no-spec/MTP1 windows included the heavy
  full-context prefill cells; coverage differs, so treat mode-vs-mode draw
  deltas as directional, not matched (§8.6 paired-power rule).
- Lane 1 selection: MTP4 at 230 W, fp8 KV, 128K is the dense production
  candidate for decode-heavy serving; the Pi flow shows 44-56 t/s realistic
  decode with 91% cache reuse on resident sessions.

**Compiler/render:** `scripts/tmp/b70-compile-dense27-summary.py` →
`summary.json` (60 rows); `scripts/tmp/b70-render-dense27-tables.py` →
`tables.md`; `scripts/tmp/render-dense/render-dense27-dashboard-svg.py` →
`b70-dense27-4mode-dashboard.svg` (XML-valid, chromium-rendered check).

---

## Run 30 — Dense prefill: 230W power lever (+52%) and W4A8 kernel track (+41-43%) (2026-08-09)

**Status:** E2 provisional self-report with raw evidence. Not independently
reproduced. W4A8 accuracy caveat in force — see below.

**What:** Dense-track prefill optimization A/B on the pinned nightly
(vLLM 0.26.1rc1.dev457, vllm-xpu-kernels 0.1.12, dense Qwen3.6-27B GPTQ-Int4
MTP-Preserved, fp8 KV, MTP4, budget 8192, 128K). Two levers isolated:
(1) power cap 165W → 230W (dense prefill is compute-bound, clock-scales with
power — the MoE's "power is dead" finding does not apply to dense); (2) a
dense-build-track W4A8 kernel path (per-token int8 activation quant + oneDNN
`int4_gemm_w4a8`, same GPTQ u4 weights, no repack). The MoE grouped-GEMM wheel
and build are untouched — this is a separate dense track.

**Evidence roots:**
- Microbenchmarks (kernel rates, M=4096): W4A16 oneDNN 96.6-129.6 TFLOPS;
  **W4A8 oneDNN 186.6 TFLOPS (1.93×)**; cutlass single-group dense INT4 115.5
  TFLOPS (SLOWER — MoE kernel is not a dense win).
- 165W W4A16 baseline: `results/vllm-dense27-int4-budget-ab-b8192-20260809T120157Z-361073/`
- 230W W4A16: `results/vllm-dense27-int4-budget-ab-b8192-20260809T143533Z-27690/`
- 230W W4A8: `results/vllm-dense27-int4-budget-ab-b8192-20260809T144410Z-32099/`
- Patch: `scripts/tmp/patch_dense_w4a8.py` (env-gated `B70_DENSE_W4A8=1`, marker
  `B70_DENSE_W4A8_PATCH`, byte-faithful port of upstream `XPUW4A8IntLinearKernel`).
- Accuracy parity: `results/vllm-dense27-int4-mtp4-20260809T143714Z-28637/w4a8-accuracy.json`
  vs `...-20260809T144618Z-33149/w4a16-accuracy.json`.

**Prefill input rate (actual endpoint tokens / client TTFT, g1, C1, cold, zero
cache hits, median n=5):**

| Config | p2048 | p4096 | p8192 |
|---|---:|---:|---:|
| 165W W4A16 (Run 29 baseline) | 1,179 | 1,154 | 1,116 |
| 230W W4A16 | 1,798 | 1,754 | 1,702 |
| **230W W4A8** | **2,576** | **2,470** | **2,395** |

| Delta | p2048 | p4096 | p8192 |
|---|---:|---:|---:|
| 230W vs 165W (power, W4A16) | +52.4% | +52.0% | +52.5% |
| W4A8 vs W4A16 (kernel, at 230W) | +43.3% | +40.8% | +40.7% |
| **Combined 230W W4A8 vs 165W W4A16** | **+118.4%** | **+114.0%** | **+114.5%** |

**Measured power (energy1_input deltas, 0.5 s interval avg):** 230W W4A8
campaign mean 132 W / max 260 W (mean includes idle gaps between cells; max
spikes above the 230 W cap on short prefill bursts — recorded as measured).

**Accuracy gate (mandatory — W4A8 changes activations to int8, speed must not
stand alone):** identical prompt, temp 0, 200 tokens. Reasoning text 100%
identical; visible content **diverges at word 15, only 32.1% SequenceMatcher
similarity**. W4A8 is NOT output-identical to W4A16 — per-token int8 activation
quantization flips near-tie greedy logits. This is a research/kernel A/B, not a
publication default. A full KL/perplexity audit is required before W4A8 could be
recommended; until then it stays env-gated off by default.

**Architect verification of model choice (b70-architect, 2026-08-09):** keep
`llmfan46` GPTQ-Int4 MTP-Preserved — the only checkpoint that is simultaneously
loadable on the pinned nightly, W4A8-compatible (sym, no g_idx, g128),
128K+fp8+MTP4-feasible, and MTP-BF16-preserved. Alternatives die on verified
gates: AutoRound not registered in the nightly (cannot load); AWQ unverified on
host + VRAM abort band; MXFP4 no dense checkpoint + measured non-competitive;
FP8 no XPU dense path + 0.5× rate + VRAM. Honest expectation: dense prefill
lands ~1.5-2.0K t/s with W4A8 alone; ~2.4-2.6K with 230W+W4A8 (measured here);
3K+ prefill remains the MoE track's headline (8.3K t/s). Dense's differentiators
are decode (73 t/s MTP) and 128K fp8 KV. Full verdict in the Run 30 context.

**Interpretation:**
- Power is a real dense-prefill lever: +52% at 230W, zero quality cost. The
  AGENTS §9.1 prediction ("power scaling helps dense, not MoE") is now measured
  on prefill: dense is compute-bound and frequency-scales; MoE self-limits ~140W.
- The oneDNN FP16 W4A16 path was the dense GEMM ceiling (~96-130 TFLOPS). The
  W4A8 int8-dpas path doubles the GEMM rate but costs output fidelity at greedy
  sampling — a classic speed/quality trade, not a free win.
- 3K t/s for dense w4a16 is mathematically unreachable (GEMM floor alone 1.26s
  at 85% FP16 peak vs 1.37s budget for 3K); the W4A8+230W combination is the
  only measured path approaching it (~2.4-2.6K).

---

## Run 29 — Dense Qwen3.6-27B GPTQ-INT4 MTP-Preserved: first vLLM XPU campaign (2026-08-09)

**Status:** E2 provisional self-report with raw evidence. Not independently
reproduced.

**What:** First load and C1 benchmark of the dense 27B twin of the proven 35B MoE
recipe on the pinned nightly — dense GPTQ-INT4 linear (XPUwNa16LinearKernel),
BF16 MTP head, fp8 KV cache, MTP4, exact-128K and 200K/256K context gates, and a
realistic Pi multi-turn flow. Nothing public until claims audit.

**Evidence roots:**
- Launch/smoke: `results/vllm-dense27-int4-mtp4-20260809T103223Z-327957/`
  (and sibling `-T110155Z-339865` for the flow server)
- C1 matrix: `results/vllm-dense27-int4-bench-20260809T104145Z-332156/`
- Exact-128K gate: `results/vllm-dense27-int4-exact128k-20260809T104326Z-332600/`
- Pi flow: `results/vllm-dense27-int4-flow-20260809T110641Z-342371/`
- 256K/200K VRAM gates: `results/vllm-dense27-int4-mtp4-20260809T105325Z-335186/`,
  `-T105629Z-336817/`, `-T110844Z-343136/`

**Stack:** image
`vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97`;
observed vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`; `vllm-xpu-kernels 0.1.12`;
model `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4`
(heretic-v2 uncensored fine-tune, 19.58 GB, GPTQ-Int4 g128 sym desc_act=false,
15 `mtp.*` tensors BF16 in `model-auxiliary.safetensors` — verified in index).
Patches: `patch_mtp_nightly.py` SHA-256
`4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14`,
then `patch_mtp_boundary.py` SHA-256
`41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50` — both
applied cleanly to the dense model (BF16 draft gate hit in `qwen3_5_mtp.py`,
GDN boundary patch hit in `gdn_attn.py`). MTP head detected and weights shared.

**Methodology:** C1; p512/p2048/p4096/p8192 and exact-128K cells = median of 5
measured requests after one full-output same-shape warmup (exact-128K gate: n=1
after full-output warmup); scheduler budget 8,192; prefix caching enabled with
entropy-first unique cold prefixes and zero `vllm:prefix_cache_hits` delta;
configured cap 165 W during campaign, restored 150 W on exit; timing = client
monotonic SSE; `input_tokens_per_ttft_s` is endpoint tokens / client TTFT
(cold input rate, not engine prefill); decode is client post-first
`(completion-1)/(end-first_token)`.

**Config:** `--quantization gptq --dtype float16 --max-model-len 131072
--gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --max-num-seqs 64
--max-num-batched-tokens 8192 --enable-prefix-caching --language-model-only
--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`.

**Key results (all cold, zero cache hits, C1):**

| Cell | TTFT (s) | Cold input rate (t/s) | Decode post-first (t/s) | MTP accept | n |
|---|---:|---:|---:|---:|--:|
| p512/g128 | 0.387 | 1,322 | **73.17** | 85.34% | 5 |
| p2048/g1 | 1.555 | 1,317 | — | — | 5 |
| p4096/g1 | 3.190 | 1,284 | — | — | 5 |
| p8192/g1 | 6.584 | 1,244 | — | — | 5 |
| exact-128K p130944/g128 | 239.4 | 547 | 49.48 | 90.0% | 1 |

KV capacity at 128K config: 156,745 tokens (logged), 6.28 GiB KV, 529 MiB free
after load (isolated-C1 band). Exact-128K gate completed all 131,072 tokens —
the GDN partial-final-group boundary patch works on the dense path too.

**Realistic Pi flow (six scenarios, exact Pi system prompt, C1 sequential):**

| Scenario | Prompt | TTFT (s) | Cache hits | Hit % | Decode post-first (t/s) | MTP accept |
|---|---:|---:|---:|---:|---:|---:|
| cold new conversation | 595 | 0.491 | 0 | 0% | 48.98 | 49.4% |
| warm shared system prefix | 591 | 0.544 | 0 | 0%* | 55.37 | 59.9% |
| warm multi-turn session | 753 | 0.627 | 0 | 0%* | 44.56 | 44.0% |
| RAG payload on warm session | 930 | 0.776 | 0 | 0%* | 44.27 | 44.6% |
| warm system + cold 32K doc | 32,640 | 35.30 | 0 | 0% | 53.37 | 69.9% |
| resident 32K follow-up | 32,797 | **3.472** | **29,952** | **91.3%** | 51.79 | 68.3% |

\*Pi system prefix (~300 tokens) is shorter than the 1,088-token cache page, so
semantically warm short turns correctly show zero hits (same documented behavior
as the MoE campaign). Resident follow-up: 10.2× TTFT reduction from cache reuse;
new-token prefill rate ~819-925 t/s in both cold and cached cells — cache
eliminates tokens, it does not speed up prefill itself.

**Matched cache on/off resident-session comparison (coding-session workload;
prep a fresh 32,640-token Pi-prompt session, then 5 follow-up questions, C1,
n=5 per mode, exact 128-token output via ignore_eos):**

| Metric | Cache ON | Cache OFF | Speedup |
|---|---:|---:|---:|
| Follow-up TTFT median (s) | 3.728 | 35.363 | **9.5×** |
| Follow-up E2E median (s) | 6.526 | 38.593 | **5.9×** |
| Token reuse per follow-up | 29,952 / 32,720 (91.5%) | 0 / 32,765 (0%) | — |
| Decode post-first range (t/s) | 41-51 | 37-47 | ~same |

Novel-token prefill rate is ~740-925 t/s in both modes — caching eliminates
tokens, it does not accelerate prefill itself; decode is unaffected. Evidence:
`results/vllm-dense27-int4-resident-cache-on-20260809T121736Z-368981/` and
`...-cache-off-20260809T122443Z-372205/`. Cache-off server launched with
`--no-enable-prefix-caching` (verified `enable_prefix_caching=False` in log).

**Prefill-budget A/B (negative result):** relaunching with
`--max-num-batched-tokens 16384` (MTP4, fp8 KV) crashes the engine core on the
known GDN boundary bug — `causal_conv1d does not support spec-decode and
non-spec (prefill + decode) tokens in the same invocation` (same class as MoE
Run 26). Budget 8192 remains the only viable scheduler budget for MTP4 on this
pinned nightly. Evidence: `results/vllm-dense27-int4-budget-ab-b16384-*`.

**Context gates (fp8 KV):**

| Target | Result |
|---|---|
| 128K (131,072) | ✅ exact completion, MTP4, boundary patch applied |
| 200K (204,800) | ⚠️ loads (KV cap 207,684) but free=3 MiB after load → abort zone, not run |
| 256K (262,144) | ❌ VRAM-infeasible: 9.5 GiB KV needed > 7.82 GiB available; est. max 209,664 tokens (U=0.95). U=0.98 cannot init (driver reserves ~2.3 GiB above visible VRAM) |

**Interpretation:**
- Dense GPTQ-INT4 linear on XPU works and is fast: 73.17 t/s C1 decode at p512
  (85% MTP accept) vs ~24-29 t/s GGUF Q4/Q6-MTP — a 2.5-3× win over the llama.cpp
  MTP stack. Realistic Pi short-turn decode is 44-56 t/s (MTP accept 44-60% —
  real prompts less predictable than synthetic filler).
- Long-context prefill collapses (1,284 → 547 t/s at 128K): full-attention O(N²)
  term dominates; 4.1% of INT4 XMX peak vs 9.7% at p4096.
- Dense prefill is **compute-bound, not memory-bound**: ~71 TFLOPS at p4096 =
  ~10% of the 734 TOPS INT4 XMX peak; weight streaming is 3-48 GB/s against a
  608 GB/s ceiling (~1% at p4096). Levers per architecture doc: `-ze-opt-large-register-file`
  in the wheel, `--max-num-batched-tokens` 16384/32768 A/B, chunked-prefill config.
- "Fill more context": safe ceiling is 128K for dense 27B + MTP4 + fp8 KV.
  200K loads with zero headroom (abort); 256K impossible. Quantized-MTP head
  (~0.6 GB reclaim) would buy only ~2-3K tokens — not worth pursuing.
- Cache-hit semantics confirmed: hits eliminate tokens (91.3% hit → 10.2× TTFT);
  per-new-token prefill rate unchanged (~820-925 t/s).

**Complete decode generation-length matrix (client post-first t/s, ignore_eos,
cold, zero cache hits, C1, median of n=5; max observed in parens):**

| Prompt | g32 (peak) | g128 (representative) | g256 | g512 |
|---|---:|---:|---:|---:|
| p512 | 62.11 (80.62) | 68.65 (73.43) | 67.82 (71.17) | 60.83 (63.71) |
| p8192 | 73.97 (74.09) | 68.42 (73.47) | 64.70 (68.31) | 61.74 (72.06) |

MTP acceptance falls with generation length (p512: 75.0% g32 → 68.9% g512;
p8192: 100% g32 → 75.8% g512) — the same per-position draft compounding seen
on the MoE. Evidence: `results/vllm-dense27-int4-decode-sweep-20260809T114630Z-356008/`
(each cell used disjoint entropy-first prompt sets — the first g128 attempt
contaminated on reused prefixes and was correctly rejected by the zero-hit
guard; rerun with fresh per-cell prompts).

**Complete prefill/input matrix (actual endpoint tokens / client TTFT, g1):**

| pp512 | pp2048 | pp4096 | pp6144 | pp8192 | exact-128K p130944 |
|---|---:|---:|---:|---:|---:|---:|
| 1,322 | 1,317 | 1,284 | 1,124 | 1,244 | 547 |

**Power/temp provenance** (hwmon4 `xe`, `energy1_input` deltas, 0.5 s interval
average, configured cap 165 W during campaign): C1 matrix mean 164 W / max 197 W
/ VRAM max 74 °C; exact-128K gate mean 165 W / max 209 W / VRAM max 78 °C; Pi
flow mean 151 W / max 210 W / VRAM max 76 °C. Restored 150 W on exit.

**Note:** model is a heretic-v2 uncensored fine-tune, not stock Qwen; the
prompt-reuse caveats and non-public status apply. Older-generation tables are
unchanged.

---

## Run 22 — Multi-turn 128K with prefix caching (2026-08-06) — 28× warm TTFT

**Why:** Run 21 showed 40s TTFT at 128K and we framed it as "batch/RAG". But
that's only the COLD load. The real question: once the context is in KV cache,
are follow-ups fast? Initial test WITHOUT prefix caching: every turn re-prefilled
the full context (TTFT ~37s each turn — confirmed `enable_prefix_caching=False`,
`Prefix cache hit rate: 0.0%`).

**Fix:** add `--enable-prefix-caching` to the 128K launch.

**Multi-turn test (5 turns, full 122K context resident, MTP on, 150W):**

| Turn | Context tokens | TTFT | Decode t/s | Wall |
|------|---------------:|-----:|-----------:|-----:|
| 1 (cold load) | 122,531 | **39.6s** | 75.3 | 41.3s |
| 2 (warm) | 122,561 | **1.42s** | 78.2 | 2.2s |
| 3 (warm) | 122,587 | 1.43s | 82.4 | 2.2s |
| 4 (warm) | 122,611 | 1.40s | 82.4 | 2.2s |
| 5 (warm) | 122,636 | 1.42s | 81.9 | 2.2s |

**Result: warm follow-up TTFT = 1.42s at FULL 122K context — 28× faster than
the cold load.** Decode stays ~80-82 t/s. This makes 128K genuinely interactive
for multi-turn sessions: load the document once (40s), then chat with the whole
thing at ~1.4s response latency.

**This reframes Run 21's "batch/RAG" verdict:** 128K with prefix caching is a
usable interactive pattern for document-grounded conversation. Cold single-turn
is still slow (40s), but warm multi-turn is snappy.

**Flags now in the launch scripts:** `--enable-prefix-caching` is REQUIRED for
multi-turn 128K (without it, every turn re-prefills — no KV reuse).

**Artifacts:** `results/vllm-mtp-128k-multiturn-pc.json`,
`results/vllm-mtp-128k-multiturn-pc.out`; harness
`scripts/tmp/b70-multiturn-128k-test.py`; launch `scripts/tmp/launch-mtp-128k.sh`.

---

## Run 21 — vLLM MTP context scaling to 128K (2026-08-06)

**Why:** Does MTP + the patched GDN path hold up as context fills to 128K? Does
decode degrade, does prefill collapse, does it OOM?

**VRAM fit (measured):** Model 19.79 GiB + KV cache **7.75 GiB available** →
GPU KV cache size **349,869 tokens** (341K headroom — MoE's tiny 3B-active
attention makes KV nearly free, per AGENTS.md §9.1). 128K context fits with
213K tokens to spare. No OOM.

**Context-scaling sweep (single-stream, MTP on, 150W, gen=64):**

| Context | Prefill t/s | Decode t/s | TTFT | Wall |
|--------:|------------:|-----------:|-----:|-----:|
| 4K | 5,423 | **120.9** | 714ms | 1.2s |
| 10K | 7,098 | 107.5 | 1.4s | 2.0s |
| 20K | **7,325** | 116.0 | 2.6s | 3.2s |
| 40K | 5,877 | 100.0 | 6.6s | 7.2s |
| 65K | 4,418 | 104.7 | 14.3s | 14.9s |
| **128K** | **3,064** | **92.5** | **40s** | 40.7s |

**Findings:**
- **Decode degrades mildly**: 121 → 92 t/s (4K → 128K), **-24%**. Even at full
  128K context, MTP delivers ~92 t/s — still above the no-spec 73 t/s ceiling.
  Attention math grows with KV length but MoE's small attention keeps it cheap.
- **Prefill degrades harder past 20K**: peaks ~7.3K (20K ctx), falls to 3,064
  at 128K (**-58% from peak**). This is the O(n²) attention cost during prefill
  (building KV cache for every token). Still, 3K t/s at 128K beats llama.cpp's
  peak prefill of 1.7K.
- **128K TTFT = 40s**: fine for batch/RAG/document processing; painful for
  interactive chat. **≤32K is the interactive sweet spot** (TTFT < 7s).

**Launcher guidance:** vLLM MTP @128K is a **batch/RAG workload**, not an
interactive one. For the pi-telegram-bridge (single-user chat), llama.cpp
@128K dense (ThinkingCap Q6-MTP, the production profile) stays the right call —
lower TTFT, no patched-engine correctness risk. vLLM MTP wins the moment the
workload is long-context batch or multi-user API serving.

**Artifacts:** `results/vllm-mtp-128k-ctx-scaling.json`,
`results/vllm-mtp-128k-serve.log`; harness
`scripts/tmp/b70-ctx-scaling-sweep.py`; launch `scripts/tmp/launch-mtp-128k.sh`.

---

## Run 20 — MTP prefill fix: --max-num-batched-tokens 8192 (2026-08-06)

**Why:** Run 18-19's MTP prefill (~6.2-7.5K) was below Run 17's no-spec 9,094.
Root cause: MTP caps `max_num_scheduled_tokens` to **2048** by default, chunking
any prompt >2048 tokens. vLLM logged the warning but we missed it.

**Fix:** add `--max-num-batched-tokens 8192` to the launch flags. Warning gone,
chunked-prefill ceiling lifted.

**Result (single-stream, 150W, full grid):**

| Prompt | Decode (best g32) | Prefill t/s | vs Run 18/19 |
|---|---:|---:|---|
| short (55 tok) | **132.9** | 748 | decode +5% |
| p512 | 126.4 | 3,519 | — |
| p1k | 118.1 | 6,184 | — |
| p2k | 119.8 | **7,535** | prefill +21% |
| p4k | 123.4 | **8,484** | prefill +28% |
| p8k | 119.3 | **8,436** | prefill +12% |

**Decode up across the board** (133 short/g32, +5% over Run 18's 127).
**Prefill recovered 21-28%** on p2k-p4k; p8k best cell hits **8,718 t/s** —
back in Run 17's 9,094 ballpark. Remaining gap (p2k 7,535 vs 9,094) is the
genuine MTP draft-forward cost during prefill (unavoidable).

**This flag is now in the launch script + AGENTS.md + cookbook repo.** Without
it you silently lose 21-28% long-prompt prefill.

**Artifacts:** `results/moe-vllm-mtp-prefillfix-150w-grid.json`;
launch `scripts/tmp/launch-mtp-bf16draft.sh` (+ the cookbook copy).

---

## Run 19 — Engine + power sweep: vLLM MTP vs llama.cpp, MoE vs Dense (2026-08-06): vLLM MTP vs llama.cpp, MoE vs Dense (2026-08-06)

**Why:** Run 18 unlocked MTP at one operating point. This run maps the full
prefill×gen surface for both engines on MoE, adds a power sweep (150/230W) to
find the sweet spot, and tests the dense 27B on both engines.

### MoE 35B — full grid (single-stream, 150W sweet spot)
Decode best steady-state t/s — **vLLM GPTQ-Int4+MTP** vs **llama.cpp Q4_K_XL**:

| Prompt \ Gen | g32 | g128 | g256 | g512 |
|---|---|---|---|---|
| short | **127**/74 (1.73×) | **118**/72 (1.64×) | **113**/67 (1.67×) | **110**/72 (1.53×) |
| p512 | **121**/73 (1.66×) | **116**/72 (1.62×) | **115**/72 (1.61×) | **113**/72 (1.58×) |
| p1k | **113**/73 (1.55×) | **114**/64 (1.79×) | **114**/70 (1.62×) | **105**/70 (1.50×) |
| p2k | **111**/70 (1.58×) | **126**/69 (1.82×) | **116**/69 (1.67×) | **118**/63 (1.87×) |
| p4k | **130**/66 (1.97×) | **114**/65 (1.77×) | **116**/65 (1.80×) | **116**/65 (1.80×) |
| p8k | **126**/59 (2.14×) | **111**/58 (1.92×) | **114**/58 (1.97×) | **114**/58 (1.96×) |

Prefill t/s (avg): vLLM **5.6K–7.5K** vs llama.cpp **104–1,728** (vLLM **3.8–8.5×** faster).
vLLM MTP advantage *grows with prompt length* (1.5× short → 2.1× p8k).

### Power sweep (MoE p2k/g128, single-stream)
| Wattage | vLLM MTP decode | llama.cpp decode | temp peak |
|---|---:|---:|---:|
| **150W** | **125.7** | 69 | ~58°C |
| 230W | 115.4 | — | ~58°C |

**MoE sweet spot = 150W.** 230W gives -8% (noise/thermal); MoE self-limits to ~140W.
**Run MoE at 150W** — cooler, same speed, 80W less heat.

### Dense 27B — power sweep (llama.cpp Q4_K_M, single-stream)
vLLM dense FP8 **has no XPU kernel** — `KeyError: PlatformEnum.XPU` in
`choose_scaled_mm_linear_kernel` (confirmed 2026-08-06). llama.cpp is the only
working dense path.

| Wattage | short/g32 | p2k/g128 | temp peak |
|---|---:|---:|---:|
| 150W | 22 | 18 | 71°C |
| **230W** | **26** | **23** | 79°C |

**Dense scales with power** (+18-30% from 150→230W, unlike MoE). **Dense sweet spot = 180W**
(efficiency 0.155 t/s/W per AGENTS.md §6); 230W for bursts only (79°C sustained).
Dense runs much hotter than MoE (71-79°C vs 58°C) — denser weight reads.

### Cross-model summary (single-stream, sweet spot)
| Model | Engine | Config | Decode (p2k/g128) | Prefill (p2k) | Power | Temp |
|---|---|---|---:|---:|---:|---:|
| MoE 35B | **vLLM MTP** | GPTQ-Int4+MTP | **126** | **6,217** | 150W | 58°C |
| MoE 35B | llama.cpp | Q4_K_XL | 69 | 1,498 | 150W | 58°C |
| Dense 27B | llama.cpp | Q4_K_M | 23 | 1,007 | 230W | 79°C |
| Dense 27B | vLLM | FP8 | ❌ no kernel | — | — | — |

**Findings:**
1. MoE decode is **5-6× faster than dense** (126 vs 23 t/s) — MoE reads ~3GB/token.
2. vLLM MTP wins MoE (1.8× decode, 4.2× prefill over llama.cpp).
3. llama.cpp wins dense by default (vLLM has no dense XPU FP8 kernel).
4. Power: **MoE=150W (self-limits), Dense=180W sweet / 230W burst** (+18-30%, thermal cost).

**Artifacts:** `results/engine-comparison-full-20260806.md`, `results/moe-{vllm-mtp,llamacpp}-*-grid.json`,
`results/dense-llamacpp-q4km-{150,230}w-grid.json`, `results/dense-vllm-fp8-serve.log`;
harness `scripts/tmp/b70-moe-sweep{,-llamacpp}.py`.

---

## Run 18 — vLLM XPU MTP speculative DECODED (2026-08-06) — single-stream 123 t/s

**Why:** A14 (ngram) and the GPTQ-MTP load path were both *patchable*, not
fundamental. The MTP-preserved GPTQ checkpoint
(`llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4`)
stores `mtp.*` as BF16 fused experts; the XPU GDN kernel already takes explicit
spec tensors (`num_spec_decodes`, `spec_query_start_loc`, `spec_token_indx`,
`spec_state_indices_tensor`) — the boolean `spec_sequence_masks` it asserts on
is metadata-only and never reaches the SYCL kernel.

**Stack:** local-only `intel/vllm:0.21.0-xpu-int4moe` image (never published) + `patch_xpu_int4_moe_v4.py`
(native int4 target) + `patch_mtp_bf16_draft.py`:
  1. `Qwen3_5MultiTokenPredictor.__init__` / `Qwen3NextSparseMoeBlock` / `FusedMoE`
     strip `quant_config` for any `mtp` prefix → BF16 fused-expert draft.
  2. `xpu_moe.py` `XpuFusedMoe(...)` call stripped of `is_fp8` / `is_mxfp4`
     (kernel auto-detects dtype).
  3. `_gdn_attention_core_xpu_impl` assert → proceed-with-warning.

### Single-stream (streaming include_usage, Concurrent 1, 230W)
| Config | tg32 | pp2048 | tg128 | vs Reddit 145 / 7975 |
|---|---:|---:|---:|---|
| Native int4 v4 (no spec, Run 17) | 72.6 | 9,094 | 70.2 | baseline |
| **Native int4 v4 + MTP (Run 18)** | **123.3** | **7,261** | **113.1** | **85% / 91%** |
| Reddit claim | 145.5 | 7,975 | — | — |

- **Decode +70% over no-spec** (72.6 → 123). First vLLM XPU result to **beat
  llama.cpp MoE single-stream parity** (llama.cpp Q4_K_XL ~73 t/s, no usable
  MTP on MoE).
- 1.69× effective speedup ⇒ ~69% implied draft-token acceptance on hybrid GDN.
- `num_speculative_tokens=2` requested, clamped to 1 (checkpoint has 1 MTP
  layer) → 123 t/s is the **single-layer MTP ceiling**.

### Caveat
GDN spec assert removed without a correctness audit. Output coherent on spot
checks; **KL-divergence / acceptance-rate audit vs eager path required** before
any production use. A14's "XPU GDN incompatible with spec decode" verdict was
**wrong for MTP** — overcautious assert, not a kernel limit.

### Dense
Unchanged from Run 16-17: block-FP8 dequant only (~0.75 t/s).

**Artifacts:** `research/vllm-021-campaign-20260806.md` A15;
`results/vllm-021-mtp-native-int4-bench-v2.json`;
`results/vllm-native-mtp-bf16draft-serve.log`; launch
`scripts/tmp/launch-mtp-bf16draft.sh`; patches
`scripts/tmp/vllm-xpu-int4-patch/{patch_xpu_int4_moe_v4,patch_mtp_bf16_draft}.py`.

---

## Run 17 — vLLM native INT4 MoE unlocked (2026-08-06)

**Why:** A11 native path failed with `ptr_A.size(1) must match ptr_B.size(1)`. Root cause:
C++ int4 path requires `torch.int8` (`at::kChar`); v3 left `uint8` so kernel used BF16 layout checks.

**Stack:** local-only `intel/vllm:0.21.0-xpu-int4moe` image (never published) + `patch_xpu_int4_moe_v4.py` (implement_zp→int8,
skip re-convert, route MoeWNA16→XpuFusedMoe). Graphs PIECEWISE, gptq float16,
`--language-model-only`, model `Qwen3.6-35B-A3B-GPTQ-Int4`.

### Single-stream (streaming tg32/pp2048, Concurrent 1)
| Config | tg32 decode | pp2048 prefill | tg128 |
|---|---:|---:|---:|
| Triton GPTQ (Run 16) | 57.9 | 5.1–5.3K | ~55 |
| Native v4 @150–180W | **65.5–68.3** | **7.1–8.4K** | **65.6** |
| **Native v4 @230W** | **72.6** | **9,094** | **70.2** |
| Reddit claim | 145.5 | 7,975 | — |

- **Prefill target met/beaten** at 180–230W.
- **Decode single-stream ceiling ~73 t/s** = bandwidth parity with llama.cpp MoE on B70.
  145 Concurrent-1 still not reproduced; remains multi-user or different stack/measurement.

### Concurrency (native v4, 180W, max-num-seqs=64)
C1=64, C4=225, C8=424, C16=**694** wall-clock aggregate tok/s.

### Dense
Unchanged from Run 16: block-FP8 dequant only (~0.75 t/s).

**Artifacts:** `research/vllm-021-campaign-20260806.md` A13;
`results/vllm-021-native-int4-v4-*.json`; patch `scripts/tmp/vllm-xpu-int4-patch/patch_xpu_int4_moe_v4.py`.

**A14 follow-up:** ngram speculative (`num_speculative_tokens=5`) serves but crashes on first decode — XPU GDN kernel asserts `spec_sequence_masks is None`. No MTP tensors in GPTQ shards. Single-stream ceiling unchanged at 72.6 / 9,094.

---

## Run 16 — vLLM 0.21.0-xpu campaign close (2026-08-06)

**Goal:** Reproduce Reddit Dolboyob77 GPTQ-Int4 numbers (tg32 145 / pp2048 7975, Concurrent 1)
on Arc Pro B70; then port dense ThinkingCap 27B FP8.

**Stack:** `intel/vllm:0.21.0-xpu-full` (vLLM 0.21.1.dev18, torch 2.13.0+xpu, kernels 0.1.12.2).
Power 180W unless noted. OOM guards: stop llama-profile, VRAM/RAM checks, named containers.

### MoE — Qwen3.6-35B-A3B-GPTQ-Int4
- Working path: Triton `fused_moe_kernel_gptq_awq` + FLASH_ATTN PIECEWISE graphs.
- **Single-stream ceiling:** tg32 **57.9 t/s**, pp2048 **5,122–5,349 t/s** (not 145 / 7975).
- **Concurrency aggregate gen t/s:** C1=49.4, C4=98.4, C8=133.0, C16=182.5.
  Reddit 145 sits between C=8 and C=16 → multi-user aggregate, not Concurrent-1 single-stream.
- 230W single-stream: ~52.7 t/s (no help; MoE self-limits).
- Native `XpuFusedMoe` int4: A9 OOM (−6 to −7 GiB KV); A11 lean prep clears OOM then
  **profile shape crash** `ptr_A.size(1) must match ptr_B.size(1)` — GPTQ layout ≠ kernel.
  Triton GPTQ remains only working MoE path.

### Dense — ThinkingCap-Qwen3.6-27B-FP8 (block e4m3 128×128)
- Stock: `KeyError: PlatformEnum.XPU` — no entry in `_POSSIBLE_FP8_BLOCK_KERNELS`.
- Patch: `XPUFp8BlockDequantKernel` (runtime dequant + BF16 linear).
- Serve OK with `--language-model-only --max-model-len 2048 --gpu-memory-utilization 0.97 --enforce-eager`.
  Weights 27.57 GiB; KV 1.6 GiB free (10.5K tokens). Without LM-only: KV −1.75 GiB (vision encoder cache).
- **Decode ~0.75 t/s** (tg32 steady-state) — dequant path only; not competitive vs llama.cpp dense ~24–30 t/s MTP.

### Verdict
Single-stream Reddit 145 **not achieved** on this stack. Best MoE single-stream **57.9 t/s**.
145 is explained by multi-user aggregate throughput. Dense block-FP8 needs a real XPU
block kernel (not dequant) to be useful.

**Artifacts:** `research/vllm-021-campaign-20260806.md` A7–A10;
`results/vllm-021-gptq-concurrency.json`, `results/vllm-021-dense-fp8-bench.json`;
patch `scripts/tmp/vllm-xpu-fp8-block-patch/`.

## Run 15 — 2026-08-05 | Concurrency head-to-head: vLLM MXFP4 vs llama.cpp — the "150 t/s" claim EXPLAINED

**Why:** people say "vLLM is better" — the claims are aggregate multi-user throughput,
not single-stream. Measured the exact thing they measure: same HTTP API, same
prompt mix, 1/4/8/16 concurrent users, 512 gen tokens each, 165W cap, B70.
Engines: vLLM 0.17.0-xpu (patched, MXFP4 native ckpt) vs llama.cpp b10255 SYCL
(Q4_K_XL, --parallel 16, 2K ctx/slot). Wall-clock aggregate tokens/s.

| Users | vLLM agg tok/s | vLLM gen t/s | vLLM TPOT | llama agg tok/s | llama gen t/s | llama TPOT | agg Δ (llama/vLLM) |
|-------|----------------|--------------|-----------|-----------------|---------------|------------|--------------------|
| 1     | 18.0           | 10.4         | 96ms      | 112.9           | 65.3          | 15ms       | 6.3× (llama wins)  |
| 4     | 68.6           | 39.7         | 101ms     | 134.7           | 77.9          | 51ms       | 2.0× (llama wins)  |
| 8     | 134.4          | 77.8         | 103ms     | 159.5           | 92.4          | 87ms       | 1.2× (llama wins)  |
| 16    | 264.9          | **153.4**    | 104ms     | 146.8           | 85.0          | 188ms      | 0.55× (**vLLM wins**) |

**Findings (the answer to "why do people say vLLM is better"):**
- The Reddit "150 t/s" figure = vLLM at ~16 concurrent users. Measured: 153.4 gen t/s
  aggregate on OUR B70. People quote aggregate throughput, not single-stream speed.
- vLLM's continuous batching: aggregate scales ~linearly with users (10 → 153 gen t/s)
  while per-user latency stays FLAT (TPOT 96 → 104 ms). The FLA per-step cost is
  shared across the batch — GPU stays saturated.
- llama.cpp --parallel 16 serializes slot decode: aggregate SATURATES at ~85-92 gen t/s,
  per-user TPOT degrades linearly (15 → 51 → 87 → 188 ms).
- Single user: llama.cpp is 6.3× faster (65.3 vs 10.4 gen t/s) with 6× better latency.
- Community corroboration: PMZFX/intel-arc-pro-b70-benchmarks engine-comparison —
  vLLM XPU wins prefill 2.4-15× (XMX/DPAS FA); decode ~tie on non-hybrid models;
  llama.cpp wins quantized memory + model coverage (vLLM XPU fails on Qwen 3.5 GDN).
  Level1Techs: Qwen3-30B-A3B FP8 TP2 dual B70 = 912 out tok/s @ 50 concurrent.
- Production implication: pi-telegram-bridge (single user) = llama.cpp is correct.
  Multi-user API serving (≥8 concurrent) = vLLM XPU MXFP4 becomes competitive;
  note UD hybrid model caps vLLM decode (FLA triton kernels) — a non-hybrid model
  would widen vLLM's concurrency advantage further.

Artifacts: results/vllm-mxfp4-concurrency-20260805.json, results/llamacpp-concurrency-20260805.json.

---
## Run 14 — 2026-08-05 | vLLM 0.17.0-xpu MXFP4 — SERVED (7 patches) but decode 7x slower than llama.cpp

**Why:** Run 13's verdict was "dead end on 0.14 image". Retried with the newer
`intel/vllm:0.17.0-xpu` (vllm-xpu-kernels v0.1.4) + the native MXFP4 checkpoint
(/mnt/<models-disk>/Qwen3.6-35B-A3B-MXFP4-native, 22.4 GB). Model SERVES and generates
correct output after 7 in-container engine patches:

| # | File | Bug | Fix |
|---|------|-----|-----|
| A | fused_moe/layer.py | mxfp4 branch expects combined 3D (gpt-oss); qwen3_5 slices per-expert 2D → IndexError | 2D per-expert copy + w1/w3 half-offset (w3 = second half of fused gate_up) |
| B | models/qwen3_5.py | fused_expert_params_mapping has no scale entries → scale tensors hit generic 2-arg loader → TypeError | add w13/w2_weight_scale (+input_scale) mapping entries |
| C | quantization/mxfp4.py | assert swiglu_oai-only (PR #22700) blocks silu models | allow SILU (kernel supports it) |
| D | mxfp4_utils.py | `_can_support_mxfp4` same swiglu_oai gate | allow SILU |
| E | fla/ops/utils.py | `torch.cuda.device` on XPU build → "PyTorch compiled without CUDA" | use torch.xpu.device for xpu tensors |
| F | vllm_xpu_kernels/flash_attn_interface.py | maybe_contiguous only checks last stride → "k must be contiguous" | use is_contiguous() |
| G | model_executor/models/config.py | hybrid alignment sets FA block 1056; XPU FA kernel supports 64/128 only → "Unsupported page size for fmha" | force block_size=64, pad mamba page UP to multiple of attn page (vllm PR #37467 behavior) |

Launch flags that matter: `--block-size 64` (REQUIRED), `--enforce-eager`,
`--quantization mxfp4`. Patch set: `scripts/tmp/vllm017-full-patch.py`.

**Output quality:** VERIFIED CORRECT — coherent reasoning traces, math (17×23), knowledge
(Tokyo), long prose. Weights/scales/activation layout all correct (w1/w3 offset matters!).

**Benchmark (wall-clock, 4 prompts × 3 reps + isolated tests, 165W, temp 62°C, 32.7GB VRAM):**

| Metric | vLLM 0.17 MXFP4 | llama.cpp b10255 Q4_K_XL (Run 9, 150W) | Δ |
|--------|----------------|----------------------------------------|---|
| Decode | 10.4 t/s | 72.6 t/s | **7.0× slower** |
| Prefill (pp4K) | ~1738 t/s | 2128 t/s | −18% |

- Decode is per-step kernel-bound: the hybrid UD model's linear-attention layers run
  GENERIC triton chunk kernels on XPU (no hand-tuned SYCL) → ~96 ms/step cap.
- Prefill is batched → triton kernels efficient → within 20% of llama.cpp.
- First-call JIT: ~4 min; steady ~49 s per 1.5K-prompt + 512-gen request.
- Reddit claim (150 tg / 11K pp vLLM MXFP4 on B70) NOT reproducible on this stack.

**Final verdict:** vLLM XPU MXFP4 = prefill parity, decode 7× worse than llama.cpp SYCL.
llama.cpp remains production. Artifacts: results/vllm-mxfp4-bench-20260805.json,
results/vllm-mxfp4-summary-20260805.md, scripts/tmp/vllm017-full-patch.py,
checkpoint /mnt/<models-disk>/Qwen3.6-35B-A3B-MXFP4-native.

---

## Run 13 — 2026-08-05 | vLLM XPU MXFP4 path — TESTED, verdict: not usable on this stack

**Why:** community claims vLLM XPU gives 10,000+ pp / 150 tg on Qwen3.6-35B-A3B (MXFP4)
on B70. Tested the full path with the kept `intel/llm-scaler-vllm:0.14.0-b8.3.2` image.

**What was done (all artifacts on /mnt/<models-disk>):**
1. Image boots on B70 with /dev/dri passthrough (VLLM_TARGET_DEVICE=xpu) - engine inits.
2. FP8 checkpoint (Qwen/Qwen3.6-35B-A3B-FP8) = 37.5 GB > 32 GB VRAM - impossible.
3. Community MXFP4 (`pahajokiconsulting/...-MXFP4`, 23 GB, compressed-tensors, 256 experts
   per layer, per-expert unfused naming) - incompatible with the native mxfp4 path.
4. Built a NATIVE-format checkpoint ourselves (BF16 dense 5.9 GB + fused MXFP4 experts
   16.5 GB = 22.4 GB fits): fused per-expert gate/up -> w13_weight_packed, verified the
   MXFP4 dequant empirically against BF16 ground truth (MSE 8.8e-5; nibble low-first,
   E8M0 bias 127). Script: scripts/tmp/b70-mxfp4-merge.py. Output:
   /mnt/<models-disk>/Qwen3.6-35B-A3B-MXFP4-native (config quant_method=mxfp4).
5. Load attempt with `--quantization mxfp4` exposed ENGINE BUGS in this build
   (llm-scaler 0.14.0-b8.3.2):
   a. fused_moe mxfp4 branch expects the full 3D fused tensor, but qwen3_5.py slices per
      expert (2D) - `IndexError: tuple index out of range`. Source has
      "(FIX-ME) for gpt-oss all experts are combined" - path written for gpt-oss only.
   b. After patching the loader, scale keys fail: fused_expert_params_mapping has NO
      scale entries, so `down_proj.weight_scale` maps to a non-existent param
      (`w2_weight.weight_scale` KeyError). No checkpoint key naming can satisfy it.

**Verdict:** the 150 tg / 11 K pp MXFP4 claims require a NEWER vLLM XPU build (the
qwen3_5 + mxfp4 MoE integration landed after this image) plus Intel's llm-compressor
quantization toolchain. On the available stack the path is a dead end without a
multi-hour source build. llama.cpp SYCL (72.6 t/s MoE, MTP-4 dense) remains production.
BF16 source (67 GB) + native-MXFP4 artifact (22.4 GB) kept on /mnt/<models-disk> for a future
test with a newer vLLM build.

---
## Run 12 — 2026-08-05 | Fleet MTP-4 dense @200W + VRAM ceiling findings + community 81 t/s analysis

**Why:** localmaxxing dense submissions must use the fleet quants (Q5_K_M ACTIVE, Q6_K
production) with MTP-4 — not the Run 9 A/B model (Q4). Also analyzed a community B70
run claiming 81 t/s MoE decode.
**Method:** llama-server 0804 + `--spec-type draft-mtp --spec-draft-n-max 4`, 200W cap,
128K ctx, q8_0/q4_1 KV. Engine rate (`timings.predicted_per_second`), 4 diverse
~1.5K-token prompts × 3 reps, steady-state = reps 2-3. Client:
`scripts/tmp/b70-mtp-dense-client.py`. Raw: `results/mtp-{q4,q5,q6}-*.json` +
`mtp-{dense,fleet,q6}-session-20260805.log`.

### Dense MTP-4 decode (all ThinkingCap 27B, 128K ctx, 200W)

| Quant | Decode (MTP-4) | VRAM free after load |
|-------|----------------|----------------------|
| Q4_K_M | 29.75 t/s | — |
| Q5_K_M-MTP | 26.3 t/s | 3001 MiB |
| Q6_K-MTP | 27.85 t/s | 404 MiB (works; production runs this config) |

### VRAM ceiling findings (MTP spec buffers exceed the llama-bench boundary table)

- **Q4_K_M @ 256K + MTP = OVERFLOW** (0 MiB free → garbage numbers: decode 0.07-20 t/s,
  prefill 4-18 t/s). §9.3's "0.9 GB free @256K" was measured without MTP buffers.
- **Q5_K_M-MTP @ 200K + MTP = OVERFLOW** (user-confirmed "too tight"; ~0 free).
- **Dense + MTP-4 ceiling = 128K ctx** for all quants (Q6 fits with only 404 MiB spare).
- On this driver, VRAM overflow degrades silently (no OOM error) and the Q4@256K episode
  ended in a **system-wide OOM shutdown** — treat <1 GiB free as unsafe.

### Community "81 t/s" B70 run (Windows, analyzed — NOT adopted)

`llama-bench -p 4096 -n 128 -d 0 -r 2 -b 8192 -ub 1024 -t 15 -ctk f16 -ctv f16 -fa on
-ngl 99 -ncmoe 0 -nopo 1` on Qwen3.6-35B-A3B-UD-Q5_K_XL, "Turbo StateTree" custom
package (Windows 11, oneAPI 2026).
- **The math checks out — no VRAM miracle:** their row's "Context 262,144" is the
  model's `n_ctx_train` (metadata), not the bench ctx; llama-bench without `-c` benches
  at the prompt length (4K). F16/F16 KV on MoE = ~4.7 GiB per 128K (~2.56× q8_0/q4_1):
  at 262K it would need ~9.2 GiB KV + ~25 GB weights = 34+ GiB — does NOT fit 32 GB.
- F16 KV is why #25222 XMX FA's full potential needs ~4.7 GiB/128K on MoE; q8_0/q4_1
  (1.84 GiB/128K) is what enables 512K. **Fleet standard kept.**
- Adoptable-but-unmeasured here: `-ub 1024`, `-t 15` pinning, `-nopo 1`, Q5_K_XL quant
  (no GGUF indexed on HF). Not adopted — user preference: keep our config.

### Submissions made (localmaxxing, all APPROVED)

- Q5_K_M-MTP dense: 26.3 t/s → `cmsfsouer00ygpp01uhyfa9ck` (128K, 200W)
- Q6_K-MTP dense: 27.85 t/s → `cmsfsq8u300ykpp01p5y8g9l2` (128K, 200W)
- Q4_K_M-MTP dense (29.75) deleted — Q4 not a fleet quant
- MoE Q4_K_XL 72.6/2128 (Run 9) kept — best measured; 128K ctx would be 67.3

---

## Run 11 — 2026-08-04 | Long-context extension: pp65536 / pp131072 (b10222 vs master 0804)

**Why:** extend the Run 9 A/B to 64K/128K prefill — the regime where the TILE path's
O(n²) collapse should make #25874's XMX FA win the most.
**Method:** same config as Run 9 (`-p 65536,131072 -n 128 -r 2 -b 8192 -ub 4096
-ctk q8_0 -ctv q4_1 -fa on -mmp 0 -t 8 -dev SYCL0`), 150W. Reps r=2 (long prompts;
~90 min session total). Cooldown threshold relaxed to ≤56°C — idle temp pins at
53-54°C in current ambient and the standard <52°C rule never trips; metric is t/s,
not temperature (deviation documented, no other protocol changes). tg128 is captured
after the 128K-prompt tests (KV carries full context). Script:
`scripts/tmp/b70-longctx-sweep.sh` (scratch; promote if reused).
**Raw:** `results/longctx-{0804,0801}-{m,d}-20260804.txt` + `results/longctx-session-20260804.log`.

### MoE — Qwen3.6-35B-A3B Q4_K_XL (20.81 GiB)

| Test | b10222 | master 0804 | Δ |
|------|--------|-------------|---|
| pp65536 | 985.77 ± 1.98 | **1503.76 ± 7.82** | **+52.5%** |
| pp131072 | 673.45 ± 0.19 | **1210.79 ± 1.76** | **+79.8%** |
| tg128 (after 128K prompt) | 71.50 ± 0.18 | 67.28 ± 0.01 | — (see note) |

### Dense — ThinkingCap-27B Q4_K_M (15.65 GiB)

| Test | b10222 | master 0804 | Δ |
|------|--------|-------------|---|
| pp65536 | 417.08 ± 0.22 | **651.30 ± 0.26** | **+56.2%** |
| pp131072 | 287.61 ± 0.03 | **545.94 ± 0.07** | **+89.8%** |
| tg128 (after 128K prompt) | 18.65 ± 0.28 | 18.99 ± 0.27 | +1.8% |

**Key findings:**
- **The TILE→XMX win grows with context, exactly as predicted.** MoE +52% @64K →
  +80% @128K; dense +56% @64K → **+90% @128K** (287.61 → 545.94 t/s, nearly 2×).
- Prefill declines from the ~4K peak on both builds (KV write/attention growth), but
  0804 declines gracefully (2085 @8K → 1211 @128K) while b10222 collapses (1620 @8K →
  288-673 @128K).
- Decode with full 128K KV: dense drops ~10% vs short-ctx on BOTH builds (18.99/18.65
  vs 21.3/20.8) — consistent, likely the KV-read cost at 128K. MoE: b10222 71.50
  (-1.5% vs Run 9's 72.6) but 0804 67.28 (-7.3%) — flag as outlier (first run of the
  session: JIT warmup + hottest); do not over-interpret a single point.

---

## Run 10 — 2026-08-04 | OVMS dense RE-VERIFY on master 0804 (post-#25874)

**Why:** Run 8's OVMS-dense verdict (OVMS wins dense) used llama.cpp b10222. After the
0804 promotion, re-measure OVMS dense INT4 vs the new build's dense numbers (Run 9).
**Models:** OVMS `OpenVINO/Qwen3.6-27B-int4-ov` (INT4, gs128, 15 GB, /mnt/<models-disk>).
**Method:** same as Run 8 — OVMS 2026.2.1-gpu docker, `/v3/chat/completions`,
`enable_thinking=false`, 4 prompts × 3 reps (decode), 1592-token prompt × 3 (prefill),
3 streamed (TTFT). llama.cpp dense ref = Run 9 (llama-bench, 150W). **Power:** 150W.

| Metric | OVMS dense (re-verify) | llama.cpp 0804 | Winner |
|--------|----------------------|----------------|--------|
| **Decode (t/s)** | **27.5** (27.1-27.6) | 21.3 base / 24-29 MTP-4 | OVMS +29% over base |
| **Prefill** | ~915 t/s @1592 tok (steady-state; rep1 530 = JIT warmup) | **936 t/s** @ pp4096 | ~tie → llama.cpp at longer prompts |
| TTFT | 213 ms | — | — |

**Verdict update:** dense decode still favors OVMS (+29% over llama.cpp base; ~tie with
llama.cpp+MTP-4) — unchanged from Run 8 (29.8 vs 27.5 now, session variance). The dense
**prefill** gap is gone: 0804's #25874 XMX FA (936 @ pp4096) now matches/edges OVMS
(~915 @ 1.6K). OVMS remains the dense-decode engine; llama.cpp is now competitive on
both prefill and (with MTP) decode. Raw: `results/ovms-dense-reverify-20260804.txt` (if saved).

---

## Run 9 — 2026-08-04 | b10222 vs master(0804) | llama.cpp SYCL upgrade A/B (XMX quantized-KV FA)

**The build upgrade.** Rebuilt llama.cpp SYCL from upstream master (`071327508`, b10255+) in
`build-sycl-0804`, same config as b10222 (DNN=ON, F16=ON, GRAPH=ON, oneAPI 2026.0). Adds:
- **#25874** — oneDNN SDPA extended to **quantized KV** (Q4_0-Q8_0, FP32) → XMX flash attention on our q8_0/q4_1 cache
- **#25880** — fix SDPA scale use-after-return (multi-turn corruption) + restores no-sync single-GPU fast path
**Method:** `llama-bench -p 512,4096,8192,32768 -n 128 -r 3 -fa 1 -ctk q8_0 -ctv q4_1 -b 8192 -ub 4096 -mmp 0 -t 8 -dev SYCL0`
**KV:** q8_0 K + q4_1 V (fleet standard). **Power:** 150W.

### MoE — Qwen3.6-35B-A3B Q4_K_XL (20.81 GiB)

| Test | b10222 | master 0804 | Δ |
|------|--------|-------------|---|
| pp512 | 1061 | 1134 | +6.9% |
| **pp4096** | 1691 | **2128** | **+26%** |
| **pp8192** | 1620 | **2085** | **+29%** |
| **pp32768** | ~780 (TILE) | **1871** | **+140%** |
| tg128 | 70.1 | **72.6** | +3.6% |

### Dense — ThinkingCap-27B Q4_K_M (15.65 GiB)

| Test | b10222 | master 0804 | Δ |
|------|--------|-------------|---|
| pp512 | 640 | 685 | +7% |
| **pp4096** | 795 | **936** | **+18%** |
| **pp8192** | 758 | **921** | **+21%** |
| tg128 | 20.8 | **21.3** | +2.4% |

### Consolidated delta — all measures (b10222 → master 0804)

Same build config, same day, 150W, q8_0 K + q4_1 V KV, llama-bench. All values t/s.

| Model | Measure | b10222 | master 0804 | Δ |
|-------|---------|--------|-------------|---|
| **MoE 35B Q4_K_XL** | prefill pp512 | 1061 | 1134 | +6.9% |
| | prefill pp4096 | 1691 | **2128** | **+26%** |
| | prefill pp8192 | 1620 | **2085** | **+29%** |
| | prefill pp32768 | ~780 | **1871** | **+140%** |
| | decode tg128 | 70.1 | **72.6** | +3.6% |
| **Dense 27B Q4_K_M** | prefill pp512 | 640 | 685 | +7% |
| | prefill pp4096 | 795 | **936** | **+18%** |
| | prefill pp8192 | 758 | **921** | **+21%** |
| | decode tg128 | 20.8 | **21.3** | +2.4% |

**Key findings:**
- **#25874 is the big one.** XMX flash attention on quantized KV gives +18-29% prefill at 4K-8K, and **+140% at 32K** (1871 vs ~780 TILE). The win scales with context length.
- Decode improves modestly (+3.6% MoE, +2.4% dense) — still bandwidth-bound, near ceiling.
- Tuning env (`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`, `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`) → **no measurable change** at bench level. The XMX FA does the work.
- **Recommendation: promote `build-sycl-0804` to production.** It's faster everywhere AND fixes the multi-turn corruption bug (#25880) present in b10222.

---

## Run 8 — 2026-08-04 | OVMS/OpenVINO vs llama.cpp b10222 | DENSE engine A/B (Qwen3.6-27B)

**Cross-engine head-to-head on DENSE models.** Complementary to Run 7 (MoE).
**Models:** llama.cpp `ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf` vs OVMS `OpenVINO/Qwen3.6-27B-int4-ov` (INT4, gs128).
**Method:** llama.cpp via `llama-bench` (engine rate); OVMS via `/v3/chat/completions` (wall, enable_thinking=false).
**Note:** llama.cpp dense decode shown is BASE (no MTP). With MTP-4 production reaches ~24-29 t/s.

| Metric | llama.cpp SYCL (base) | OVMS/OpenVINO | Winner |
|--------|----------------------|---------------|--------|
| **Decode (t/s)** | 20.8 | **29.8** | OVMS +43% |
| Prefill pp512 | 640 | — | — |
| Prefill pp4096 | 795 | — | — |
| Prefill ~1.6K | ~800 | ~900 | ~tie |
| TTFT | — | 195 ms | — |
| VRAM after load | ~21 GB (Q6) | 17 GB | OVMS lighter |

**KEY FINDING: dense flips the MoE result.** OVMS INT4 dense decode (29.8 t/s) beats llama.cpp base dense (20.8 t/s) by +43%, because INT4 dense reads fewer bytes/token on a bandwidth-bound dense model. llama.cpp closes most of the gap only with MTP-4 (24-29 t/s). For dense-only workloads, OVMS is now the faster single-stream engine.

**Pattern summary (Run 7 + Run 8):**
- **MoE:** llama.cpp wins decode (+20%). OVMS wins thermals/VRAM/concurrency.
- **Dense:** OVMS wins decode (+43% over llama.cpp base; ~equal to llama.cpp+MTP).
- Recommendation: keep llama.cpp as the default engine; consider OVMS specifically for dense/multimodal or high-concurrency serving.

**UPDATE after Run 9 (master 0804 promoted, 2026-08-04):** llama.cpp dense prefill
improves to **936 t/s @ pp4096** (+18% over the 795 shown above) on build-sycl-0804
(#25874 quantized-KV XMX FA), now **beating OVMS's ~900 t/s** — dense prefill verdict
flips to llama.cpp. Dense decode unchanged in spirit (21.3 base on 0804 vs OVMS 29.8):
OVMS remains the faster dense single-stream decoder; llama.cpp+MTP-4 stays ~tie.
Re-verified on 0804: see `benchmarks/ovms-vs-llamacpp-qwen35b-moe-2026-08-04.md` and
Run 10 (if appended).

---

## Run 7 — 2026-08-04 | OVMS/OpenVINO vs llama.cpp b10222 | engine A/B (Qwen35B MoE)

**The first cross-engine head-to-head on B70.** llama.cpp SYCL (production) vs OpenVINO Model Server 2026.2.1-gpu.
**Models:** llama.cpp `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` vs OVMS `OpenVINO/Qwen3.6-35B-A3B-int4-ov` (INT4).
**Method:** llama.cpp via `llama-bench` (engine rate); OVMS via `/v3/chat/completions` (wall, enable_thinking=false).
**Report:** `benchmarks/ovms-vs-llamacpp-qwen35b-moe-2026-08-04.md`

| Metric | llama.cpp SYCL | OVMS/OpenVINO | Winner |
|--------|----------------|---------------|--------|
| **Decode (t/s)** | **70.1** | 58.4 | llama.cpp +20% |
| Prefill pp512 | 1061 | — | — |
| Prefill pp4096 | 1691 | — | — |
| Prefill ~1.6K | ~1600 | 1474 | ~tie |
| TTFT | ~69 ms | 195 ms | llama.cpp |
| Sustained temp | 63-70°C | **58-62°C** | OVMS cooler |
| VRAM after load | ~24.6 GB | 20.8 GB | OVMS lighter |

**Verdict:** llama.cpp SYCL stays production for single-stream inference (decode +20%, faster TTFT). OVMS wins on thermals, VRAM, concurrency, and native vision — worth revisiting only for multi-client serving or prefix-caching-heavy agent workloads. vLLM/SGLang XPU not recommended for single-stream (vLLM TPOT regression on Qwen3-A3B; SGLang source-only/experimental).

---

## Run 6 — 2026-08-03 | b10222 | boundary sweep (VRAM/context limits)

**The definitive VRAM boundary map.** Tests every quant at every context to find
exactly where each config fits and where it OOMs.
**Method:** `llama-bench -p 512,4096 -n 128 -r 3 -fa 1 -ctk q8_0 -ctv q4_1 -b 8192 -ub 4096`
**Raw data:** `results/boundary-sweep-20260803-1744.txt` | **Report:** `benchmarks/boundary-sweep-2026-08-03.md`

### ThinkingCap 27B (dense + MTP)

| Quant | Context | VRAM free | pp4096 | tg128 (base¹) | Verdict |
|-------|---------|-----------|--------|-------------|---------|
| Q4_K_M | 256K | 930 MiB | 700 t/s | 18.4 t/s | ✅ TIGHT |
| Q5_K_M | 128K | 3250 MiB | 706 t/s | 16.2 t/s | ✅ Comfortable |
| Q5_K_M | 200K | 2677 MiB | 706 t/s | 16.3 t/s | ✅ WORKS (better than predicted!) |
| Q6_K | 128K | 687 MiB | 715 t/s | 16.0 t/s | ✅ TIGHT (absolute max) |

¹ tg128 = base decode WITHOUT MTP. With MTP-4 via server: +35-50% → ~24-29 t/s.

### Qwen 35B MoE

| Quant | Context | VRAM free | pp4096 | tg128 | Verdict |
|-------|---------|-----------|--------|-------|---------|
| Q4_K_XL | 512K | 1593 MiB | **1659 t/s** | **70.2 t/s** | ✅ Context champion! |
| Q5_K_M | 512K | — | — | — | ❌ OOM CRASH |

**Key findings:**
- Q5_K_M @ 200K works (2677 MiB free) — linear model was too pessimistic
- Q6_K @ 128K is the absolute quality ceiling (687 MiB — zero headroom)
- MoE Q4 @ 512K = best context+throughput config (1593 MiB free, 1659 t/s prefill)
- Dense decode is flat ~16-18 t/s across quants (bandwidth-bound; quality differs, not speed)

---

## Run 5 — 2026-08-03 | b9853 vs b10222 A/B | llama-bench (VALID prefill)

**The definitive A/B: same model, same config, two builds, llama-bench (synthetic prompts).**
**Method:** `llama-bench -p 512,4096,8192 -n 128 -r 5 -d 0 -fa 1 -ctk q8_0 -ctv q4_1 -b 8192 -ub 4096`
**Model:** Qwen3.6-35B-A3B Q4_K_XL (MoE), 128K context, 150W
**Raw data:** `results/ab-llama-bench-20260803-1648.txt`

### Results

| Test | b9853 (baseline) | b10222 (current) | Improvement |
|------|-----------------|------------------|-------------|
| **pp512** | 963 ± 8.9 t/s | **1041 ± 6.5 t/s** | **+8.1%** |
| **pp4096** | 1525 ± 6.0 t/s | **1621 ± 12.2 t/s** | **+6.3%** |
| **pp8192** | 1375 ± 6.8 t/s | **1551 ± 8.5 t/s** | **+12.8%** |
| **tg128** (decode) | 61.8 ± 0.2 t/s | **69.5 ± 0.1 t/s** | **+12.5%** |

### Verdict: upstream SYCL improved BOTH prefill AND decode meaningfully

The b9853→b10222 delta (169 commits) delivered:
- **Prefill: +6.3% to +12.8%** — the oneDNN XMX FA (#25222) and oneMKL GEMM FA (#25025) commits are real and measurable. Improvement **scales with prompt length** (longer prompts benefit more from the fused attention kernels).
- **Decode: +12.5%** — surprisingly large; likely from the fused top-k MoE (#25217) + RMS_NORM fusion (#26015) + contiguous elementwise fast path (#25946).
- pp4096 on b10222 (1621 t/s) matches your localmaxxing-submitted 1601 t/s (which used Q5 weights + PR25690 patch on b10053 — Q4 is slightly faster).

**Note:** b9853 `mmap=0`, b10222 shows `mmap=none` (the `--no-mmap`/`-mmp 0` flag was deprecated between builds — `mmap` column changed name but behavior is identical: model loaded to VRAM, no paging).

---

## Run 4 — 2026-08-03 | b10222 | all-model 128K+256K re-bench

**Build:** b10222 (`a7a6d0d26`, `build-sycl-0801`, built 2026-08-01)
**Engine:** llama.cpp SYCL, IntelLLVM 2026.0.0
**Method:** 4 diverse prompts, warmup discarded, engine decode rate, cooldown ≤55°C
**KV:** q8_0 K + q4_1 V
**Raw data:** `results/all-models-128k-256k-20260803-1452.txt`
**Context:** See also `b70-2026-08-03-longcat-llamacpp-lvm-bench.md`

| Model | Quant | Type | Context | Power | Decode t/s | Prefill t/s¹ | VRAM free |
|-------|-------|------|---------|-------|-----------|--------------|-----------|
| Qwen3.6-35B-A3B | Q4_K_XL | MoE | 128K | 150W | 69.8 | 140² | 7950 MiB |
| Qwen3.6-35B-A3B | Q4_K_XL | MoE | 256K | 150W | 70.0 | 143² | 6103 MiB |
| Qwen3.6-35B-A3B | Q5_K_M | MoE | 128K | 150W | 67.9 | 132² | 4061 MiB |
| Qwen3.6-35B-A3B | Q5_K_M | MoE | 256K | 150W | 67.7 | 144² | 2197 MiB |
| ThinkingCap Qwen3.6-27B | Q4_K_M | Dense+MTP | 128K | 180W | 29.0 | 76² | 6319 MiB |
| ThinkingCap Qwen3.6-27B | Q4_K_M | Dense+MTP | 256K | 180W | 29.4 | 63² | 1135 MiB |
| Laguna-S-2.1 | IQ2_XXS | Dense | 128K | 150W | — | — | CRASH³ |

¹ **Prefill numbers are INVALID** — measured with 34-43 token prompts (overhead-dominated).
   See `benchmark-protocol.md §1.5`. Real prefill is 5-12× higher. Do NOT cite these.
   Correct prefill data from Run 3 below (long prompts).
² These prefill values are overhead-amplified and should not be compared to Run 3.
³ 34 GB model > 32 GB VRAM → hard system crash. See `benchmark-protocol.md §7`.

**Key findings:**
- Context scaling penalty = ZERO (128K → 256K decode within noise)
- b10222 is +1–1.7% faster on MoE than documented b9853 baseline
- Dense 27B @ 256K works but VRAM-tight (1135 MiB headroom)

---

## Run 3 — 2026-07-16 | b10053 | clean-suite TG/PP matrix

**Build:** b10053 (`0dc74e332`, `build-sycl-0717`)
**Engine:** llama.cpp SYCL
**KV:** q8_0 K + q4_1 V
**Source:** `inference/BENCHMARKS-TG-PP-2026-07-16.md`, blog post
[arc-pro-b70-gemma31-mtp-moe-clean-suite](https://sergiiob.dev/posts/arc-pro-b70-gemma31-mtp-moe-clean-suite/)

| Model | Quant | Type | Context | Power | Decode t/s | Prefill t/s¹ |
|-------|-------|------|---------|-------|-----------|--------------|
| Ornith 35B | Q5 | MoE | 256K | 150W | 69.2 | 1300 (~4K prompt) |
| Qwen 35B | Q5 | MoE | 256K | 150W | 61.5 | 1690 (~3K prompt) |
| Qwen 35B | Q4 | MoE | 256K | 150W | 62.7 | 1682 (~4K prompt) |
| Qwen 27B | Q5 | Dense+MTP | 256K | 165W | 25.1 | 1359 (~4K prompt) |
| Gemma 4 31B | Q4 | Dense+MTP | 256K | 165W | 24.8 | 361 (~4K prompt) |

¹ Prefill measured correctly with long prompts (2000-4000 tokens). These are valid.
   Short-prompt prefill (458-621 t/s) shown in the blog is overhead-inflated.

**Key findings:**
- "B70 SYCL MoE is the prefill card" — 1.2k-1.7k t/s on long prompts
- MTP-4 gives +51% decode on dense Gemma 31B (16.4 → 24.8 t/s)
- Vision mmproj overhead "basically free" on text path

---

## Run 2 — 2026-07-05 | b10053 | engine sweep (per-watt telemetry)

**Build:** b10053
**Source:** `docs/b70-engine-sweep-results-2026-07-05.md`
**Note:** Rigorous per-watt card-power telemetry + vLLM/OVMS scout status.

| Model | Quant | Context | Power | Decode t/s | Temp |
|-------|-------|---------|-------|-----------|------|
| Qwen 35B Q4 | 256K | 150W | 68.8 | 66°C |
| Qwen 35B Q5 | 256K | 150W | 67.1 | 67°C |
| Ornith 35B Q4 | 256K | 150W | 73.7 | 66°C |
| Gemma 4 26B Q4 | 256K | 150W | 59.6 | 63°C |

---

## Run 1 — 2026-07-01 | b9853 | baseline (lmx-submitted)

**Build:** b9853 (`7af4279f4`, `build-sycl-latest-l0`)
**Engine:** llama.cpp SYCL
**KV:** q5_0-q4_1 (pre-q8_0-upgrade)
**Source:** lmx submissions to localmaxxing.com (APPROVED)

| Model | HF ID | Quant | Context | Power | Decode | Prefill | TTFT |
|-------|-------|-------|---------|-------|--------|---------|------|
| Qwen 35B | Qwen/Qwen3.6-35B-A3B | Q5_K_M | 256K | 150W | 67.0 | 464.9 | 69ms |
| Qwen 27B MTP | Qwen/Qwen3.6-27B | Q5_K_M | 128K | 230W | 30.4 | 102.4 | 312ms |
| Ornith 35B | deepreinforce-ai/Ornith-1.0-35B | Q4_K_M | 256K | 150W | 74.3 | 440.8 | 73ms |
| Agents-A1 | InternScience/Agents-A1 | Q4_K_M | 256K | 150W | 73.2 | 416.1 | 77ms |

**Note:** Prefill numbers from lmx may also be short-prompt-inflated. Treat as
approximate. The 464.9 t/s for Qwen 35B is likely between the overhead-floor and
the real long-prompt ceiling (~1300-1700 t/s per Run 3).

---

## Run 23 — MTP + concurrency test (2026-08-06) — GDN causal_conv1d blocks it

**Why:** with MTP unlocked (Run 18+), per-user decode is ~1.8× higher, so the
C16=694 aggregate should scale up. Tested C1/C4/C8/C16 with MTP on, 150W.

**Result:** C1 works (78.3 t/s). **C2+ crashes the EngineCore:**

```
RuntimeError: causal_conv1d does not support spec-decode and non-spec
(prefill + decode) tokens in the same invocation; the spec path and the
non-spec path are mutually exclusive
```

**Root cause:** the XPU GDN kernel's `causal_conv1d` (the convolution state
machine inside GDN attention) cannot mix speculative and non-speculative tokens
in a single forward pass. With C1, every batch is either all-spec (decode) or
all-non-spec (prefill) — works. With C2+, vLLM batches a mix (some users
decoding with spec + some in prefill) → the kernel rejects the mixed batch.

**Verdict:** **MTP + concurrency is blocked on the XPU GDN kernel.** This is
the third XPU GDN kernel limitation found (after the spec_sequence_masks assert
in A14, and the GDN spec assert in A18). The no-MTP concurrency numbers
(C16=694 wall-agg, Run 19) stand as the multi-user ceiling.

**Implication for serving:** choose one — MTP (single-user max speed, 133 t/s)
OR concurrency (no MTP, C16=694 aggregate). Can't have both until the XPU GDN
causal_conv1d kernel supports mixed spec/non-spec batches.

**Raw:** `results/vllm-mtp-concurrency-150w.json` (partial), crash in container logs.

---

## Run 24 — Nightly upstream adoption + spec-N curve + power A/B (2026-08-07)

**Build:** vLLM `v0.26.1rc1.dev457+gc810e5ee9` (`vllm/vllm-openai-xpu:nightly`)
+ `patch_mtp_nightly.py` (env-gate BF16 draft) + graphs on (`VLLM_XPU_ENABLE_XPU_GRAPH=1`)
**Model:** Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 (same checkpoint as Run 18-20)
**Engine flags:** `--quantization gptq --dtype float16 --max-model-len 16384
--gpu-memory-utilization 0.90 --max-num-seqs 64 --max-num-batched-tokens 8192
--enable-prefix-caching --language-model-only`
**Goal:** upstream nightly adopted (most 0.21-xpu patches now obsolete; only the
MTP BF16-draft env-gate remains). Confirm reference ~200 decode / ~8700 prefill,
then probe levers: spec token count, power, clocks, gpu-mem-util.

### A. Baseline reproduced (MTP-4, 230W)

| Metric | Value |
|--------|------:|
| Decode short/g32 (median of 4 prompts) | **198.2 t/s** |
| Decode short/g64 | 206.4 t/s |
| Prefill p4k (honest, no cache) | **8,153 t/s** |
| Prefill p8k (honest, no cache) | **8,393 t/s** |

Matches the ~200 / ~8700 reference. ✓

### B. MTP spec acceptance (measured from `vllm:spec_decode_*` counters)

`num_speculative_tokens=4` cumulative counters: 1,697 drafts, 6,788 draft
tokens, 5,402 accepted → **80.1% overall**. Per-position:

| pos0 | pos1 | pos2 | pos3 |
|-----:|-----:|-----:|-----:|
| 92.6% | 84.2% | 72.7% | 68.9% |

The falloff is **fundamental to single-layer MTP drafting** — each draft token
is an autoregressive guess off the previous guess, so errors compound down the
chain. The engine logs this explicitly at startup: *"Enabling
num_speculative_tokens > 1 will run multiple times of forward on same MTP layer,
which may result in lower acceptance rate."*

### C. Spec-N throughput curve (the headline finding) — N=4 wins

| num_spec | Decode t/s | Overall accept | Speedup vs no-spec (72.6) |
|----------|----------:|---------------:|--------------------------:|
| 1 | 141.8 | **97.1%** | 1.95× |
| 2 | 174.5 | 92.1% | 2.40× |
| **4** | **198.2** | 80.1% | **2.73×** |

**This answers "can we get 98-99% acceptance":** YES — at N=1 (97.1%). But it
**costs 30% throughput** (141.8 vs 198.2). N=4 is the throughput optimum on a
single-layer draft head; chasing higher acceptance is the wrong objective for
this checkpoint. The only way to combine high-N AND high-acceptance is a
multi-layer draft model (DeepSeek-V3-style 3-layer MTP), which this GPTQ
checkpoint does not have (`mtp_num_hidden_layers: 1`).

**Implication:** `num_speculative_tokens=4` stays optimal for single-stream
serving. Don't lower it to chase acceptance %.

### D. Power A/B prefill — paired, alternating, REFUTES prior claim

**Prior claim (Run 20/A16b implicit):** raising the cap 150W→230W helps prefill
(based on the unpaired 7,535→9,094 jump, which was actually the
`--max-num-batched-tokens` flag, not power).

**Test:** paired A/B, alternating 150W and 230W back-to-back on the SAME warm
MTP-4 server, cooled <52C between rounds, 3 rounds, honest cold prefill (unique
random prefix per call so prefix-cache never fires). Harness:
`scripts/tmp/b70-power-ab-prefill.py`. Raw: `results/power-ab-prefill-150v230.json`.

| cell | 150W mean | 230W mean | delta | delta% |
|------|----------|----------|-------|--------|
| p2k | 7,216 | 7,207 | -9.7 | **-0.1%** |
| p4k | 8,140 | 8,135 | -5.4 | **-0.1%** |
| p8k | 8,384 | 8,403 | +18.4 | **+0.2%** |

**Verdict: NO measurable prefill gain from 230W.** ±0.2% across all prompt
lengths = within noise. The dynamic-power strategy (boost prefill, relax decode)
is therefore **refuted for MoE+MTP** — don't deploy it for this workload. (It
may still help *dense* llama.cpp decode, per Run 19's 150→230W +18-30%.)

Measured live card draw confirms why: prefill p8k draws ~171W, decode draws
~113W, idle ~47W. Raising the cap to 230W does NOT raise prefill draw — the
MoE grouped-GEMM is gated by kernel efficiency / bandwidth, not the power
envelope, at these token counts.

### E. GT clock locking — BLOCKED (driver)

`intel_gpu_frequency` and `xpu-smi` ARE installed, but both fail: the B70 uses
the **`xe` driver** (not i915), which exposes **no clock-control sysfs/debugfs**
— only read-only PMU counters (`/sys/devices/xe_0000_0b_00.0/events/gt-*-frequency`).
There is no `gt_min`/`gt_max`/RPS knob. **GT clock locking is impossible on the
current xe stack.** (AGENTS.md §6 note "xpu-smi does NOT work" was correct for a
different reason; the real reason is the xe driver, not the tool.)

### F. Methodology fix — honest prefill probe

`b70-mtp-prefill-sweep.py` and `b70-mtp-grid-bench.py` use a CONSTANT filler
string, so reps 2-3 (and any later run) hit prefix caching and report inflated
prefill (observed p8k: 8,640 → 42,733 t/s once cached — a 5× lie). The
"steady-state = reps 2-3" averaging made it worse. New harness
`scripts/tmp/b70-prefill-honest.py` prepends a unique random prefix per call so
the cache can never match; every rep is true cold prefill.

**Anyone citing prefill numbers: use `b70-prefill-honest.py`, not the old sweep.**

### Net for this session

| Lever | Result | Action |
|-------|--------|--------|
| num_speculative_tokens | N=4 optimal (198 t/s) | keep at 4 |
| Power cap 150 vs 230 | flat (±0.2%) | keep at 150W (saves 80W heat) |
| GT clock locking | impossible (xe driver) | n/a |
| Prefix-cache in prefill bench | inflated old numbers 5× | use honest probe |
| 97-99% acceptance | reachable at N=1 but -30% t/s | not worth it |
| `-ze-opt-large-register-file` | **already set** (GRF=128, auto-large-GRF in .so) | closed — not a lever |
| `vllm-xpu-kernels` version | **0.1.12** (latest; small-K tiling banked) | closed |
| max-num-batched-tokens | 8192 vs higher | deferred (requires restart; low confidence) |

**Production unchanged:** the patched nightly MTP-4 server reproduces the
200 decode reference. Decode is rock-solid (193-207 t/s median across instances).

### G. Prefill variance is FREQUENCY-driven (the hidden variable)

After restarting the MTP-4 server (same config), honest prefill dropped from
**p8k 8,390 → 6,840 t/s (-18%)** while decode stayed at 193-207 t/s (no change).
Same config, same container image, same graph capture (48 PIECEWISE + 37 FULL).
Root cause found via the read-only GT-frequency PMU:
**the card runs at ~2.0 GHz under prefill load — well below the 2.8 GHz boost
and below the 2.28 GHz base clock.** The `xe` driver self-throttles frequency
based on its own internal policy (not the power cap, not temperature — the card
was at 64°C/165W, well within limits). Prefill is compute-bound, so frequency
maps directly to throughput: 2.0/2.8 ≈ 0.71, close to the observed 6,840/8,390
≈ 0.82 ratio (the rest is warm-cache / scheduling variance).

**Implication:** prefill on this card varies ~18% depending on the driver's
frequency state at measurement time, and we CANNOT control it (no clock sysfs on
`xe`). This means:
- **Single prefill numbers are unreliable to ±18%.** Always report a range or
  median-of-many, and note the GT frequency if you can read it (`perf stat -e
  xe_0000_0b_00.0/gt-actual-frequency/`).
- **The "8,715 p4k / 8,390 p8k" reference was a high-frequency-state run.** The
  steady-state band is 6,800-8,400 t/s depending on driver boost behavior.
- **Decode is unaffected** — it's bandwidth-bound at 608 GB/s, so frequency
  matters less (consistent with the stable 193-207 t/s across instances).

This closes the prefill-optimization investigation: **there is no remaining
tunable lever on the current stack.** The two viable candidates from the
architecture research (`-ze-opt-large-register-file`, v0.1.12 tiling) are
already satisfied. The next real lever is upstream kernel work
(grouped-GEMM autotuning against real token distributions,
intel-xpu-backend-for-triton #6389) — not a config change.

**Raw:** `results/sanity-mtp4-150w.json`, `results/prefill-batch8192-{baseline,warm}.json`.
**Analysis:** `research/b70-xe2-architecture-prefill-analysis-20260807.md`.

---

## Run 25 — Wheel rebuild: w4a16_policy_n_128 tile tuning — BUILT + TESTED, FLAT (2026-08-08)

**Question:** would a narrower N tile (128×128×32 vs stock 128×256×32) for the
int4 grouped-GEMM give more N-parallelism (8 tile-columns vs 4 on gemm1 N=1024)
and speed up prefill?

**What was done (full pipeline, ~7 hrs incl. 2 reboots):**
1. Cloned `vllm-project/vllm-xpu-kernels` → `/mnt/<models-disk>/vxk-build` (persistent).
2. Added env-gated `w4a16_policy_n_128` to `gemm_xe2_policy.hpp` +
   `grouped_gemm_xe2_interface.hpp` dispatch (A_avg_M 128-512 range,
   `B70_W4A16_N128=1` to opt in).
3. Built the wheel in a container: `vllm/vllm-openai-xpu:nightly` + host
   `/opt/intel` oneAPI 2026.0 (icpx) mounted. SYCL AOT for BMG-G31.
4. Wheel `vllm_xpu_kernels-0.1.12+b70n128-cp312-cp312-linux_x86_64.whl` (469 MB)
   → derived image `vllm-openai-xpu:nightly-b70n128`. Verified
   `MoEGEMMLauncher<..., w4a16_policy_n_128, ...>` symbols (BF16+FP16).
5. **Clean A/B** (same wheel, same session, env ON vs OFF, honest cold prefill):

| cell | stock 128×256 | n128 tile | delta |
|------|--------------|-----------|-------|
| p500 | 4,107 | 4,826 | **+17.5%** |
| p1k | 6,447 | 6,311 | -2.1% |
| p2k | 6,345 | 6,304 | -0.7% |
| p4k | 6,840 | 6,855 | +0.2% |
| p8k | 6,882 | 6,834 | -0.7% |

**Verdict: FLAT for long prompts (p2k-p8k ±1%, noise).** The +17% at p500 is
launch-overhead reduction at tiny M — irrelevant for real prefill. The stock
128×256 tile is already well-tuned for this MoE: the prefill bottleneck is
L2/DRAM weight streaming (int4 weights), which is tile-independent, NOT the
N-column count. Hypothesis "more N-parallelism = faster" REFUTED for long
prompts.

**Frequency-variance note:** both kernels landed in the 6.8-6.9K p8k band (vs
8.4K on the high-frequency-state reference of Run 24) — the comparison is fair
because both were measured in the same session/band.

**Build gotchas (for future wheel work):**
- `grouped_gemm_xe2.cpp` single-TU OOMs at >18 GB — compile with `-j1` + 24g
  limit for that file (rest at -j3).
- `/tmp` is tmpfs — a host reboot wipes builds. Use `/mnt/<models-disk>` (persistent).
- Total build ~4-5 hrs at -j3: 1232 objects (oneDNN+cutlass deps, grouped-GEMM,
  462 attention TUs, gdn, mhc, mqa).

**Status: tile tuning CLOSED (negative result).** Env-gate remains in the wheel
(inert by default) for future experiments.

**Raw:** `results/prefill-b70n128-{tile,stock-same}.json`.
**Modified sources:** `research/b70-wheel-n128/`.
**Artifacts:** `/mnt/<models-disk>/vxk-build/` (source + wheel + image).
**Analysis:** `research/b70-patch-candidates-20260807.md` §4.


---

## Run 26 — Real Pi workloads, exact 128K MTP4 repair, and mixed load (2026-08-08)

**Status:** E2 — provisional self-report with raw evidence. Not independently reproduced.

**Stack:** public image `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97`; observed vLLM `v0.26.1rc1.dev457+gc810e5ee9` and stock `vllm-xpu-kernels 0.1.12`; Qwen3.6-35B-A3B GPTQ-INT4 target; `patch_mtp_nightly.py` then `patch_mtp_boundary.py` for exact MTP4 128K; exact Pi system prompt; configured 165 W. The historical `intel/vllm:0.21.0-xpu-int4moe` local image is not part of this run and was never published.

### C1 scheduler: exact p9445/g128

Five cold-prefix samples per cell, zero cache-hit delta:

| Budget | TTFT median (s) | Input tokens / TTFT median (tok/s) | Client post-first median (tok/s) |
|---:|---:|---:|---:|
| **8,192** | **1.2600** | **7,495.93** | 158.83 |
| 9,600 | 1.2645 | 7,469.27 | 159.50 |
| 10,240 | 1.2646 | 7,469.03 | 160.35 |
| 12,288 | 1.2643 | 7,470.81 | 157.72 |
| 16,384 | 1.2672 | 7,453.66 | 159.18 |

Keep 8,192 for C1. Larger budgets did not improve the matched cell.

### Exact long-context completion

One capacity/correctness observation per cell:

| Spec | Prompt | Output | Total | TTFT (s) | Client post-first (tok/s) | MTP accept | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| MTP4 | 16,256 | 128 | 16,384 | 2.403 | 161.23 | 85.34% | Completed |
| MTP4 | 32,640 | 128 | 32,768 | 5.785 | 139.99 | 71.21% | Completed |
| MTP4 | 65,408 | 128 | 65,536 | 15.093 | 111.17 | 58.55% | Completed |
| MTP4 | 98,176 | 128 | 98,304 | 28.078 | 117.36 | 76.56% | Completed |
| MTP4 | 122,880 | 128 | 123,008 | 44.057 | 95.13 | 62.84% | Completed |
| MTP4 + boundary patch | 130,944 | 128 | **131,072** | 48.601 | 96.87 | 72.31% | **Completed** |
| MTP2, unpatched fallback | 130,944 | 128 | **131,072** | 48.559 | 103.63 | 86.96% | Completed |

The unpatched exact MTP4 request emitted 124 tokens, then failed because the last four-token sequence space truncated a required five-token MTP4 GDN group (`1 target + 4 drafts`). `scripts/tmp/patch_mtp_boundary.py` reclassifies only that truncated final group as stateful non-spec prefill. It does not pad past 131,072 or reduce output. The patched request completed all 128 output tokens with zero cache hits, 2,619 MiB free after load, and logged KV capacity 165,961 tokens.

MTP2 was 6.98% faster than MTP4 by client post-first rate in these single exact-128K observations. This is a routing signal, not a median A/B.

### Real Pi flow

| Scenario | Prompt | Output | TTFT (s) | TPOT (ms) | E2E (s) | Cache hits |
|---|---:|---:|---:|---:|---:|---:|
| Cold new conversation | 595 | 128 | 0.811 | 7.36 | 1.746 | 0 |
| Warm shared system | 591 | 128 | 0.148 | 7.30 | 1.075 | 0 |
| Warm multi-turn | 753 | 128 | 0.157 | 8.93 | 1.291 | 0 |
| RAG/tool append | 930 | 128 | 0.236 | 6.23 | 1.027 | 0 |
| Cold 32K document | 32,640 | 128 | 5.802 | 7.03 | 6.694 | 0 |
| Resident 32K follow-up | 32,795 | 128 | 0.676 | 8.26 | 1.726 | 30,464 |

The Pi system prompt is shorter than the model's 1,088-token cache page. Short warm requests can correctly have zero cache hits. The resident 32K document follow-up reused 30,464 tokens.

### Mixed load

MTP4 failed when one p65408/g128 prefill overlapped 20 short g64 requests:

```text
causal_conv1d does not support spec-decode and non-spec
(prefill + decode) tokens in the same invocation
```

No-spec completed all 21 requests, but budget 8,192 starved short chats:

| Workload | Completed | TTFT p50 / p95 (s) | TPOT p50 / p95 (ms) |
|---|---:|---:|---:|
| Sequential short baseline | 10/10 | 0.112 / 0.284 | 10.34 / 10.38 |
| Mixed short | 20/20 | 12.855 / 13.674 | 34.88 / 50.49 |
| Mixed 64K document | 1/1 | 15.345 | 24.42 |

Mixed aggregate output was 74.46 tok/s: 1,374 successful output tokens divided by 18.452 seconds. This denominator includes the 64K prefill. It is not per stream and not directly comparable to historical all-short no-spec C16 at 694 aggregate tok/s.

### Power/thermal note

Exact-128K patched MTP4 full harness window: 160.23 W average card draw, 210.24 W maximum 0.5-second interval average, 74°C package maximum, 76°C VRAM maximum. Configured cap: 165 W.

### Evidence

- Full report: `benchmarks/vllm-pi-realworld-and-long-context-2026-08-08.md`
- Scheduler: `results/vllm-realworld-scheduler-20260808T193702Z-13490/`
- Capacity: `results/vllm-realworld-mtp4-capacity-20260808T201134Z-31970/`
- Long context: `results/vllm-pi-mtp4-longctx-20260808T202455Z-38872/`
- MTP2 fallback: `results/vllm-pi-128k-mtp-fallback-20260808T203735Z-44148/`
- Patched MTP4 exact 128K: `results/vllm-pi-mtp4-128k-boundary-fix-20260808T214938Z-77748/`
- Pi flow: `results/vllm-pi-realworld-flow-20260808T210237Z-56507/`
- Mixed MTP4 failure: `results/vllm-pi-mixed-load-20260808T211221Z-60972/`
- Mixed no-spec: `results/vllm-pi-mixed-load-nospec-20260808T212657Z-68780/`

---

## Run 27 — Matched exact-128K cache × no-spec/MTP1/MTP2/MTP4 matrix (2026-08-08)

**Status:** E2 — provisional self-report with raw evidence. Not independently reproduced.

**Stack:** public image `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97`; observed vLLM `v0.26.1rc1.dev457+gc810e5ee9`; stock `vllm-xpu-kernels 0.1.12`; Qwen3.6-35B-A3B GPTQ-INT4 target + preserved BF16 MTP draft; `patch_mtp_nightly.py` then `patch_mtp_boundary.py`; `max_model_len=131072`; scheduler budget 8,192; `gpu_memory_utilization=0.85`; C1; configured 165 W.

Every cell used the same six calibrated p130944 prompts: one same-shape warmup and five measured g128 requests. Cold prefixes carried unique leading entropy and recorded zero cache-hit tokens. Each server then prepared the same exact 120,000-token Pi session and handled five different ~120,148-token g128 follow-ups. Values below are medians.

### Cold exact p130944/g128

| Mode | Cache | TTFC (s) | E2E (s) | Client post-first (tok/s) | MTP accept | Free after load (MiB) | Logged KV capacity |
|---|---|---:|---:|---:|---:|---:|---:|
| No spec | On | **41.589** | **43.793** | 57.57 | n/a | 4,824 | 283,817 |
| No spec | Off | 42.192 | 44.413 | 57.19 | n/a | 4,829 | 290,683 |
| MTP1 | On | 48.865 | 50.358 | 85.10 | 90.36% | 3,414 | 177,451 |
| MTP1 | Off | 45.262 | 46.749 | 86.21 | 90.69% | 3,419 | 181,643 |
| MTP2 | On | 48.564 | 49.946 | 98.81 | 80.54% | 2,958 | 173,448 |
| MTP2 | Off | 45.347 | 46.653 | **101.68** | 82.77% | 2,963 | 177,451 |
| MTP4 | On | 48.761 | 50.011 | **102.30** | 61.22% | 2,616 | 165,961 |
| MTP4 | Off | 45.473 | 46.865 | 93.60 | 62.31% | 2,624 | 169,622 |

`Client post-first` is `(completion_tokens - 1) / (request_end - first_generated)`. It is client-observed and not an engine-native vLLM decode field.

### Changed follow-ups over a resident 120K session

| Mode | Cache | Reused / recomputed median | TTFC (s) | E2E (s) | Client post-first (tok/s) |
|---|---|---:|---:|---:|---:|
| No spec | On | 119,680 / 468 | **0.554** | 2.671 | 59.90 |
| No spec | Off | 0 / 120,148 | 36.770 | 38.921 | 59.04 |
| MTP1 | On | 118,592 / 1,556 | 1.222 | 2.666 | 88.57 |
| MTP1 | Off | 0 / 120,148 | 39.342 | 40.808 | 86.64 |
| MTP2 | On | 118,592 / 1,556 | 1.251 | **2.504** | 101.39 |
| MTP2 | Off | 0 / 120,148 | 39.408 | 40.634 | 104.62 |
| MTP4 | On | 118,592 / 1,556 | 1.256 | 2.517 | 104.37 |
| MTP4 | Off | 0 / 120,148 | 39.508 | 40.657 | **110.48** |

| Mode | Cache TTFC speedup | Cache E2E speedup | Cold cache-on TTFC change |
|---|---:|---:|---:|
| No spec | 66.32× | 14.57× | -1.43% |
| MTP1 | 32.19× | 15.31× | +7.96% |
| MTP2 | 31.49× | 16.23× | +7.09% |
| MTP4 | 31.46× | 16.15× | +7.23% |

For resident long sessions, MTP2 + cache on had the best E2E median. No-spec + cache on had the fastest TTFC. For a cold exact-128K request with only 128 output tokens, no-spec completed fastest because the higher MTP output rate did not repay its prefill/first-token overhead.

### Full-cell telemetry

Telemetry covers the cold and resident request windows for each cell, including the 120K session preparation. Maximum power is a 0.5-second interval average, not an instantaneous peak.

| Mode | Cache | Average card draw (W) | Max 0.5 s interval average (W) | Max package (°C) | Max VRAM (°C) |
|---|---|---:|---:|---:|---:|
| No spec | On | 161.43 | 201.82 | 75 | 78 |
| No spec | Off | 162.95 | 215.07 | 75 | 78 |
| MTP1 | On | 162.30 | 214.10 | 76 | 78 |
| MTP1 | Off | 163.35 | 207.05 | 75 | 78 |
| MTP2 | On | 162.54 | 214.25 | 76 | 78 |
| MTP2 | Off | 163.55 | 211.72 | 76 | 78 |
| MTP4 | On | 162.70 | 205.76 | 76 | 78 |
| MTP4 | Off | 163.49 | 214.08 | 76 | 80 |

### Exclusion and correction

The first attempted no-spec/cache-off cell was rejected. This vLLM V1 build defaults prefix caching to on, so omitting `--enable-prefix-caching` did not disable it. Unique cold prefixes still produced zero hits, but the resident request exposed 119,680 reused tokens and the harness stopped. Accepted cache-off cells use `--no-enable-prefix-caching`, show `enable_prefix_caching: False` in server logs, and record zero hits.

### Evidence

- Raw root: `results/vllm-pi-128k-cache-spec-matched-20260808T224033Z-101198/`
- Compiled summary: `results/vllm-pi-128k-cache-spec-matched-20260808T224033Z-101198/summary.json`
- Runner: `scripts/tmp/b70-pi-128k-spec-matched.sh`
- Cold harness: `scripts/tmp/b70-realworld-context-harness.py`
- Resident harness: `scripts/tmp/b70-resident-session-harness.py`
- Compiler/auditor: `scripts/tmp/b70-compile-cache-spec-matrix.py`

---

## Run 28 — Phase-separated no-spec/MTP1/MTP2/MTP4 C1 matrix (2026-08-09)

**Status:** E2 provisional self-report with raw evidence. Not independently
reproduced.

**Canonical evidence:**
`results/vllm-pi-prefill-decode-matrix-20260809T060920Z-238794/summary.json`.
Public compact copy:
`~/intel-arc-pro-b70-inference-cookbook/results/prefill-decode-matrix-20260809-summary.json`.

**Stack:** image
`vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97`;
observed vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`;
`vllm-xpu-kernels 0.1.12`; Qwen3.6-35B-A3B preserved-MTP GPTQ-INT4;
`patch_mtp_nightly.py` SHA-256
`4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14`,
then `patch_mtp_boundary.py` SHA-256
`41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50`.
The newer PyPI `vllm-xpu-kernels 0.1.12.2` was not installed or tested.

**Methodology for every table below:** C1, median of five measured requests after
one full-output same-shape warmup; context 131,072; scheduler budget 8,192;
`gpu-memory-utilization=0.85`; prefix cache enabled; entropy-first unique cold
prefixes; zero cache-hit delta; configured cap 165 W. Timing source was client
monotonic SSE. Measured draw is not part of these tables. Standard p512 through
p8192 cells used the compact benchmark system prompt. The p9445 control and
full-context cells used the full Pi system prompt.

### Cold input rate: actual endpoint input tokens / TTFT (tok/s)

| Mode | p512 | p2048 | p4096 | p6144 | p8192 | Full p131071 |
|---|---:|---:|---:|---:|---:|---:|
| No spec | 5,156 | 6,674 | 7,197 | 7,451 | 7,576 | 3,144 |
| MTP1 | 4,840 | 7,377 | 6,999 | 7,189 | 7,264 | 2,679 |
| MTP2 | 4,843 | 7,341 | 7,002 | 7,140 | 7,229 | 2,683 |
| MTP4 | 4,532 | 7,401 | 6,868 | 7,057 | 7,197 | 2,678 |

This input rate includes scheduling, uncached input processing, and first-token
work. It is not isolated engine prefill and not llama-bench `pp`.

### Decode at p512: client post-first median (tok/s)

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| No spec | 97.43 | 96.79 | 96.60 | 96.13 |
| MTP1 | 122.21 | 124.57 | 123.82 | 120.58 |
| MTP2 | 162.90 | 153.17 | 148.31 | 141.80 |
| MTP4 | 178.34 | 170.91 | 167.85 | 148.35 |

### Decode at p8192: client post-first median (tok/s)

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| No spec | 85.92 | 90.34 | 90.91 | 91.26 |
| MTP1 | 108.41 | 118.41 | 118.49 | 117.45 |
| MTP2 | 143.95 | 145.43 | 143.82 | 135.61 |
| MTP4 | 156.28 | 164.36 | 163.89 | 138.03 |

### Historical control: p9445/g128

| Mode | Client post-first median (tok/s) |
|---|---:|
| No spec | 89.68 |
| MTP1 | 116.85 |
| MTP2 | 142.02 |
| MTP4 | 160.42 |

MTP4 at 160.42 tok/s reproduces Run 26's 158.83 tok/s control within 1.0%.
This does not make the exact-128K workload equivalent.

### Full-context decode

| Mode | p130944/g128 (tok/s) | MTP accept | p130560/g512 (tok/s) | MTP accept |
|---|---:|---:|---:|---:|
| No spec | 57.35 | n/a | 57.14 | n/a |
| MTP1 | 84.88 | 89.22% | 82.74 | 85.32% |
| MTP2 | 101.64 | 85.81% | 94.01 | 76.45% |
| MTP4 | 93.53 | 66.91% | 93.83 | 59.81% |

`Client post-first` is
`(completion_tokens - 1) / (request_end - first_generated)`. It is request-side
and not engine-native vLLM decode or llama-bench `tg`.

### Exclusions and correctness scope

The original no-spec p130560/g512 cell is retained at
`no-spec/decode-full-p130560-g512/results.json` and excluded because three of five
requests stopped at EOS before 512 tokens. The accepted 57.14 tok/s result comes
from `no-spec/decode-full-p130560-g512-forced-exact/results/results.json`. Future
decode requests use `ignore_eos=true`.

Prompt hashes matched across all four modes. Exact output text did not: depending
on the longer-decode cell, only 0–4 of five repetitions matched across all four
modes. These results establish measured speed and exact request shape, not token,
logit/KL, or task-quality parity.

**Evidence layout:** the raw root contains `manifest.txt`,
`package-versions.txt`, calibrated `prompts/`, per-mode server/metrics/monitor/
VRAM/harness/request evidence, the retained exclusion and replacement,
`summary.json`, and rendered `tables.md`. Compiler:
`scripts/tmp/b70-compile-prefill-decode-matrix.py`; renderer:
`scripts/tmp/b70-render-prefill-decode-tables.py`.
## Run 29b — MoE 12.4k prefill claim reproduction screen (2026-08-09)

**Status:** E2 provisional self-report with raw evidence. Config replica, NOT an
exact model/build reproduction of the LocalMaxxing
`palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` 12,400 prefill tok/s entry.

**Why not exact:** palmfuture's `mtp.safetensors` (785 MTP keys) and
`model.safetensors.index.json` / `quantize_config.json` are absent locally (the
six model shards match byte-for-byte, but MTP weights are missing, so the exact
checkpoint is not fully present); vLLM commit `568afb3a1` is unavailable (pinned
image `gc810e5ee9`, vLLM `0.26.1rc1.dev457+gc810e5ee9`, `vllm-xpu-kernels`
version unresolved in container); Linux/XPU `xe` driver stack vs the claimed
Windows build. Used the local MTP-Preserved GPTQ-INT4 checkpoint as the closest
config replica.

**Canonical evidence:**
`results/vllm-moe-12k-repro-20260809T183124Z-127886/` (manifest, per-phase
server.log, monitor.jsonl, package-versions.txt, per-rep SSE, `results.json`,
`summary.json`, `tables.md`).

**Stack both phases:** pinned nightly image; model
`/mnt/<models-disk>/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4` (config sha256
`7342c56e…`); `patch_mtp_nightly.py` (SHA-256
`4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14`); control
additionally applies `patch_mtp_boundary.py` (SHA-256
`41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50`); dtype
float16, `--quantization gptq`, kv-cache fp8, block-size 64,
`--language-model-only --trust-remote-code`, `--no-enable-prefix-caching`
(`enable_prefix_caching: False` verified in both server logs), MTP4
`{"method":"mtp","num_speculative_tokens":4}` (counters live, no fallback in
either phase), configured cap 230 W.

**Methodology per cell:** C1, temperature 0, one generic + one same-shape warmup
discarded, five measured repetitions of exact-4096-token entropy-first cold
prompts (6 prompts calibrated in-image to 4096 rendered tokens; same prompt set
in both phases), zero `vllm:prefix_cache_hits` delta every cell (cache manager
off, so query counter also stays at 0 — consistent with disabled cache). Timing
source: client monotonic SSE. Prefill cell = actual endpoint prompt tokens /
TTFT (cold input rate, not llama-bench pp). Decode = client post-first tok/s,
`(completion-1)/(end-first_generated)` (not engine-native vLLM decode).

### Comparison: public shape vs normal 128K recipe (230 W, n=5 median)

| Metric (p4096) | Repro: public shape | Control: normal recipe |
|---|---:|---:|
| Config | 32K ctx · batch 16384 · seqs 16 · MTP4 | 128K ctx · batch 8192 · seqs 64 · MTP4 + boundary |
| Prefill input rate median | 8,660.5 t/s | 7,086.6 t/s |
| Prefill TTFT median | 0.473 s | 0.558 s |
| g32 post-first median | 156.0 t/s | 153.1 t/s |
| g128 post-first median | 175.1 t/s | 171.6 t/s |
| MTP accept (g128 total) | 85.9% (498/580) | 88.3% (498/564) |
| Measured avg card draw | 176.1 W | 150.5 W |
| Peak interval-avg W | 231.9 W | 209.7 W |
| pkg temp max | 71.0 °C | 69.0 °C |
| VRAM free after load | 4,248 MiB | 1,048 MiB |

**Findings:** the public-shape config is faster on cold prefill (+22% vs the
128K recipe on this cell), but does **not** reproduce the claimed 12,400 t/s
(8,660 t/s median at p4096/g1, C1, cache off). Comparison vs the claim is
`directional_only`: checkpoint, build, and OS differ. MTP is real in both
phases; no no-spec fallback. The 128K control loads with 1,048 MiB free (the
1,024–2,047 staged-C1-research band); the 32K repro leaves 4,248 MiB.

**Correctness scope:** measured speed and exact request shape only; not token,
logit/KL, or task-quality parity with the claimed run.

**Harness note:** campaign-local copy `scripts/tmp/b70-moe-12k-harness.py` adds
`delta.get("reasoning")` alongside `reasoning_content` so the `--reasoning-parser
qwen3` stream (public command shape) is counted; the shared
`b70-realworld-context-harness.py` is unchanged.

## Run 29c — MoE 12.4k reproduction, EXACT palmfuture checkpoint, focused (2026-08-09)

**Status:** E2 provisional self-report with raw evidence. Checkpoint is
**byte-exact** vs `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` (all six shards +
`mtp.safetensors` sha256 == HF LFS oids; 785 MTP tensors, per-expert split
format, verified in `mtp.safetensors` header). Still NOT an exact **build**
reproduction: vLLM commit `568afb3a1` unavailable (pinned image `gc810e5ee9`,
vLLM `0.26.1rc1.dev457+gc810e5ee9`), Linux/XPU `xe` driver vs their Windows
build. No `--quantization` flag passed (auto-detected from `quantize_config.json`
per the palmfuture card's warning that forcing the flag breaks the MTP loader).

**Canonical evidence:** `results/vllm-moe-12k-exact-20260809T190246Z-13717/`
(manifest, prompts, repro/ + control/ + iso16384/ with server.log,
monitor.jsonl, package-versions.txt, per-cell results.json + raw SSE,
summary.json, tables.md).

**Stack:** pinned nightly image; model
`/mnt/<models-disk>/Qwen3.6-35B-A3B-GPTQ-Int4`; `patch_mtp_nightly.py` +
`patch_mtp_boundary.py`; dtype float16, kv-cache fp8, block-size 64,
`--language-model-only --trust-remote-code`, `--no-enable-prefix-caching`
(`enable_prefix_caching: False` verified), MTP4 (counters live, no fallback in
any phase), configured cap 230 W. The repro phase additionally carries the
public command-shape flags `--enable-auto-tool-choice --tool-call-parser
qwen3_coder --reasoning-parser qwen3`; the control/iso phases do not.

**Methodology per cell:** C1, temperature 0, one generic + one same-shape
warmup discarded, five measured repetitions of exact-4096-token entropy-first
cold prompts (same prompt set across phases), zero prefix-cache-hit delta every
cell. Prefill = endpoint prompt tokens / client TTFT; decode = client post-first
tok/s. Timing: client monotonic SSE.

### Focused comparison (p4096, C1, n=5 median, 230 W configured)

| Metric | Repro: public shape | Control: 128K batch 8192 | Iso: 128K batch 16384 |
|---|---:|---:|---:|
| Config | 32K · batch 16384 · seqs 16 | 128K · batch 8192 · seqs 64 | 128K · batch 16384 · seqs 64 |
| Prefill input rate | 7,740 t/s | 6,525 t/s | 7,672 t/s |
| Prefill TTFT | 0.53 s | 0.63 s | 0.53 s |
| g128 post-first | 135.1 t/s | 133.1 t/s | 149.1 t/s |
| MTP accept (g128) | 87.8% | 88.6% | 89.1% |
| Measured avg draw | 173.1 W | 164.9 W | 168.2 W |
| Peak interval-avg W | 229.3 W | 227.5 W | 226.3 W |
| pkg temp max | 72.0 °C | 71.0 °C | 72.0 °C |
| VRAM free after load | 3,580 MiB | (see control) | (see iso) |

### Findings

- **Batch isolation at the same 128K recipe:** 16384 vs 8192 gives
  **+17.6% prefill** (6,525 → 7,672 t/s) and **+12.0% g128 decode**
  (133.1 → 149.1 t/s) at p4096. Both budgets prefill p4096 in one chunk, so the
  gain is scheduler/memory-layout, not chunk count. Not yet validated for mixed
  long-prefill + short-chat workloads (head-of-line risk) — do not blanket-adopt.
- **Exact checkpoint does NOT reproduce 12,400 t/s:** 7,740 t/s median on the
  public-shape config, below the 8,660 t/s of the MTP-Preserved config replica
  (Run 29b). Comparison vs the claim remains `directional_only` (build/OS differ).
- Public-shape (32K) and 128K-batch-16384 prefill are nearly identical
  (7,740 vs 7,672 t/s), isolating the earlier Run 29b 8,660-vs-7,087 gap as
  mostly batch+context effects rather than the reasoning flags.

**Correctness scope:** measured speed and exact request shape only; not token,
logit/KL, or task-quality parity with the claimed run.

## Run 29d — MoE 12.4k claim: context-length probe (2026-08-09)

**Status:** E2 provisional self-report. Focused probe answering "does lower
context explain the 12,400 prefill tok/s claim?" Exact palmfuture checkpoint,
batch 16384, seqs 16, MTP4, cache off, 230 W, same prompt set as Run 29c.

**Canonical evidence:** `results/vllm-moe-12k-lowctx-20260809T193408Z-64922/`
(manifest, prompts-4096 + prompts-8192, ctx8192/ + ctx16384/ cells with raw
SSE + results.json + monitor.jsonl + server.log, summary.json).

**Result — prefill input rate (pX/g1, client TTFT-based, n=5 median):**

| Context | p4096 prefill | p8192 prefill | g128 decode |
|---|---:|---:|---:|
| 8,192 | 7,727 t/s | — | 138.7 t/s |
| 16,384 | 6,622 t/s | 6,956 t/s | 141.3 t/s |
| 32,768 (claim config) | 7,740 t/s | — | 135.1 t/s |
| 131,072 | 7,672 t/s | — | 149.1 t/s |

**Finding:** prefill is essentially flat across context (7.6-7.7K t/s at
p4096, 8K-128K); the 16K point dips slightly (6.6K) but does not rise toward
the claim. Context length does NOT explain the 12,400 t/s claim. The claim's
own config (context 32,768, p4096, batch 1, MTP4, fp8 KV) measured 7,740 t/s
here — a 1.6x gap attributable to build (their v0.26.1.dev0 Windows XPU build
vs our 0.26.1rc1.dev457 Linux pinned nightly) or their measurement methodology
(their implied TTFT = 0.330 s vs our 0.529 s). `568afb3a1` is an upstream
macOS-CI commit (#49901), not an XPU kernel change — not a meaningful build
target for reproduction. Claim remains `directional_only`, not reproduced.

## Run 33 — Muse Glimmer-30B FIRST B70 run: llama.cpp SYCL + DFlash tuning + power A/B (2026-08-10)

**Status:** E2 provisional self-report, first run of a new architecture on this
host. Muse-Glimmer-30B (Meta, Apache-2.0, dense 27.85B text + ~1.8B ViT-G/14
vision, 128K trained ctx) — llama.cpp arch support landed upstream in PR #26841
(merge `62bf73d2`, master `d2f83055d` / build 284), built here as
`build-sycl-muse-0810` (IntelLLVM 2026.0, oneAPI 2026.0).

**Artifact:** `unsloth/Muse-Glimmer-30B-GGUF` `UD-Q4_K_XL` (15.88 GB) +
`mmproj-kquant` (1.40 GB) + `dflash-kquant` (1.63 GB).

**Canonical evidence:** `results/muse-glimmer-q4kxl-dflash-20260810T143831-2493876/`
(manifest.txt, screen-results.txt, llamabench-150w-ub4096.txt, 4 server logs).

**Config:** `-ngl 99 -ncmoe 0 -fa on -ctk q8_0 -ctv q4_1 -t 8 --no-mmap -b 8192`
ctx 65,536, 4 slots. DFlash: `--spec-type draft-dflash --spec-draft-model
dflash-kquant.gguf --spec-draft-n-max 2`. C1, cold, unique 64+ char entropy
prefixes per request; decode = engine `timings.predicted_per_second` (HTTP).
Reasoning model: generates a `reasoning_content` channel; short max_tokens
returns empty `content` (needs 512+).

**Lane 0 smoke — PASS:** text coherent + correct ("2+2 → 4", full reasoning);
vision correct ("The image is a solid, vivid red.") via mmproj; DFlash outputs
coherent at all n_max tested.

**Mode screen (C1 decode t/s, n=3 median, 150 W, ub 4096):**

| Mode | p512/g128 | p8192/g128 |
|---|---:|---:|
| no-spec | 21.71 | 16.29 |
| DFlash n_max=3 | 26.37 | 19.09 |
| **DFlash n_max=2** | **25.77** | **22.03** |
| DFlash n_max=15 | 11.74 | ~6.7 (aborted) |

DFlash n_max=15 collapses: acceptance 0.18-0.45, mean accepted len 3.4-6.7 —
draft compute wasted. Small windows win (same lesson as vLLM spec-N, §15.8).

**Matched power A/B (DFlash n=2, 150→230→150, fresh prompts, cooldown ≤55 °C):**

| Cap | p512/g128 | p8192/g128 | measured draw |
|---|---:|---:|---:|
| 150 W | 25.77 / 24.21 | 22.03 / 20.20 | 149 W (157 s window) |
| 230 W | 29.82 | 23.68 | 206 W (13 s decode window), max 75 °C |

230 W vs 150 W-mean: **+19% p512, +12% p8192** — dense power scaling confirmed
on this arch (mirrors dense-27B llama.cpp Run 19 behavior).

**Final config n=5 (DFlash n=2, ub 8192, 230 W): p512/g128 median 29.87
(max 31.41); p8192/g128 median 24.11 (max 26.35).** vs no-spec@150W/ub4096:
**+38% / +48%**. `-ub 8192` (vs fleet 4096) added +8% prefill and +10% decode
on this arch (pp4096 293.3 vs 270.9; tg128 21.7 vs 19.6).

**llama-bench engine card (no-spec, 150 W, r=5, ub 4096):** pp512 225.2 ± 3.8 |
pp2048 267.0 ± 2.3 | pp4096 270.9 ± 0.4 | pp8192 269.3 ± 0.1 | tg128 19.61 ± 0.14.

**Findings:**
1. Prefill is this arch's weak spot on SYCL: ~225-293 t/s vs dense-27B Q4_K_M's
   ~936 t/s pp4096 — a 3-4x gap. Curve is flat pp2048-8192 (~270), i.e. a fixed
   per-token cost dominates, not attention O(N²) (SWA pattern caps that). Likely
   Q4_K_XL dequant + the SWA/QK-norm/gate graph on SYCL kernels — a llama.cpp
   kernel-optimization target, not a flag fix (only `-ub 8192` helped, modestly).
2. DFlash works on the new arch and is worth ~+20-40% decode at the right n_max;
   over-drafting (n_max=15) destroys it (acceptance collapse).
3. Vision + reasoning + DFlash all verified functional — the full multimodal
   stack runs on the B70.
4. 128K ctx untested (VRAM-gated: Q4_K_XL + mmproj + draft at 64K left ~2-5 GiB
   free; doubling KV risks the §8.1 abort zone). Next expansion: 128K capacity
   check, vision throughput cell, Q5_K_M quant when disk allows, and SYCL
   prefill-kernel investigation.

## Run 34 — Muse Glimmer prefill FIX: missing GGML_SYCL_F16=ON was a 3.4× prefill penalty (2026-08-10)

**Status:** E2 provisional self-report. CORRECTION to Run 33. Run 33's llama-bench
card (pp ~225-293 t/s) was measured on a build that MISSED `GGML_SYCL_F16=ON`
(production build-sycl-0804 has it; my reconstitution of cmake flags from the
cache dropped it). Same model/flags/machine: dense-27B Q4_K_M pp4096 = 336 t/s
(F16 off) vs 1178 t/s (F16 on) vs 1138 t/s (production build-0804). The "Muse
prefill is 3-4× slow" conclusion was a build artifact, not an arch problem.

**Fix:** reconfigured `build-sycl-muse-0810` with `-DGGML_SYCL_F16=ON`, rebuilt.

**Corrected llama-bench card (Muse Q4_K_XL, 230 W, ub 8192, r=5, F16 build):**

| test | t/s |
|---|---:|
| pp512 | 857.71 ± 3.91 |
| pp2048 | 1277.94 ± 39.45 |
| pp4096 | 1301.38 ± 7.00 |
| pp8192 | 1171.84 ± 17.92 |
| tg128 | 25.26 ± 0.11 |

Muse pp4096 (1301) now BEATS the dense-27B control (1178) on the same build.
Prefill is no longer the weak spot. Decode also improved: tg128 19.6 → 25.3
(+16% from the flag; bandwidth-bound decode less affected than compute-bound
prefill).

**Corrected final decode (C1, engine `timings.predicted_per_second`, DFlash
n_max=2, ub 8192, 230 W):** p512/g128 median **29.20** (max 31.84, n=5);
p8192/g128 median **22.28** (max 22.75, n=4; 5th rep lost when the server
received an external stop signal ~25 min after launch — see note below).

**Operations note:** llama-server on :8765 receives an external stop signal
("cleaning up before exit") roughly 20-30 min after launch — matches the
profile-manager automation that keeps the vLLM Pi-bridge server alive. A Muse
llama-server on 8765 will be periodically killed by it. For sustained Muse
serving, run on a different port or coordinate with the profile manager.

**Evidence:** same run root as Run 33 (`results/muse-glimmer-q4kxl-dflash-20260810T143831-2493876/`)
plus `/tmp/muse-server-f16.log` (lost on reboot — tmpfs). Re-bench raw lines
captured in this session's logs. The 3.4× build-flag lesson is now in
`muse-glimmer-adoption` SKILL and AGENTS.md §9.7 (GGML_SYCL_F16 must stay ON).

## Run 35 — Muse Glimmer DFlash n_max sweep 1-8 + 32K long-context curve (2026-08-10)

**Status:** E2 provisional self-report. Full DFlash depth screen + long-context
curve for Muse-Glimmer-30B Q4_K_XL at **128K ctx (1 slot)**, 230 W, `-ub 8192`,
F16 SYCL build (d2f83055d). Complements Run 33/34 (64K-era + flag fix).

**Canonical evidence:** `results/muse-dflash-sweep-manual-20260810T164735/`
(FINAL-TABLE.md, MANIFEST.txt, per-config dirs with raw jsonl, n2-n5/n3-n5,
prefill/, decode-winner/, decode-nospec/).

**Method:** C1 cold, unique entropy-first exact prompts (512/8192/32768 targets
+ ~50 chat-template tokens verified), 1 warmup + n=3 per cell (n=5 for the top
two depths), engine `timings.predicted_per_second`, `energy1_input` draw,
hwmon4 temp, acceptance from server logs.

**DFlash n_max screen (n=3, p512/p8192 engine t/s):** no-spec 22.5/17.2 · n1
24.8/20.0 · **n2 27.6/21.4** · n3 26.3/21.3 · n4 27.5/18.2 · n5 21.3/13.8 ·
n6 23.0/15.2 · n7 23.2/13.6 · **n8 ABORTS the server at launch** (core dump —
unsupported at 128K; excluded). Acceptance decays 0.91→0.30 with depth.

**n=5 confirm (fresh):** winner **n2**: p512 26.8 (max 28.9, TTFT 1.1 s), p8192
22.9 (max 23.1, TTFT 12.3 s), accept 0.85/0.75. Runner-up n3: 25.6/20.3.

**Long-context (winner n2 vs no-spec):** p32768/g128 → **21.1 vs 13.9 t/s
(+52%)**, TTFT 48.1 s; p8192 +33%; p512 +19%. DFlash gain GROWS with context
(draft hides more of the KV-read cost).

**Prefill (llama-bench pp, no-spec; DFlash does not change prefill):** pp32768 =
**865 ± 4.5 t/s** (128K ctx, r=3); short-pp from Run 34 (auto-ctx): pp512 858,
pp2048 1278, pp4096 1301, pp8192 1172. Long-prompt prefill degrades (pp4096 1301
→ pp32768 865): SWA caps attention cost but memory access grows.

**Findings:**
1. DFlash n_max=2 is the 128K optimum (same as 64K) — small windows win; n8
   crashes, n5-7 collapse via acceptance decay.
2. 128K ctx fits on 32 GB with `--parallel 1` (~2.3-4 GiB free) but decode and
   prefill both pay a long-context tax (p512 decode 26.8 at 128K vs 29.2 at 64K).
3. Operations: the vLLM profile manager repeatedly stole :8765 mid-run (killed
   per phase; unit renamed `.disabled-by-sweep`). llama-bench hangs after
   printing (kill after results). Long llama-bench runs buffer output — give
   generous timeouts or they die empty.

## Run 36 — Nemotron 3.5 Lightning DFlash on vLLM XPU: isolated n=5 Lane 1 (2026-08-13)

**Status:** E2 self-report with raw evidence. Isolated C1 n=5 Lane 1 card.
Eligible as a LocalMaxxing self-report. **Not independently reproduced.**
The earlier n=3 screen (`results/nemotron-dflash-bf16-n7-screen-20260813T080234Z-1389483/`)
is **superseded** and must not be used for public medians.

**What:** first isolated publication-grade proof that NVIDIA DFlash
(`method=dflash`, `num_speculative_tokens=7`) loads, drafts, and accepts on
vLLM XPU against the local Nemotron-3.5-Lightning GPTQ-INT4 G64 target.

**Canonical evidence:**
`results/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/`
(`results.json`, `n5-summary.json`, `manifest.txt`, per-rep raw SSE + metrics
snapshots, `monitor.jsonl`, `correctness-summary.json`, `spec-counters.json`,
`localmaxxing-payload.json`).

**Stack:** local derived image `vllm-openai-xpu:1da0a954-det0123` from
`vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57`;
vLLM `0.26.1rc1.dev668+g3ee2df303`, kernels `0.1.12.3`. Native grouped-topk v2,
SSU B8/W4, `VLLM_XPU_ENABLE_XPU_GRAPH=1` PIECEWISE+FULL, `--async-scheduling`,
`--quantization gptq --dtype float16 --max-model-len 16384 --max-num-seqs 1
--no-enable-prefix-caching`. Target:
`/mnt/<models-disk>/nemotron-lightning-gptq-sym64-20260812` (local GPTQ INT4 G64
symmetric). Draft: local NVFP4 E2M1→BF16 reconstruction at
`/mnt/<models-disk>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16`.

**Isolation:** production `vllm-dense-profile.service` pinned `Restart=no` for
the whole load+matrix window; `b70dense` did not respawn during measured cells.
Empty-card `visible_avail` prelaunch ~32k MiB; after load **5826 MiB** free.
Production restored after cleanup.

**Method:** C1, cache explicitly off (0 hits / 0 queries on every measured
cell), entropy-first exact chat-template prompts (5 families), one same-shape
warmup discarded per cell, `ignore_eos` on g128, client monotonic SSE. Decode
metric = `(completion_tokens-1)/(end-first)`. Input metric = `prompt_tokens/TTFT`
(cold input rate, **not** isolated engine prefill). Configured cap **150 W**.

| Cell | Metric | n | median | min | max | acceptance |
|------|--------|--:|-------:|----:|----:|-----------:|
| p2048/g1 | cold input (tok/s) | 5 | 6455.6 | 6262.9 | 7353.3 | — |
| p8192/g1 | cold input (tok/s) | 5 | **7160.1** | 7117.3 | 7226.3 | — |
| p512/g128 | C1 client post-first | 5 | 194.61 | 140.20 | 220.01 | 45.1% (492/1092) |
| p2048/g128 | C1 client post-first | 5 | **186.61** | 174.60 | 201.83 | 56.5% (518/917) |
| p8192/g128 | C1 client post-first | 5 | 157.92 | 143.50 | 170.25 | 53.0% (508/959) |

Window spec: **1830/3521 = 52.0%** accepted, 503 drafts. Deterministic
raw-completion replay **exact_match=true** (smoke only). Measured cell-window
averages ~149–160 W; peak interval-average **179.3 W** (p8192/g128); pkg max
**68.0 °C**.

**Representative LocalMaxxing scalars (auditor-approved):**
- `tokSOut` = **186.6** (p2048/g128 n=5 median). p512 194.6 is **not** the
  headline (41% family range, 140–220).
- `tokSPrefill` = **7160** (p8192/g1 n=5 median cold input rate from TTFT).

**Matched-except-speculation** vs prior no-spec graph n=5
(`results/nemotron-gptq-graph-n5-20260812T2045Z/`, p8192/g128 87.25 t/s):
157.92 / 87.25 = **1.81× on that cell only**.

**Findings:**
1. DFlash is real on this XPU stack (~52% acceptance), unlike native MTP (0%).
2. Isolated n=5 decode is slower and tighter than the contaminated n=3 screen
   (p8192 157.9 vs screen 185.1) — the screen is not reusable.
3. p512 remains family-sensitive; use p2048/g128 as the representative decode.
4. Speed is not task-quality or official-quant parity. Both artifacts are local
   conversions. Native MTP is still unfixed.

**LocalMaxxing:** submitted **once** 2026-08-13T08:40:16.496Z as
`cmsr9po4w000ams01e4fc5qhj`, platform status `APPROVED` (published self-report,
not independently verified). Displayed: `tokSOut=186.6`, `tokSPrefill=7160`,
`engineName=vllm`, quant
`GPTQ-INT4-G64-sym-local+DFlash-BF16-local`, notes survived (DFlash n=7, cold
input disclaimer, local conversions, n=3 superseded). Platform flag parser
limitation: `specDecoding=false` / `specMethod=null` / `mtpEnabled=false` even
though `commandSnippet` and notes carry `method=dflash` n_spec=7 — do not treat
	those parsed flag fields as evidence that speculation was off.

## Run 37 — Nemotron DFlash scheduler-budget 16384 cold-input probe (2026-08-13)

**Status:** E2 self-report. **Does not replace Run 36.** The n=3 16K-budget
spike is **superseded / not a headline**. Public Lane 1 input remains Run 36
**7160** at p8192/g1 n=5, `max-num-batched-tokens=8192`.

**Question:** does raising `--max-num-batched-tokens` from 8192 to **16384**
move DFlash cold input (`prompt_tokens / client TTFT`) past 7160?

**Evidence:**
- n=3 screen: `results/nemotron-dflash-prefill-budget16k-20260813T100159Z-1477342/`
- fresh n=5 confirmation: `results/nemotron-dflash-prefill-budget16k-n5-20260813T101252Z-1486488/`

Same stack as Run 36 except `max-num-batched-tokens=16384`. C1, cache
explicitly off (resident-prefix proof 0 hits; every measured cell 0 hits),
150 W cap, isolated production pin, client monotonic SSE, exact tokens.

| Campaign | Cell | n | cold-input median | range | decode post-first |
|---|---|--:|---:|---|---:|
| Screen (PROVISIONAL) | p8192/g1 | 3 | **9873.61** | 9804.60–9874.79 | — |
| Screen | p2048/g1 | 3 | 7614.91 | 7464.87–7906.86 | — |
| Screen | p2048/g128 | 3 | 8108.37 (TTFT-derived) | — | 189.22 |
| **n=5 confirmation** | **p8192/g1** | **5** | **6095.62** | 5731.93–7006.17 | — |
| n=5 | p2048/g1 | 5 | 5597.54 | 5250.18–5738.20 | — |
| n=5 | p2048/g128 | 5 | 6788.20 (TTFT-derived) | — | **167.16** (136.80–188.13) |

n=5 vs published Run 36 p8192/g1 7160: **−14.9%**. n=5 decode 167.16 vs
Run 36 186.61 at the same cell. After-load `visible_avail` 6637 MiB (vs
Run 36 5826 at batched=8192).

**Verdict:** the 16K scheduler budget is **not** a confirmed DFlash cold-input
lever. The n=3 9874 cell did not survive a fresh n=5. Keep the published
recipe at batched=8192 and tokSPrefill **7160**. Do not headline 9874, 10k,
or “prefill got further.”

**Comparison class:** `matched_except` scheduler budget vs Run 36; n=3 vs n=5
are not interchangeable.

## Run 38 — Nemotron DFlash staged context-capacity ladder to 120000 (2026-08-13)

**Status:** E2 self-report. **Capacity claim only.** Does **not** replace Run 36
speed numbers (186.61 / 7160). Not a 128K claim. Diagnostic TTFT / post-first
rates on g32 cells are **not** headlines.

**Question:** does this isolated DFlash stack complete real requests past the
published 16K serving card if `--max-model-len` is raised to **120000**
(fallback 100000)?

**Evidence:**
`results/nemotron-dflash-ctx-capacity-20260813T103627Z-1501089/`
(`manifest.txt`, `results.json`, `server-120000.log`, `kv-120000.txt`,
`vram-after-load-120000.txt`, exact prompts, per-cell raw SSE, `monitor.jsonl`).

**Stack:** same isolated DFlash path as Run 36
(`vllm-openai-xpu:1da0a954-det0123`, GPTQ-INT4 G64 + DFlash BF16 n=7, graphs,
native grouped-topk v2, SSU B8/W4, cache explicitly off, C1, batched=8192,
U=0.90, configured **150 W**). Only `max-model-len` changed: **120000**.

**Load gates at 120000 (no fallback):** after-load `visible_avail` **5328 MiB**;
logged **GPU KV cache size 295,000 tokens** (4.43 GiB KV memory, 2.46×
concurrency at 120000). Same 4.43 GiB as Run 36's 120,832-token line: hybrid
FLASH_ATTN set attention block size to **2176** so the reported token count
changes with `max-model-len`. Trust the completed 119,904-token request, not
a naive 120832→128K extrapolation. Empty-card prelaunch ~31k+ MiB. Production
pinned `Restart=no` for the window and restored after.

**Method:** protocol §12 staged ladder. Exact rendered chat-template prompts
(`len(encoded["input_ids"])`). One same-shape warmup discarded + **one**
measured C1 completion per rung (`ignore_eos`, `finish_reason=length`). Cache
off proof 0 hits; every measured cell 0 hits. Boundary cell is p119872/g32
(120000 − 128).

| Cell | Prompt + output | finish | n | Class |
|---|---:|---|--:|---|
| p32768/g32 | 32768 + 32 = 32800 | length | 1 | capacity completion |
| p65536/g32 | 65536 + 32 = 65568 | length | 1 | capacity completion |
| p98304/g32 | 98304 + 32 = 98336 | length | 1 | capacity completion |
| **p119872/g32** | **119872 + 32 = 119904** | length | 1 | **max completed** |

**Verdict:** `--max-model-len 120000` **loads and completes** through
119,904 tokens on this isolated DFlash stack. The published **speed** recipe
remains Run 36 at 16K / batched=8192 (186.61 / 7160). **128K was not run.**
Logged KV 295k would *admit* a later 128K attempt; that is not a 128K result.

**Comparison class:** `not_comparable` to Run 36 decode/input medians (different
`max-model-len`, g32 vs g128, n=1 vs n=5).

## Run 39 — Qwen3.8-27B day-0: llama.cpp GGUF Lane 1 + first-ever GPTQ-Int4/MTP-BF16 artifact on vLLM XPU (2026-08-14)

**Status:** E2 self-report, PROVISIONAL pending claims audit for any public
wording. Two LocalMaxxing submissions accepted (llama.cpp cells only). Full
day-0 narrative: `research/qwen38-27b-day0-adoption-20260814.md`.

**Evidence:** `results/qwen38-gguf-20260814T150404/` (manifest.txt,
llama-bench n=3/n=5 logs @150W/230W, server load+smoke log, GPTQ quantize
run2/run3 logs with failure evidence, vLLM no-spec/MTP1/2/4/8 + async/16K/
newest-nightly/W4A8 screen logs, lmx payloads + submission records).

**Release:** `Qwen/Qwen3.8-27B` completed 2026-08-14T15:00:01Z (sha
`1d4bf0f2`, 55.59 GB BF16). Config diff vs Qwen3.6-27B: zero structural
deltas (same `qwen3_5_text`, same GDN hybrid, same 1-layer MTP head tensor
names) — every 3.6 recipe transferred unchanged.

**Artifacts:**
- llama.cpp: community `unsloth/Qwen3.8-27B-GGUF` Q4_K_M (17.11 GB, SHA-256
  `7b2aec3b…cc89f1b`), MTP head stripped by converter.
- vLLM: **self-quantized** `/qw38-gptq-out/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
  (gptqmodel 7.3.2 on B70 XPU, wikitext-2 128-text calibration, bits=4
  sym=true group=128 desc_act=false, `mtp.*` excluded → BF16; verified 400
  qweights / 15 BF16 mtp tensors / 0 leaks / 19.56 GB). Quantization cost:
  one host OOM reboot (run1), one ENOSPC at save (run2, swapfile/output
  space collision), success on run3.

**Correctness:** rungs 1-4 pass on both paths (load, greedy smoke 5/5,
deterministic replay, g128 token exactness). Reference parity and task
quality NOT run.

**llama.cpp (b10255+ build 0804, q8_0/q4_1 KV, FA on, C1, llama-bench
mean±stdev):**

| Cell | 150W n=5 | 230W n=5 |
|---|---|---|
| tg128 | 17.78 ±0.17 | **22.30 ±0.01** |
| pp2048 | 785 ±21 | 1116 ±0.96 |
| pp4096 | 808 ±2.1 | 1137 ±1.9 |
| pp8192 | 787 ±1.7 | 1121 ±1.9 |

Submitted to LocalMaxxing at 15:41:52Z (150W) and 15:50:32Z (230W) after a
claims-audit correction (mean not median; 3.6-comparison n-mismatch
disclosed). Matched-cap direction: 3.6 Q4_K_M @150W tg128 21.3 (Run 9, n=3)
— 3.8 ~16% slower on llama.cpp at matched conditions.

**vLLM XPU (pinned `2c427ef` digest + patch_mtp_nightly + patch_mtp_boundary,
`VLLM_XPU_ENABLE_XPU_GRAPH=1`, fp8 KV, batched 8192, `--max-num-seqs 64`
(GDN Mamba-cache caps decode seqs at 72), 230W, C1, cache off, exact tokens
via /tokenize, entropy-first unique prefixes, client post-first rate):**

| Mode | p512/g128 | p8192/g128 | cold input p8192 |
|---|---|---|---:|
| no-spec (150W, eager, n=3) | 24.4 | 24.7 | 1146 |
| MTP1 (99.0% accept, n=3) | 47.5 | 46.0 | 1705 |
| MTP2 (97.0%, n=3) | 64.3 | 59.5 | 1740 |
| **MTP4 (n=5)** | **81.26 median** (max 81.39; dip 56.6 retained) | **77.04** (75.3-77.2) | **1731** |

**Rejected levers (all screened, evidence retained):** MTP8 71.0/75.1
(acceptance decay); async-scheduling 75.8/74.3 (dense regression);
16K batched budget 78.5/74.5; newest nightly `ac7509e` 78.9/74.6 (below
pinned); W4A8 patch — no prefill gain AND output corruption (g_idx artifact;
path requires no-g_idx) — W4A8 dead for this family.

**Headline (PROVISIONAL):** Qwen3.8-27B GPTQ-Int4/MTP-BF16 on vLLM XPU:
**81.26 t/s** decode median p512/g128 n=5 C1 230W; **1,731 t/s** cold input
p8192. vs 3.6 Run 31 MTP4 (69.3, same class): +17% decode. Bottleneck
verdict (architect analysis appended to the research doc): MTP4 decode runs
at the 608 GB/s bandwidth ceiling; dense prefill is XMX-compute-bound —
95 decode / 3000 prefill require a multi-layer MTP checkpoint / the MoE
track, not kernel tuning.

**Comparison class:** 3.6 Run 31 = matched-except-checkpoint (same digest,
same flags class, same power); llama.cpp 3.6 Run 9 = matched-except-checkpoint
at 150W with n-mismatch. Never cross-cap without labels.

---

## Run 40 — Qwen3.8-27B 4-mode matrix on the LATEST vLLM XPU nightly (2026-08-15)

**Stack (new generation):** `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
(local tag `nightly-ac7509e2b...`), observed vLLM **0.27.2rc1.dev77+gac7509e2b**,
vllm-xpu-kernels **0.1.12.3**. Same two patches as Run 39 (patch_mtp_nightly
`f1db50bf...` then patch_mtp_boundary `41d2f74e...`). Model
`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (uploaded 2026-08-15,
16 files). fp8 KV, scheduler 8192, `--max-num-seqs 64`, graphs on,
`B70_MTP_BF16_DRAFT=1`, cache OFF (`--no-enable-prefix-caching`, zero
prefix-cache-query delta — runtime proof), 230 W configured cap, C1, client
monotonic timing, n=5 (full-context n=1), exact tokens via /tokenize.

**Capacity:** 131,072 ctx at U=0.88 (MTP modes) / 0.90 (no-spec): 997 MiB free
after load, logged KV 145,015 tokens. batched 16384 at 128K = EngineCore OOM
(VRAM-bound; lever closed on this image).

**Results (median, tok/s):**

Cold input (tokens/TTFT): no-spec 1851/1848/1809/**1774**; MTP1 1817/1813/1776/1738;
MTP2 1810/1810/1770/1736; MTP4 1795/1800/1767/1728 (p2048..p8192/g1).

Decode p512/g128 (post-first): no-spec 32.9 · MTP1 52.0 · MTP2 65.8 · **MTP4 83.7** (78.3–84.4).
Decode p8192/g128: 31.5 · 50.0 · 62.9 · **77.1**.
Control p9445/g128: 31.4 · 49.9 · 62.3 · **79.0**.
Full-context p130944/g128 (n=1): 23.2 · 38.9 · 44.4 · **56.3** (exact 131,072).
MTP acceptance: MTP1 100.0/99.7/100.0; MTP2 99.3/97.7/96.5; MTP4 95.0/93.7/96.2.

Power (campaign window, includes warmups): mean 197/198/199/196 W;
max 0.5 s interval-avg 274/276/275/274 W.

**Headline (PROVISIONAL, E2, audited):** Qwen3.8-27B MTP4 on the latest
nightly: **83.7 t/s** decode p512/g128 n=5 C1 230 W; **1,774 t/s** cold input
p8192/g1 (no-spec); exact-128K full-context **56.3 t/s** (n=1). Best
cold-input median anywhere: 1,851 t/s (no-spec p2048) — **2000 t/s prefill
NOT reached**; the new nightly is +1.7% cold / +3% decode over pinned Run 39.

**Comparison class:** vs Run 31 Qwen3.6-27B (69.3 @ p512/g128) =
`directional_only` (different model checkpoint, build 0.27.2rc1 vs
0.26.1rc1.dev457, U 0.88 vs 0.90, cache off vs on-zero-delta). Not a
model-alone gain claim. MTP4 wins 10 of 11 decode cells (p130560/g512 ties
MTP2 36.2 vs 36.3 — not a win).

**Evidence:** `results/vllm-qwen38-4mode-230w-20260815T170840Z-581426/`
(summary.json, tables.md, claims-audit-20260815.md, per-mode raw cells +
monitors). LMX submission `cmsur82fz06svms01ga1f0z83` (APPROVED, 83.7 /
1774 / 131072). Cookbook: docs/qwen38-27/QWEN38-VLLM-XPU.md + dashboard SVG.
Status: PROVISIONAL — NOT FOR PUBLIC HEADLINE until independent reproduction.

## Run 41 — Qwen3.8 GDN mixed-split v5 + autoresearch config exhaustion (2026-08-18)

**Not a speed generation.** Same image/model/flags as Run 40 champion
(`f01e24f6…`, GPTQ-INT4-sym-G128 + BF16 MTP4, U=0.88, 131K, cache off,
230 W). Adds `scripts/tmp/vllm-xpu-causalconv-patch/patch_gdn_mixed_split_v5.py`
so a mixed spec+non-spec `gdn_attention` step is two fused calls instead of
the exclusive host `TORCH_CHECK` (vllm-xpu-kernels#510).

**Correctness:** concurrent probe completed `prefill=1 spec=1 n_ns=1187 n_sp=5`,
health 200. Must patch **every** `_xpu_ops.py` (EngineCore used site-packages
while `/workspace/vllm` was the first candidate). CPU unit test 16/16.
Sequential autoresearch agentic is **not** a mixed-step proof.

**C1 n=3 screen (PROVISIONAL, not a Run 40 replacement):** prose 83.77 /
code 81.1 / agentic 38.76 / prefill 1734.7 vs last keep `8924799`
83.49 / 80.94 / 38.57 / 1720.1. Speed-flat; adopted as infra.

**Autoresearch:** config-space exhausted. Last keep remains `8924799`.
Closed after the overlay: seqs32, seqs128 (Mamba 104), 64K@0.88, 64K@U=0.90
(296 MiB < 500), MTP3, batch4096, MTP1/2, prefix-cache cold, bf16, DSpark
(no served cell). Production `qwen38-pi-serve.sh` is now MTP4 + overlay
(`NOSPEC=0`, `GPU_UTIL=0.88`).

**Evidence:** `research/qwen38-gdn-mixed-split-v5-20260818.md`,
`results/qwen38-gdn-mixed-split-v5-20260818/`, `~/autoresearch/results.tsv`.
Status: PROVISIONAL — NOT FOR PUBLIC HEADLINE.

## Run 42 — Qwen3.8 S+M1 generation curve + isolated C1 128K (2026-08-19)

**Status:** E2 provisional self-report with raw evidence.
**PROVISIONAL — NOT FOR PUBLIC HEADLINE.**

**What:** Lane 3 expansion of the research S+M1 arm (v5 mixed-split +
`B70_DRAFT_LMHEAD_INT4=1` + `B70_DRAFT_MTP_INT4=1`) on the champion image.
User ask was max context (128K if it fits). After-load `visible_avail` =
**870 MiB** → isolated C1 capacity only (500–1,023 MiB gate). Not preferred
C1 performance (≥2,048 MiB) and not mixed/concurrent.

**Stack:**
- Image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
- Model `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, MTP4, U=0.88, max-len 131072
- Overlay `scripts/tmp/patch_champion_stack_overlay.py` (includes v5)
- Prefix cache off; C1; `ignore_eos=true`; configured **230 W**
- Timing: client post-first `(completion-1)/(t_end-first)`; cold input =
  `prompt_tokens / TTFT`
- Cache: `warm_model_cold_prefix`; every measured cell `prefix_cache_hits`
  delta = 0
- Statistic: median of n=5 after same-shape warmup; exact p/g tokens;
  `finish_reason=length` on every row; 0 exclusions
- Cleanup: cap restored 150 W; empty-card `visible_avail` 31,919 MiB

**Reused Lane 1 (not remeasured this run)** from
`results/qwen38-pr-n5-20260818T214749Z` research-mtp4-v5-draftint4:
p512/g128 **112.65** tok/s; p8192/g128 **103.63**; p2048/g1 input/TTFT 1799.5;
p8192/g1 input/TTFT 1696.0.

**Generation curve (this run, C1, 230 W cap):**

| cell | median post-first (tok/s) | max | range | accept | median W | n |
|---|---:|---:|---|---:|---:|--:|
| p512/g32 | 101.57 | 102.73 | 89.59–102.73 | 84.9% | 215.5 | 5 |
| p512/g256 | 110.76 | 112.87 | 85.13–112.87 | 89.0% | 195.6 | 5 |
| p512/g512 | 101.20 | 112.71 | 73.24–112.71 | 81.2% | 193.8 | 5 |
| p8192/g32 | 93.92 | 94.64 | 82.50–94.64 | 86.5% | 229.1 | 5 |
| p8192/g256 | 75.75 | 80.33 | 69.00–80.33 | 62.5% | 215.1 | 5 |
| p8192/g512 | 64.96 | 72.22 | 56.81–72.22 | 51.6% | 206.3 | 5 |

**Staged long context, g128 (this run):**

| cell | median post-first | max | range | input/TTFT | accept | median W | n |
|---|---:|---:|---|---:|---:|---:|--:|
| p16384/g128 | 94.92 | 95.41 | 90.52–95.41 | 1495 | 92.6% | 226.0 | 5 |
| p32768/g128 | 88.25 | 91.59 | 85.00–91.59 | 1269 | 93.8% | 227.6 | 5 |
| p65536/g128 | 76.72 | 79.79 | 76.53–79.79 | 1000 | 94.8% | 228.6 | 5 |
| p98304/g128 | 67.93 | 70.50 | 65.24–70.50 | 827 | 93.4% | 229.0 | 5 |
| p130944/g128 | 62.52 | 65.57 | 58.74–65.57 | 708 | 92.6% | 229.1 | 5 |

p130944+g128 completed all 131,072 tokens on every measured rep. Mid-request
`visible_avail` dipped to tens of MiB; the server stayed healthy. Do **not**
headline 128K as a preferred-perf or production-safe serving point.

**Interpretation:** S+M1 MTP4 still drafts at 128K (acceptance ~93% on g128
long cells). Decode falls with prompt length (112.65 @ p512 → 62.52 @
p130944). On p8192, longer generations lose acceptance (86.5% @ g32 → 51.6%
@ g512) and rate (93.92 → 64.96). g32 is not a sustained number; g512 is the
generation-length diagnostic, not a predeclared sustained window.

**Comparison:** vs the sibling Lane 1 S+M1 g128 cells is `matched_exact` for
stack/cache/cap, different output length. vs cookbook BF16-draft MTP4 81.20
@ p512/g128 remains the prior n=5 matched arm, not this run.

**DFlash 2 (same day, separate smoke):** loads + coherent output with overlay
+ v5 + eager; **acceptance stays 0%** (896 drafts / 0 accepted; selector
scores NaN then sanitized to −1e4). Not a serving path. Note:
`research/qwen38-dflash2-smoke-20260819.md`.

**Evidence:** `results/qwen38-sm1-ctx-gen-20260819T082145Z/`
(`measure/research-mtp4-v5-draftint4-ctx-gen.json`, `server/`, `manifest.txt`).
Harness: `scripts/tmp/qwen38-sm1-ctx-gen.sh` + `qwen38-sm1-ctx-gen.py`.
Claims audit 2026-08-19: Lane 1 112.65 vs 81.20 is cookbook-eligible
(`matched_except_quant`). Run 42 curve/128K may appear only as isolated C1.
LMX 83.7 row untouched. Cookbook tables +
`docs/assets/b70-qwen38-draft-int4-ctx-gen.svg` updated.
Status: E2 self-reported; 128K remains isolated-C1, not a serving headline.

---

## Run 43 — Qwen3.8 prefix-on agentic A/B: S+M1 vs cookbook MTP4 (2026-08-19)

**Status:** E2 self-report with raw evidence. Cookbook-eligible for the
**prefix-on agentic table only**. Isolated C1 (after-load 865 / 871 MiB).
LMX 83.7 row untouched.

**What:** Matched two-arm agentic sweep with prefix caching **on**. Decide
whether draft-INT4 S+M1 still beats cookbook BF16-draft MTP4 on multi-turn
coding. Publish if it wins those cells.

**Stack:**
- Image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
  (vLLM `0.27.2rc1.dev77+gac7509e2b`, kernels 0.1.12.3)
- Model `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, MTP4, U=0.88, max-len 131072
- Cookbook arm: no overlay. S+M1: `scripts/tmp/patch_champion_stack_overlay.py`
  + `B70_DRAFT_LMHEAD_INT4=1,B70_DRAFT_MTP_INT4=1`
- Prefix cache on (`enable_prefix_caching: True` in both server logs)
- C1; configured **230 W**; restored 150 W
- Timing: client post-first `(completion-1)/(t_end-first)`
- After-load `visible_avail`: cookbook **865 MiB**, S+M1 **871 MiB** →
  isolated C1 only (500–1,023 MiB)
- Cleanup: empty-card **32,078 MiB**, cap 150 W

**Prefix-on agentic (client post-first tok/s, median):**

| Cell | Cookbook BF16 draft | Draft INT4 S+M1 | Δ | n |
|---|---:|---:|---:|---|
| short 3×5 g256 | 43.81 | **58.86** | **+34.4%** | 3 sessions, 15 turns |
| 8K start g128 | 48.04 | **66.99** | **+39.4%** | 3 sessions, 15 turns |
| 16K start g128 | 54.40 | **65.92** | **+21.2%** | 2 sessions, 8 turns |

Comparison: `matched_except_quant`. 0 crashes. Short MTP accept 43.3% vs
40.9%. Short measured_avg_W median 192.0 vs 200.9 (energy1_input, n=15).
Cache proof: short t1 hits=0, t5 hits=1664/session; long t2+ hits
4992–18304.

Do **not** headline prefix-on as faster decode than the cache-off agentic
campaign (short 43.27 vs 57.45; 8K 50.06 vs 68.73; 16K 56.29 vs 70.86).
Prefix-on vs that overlay's own cache-off decode is +2.5% / −2.5% / −7.0%
— research note only.

Quality not remeasured. Prior KEEP 12/15 vs 12/15 still stands. 128K
agentic is Run 44 (prefix-on isolated C1), not Run 42 synthetic cache-off.

**Evidence:** `results/qwen38-agentic-cacheon-20260819T093858Z/`
(`comparison.json`, per-arm `*-agentic-cacheon.json`, `server.log`).
Harness: `scripts/tmp/qwen38-agentic-cacheon.sh` + `qwen38-agentic-cacheon.py`.
Cookbook: prefix-on table + `docs/assets/b70-qwen38-draft-int4-agentic-cacheon.svg`.

---

## Run 44 — Qwen3.8 prefix-on agentic ladder 32K→128K (2026-08-19)

**Status:** E2 self-report with raw evidence. Cookbook-eligible for the
**prefix-on 32K/64K agentic rows only**. 96K is thin (`n=2` turns, 1 session).
128K is a **single** cold t1 — not a median, not a serving cell. Isolated C1
(after-load **825 / 816 MiB**). LMX 83.7 row untouched.

**What:** Same two arms as Run 43, prefix cache **on**, C1, configured **230 W**,
client post-first g128. Ladder: 32K (2 sessions × 2 follow-ups) → 64K (2×1) →
96K (1×1) → ~130560 t1 only.

**Stack:** identical to Run 43 (`f01e24f6`, MTP4, U=0.88, max-len 131072).
Cookbook: no overlay. S+M1: overlay + `B70_DRAFT_LMHEAD_INT4=1,B70_DRAFT_MTP_INT4=1`.
Comparison: `matched_except_quant`. Timing: client post-first
`(completion-1)/(t_end-first)`.

**Prefix-on agentic ladder (client post-first tok/s, median of all turns):**

| Cell | Cookbook BF16 draft | Draft INT4 S+M1 | Δ | n |
|---|---:|---:|---:|---|
| 32K start g128 | 46.34 | **61.02** | **+31.7%** | 2 sessions, 6 turns |
| 64K start g128 | 44.83 | **56.20** | **+25.4%** | 2 sessions, 4 turns |
| 96K start g128 | 40.66 | **47.85** | **+17.7%** | 1 session, 2 turns (thin) |
| ~128K t1 only | **43.82** | 37.48 | **−14.5%** | 1 session, 1 turn |

128K actual prompts: cookbook p=130680 / S+M1 p=130675, g=128 both, TTFT
215.76 s vs 209.02 s. Cache hits t1=0 (cold); follow-ups 29952 / 63232 /
96512. MTP accepted/draft on the 128K turn: cookbook 99/128, S+M1 84/176.
0 crashes both arms.

Do **not** headline S+M1 through 128K. Do **not** mix this agentic 128K
single-obs with Run 42 synthetic cache-off p130944/g128 **62.52** n=5.

**Evidence:** `results/qwen38-agentic-128k-20260819T102259Z/`
(`comparison.json`, per-arm `*-agentic-128k.json`).
Harness: `scripts/tmp/qwen38-agentic-128k.sh` + `qwen38-agentic-128k.py`.
Note: `research/qwen38-agentic-128k-20260819.md`.

---

## Run 45 — Qwen3.8 Cn concurrency on v5 + draft-INT4 S+M1 (2026-08-19)

**Status:** E2 self-report with raw evidence + 4 APPROVED LocalMaxxing
records. Cookbook-eligible (concurrency section). After-load **631 MiB**
→ Cn research class (sub-1 GiB reserve), labeled as such. LMX 83.7 row
untouched.

**What:** One arm (champion `f01e24f6`, MTP4, v5 mixed-split + S+M1 envs,
prefix cache ON, U=0.88, max-len 131072, max-seqs 64, MBT 8192, fp8 KV,
configured 230 W, measured draw 221–229 W, 82–83 °C package). Controlled Cn
cells + LocalMaxxing harness evals on the same live server.

**Controlled Cn (client wall-aggregate = successful tokens / wave wall):**

| Cell | Aggregate (median) | Per-stream post-first | TTFT p50 | Accept | OK |
|---|---:|---:|---:|---:|---|
| 5×~6K coding sessions, 3 turns, g128 | 30.13 | 17.1 | 16.5 s | 53.8% | 45/45 |
| C10 p2048/g256 | 105.80 | 31.0 | 7.5 s | 46.4% | 30/30 |
| C16 p2048/g256 | 106.81 | 25.0 | 17.0 s | 46.8% | 48/48 |
| C32 p512/g128 | 160.46 | 28.2 | 12.0 s | 55.8% | 96/96 |
| Mixed 1×p32768+4×p512 g128 | 20.99 | 8.4 | 15.5 s | 58.8% | 10/10 |

0 crashes all cells — the mixed long-prefill + spec-decode cell that kills
the unpatched stack completed 10/10 on v5.

**LocalMaxxing real evals** (their short-prompt remote harness, 256 tok,
3 iters, `--hf-id SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`;
self-reported APPROVED): C1 **56.8** (`cmt00hzaf0efams01r6rw5j14`, TTFT
171 ms), C5 **203.8** (`cmt00hzwf0effms014mdyizca`, 414 ms), C16 **200.6**
(`cmt00i05k0efims01vz3u1kl5`, 8.1 s), C32 **224.2**
(`cmt00i0eb0eflms012anb0yau`, 15.4 s). Aggregate Cn — never place beside
C1 rows (83.7 / 112.65 are separate records). The two Cn tables above use
different workloads (mine prefill-heavy wall, lmx short-prompt) — do not
merge.

**Evidence:** `results/qwen38-concurrency-20260819T110843Z/`
(`measure/sm1-v5-cacheon-concurrency.json`, `lmx/c{1,5,16,32}.json`,
`server.log`, `docker-run.out`). Harness: `scripts/tmp/qwen38-concurrency-cn.sh`
+ `qwen38-concurrency-cn.py`. Cleanup: empty-card **31,976 MiB**, 150 W.

---

## Run 46 — Qwen3.8 C5 realistic g512 + Pi-prompt isolation + current-stack C1 record (2026-08-19)

**Status:** E2 self-report with raw evidence + 1 APPROVED LocalMaxxing record.
Closes the C5 realism bracket and isolates the 71-vs-112.65 decode anomaly.

**Campaign A — C5 realistic (8K sessions, g512, prefix on, 230 W, 0 crashes):**
4 wave-sets × 5 sessions × 3 turns, 60/60 OK. Median wave aggregate
**53.96 tok/s** (51.6–68.2), per-stream post-first median **27.7** (min
18.5), TTFT p50 22.4 s, accept 47.0%, 220.4 W, hits 81,536 (t2/t3 partially
cached; assistant stub makes t2 cold — conservative). C1 anchors same
server: p512/g128 **71.5** (64.8–88.4) on INDEX filler (44% accept);
p8192/g1 cold input **1693.9** (identical to the Aug-18 record's 1696).
Evidence: `results/qwen38-c5-realistic-20260819T114142Z/`.

**Campaign B — prompt-content isolation (same stack, calibrated Pi prompt
files from the 112.65 run, cache ON):** p512/g128 **100.2** (92.9–105.1),
accept 89–96%; p8192/g128 **100.0** (96.3–104.1), accept 88–98%. Verdict:
**prompt content drives MTP acceptance** (INDEX filler 44% vs realistic text
94%); cache flag exonerated for the main effect, with a secondary ~5–11%
cache-on cost at zero hits (100.2 vs 112.65). First attempt
`qwen38-pi-isolation-20260819T115853Z` failed on a harness-filename typo
(server loaded fine, no measurements) — retained as failure evidence.
Evidence: `results/qwen38-pi-isolation-20260819T120424Z/`.

**LocalMaxxing record:** C1 current-stack **100.2** (`cmt01ygp40eg9ms016odaz6kc`),
tokSPrefill 1693.9, notes carry the prompt-sensitivity caveat and the
112.65 cache-off sibling. Harness:
`scripts/tmp/qwen38-c5-realistic.{sh,py}`, `qwen38-pi-isolation.sh`,
`qwen38-pi-prompt-isolation.py`.

**Run 46 amendment (2026-08-19, metric decomposition):** the C5-realistic
"54 tok/s aggregate" published above is the **turn-completion wall**
(includes each wave's 25–40 s prefill phase). Decomposition across all Cn
cells: per-stream median / Σ streams (n × per-stream median) / wall —
C5 8K/g512: **27.7 / 138.3 / 54.0**; C5 6K/g128: 17.1 / 85.6 / 30.1;
C10: 31.0 / 309.5 / 105.8; C16: 25.0 / 400.5 / 106.8; C32: 28.2 / 903.0 /
160.5. **Do NOT use max(e2e)−max(ttft) decode-window aggregates**: with
staggered scheduling they overstate badly (749 at C16, 1786 at C32 here).
New LMX record from the decomposition: C5 realistic **138.3** Σ-streams
(`cmt023kzf0egfms01rxx5jhv7`; a duplicate identical submission was created
at 2026-08-19T12:16:59Z by a re-run of the same submit command — CLI has no
server-side delete; both carry identical numbers/notes). Cookbook §11
rewritten with the three-metric table and this warning.

---

## Run 47 — Qwen3.8 C5-realistic v2 (real history) + C1 refresh (2026-08-19)

**Status:** E2 self-report with raw evidence + 2 APPROVED LocalMaxxing
records. Supersedes the v1 C5-realistic numbers (stub artifact).

**What:** v1's harness stubbed assistant replies, breaking prefix
continuity (full re-prefill each turn; prefill-starved decode). v2 keeps
the REAL completions in session history. Same stack as Run 45/46
(`f01e24f6`, MTP4, v5 + S+M1, prefix ON, 230 W, U=0.88; after-load 628 MiB;
two earlier launch attempts aborted: capacity floor 309 MiB, then a harness
filename typo — retained under `results/qwen38-c5v2-20260819T12{2136,3253}Z`).

**C1 (same server, calibrated Pi prompts, n=5):** median **106.72**
(103.23–111.27), accept 89–96%, TTFT ~0.335 s. Record
`cmt03mj040eh8ms01trjvhm75` (supersedes same-day 100.2 — run-to-run spread).

**C5 v2 (5×~8K sessions, 3 turns, g512, real history, 4 wave-sets, 60/60):**

| Turn | Per-stream | Σ streams | Wall agg | TTFT p50 | Hits | Accept |
|---|---:|---:|---:|---:|---:|---:|
| t1 (cold) | 33.1 | 165.5 | 65.4 | 22.6 s | 0 | 43% |
| t2 | 25.6 | 127.9 | 49.3 | 25.0 s | **0** | 54% |
| t3 | 22.9 | 114.2 | 47.9 | 22.8 s | 19,968 | 43% |
| all turns | **25.5** | **127.4** | 49.4 | 22.9 s | — | — |

**Key finding — prefix reuse largely fails at C5 on this build:** with real
completions kept, t2 lands 0 hits and t3 only ~38% of shared tokens
(19,968 / ~53K), while the same session shape at C1 hits 91% (Run 44: t2
TTFT 5.15 s). Every turn re-prefills ~45–53K tokens → TTFT stays 22–25 s.
Root cause not isolated (suspect MTP + prefix-cache + concurrency interplay
on this XPU build). Warm-session TTFT at Cn = open issue.

C5 record `cmt03mjo60ehbms0117c5i745` (127.4 Σ-streams, per-turn detail in
notes). Why C5 Σ ≈ 1.2× C1: acceptance collapse (94→43–56%), 5× resident KV
reads, and the re-prefill above; the short-prompt C5 203.8 record avoids
all three.

**Evidence:** `results/qwen38-c5v2-20260819T123851Z/`. Harness:
`scripts/tmp/qwen38-c5v2.sh` + `qwen38-c5v2-real.py`. Cleanup: 31,920 MiB,
150 W.

---

## Run 48 — Qwen3.8 at Qwen-recommended sampling + launcher wiring (2026-08-19)

**Status:** E2 with raw evidence + 2 APPROVED LocalMaxxing records.
Same stack as Run 47 (`f01e24f6`, MTP4, v5 + S+M1, prefix ON, 230 W,
after-load per launch gate; 45/45 C5 requests OK).

**Qwen3.8-27B model-card presets:** non-thinking = temp 0.7 / top_p 0.80 /
top_k 20 / presence_penalty 1.5; thinking = temp 1.0 / top_p 0.95 / top_k 20
(checkpoint `generation_config.json` ships the thinking defaults).

| Cell | Greedy (Run 47) | Recommended non-thinking | Recommended thinking |
|---|---:|---:|---:|
| C1 p512/g128 median (accept) | 106.7 (~93%) | **102.61** (95.7–106.7, 91.8%) | 99.32 (81.9–107.8, 82.7%) |
| C5 realistic Σ-streams (per-user) | 127.4 (25.5) | **107.8** (21.6; t1 134 / t2 106 / t3 101; acc 34–41%) | — |

**Correction of the 2026-08-19 temperature caveat:** the earlier "expect
lower tok/s at sampling temperatures" was too pessimistic at C1 — with
top_k/top_p truncation, recommended-sampling acceptance stays 91.8% (C1).
The real sampling cost is at C5 (−15% Σ). All public numbers now state
their sampling.

**Records:** C1 recommended `cmt051y730eiims01prf8laxc` (102.61), C5
recommended `cmt051ytx0eilms01fdxrujeh` (107.8 Σ). **Wiring:** Desktop
launcher `~/Desktop/b70-launcher.sh` (champion stack + presets banner +
`--generation-config auto` + isolation gates/150 W restore); AGENTS.md §15.14
sampling rule; recommended-configs sampling block; harness
`scripts/tmp/qwen38-sampling-real.py` (`NONTHINK`/`THINK` constants).
**Evidence:** `results/qwen38-sampling-20260819T131817Z/`.

---

## Run 49 — Qwen3.8 DFlash2 overlay n=5 screen (2026-08-19)

**Status:** E2 provisional self-report with raw evidence.
**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Labeled n=5 screen, not a
Lane 1 card, not a cookbook recipe.

**Question:** after mixed-dtype + causal-fallback made DFlash2
acceptance non-zero (fix12 n=1 15.7%), what is C1 median client
post-first tok/s and window acceptance on Run 40 calibrated prompts?

**Stack:** image
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
(vLLM `0.27.2rc1.dev77+gac7509e2b`); target GPTQ-Int4 + BF16 MTP;
draft `/mnt/<models-disk>/Qwen3.8-27B-DFlash2`; `method=dflash` n=7
probabilistic; overlays `patch_mtp_nightly.py` +
`patch_mtp_boundary.py` + `patch_gdn_mixed_split_v5.py` + PR #52816
DFlash2 + mixed-dtype recast + **causal fallback**; graphs off /
`--enforce-eager`; `--no-enable-prefix-caching` (log
`enable_prefix_caching=False`); max-len 16384; U=0.88; fp8 KV; MBT
8192; max-num-seqs 8.

**Cell:** C1, same-shape warmup discarded, n=5, greedy temp 0,
`enable_thinking=false`, `ignore_eos=true`, finish=length, g=128.
Prompt JSON target 512; **actual endpoint prompt_tokens=476** every
sample. Timing: client post-first via `stream_options.include_usage`.
Spec/cache: per-request `/metrics` diffs.

| i | p | g | post-first tok/s | acc/draft | hits | measured avg W |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 476 | 128 | 22.412 | 81/336 | 0 | 128.0 |
| 1 | 476 | 128 | 23.085 | 82/315 | 0 | 128.1 |
| 2 | 476 | 128 | 23.010 | 84/315 | 0 | 127.3 |
| 3 | 476 | 128 | 21.924 | 82/322 | 0 | 126.1 |
| 4 | 476 | 128 | 19.866 | 77/357 | 0 | 126.6 |

- median **22.41 tok/s** (max 23.09, min 19.87, n=5)
- acceptance **24.7%** (406/1645)
- cache: `warm_model_cold_prefix`; hits delta 0
- after load: 5152 MiB free (preferred C1 performance zone); KV 3.31
  GiB / 26,014 tokens; concurrency at 16,384: 1.59x
- power: configured **230 W**; per-request energy median **127.3 W**;
  campaign monitor (includes load) pkg max 72.0 °C, vram max 70.0 °C,
  peak ~1 s interval-average 226.2 W (`hwmon4` xe `0000:0b:00.0`)
- correctness: coherent-output smoke only

**Comparison to Run 47 MTP4 106.7:** `not_comparable` (different spec
method, draft, context, cache, overlay). Do not headline a
multiplier. Causal fallback is the wrong mask for block diffusion;
SDPA n=1 was slower (12.53 tok/s, 6.8%).

**Decision:** keep MTP4 as the serving spec. DFlash2 stays a research
overlay.

**Evidence:** `results/qwen38-dflash2-n5-20260819T151054Z/`. Harness:
`scripts/tmp/qwen38-dflash2-n5.sh` + `qwen38-dflash2-n5-client.py`.
Writeup: `research/qwen38-dflash2-smoke-20260819.md`. Cleanup:
31,640 MiB visible_avail, 150 W.

## Run 50 — Ornith-1.5-35B-A3B day-0 GPTQ + MTP screen (2026-08-19/20)

**PROVISIONAL n=3 — NOT FOR PUBLIC HEADLINE.** Pre-authorized 18:30 job.

- **Model:** `ornith-ai/Ornith-1.5-35B-A3B` (sha fbb995a7) — arch MATCH to
  Qwen3.6-35B-A3B (`qwen3_5_moe`, hybrid GDN, 1-layer MTP). No official GPTQ.
- **Self-GPTQ:** gptqmodel 7.3.2 in champion container; INT4 g128 sym,
  mtp/attn/shared_expert/visual/embed BF16; 128-text wikitext calib;
  offload_to_disk on /mnt/<models-disk>; 6.4 h; RTN fallback ~6-9% expert modules.
  Artifact 24.42 GB `/mnt/<models-disk>/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16`,
  contract verified (30,720 qweight, 785 MTP BF16, 0 leaks).
- **Stack:** champion `f01e24f6` + patch_mtp_nightly + patch_mtp_boundary;
  f16, gptq, language-model-only, block 64, U=0.85, cache OFF, 230 W cap.
  32K no-spec: KV 169,622 tok, 6,024 MiB free after load. 16K MTP cells.
- **C1 greedy-diagnostic, client SSE post-first medians (n=3):**

| Mode | p512/g128 | p~2.9K/g1 cold input | acceptance |
|---|---:|---:|---|
| no-spec 32K | 69.0 | 9,537 | — |
| MTP4 16K | 65.2 filler / 69.8 real | 9,330 | 24.8% (pos 81/15/2.5/0.5%) |
| MTP1 16K | **94.3** filler / 94.3 real | — | 80.3% pos0 |

- **Spec verdict:** MTP works; **MTP1 optimal (+37% vs no-spec)** — depth-4
  chain dies past pos0 on this head (inverse of Qwen3.8 dense MTP4).
- **Failures:** /dev/accel docker 127; serve attempt 1 without
  language-model-only/f16/block64/ZE-env pattern never healthy (logs lost to
  trap — pattern copied from champion launcher, attempt 2 healthy in 275 s).
- **Open:** 64K/128K ladder, task-quality parity vs BF16, Cn, prefix-on,
  n=5 set. Full writeup: `research/ornith15-35b-a3b-adoption-20260819.md`.
- **Evidence:** `results/ornith15-35b-20260819T1630Z/` (download, quantize,
  server1-4 logs, c1-screen jsons, spec-metrics deltas, mtp/mtp1 cells).
  Cleanup verified: 150 W, no engines, 32,373 MiB visible_avail, offload purged.
