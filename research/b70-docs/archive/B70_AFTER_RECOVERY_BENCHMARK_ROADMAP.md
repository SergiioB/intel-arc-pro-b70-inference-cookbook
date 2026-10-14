# Intel Arc Pro B70 After-Recovery Benchmark Roadmap

**Scope:** one Intel Arc Pro B70, 32 GB VRAM, 608 GB/s, Ubuntu 26.04, llama.cpp SYCL build b9853 as the frozen reference.  
**Goal:** maximize useful inference speed without regressing agent/tool-use quality, while retaining a real 128K single-request profile wherever the model fits safely.  
**Power policy:** 165 W is the efficiency/default profile; 180 W is the maximum sustained performance profile. Do not recommend or soak-test above 180 W.  
**Scheduling policy:** no heavy runs during 08:00–16:30 Europe/Madrid. Start after 17:00 Madrid or early enough to finish before 08:00. Use a lock so only one GPU benchmark suite runs at a time.

This is an execution roadmap, not a claim of new benchmark results. Historical results are useful only as sanity bounds because the production KV configuration (`K=q8_0, V=q4_1`) differs from some earlier controlled runs.

---

## 1. Priority definitions

### MUST — required before restoring or changing production

1. Establish a clean, reproducible recovery baseline against frozen b9853.
2. Prove model load, 128K context allocation, prompt processing, decode, and sustained stability with production KV.
3. Isolate build changes from runtime/config changes with controlled A/B tests.
4. Quantify KV-cache quality using token-distribution divergence and agentic/long-context task success—not speed alone.
5. Benchmark single-stream and realistic agent concurrency separately.
6. Capture complete telemetry, raw logs, exact commands, hashes, and failure classifications.
7. Apply hard quality, stability, context, thermal, and performance acceptance gates.

### SHOULD — high-value optimization after MUST gates pass

1. Tune batch/ubatch and shared-prefix prefill.
2. Tune concurrency profiles independently from the 128K deep-context profile.
3. Revalidate Qwen3.6-27B MTP speculative decoding.
4. Compare current production KV with q8_0/q8_0 for quality and q8_0/q4_1 for capacity/throughput.
5. Compare current llama.cpp SYCL against one newer pinned build and, where model formats permit fair comparison, vLLM/Intel LLM-Scaler and OpenVINO/OVMS.
6. Run a 4-hour finalist soak and replay representative real agent traces.

### EXPLORATORY — only after the production candidate is frozen

1. DFlash or compatible draft-model speculation.
2. N-gram speculation on non-MTP models; never stack it with MTP unless a future build changes the already-observed negative result.
3. oneMKL/XMX flash-attention branches not yet merged into the selected upstream build.
4. IPEX-LLM, SGLang, Vulkan fallback, TurboQuant, IQ quants, or experimental KV formats.
5. New model downloads or conversions not already validated on the card.
6. Frequency-cap tuning. If attempted, discover xe sysfs paths dynamically and verify every write by readback.

---

## 2. Non-negotiable experimental controls

- Use models from SSD (`~/models-cache`), never the HDD benchmark path.
- One benchmark process/server at a time. Use a global lockfile and verify no stale `llama-server`/`llama-bench` process owns the GPU.
- Freeze prompts, tokenization, chat template, sampler, seed, stop conditions, output limit, and API client across A/B cells.
- Use temperature 0 or deterministic greedy decoding for output-regression runs. Use teacher forcing for KV logit/KL tests.
- Source the same oneAPI environment and record every relevant environment variable. In particular, keep `SYCL_CACHE_PERSISTENT=0`, avoid `GGML_SYCL_DISABLE_OPT=1`, and test `GGML_SYCL_DISABLE_DNN=1` only as an explicit MoE A/B cell.
- Keep `-ngl 99`, `-ncmoe 0`, `-fa on`, and `--ctx-checkpoints 0` constant unless that exact factor is under test.
- Cool to at most 52–55 °C before each paired A/B block. Alternate order as ABBA or BAAB rather than running all A then all B.
- Perform one discarded warmup per model/build/runtime configuration. First-run SYCL/JIT and model-load timings are reported separately, never folded into steady-state inference.
- Use at least 5 measured repetitions for microbenchmarks and at least 3 full workload repetitions. Report median, p5/p95, standard deviation, and raw samples; do not report only the best run.
- Report engine prompt/decode timings and client wall time separately. Wall time includes queueing, network, prompt processing, and multimodal embedding; engine decode does not.
- Never compare single-stream tok/s with aggregate concurrent tok/s in the same ranking column.
- Any run with OOM, xe fault, thermal throttling, overlapping GPU activity, wrong power/frequency readback, malformed prompt count, or missing raw log is invalid—not “slow.”

---

## 3. Recovery and build A/B isolation

### 3.1 Frozen build lanes

Create separate immutable build directories and manifests:

| Lane | Purpose | Source/compiler policy |
|---|---|---|
| `R0-original-b9853` | Last known-good binary, if recoverable | Hash binaries and shared libraries; do not rebuild |
| `R1-rebuilt-b9853` | Recovery reproducibility | Exact b9853 source SHA, same oneAPI/compiler versions and CMake flags |
| `C1-current-main` | Upstream challenger | One pinned commit; never benchmark moving `main` |
| `C2-feature` | Optional single-feature branch | Exactly one PR/patch, e.g. oneMKL FA; based on a documented parent SHA |

For every lane record: Git SHA/describe, dirty status, submodule SHAs, CMake cache, compiler and oneAPI versions, kernel/Mesa/Level Zero versions, binary/library SHA-256, link dependencies, build date, build flags, and exported environment.

### 3.2 A/B sequence

**MUST**

1. **Binary recovery check:** `R0` versus historical sanity bounds; this detects system/driver changes but is not a formal build A/B if old raw data used a different KV configuration.
2. **Rebuild reproducibility:** `R0` versus `R1`, with identical runtime flags and prompts.
3. **Source-only change:** `R1` versus `C1`, with compiler, libraries, flags, model, KV, power, and workload fixed.
4. If `C1` changes both source and toolchain requirements, add an intermediate `R1-new-toolchain` build to separate compiler/runtime effects.
5. Change one runtime factor at a time after selecting the build: FA path, DNN toggle, batch, ubatch, KV, speculation, then concurrency.

**Build acceptance:** relative to `R0`/`R1`, a candidate must have no unexplained median regression greater than 3% in single-stream decode or prefill, no p95 TTFT/ITL regression greater than 5%, and no new crash, OOM, driver fault, malformed tool call, or quality-gate failure. A slower build can be retained only if a documented quality/stability improvement justifies it; it cannot be called a speed winner.

---

## 4. Baseline matrix

### 4.1 Mandatory model cells

Run the same frozen corpus and launch policy for these installed/validated candidates:

| Model | Role/hypothesis | Quant | MUST cells | Additional cells |
|---|---|---:|---|---|
| Agents-A1 | Agent-specialized speed candidate | Q4_K_M | 128K, prod KV, C=1, 165 W | C=4/8/16/32; 180 W finalist |
| Qwen3.6-35B-A3B | Quality/speed MoE candidate | Q4_K_XL | 128K, prod KV, C=1, 165 W | q8/q8 KV; DNN toggle; concurrency |
| Qwen3.6-35B-A3B | Higher-quality MoE candidate | Q5_K_M | 128K fit and headroom, prod KV, C=1, 165 W | q8/q8 if it fits; 180 W finalist |
| Ornith 1.0 35B | Maximum-throughput MoE candidate | Q4_K_M | 128K, prod KV, C=1, 165 W | DNN toggle; concurrency 4–32 |
| Gemma 4 26B-A4B | Compact MoE/vision candidate | Q4_K_M | 128K, prod KV, C=1, 165 W | vision lane; concurrency; q8/q8 |
| Qwen3.6-27B MTP | Dense quality/speculation candidate | Q5_K_M | 128K, prod KV, no-spec and MTP-4, C=1, 165 W | MTP sweep; 180 W finalist; vision |
| Ornith 1.0 9B | Lightweight/control candidate | Q6_K | 128K, prod KV, C=1, 165 W | compute-path control; q8/q8; concurrency |

“128K” means a server reports `n_ctx_slot >= 131072` for the tested request slot and successfully processes an actual long prompt; merely allocating `--ctx-size 131072` is insufficient proof.

### 4.2 Mandatory baseline dimensions

Use a staged design rather than a wasteful full factorial.

**Stage A — production restoration (all models):**

- Build: `R1-rebuilt-b9853` (and `R0` when available)
- Context: 131072 total, parallel 1
- KV: `K=q8_0, V=q4_1`
- FA: on
- Power: 165 W
- Speculation: off, except a separate Qwen27 MTP-4 cell
- Workloads: fixed 512-token prompt/256-token output decode, 8K prefill, 128K long-context proof, agentic corpus, 30-minute stability run

**Stage B — controlled reference cells:**

- KV: q8_0/q8_0 for all models where 128K fits; f16/f16 at the longest context that safely fits for KL calibration
- Power: 180 W only for Stage-A qualifiers
- Context checkpoints: 4K, 16K, 64K, 128K
- Concurrency: 1 and 4 initially

**Stage C — finalist optimization:**

- Only top 2–3 quality-qualified models
- Concurrency: 1, 2, 4, 8, 16, 32
- Batch/ubatch sweep
- cold, warm, and shared-prefix prefill
- speculative decoding where architecturally supported
- 4-hour soak and production trace replay

### 4.3 Historical sanity bands, not acceptance baselines

Previously observed single-stream figures (often with q8/q8 and different power/build details) may be used to detect catastrophic recovery failures: Agents-A1/Ornith35 about 68 tok/s, Qwen35 Q4 about 64 tok/s, Qwen35 Q5 about 62 tok/s, Gemma4-26B about 57 tok/s, Ornith9B about 49–51 tok/s, and Qwen27 MTP-4 about 24–26 tok/s at 165 W. They must not be mixed with new production-KV results as if they were matched A/B measurements.

---

## 5. Workload and quality corpus

### 5.1 Performance prompt classes

Use versioned, token-counted prompts and fixed outputs:

1. **Decode isolation:** 512 prompt tokens, 256 and 1024 generated tokens.
2. **Prefill sweep:** 512, 2K, 8K, 32K, 64K, and 128K prompt tokens; 1–16 output tokens so decode does not dominate.
3. **Mixed interactive agent:** 2K–8K prompt, 128–512 output, tool schema included.
4. **Shared-prefix fleet:** identical 4K/16K system+repository prefix with unique 256-token tails.
5. **Long-context agent:** 64K and 128K repository/log/document contexts with answer-bearing evidence at beginning, middle, and end.
6. **Vision, where supported:** fixed images and prompts; report image embedding/prefill separately from decode.

### 5.2 Agentic quality suite — hard gate

Use the exact production system prompt/chat template and replay real sanitized traces where available. Include at least:

- 50 deterministic tool-selection cases: correct tool name, required arguments, types, enums, escaping, and no extra prose when JSON-only is required.
- 30 multi-step tool trajectories: 2–6 calls with state carried between responses; score task completion and invalid/redundant calls.
- 20 planning/instruction-following cases with forbidden actions and ordering constraints.
- 20 code/shell cases with executable unit tests or deterministic assertions.
- 20 refusal/recovery cases: missing tool output, tool error, malformed return, unavailable resource, and prompt injection inside tool/document content.
- 30 long-context cases at 16K/64K/128K, including multi-needle retrieval, conflicting versions, “lost in the middle,” exact citation, and irrelevant distractors.
- Optional external cross-check: a pinned BFCL/τ-bench-style subset or equivalent open agent benchmark, with dataset/version/hash recorded.

Primary scores: task success, valid tool-call rate, exact argument accuracy, unnecessary-call rate, recovery success, long-context retrieval accuracy, instruction violations, and critical-failure count. Text similarity is diagnostic only, not proof of correctness.

### 5.3 Determinism and adjudication

- Run deterministic cases once per cell plus three seeds for cases that require sampling.
- Prefer executable checks and exact schema assertions.
- Blind-pair human review a stratified sample of disagreements between baseline and candidate.
- Do not use the candidate itself as judge. If an LLM judge is used, pin a separate judge and retain prompts/outputs; judge scores cannot override executable failures.

---

## 6. KV-cache quality and KL-divergence evaluation

The production target is `K=q8_0, V=q4_1`. Generic prior evidence suggests it can be a strong capacity/quality compromise, but each target model and long-context regime must be measured locally.

### 6.1 KV configurations

| KV configuration | Purpose | Priority |
|---|---|---|
| f16/f16 | Highest-precision reference at contexts that fit | MUST calibration |
| q8_0/q8_0 | Near-lossless operational reference | MUST |
| q8_0/q4_1 | Production candidate | MUST |
| q5_0/q4_1 | Capacity experiment; K precision risk | Exploratory only |
| q4 K variants | Expected high attention-quality risk | Do not promote; test only for research |

### 6.2 Logit collection protocol

For each model, use identical weights and teacher-forced token sequences. Only KV type changes.

1. Select 100–200 sequences from agent/tool, code, prose, multilingual, and long-context corpora.
2. Capture full-vocabulary next-token logits—not API top-k approximations—at fixed positions after 4K, 16K, 64K, and 128K cached tokens.
3. At each checkpoint capture at least 128 scored continuation positions per sequence where feasible.
4. Compute softmax in float64 after subtracting the maximum logit.
5. Compare each candidate with f16/f16 where it fits; at contexts where f16 cannot fit, use q8/q8 as the operational reference and clearly label the reference change.
6. Preserve raw logits or reproducible compressed sufficient data, token IDs, tokenizer hash, context tokens, and code version.

If the stock server cannot expose full logits, add a minimal pinned harness around the same llama.cpp library or use a verified perplexity/logit tool. Do not substitute top-5 API logprobs and call the result “full KL.” Top-k KL may be reported separately with omitted-mass accounting.

### 6.3 Metrics

Per model, context checkpoint, and corpus class report:

- Mean, median, p95, and p99 `KL(P_ref || P_kv)`
- Jensen–Shannon divergence
- ΔNLL and relative perplexity change on the forced continuation
- Top-1 and top-5 token agreement
- Reference-token rank change and log-probability delta
- Attention-sensitive task deltas: tool JSON validity, exact retrieval, multi-needle accuracy, and task success
- Divergence versus token position to detect degradation that grows with context

### 6.4 Initial KV acceptance gates

Freeze these before looking at candidate results; adjust only based on a documented predeclared calibration study, never to rescue a favored configuration.

For q8/q4_1 versus the applicable reference:

- mean full-vocabulary KL <= 0.010 at every checkpoint and no more than 2× the q8/q8 mean KL;
- p99 KL <= 0.10;
- relative NLL/perplexity degradation <= 1%;
- top-1 agreement >= 97%;
- no statistically significant downward trend exceeding 2 percentage points in agent task success or long-context retrieval;
- zero new critical malformed/unsafe tool actions.

These numerical KL gates are screening thresholds, not substitutes for task quality. Any hard agentic failure rejects the KV profile even if KL passes.

---

## 7. Prefill optimization plan

### MUST

1. Measure cold and warmed prompt processing at 512/2K/8K/32K/64K/128K.
2. Report prompt tok/s, TTFT, peak VRAM, and time spent in model load/JIT separately.
3. Test actual 128K content, not a nominal context allocation.
4. Compare `R1` and `C1` with identical `-b`/`-ub` settings.

### SHOULD

For finalists, use a coarse-to-fine sweep rather than all combinations:

1. Hold ubatch at the current default; test batch 1024, 2048, 4096, 8192.
2. At the best safe batch, test ubatch 128, 256, 512, 1024, then 2048/4096 only if supported and VRAM-safe.
3. Recheck the winner at 8K, 32K, and 128K prompts.
4. Test shared-prefix reuse separately: cold prefix, cache hit, and invalidated-prefix cases.
5. Record whether any oneMKL/XMX path activates and prove it from logs/build metadata.

**Prefill winner gate:** at least 10% median prompt-throughput or TTFT improvement at 32K/128K, with no more than 3% decode regression, no additional OOM/crash, and all quality gates passing.

---

## 8. Concurrency and production-profile design

A single 128K context and high concurrency compete for the same 32 GB VRAM. Maintain separate truthful profiles rather than claiming “128K × 32.” llama-server divides total context across parallel slots.

### Proposed profiles

| Profile | Objective | Parallelism | Required per-slot context | Power |
|---|---|---:|---:|---:|
| `deep-128k` | Maximum-context agent/research | 1 | >=131072 | 165 W default, 180 W optional |
| `agent-balanced` | Interactive + background agents | 2–4 | >=16K or 32K, model-dependent | 165 W |
| `fleet-throughput` | Aggregate short-agent throughput | 8/16/32 | >=4K–8K | 165 W; 180 W comparison |

For each server launch, verify and record `n_ctx_slot`, not only total `--ctx-size`.

### Concurrency sweep

For each finalist, test C=1, 2, 4, 8, 16, 32 with a client that truly issues simultaneous requests. Use:

- closed-loop saturation for maximum aggregate capacity;
- open-loop arrivals at realistic rates for queueing/SLO behavior;
- unique prompts for cold behavior;
- shared-prefix prompts for cache-friendly agent fleets;
- 128 and 512 output-token variants.

Record aggregate output tok/s, per-request tok/s, prompt tok/s, queue delay, TTFT p50/p95/p99, inter-token latency p50/p95/p99, end-to-end latency, request success, fairness (slowest/median request), peak VRAM, and thermal/frequency telemetry. Extract pure decode and pure prefill from server logs; do not derive both from total wall time.

**Concurrency winner gate:** >=15% aggregate goodput at the chosen SLO versus the restored production profile, request success >=99.5%, p95 TTFT and completion latency within the declared profile SLO, no request starvation, and no quality/stability regression. Maximum tok/s with unbounded queues is not a winner.

---

## 9. Speculative decoding

### MUST revalidation: Qwen3.6-27B MTP Q5

Use `llama-server` because `llama-bench` does not exercise speculative decoding.

1. Baseline: no speculation.
2. MTP `n_max=2` and `n_max=4` at fixed `p_min=0.75`.
3. For the faster setting, sweep `p_min=0.50, 0.65, 0.75, 0.85`.
4. Test four diverse prompt classes, 256 and 1024 output tokens, C=1 and C=4, 165 W; repeat the final pair at 180 W.
5. Capture proposed/accepted tokens, acceptance rate, accepted span length, verification time, engine decode rate, wall time, output token IDs under greedy decoding, and agentic quality.

Expected prior behavior—MTP-4 helping the bandwidth-bound dense 27B—is a hypothesis to revalidate, not a new result.

### Rules

- Speculation must preserve the target model distribution under the implementation’s stated exactness guarantees. Verify greedy token identity against no-spec; for sampled runs compare task metrics and distributional drift.
- Do not attempt MTP on a model that lacks compatible MTP layers/head.
- Do not stack MTP and n-gram speculation in the production path; prior B70 testing found a regression.
- Do not extrapolate a 27B dense result to 35B MoE or 9B dense models.

**Speculation winner gate:** >=10% median end-to-end decode improvement on mixed production prompts (not only repetitive prompts), acceptance >=85% or an empirically better speed/quality point, identical greedy outputs where exactness is expected, and all task/stability gates passing.

### Exploratory

- DFlash with a checksum-matched compatible drafter.
- N-gram cache on code/repetitive workloads only, reported as workload-specific.
- Compatible Gemma MTP head only after architecture/build compatibility is proven by a smoke test.

---

## 10. vLLM and OpenVINO comparison

This is a **serving-system comparison**, not automatically a kernel A/B. GGUF K-quants, AutoRound/int4, and OpenVINO IR may encode different weights; those differences can change quality.

### Fairness rules

1. Prefer the same upstream model revision, tokenizer, chat template, quantization method/bit width, prompt corpus, context, output length, and sampler.
2. Record model revision and artifact hashes for every backend.
3. If exact weights/quant cannot be shared, split reporting into:
   - **backend-controlled comparison** where artifacts are genuinely equivalent;
   - **best-deployable-system comparison** where model/quant differences are explicit and quality is re-gated.
4. Do not claim backend superiority from a model-format mismatch.

### Candidate lanes

- **llama.cpp SYCL:** restored baseline and optimized reference.
- **vLLM with Intel LLM-Scaler/XPU:** prioritize a supported Qwen/agent model with pinned HF/AutoRound artifacts. Compare at C=1, 4, 8, 16, 32 and prompt lengths 2K/16K/64K; attempt 128K only if one full slot fits with safe VRAM headroom.
- **OpenVINO GenAI/OVMS:** use a natively supported Qwen3-MoE/GPT-OSS-class or closest exact installed architecture. Start C=1/4/8 and 2K/16K/64K; continuous batching is a separate cell.

### Backend acceptance

A challenger becomes production-eligible only if:

- it passes the complete agentic and long-context suite;
- it supports the required OpenAI API/tool-call behavior and chat template without adapters that alter semantics;
- it loads reliably after reboot and survives the 4-hour soak;
- it provides >=15% aggregate goodput improvement at the selected concurrency/SLO, or >=10% prefill/TTFT improvement for long prompts, without >3% single-stream decode regression unless the target profile is explicitly throughput-first;
- 128K is retained for the deep-context profile, or the backend is explicitly limited to a separate short-context fleet profile;
- operational complexity and rollback steps are documented.

---

## 11. Telemetry and observability

Sample system telemetry at 1 Hz (or faster only if collection overhead is measured) from 30 seconds before load through cooldown.

### Required run data

- UTC and Europe/Madrid timestamps; run ID and phase
- model/source revision, exact artifact path, size, SHA-256, tokenizer/chat-template hash
- backend/build lane, binary/library hashes, driver/kernel/oneAPI versions
- full command and environment
- context total and reported per-slot context
- K/V types, FA, batch, ubatch, parallelism, speculation settings
- configured power cap and verified readback; label it **cap**, not measured power, if no `power1_input` exists
- all available GPU temperature sensors, fan RPM, VRAM usage/free, actual/max GT frequencies and throttle flags
- CPU utilization, RSS/system RAM, disk I/O during load, server/client process IDs
- model load time, JIT warmup time, prompt eval time/tok/s, decode time/tok/s
- TTFT, inter-token latency, queue time, end-to-end latency percentiles
- prompt/completion token counts, finish reason, HTTP status, request success
- speculative proposal/acceptance counters
- stderr/stdout and relevant kernel/xe fault lines before/after the run

### Derived metrics

- single-stream decode tok/s
- aggregate decode tok/s
- prompt tok/s
- goodput within SLO
- tok/s per configured watt and tokens per configured-watt-second, clearly labeled as **cap-normalized proxies**, not measured energy efficiency
- speedup with confidence interval versus paired baseline
- VRAM headroom after model load and at peak
- quality score and critical failure count

---

## 12. Acceptance criteria and decision policy

A profile is production-approved only if every hard gate passes.

### Hard gates

| Area | Gate |
|---|---|
| Stability | Zero crash/OOM/xe reset/driver fault in qualification; >=99.5% request success; clean 4-hour soak for finalists |
| 128K | `n_ctx_slot >=131072`, actual 128K prompt completes, >=1.5 GiB peak VRAM headroom preferred (2 GiB target), and no context truncation |
| Agent quality | No more than 2 percentage points absolute task-success loss versus restored baseline; valid tool-call rate no worse by >1 point; zero new critical unsafe/malformed actions |
| Long context | No more than 2 points absolute loss at 64K/128K; no systematic position-dependent failure |
| KV quality | Meets Section 6 KL/NLL/token-agreement gates and task gates |
| Build regression | Median decode/prefill regression <=3%; p95 TTFT/ITL regression <=5%, unless offset by a predeclared profile-specific gain |
| Thermal/power | Sustained cap <=180 W; recommended default 165 W; no throttle; target max <=75 °C at 165 W and <=78 °C at 180 W |
| Reproducibility | Raw logs, manifests, commands, artifacts/hashes, and validator output all present |

If a model cannot keep 128K with safe headroom, it may qualify only for `agent-balanced`/`fleet-throughput`; its model ID and documentation must state the real per-slot context.

### Optimization promotion gates

- **Single-stream speed change:** promote at >=8% paired median improvement with 95% confidence and all hard gates passing.
- **Prefill-focused change:** promote at >=10% improvement at 32K/128K with <=3% decode loss.
- **Concurrency/backend change:** promote at >=15% aggregate goodput at the target SLO with >=99.5% success.
- **Efficiency choice:** prefer 165 W unless 180 W yields >=8% useful goodput/latency improvement without violating thermal gates.
- Changes below thresholds are recorded but not added to production complexity.

### Final selection

Maintain a Pareto table rather than forcing one universal winner:

1. best 128K agent quality;
2. best 128K speed at passing quality;
3. best balanced C=4 agent profile;
4. best C=16/32 aggregate goodput;
5. best vision profile;
6. lowest cap-normalized cost per successful task.

---

## 13. Documentation and data schema

Use a dated Git workspace such as:

```text
~/research/b70-after-recovery-YYYY-MM-DD/
├── 00_admin/              # scope, schedule, approvals, changelog
├── 01_manifests/          # host, driver, build, model, tokenizer manifests
├── 02_configs/            # server profiles and backend configs
├── 03_corpus/             # prompt manifests; sanitized datasets; hashes
├── 04_scripts/            # launch, client, telemetry, KL and validation tools
├── 05_runs/<run_id>/      # immutable raw logs and request/response records
├── 06_processed/          # normalized parquet/csv/jsonl
├── 07_analysis/           # notebooks, statistics, plots
├── 08_reports/            # phase reports, scorecards, final decision
├── 09_failures/           # minimized reproducers, dmesg excerpts, classification
└── README.md
```

Keep model binaries out of Git and link to `~/models-cache`.

### `run.json` minimum schema

```json
{
  "schema_version": "1.0",
  "run_id": "20260714T180000Z__R1__qwen35-q5__kv-q8-q4_1__c1__ctx128k__165w__r01",
  "status": "success|partial|invalid|failure",
  "invalid_reason": null,
  "timestamps": {"utc_start": "", "utc_end": "", "madrid_start": ""},
  "host": {"gpu": "Arc Pro B70 32GB", "os": "", "kernel": "", "driver": "", "oneapi": ""},
  "build": {"lane": "R1", "git_sha": "", "dirty": false, "binary_sha256": "", "cmake_cache_sha256": ""},
  "model": {"name": "", "revision": "", "quant": "", "path": "", "size_bytes": 0, "sha256": "", "tokenizer_sha256": "", "chat_template_sha256": ""},
  "runtime": {"backend": "llama.cpp-sycl", "command": [], "environment": {}, "ctx_total": 131072, "ctx_per_slot": 131072, "parallel": 1, "batch": 0, "ubatch": 0, "kv_k": "q8_0", "kv_v": "q4_1", "flash_attention": true, "speculation": {}},
  "power": {"cap_w": 165, "readback_w": 165, "measured_power_available": false},
  "workload": {"corpus_version": "", "corpus_sha256": "", "prompt_class": "", "prompt_tokens": 0, "requested_output_tokens": 0, "seed": 0, "repetition": 1},
  "performance": {"load_ms": null, "ttft_ms": null, "prompt_tok_s": null, "decode_tok_s": null, "aggregate_tok_s": null, "itl_p95_ms": null, "e2e_ms": null, "queue_ms": null},
  "resources": {"vram_peak_mib": null, "vram_headroom_mib": null, "temp_max_c": null, "fan_max_rpm": null, "gt0_freq_avg_mhz": null, "throttled": false},
  "quality": {"suite_version": "", "task_success": null, "tool_valid_rate": null, "long_context_score": null, "critical_failures": 0},
  "kv_quality": {"reference": null, "mean_kl": null, "p99_kl": null, "js": null, "relative_nll_delta": null, "top1_agreement": null},
  "artifacts": {"stdout": "", "stderr": "", "telemetry": "", "responses": "", "kernel_log": ""}
}
```

### Required reports

- `recovery-baseline.md`: R0/R1 status and historical sanity comparison
- `build-ab.md`: paired build analysis and confidence intervals
- `kv-quality.md`: KL/NLL/task results by model and context
- `prefill.md`: cold/warm/shared-prefix results
- `concurrency.md`: C=1–32 goodput/SLO curves and real per-slot contexts
- `speculative.md`: acceptance, exactness, speed, and quality
- `backends.md`: controlled versus best-deployable comparisons
- `failures.md`: every invalid/failure cell and minimized reproduction
- `production-scorecard.md`: hard-gate checklist, Pareto winners, selected profiles, rollback

---

## 14. Actionable after-recovery todo

### MUST

- [ ] Confirm the host is reachable only after recovery; do not start heavy work during 08:00–16:30 Madrid.
- [ ] Inventory GPU/driver/kernel/oneAPI, model files, free disk, existing binaries, profiles, and old raw logs.
- [ ] Create the dated Git research workspace and commit the empty structure/schema.
- [ ] Hash and archive the original b9853 binary/libraries as `R0`, if recoverable.
- [ ] Rebuild exact b9853 as `R1`; capture full build manifest and smoke-test one small model, then one 20GB+ server load.
- [ ] Validate telemetry paths/readbacks and benchmark lock; capture pre/post kernel logs.
- [ ] Freeze and hash performance prompts, agentic corpus, long-context corpus, chat templates, and validation code.
- [ ] Run Stage-A 165 W production-KV baseline sequentially for all seven model/quant candidates.
- [ ] Prove real 128K per-slot behavior and VRAM headroom for every advertised deep-context candidate.
- [ ] Run R0/R1 then R1/C1 build A/B on Ornith9B (compute control), Qwen27 (dense/bandwidth), and one 35B MoE.
- [ ] Run f16, q8/q8, and q8/q4_1 KL/NLL calibration at feasible checkpoints; run q8/q8 versus q8/q4_1 through 128K where needed.
- [ ] Run the full agent/tool and long-context quality suite on every production finalist.
- [ ] Reject and document any cell that misses a hard gate; never average failures into performance.

### SHOULD

- [ ] Select the top 2–3 quality-qualified finalists before expensive sweeps.
- [ ] Tune batch/ubatch with the coarse-to-fine prefill plan.
- [ ] Run C=1/2/4/8/16/32 closed-loop and open-loop concurrency sweeps with verified per-slot contexts.
- [ ] A/B `GGML_SYCL_DISABLE_DNN=1` only on MoE finalists; do not make it a global default without model-class evidence.
- [ ] Revalidate Qwen27 no-spec/MTP-2/MTP-4 and p-min settings at 165 W, then the winner at 180 W.
- [ ] Run cold/warm/shared-prefix tests and production agent-trace replay.
- [ ] Compare vLLM/Intel LLM-Scaler and OpenVINO only on supported, pinned, quality-revalidated artifacts.
- [ ] Run 4-hour soaks for each proposed production profile and test restart/rollback after reboot.
- [ ] Publish the production scorecard and update profile names so context, KV, power, backend, and MTP are explicit.

### EXPLORATORY

- [ ] Evaluate DFlash with a compatible verified drafter.
- [ ] Test n-gram speculation only on repetitive/code workloads; do not combine with MTP.
- [ ] Test a pinned oneMKL/XMX FA branch after the stable-build scorecard is complete.
- [ ] Evaluate SGLang/IPEX-LLM/Vulkan/OpenVINO preview paths as isolated backend lanes.
- [ ] Evaluate experimental KV/TurboQuant/IQ formats only with the same KL and agentic gates.
- [ ] Consider new model candidates only after existing validated models have a complete Pareto frontier.

---

## 15. Recommended execution order and stop conditions

1. **Recovery confidence:** R0/R1, small model, then one large MoE. Stop on any xe fault or unexplained >10% regression.
2. **Broad restored baseline:** all installed candidates at 165 W, 128K, production KV, C=1. Stop testing a model if it cannot load with safe headroom or fails agent hard gates.
3. **Quality/KV qualification:** f16/q8/q8/q8-q4_1 and long-context tasks. Do not optimize a KV profile that fails.
4. **Build selection:** choose R1 or C1 before runtime tuning.
5. **Finalist optimization:** prefill, concurrency, and MTP.
6. **Backend comparison:** only after the llama.cpp finalist is a stable reference.
7. **Soak and rollout:** canary one profile, retain immediate rollback, then document the Pareto set.

This order prevents spending nights optimizing a fast but low-quality, context-truncating, or unstable configuration.
