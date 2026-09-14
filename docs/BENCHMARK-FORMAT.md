# Stable Cross-Model Benchmark Format

Use this contract when rerunning the B70 vLLM phase-separated campaign on another model. It keeps model results comparable without pretending that request-side vLLM measurements are llama-bench engine metrics.

## 1. Campaign identity

Record these values before launch:

| Field | Required value |
|---|---|
| Hardware | GPU model and visible VRAM, CPU, RAM, OS, kernel, GPU driver/runtime |
| Image | Pullable repository plus immutable digest |
| Software | Observed in-container vLLM and `vllm-xpu-kernels` versions |
| Model | Repository, exact revision, local artifact, quantization, file sizes and hashes |
| Model features | MTP layer/tensor presence, tokenizer and chat-template identity |
| Local changes | Ordered patch names and SHA-256 hashes |
| Runtime | Context, scheduler budget, GPU memory utilization, max sequences, cache mode |
| Power | Operator-selected configured cap, separate from measured draw |
| Evidence | UTC run ID, protocol version, status, harness and compiler identities |

A changed image, package, model revision, quantization, tokenizer, chat template, prompt file, patch, scheduler budget, context, or memory setting starts a new result generation.

## 2. Tokenizer calibration and prompt classes

Calibrate rendered messages inside the tested image. Count tokens with:

```python
encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)
actual_tokens = len(encoded["input_ids"])
```

Never use `len(tokenizer(prompt))` or `len(BatchEncoding)`. Save the exact messages and hashes after the chat template is applied.

Use two prompt classes:

- **Compact fixed benchmark system prompt:** all standard p512 through p8192 cells. Keep it short enough to reach p512 and hash it.
- **Full Pi system prompt:** p9445 historical control and all full-context cells. Hash it separately.

Every generated request set contains six calibrated prompts per coordinate: one same-shape warmup plus five measured prompts. Put unique entropy at the start of every cold prompt while preserving the exact target token count.

## 3. Required phase-separated matrix

### Cold input phase

| Coordinate | Output | Prompt class |
|---|---:|---|
| p512, p2048, p4096, p6144, p8192 | g1 | Compact benchmark prompt |
| p131071 | g1 | Full Pi prompt |

### Decode phase

| Prompt | Outputs | Prompt class |
|---|---|---|
| p512 | g32, g128, g256, g512 | Compact benchmark prompt |
| p8192 | g32, g128, g256, g512 | Compact benchmark prompt |
| p9445 historical control | g128 | Full Pi prompt |
| p130944 | g128 | Full Pi prompt |
| p130560 | g512 | Full Pi prompt |

Run the full matrix for no-spec and for every supported MTP depth. For the current preserved-MTP Qwen model, the modes are no-spec, MTP1, MTP2, and MTP4. Mark an unsupported mode `not supported`; never synthesize a row.

Keep cache/resident-prefix and concurrency campaigns separate from this C1 cold-prefix matrix.

## 4. Warmup and exact output

For every mode and coordinate:

1. Start from the declared server mode.
2. Run one full-output request with the same prompt and output shape.
3. Discard that warmup.
4. Run five measured C1 requests.
5. Require exact endpoint prompt tokens in every request.
6. For decode cells, set `ignore_eos=true` so all measured requests return the requested output length.

A g1 input cell does not need `ignore_eos`. If a decode request stops early, retain it as excluded evidence, rerun the complete affected cell with forced exact output, and point the compiler to the replacement. Never mix partial and exact outputs in one median.

## 5. Cache and speculation validation

The cold-prefix matrix uses prefix caching enabled so it matches the production feature state, but every measured prompt must remain cold:

- entropy comes first, before any shared text;
- the target rendered length is unchanged;
- snapshot `vllm:prefix_cache_hits` and `vllm:prefix_cache_queries` around each measured cell;
- accepted cold cells require cache-hit delta `0`;
- cache query delta must agree with actual prompt tokens;
- absolute process-wide counter values do not need to be zero.

For MTP modes, diff proposed and accepted token counters and report acceptance. For no-spec, require no proposed MTP tokens. Preserve prompt hashes across modes and report output-text parity separately. Matching prompt shape does not prove matching output.

## 6. Capacity and isolation gates

Before each model load:

- no llama.cpp or vLLM process;
- no running inference container;
- approximately 31,000 MiB or more B70 `visible_avail` on an empty 32 GB card;
- one server and one campaign harness only.

After load:

| Free VRAM | Allowed scope |
|---:|---|
| Below 500 MiB | Abort |
| 500 to 1,023 MiB | Isolated C1 capacity observation only |
| 1,024 to 2,047 MiB | Staged C1 research |
| 2,048 MiB or more | Preferred C1 performance work |
| 3,072 MiB or more | Target for mixed or concurrent work |

Logged KV capacity must exceed prompt plus requested output before a full-context cell starts. A server loading at 131,072 is not a completed 128K claim.

## 7. Timing definitions

**C1** means one active measured request and no queued request.

| Published column | Formula and source |
|---|---|
| TTFT | Client monotonic request send to first SSE output token |
| Cold input rate | Actual endpoint prompt tokens / client TTFT |
| Client post-first rate | `(completion tokens - 1) / (request end - first generated token)` |
| Client post-first TPOT | `(request end - first generated token) / (completion tokens - 1)` |
| End to end | Client monotonic request send to request completion |

Cold input rate includes scheduling, uncached prompt processing, and first-token work. It is not isolated engine prefill and must not be labeled llama-bench `pp`.

Client post-first rate is request-side. It must not be labeled engine-native vLLM decode. Do not compare either column directly with llama-bench `pp` or `tg` without a matched timing definition and explicit comparison classification.

## 8. Stable public tables

Publish six separate views:

1. Cold input rate: p512, p2048, p4096, p6144, p8192, full p131071.
2. Decode at p512: g32, g128, g256, g512.
3. Decode at p8192: g32, g128, g256, g512.
4. Historical control: p9445/g128.
5. Full-context decode: p130944/g128 and p130560/g512, with MTP acceptance.
6. Exclusions and replacements.

Near every table group, state units, C1, `n=5`, cache state, timing source, configured cap, exact image, observed vLLM/kernel versions, and E2/E4/E5 status. Report medians by default. Keep more precise values and dispersion in JSON.

## 9. Evidence layout

```text
results/vllm-pi-prefill-decode-matrix-<UTC>-<PID>/
  manifest.txt
  package-versions.txt
  prompts/<coordinate>.json
  no-spec/
    manifest.txt
    server.log
    metrics-before.txt
    metrics-after.txt
    monitor.jsonl
    <coordinate>/results.json
  mtp1/
  mtp2/
  mtp4/
  summary.json
  tables.md
```

Each per-cell result must retain raw request/SSE timestamps, exact endpoint tokens, finish reason, cache and MTP counter deltas, output text or hash, failures, and warmup separation. Each mode retains launch configuration, server log, VRAM snapshot, telemetry, and cleanup state.

Preserve failed, contaminated, superseded, and early-EOS evidence. Label it; do not delete it.

## 10. Compile before copying numbers

```bash
python3 benchmarks/b70-compile-prefill-decode-matrix.py "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.json"
python3 benchmarks/render-prefill-decode-tables.py "$RUN_ROOT/summary.json" \
  > "$RUN_ROOT/tables.md"
python3 benchmarks/render-prefill-decode-svg.py "$RUN_ROOT/summary.json" \
  --dashboard docs/assets/b70-prefill-decode-dashboard.svg \
  --method docs/assets/b70-benchmark-method.svg
```

The compiled `summary.json` is the canonical numeric source. Public tables must be rendered from it or checked cell by cell against it. Never copy a number from terminal output, an earlier post, or a nearby campaign.

## 11. Publication gate

A result may enter README, portfolio, recommendation, social, or leaderboard copy only when:

1. Every required supported cell completed with one warmup and five exact measured requests.
2. The compiler passes prompt parity, token counts, cache semantics, mode completion, exclusions, and server-error checks.
3. Image digest, observed package versions, model revision, patch order/hashes, runtime settings, power provenance, and raw evidence are present.
4. C1 input, C1 post-first output, cache/resident, and concurrency results remain separate.
5. Proposed wording labels medians, units, timing source, cache state, configured cap, evidence tier, exclusions, and correctness scope.

Speed is not correctness evidence. State output parity, token/logit/KL parity, or task-quality evidence only when directly tested.

## 12. Cookbook publication checklist

- Update README with current stack, the six stable views, workload-based mode choice, one matrix command, evidence links, dashboard SVG, and correctness limitation.
- Put the new result first in `REAL-WORLD-PI-BENCHMARKS.md`; keep previous runs below with their original status.
- Update `FULL-SETUP-COMMANDS.md` only when the tested image, packages, model, patch hashes/order, launch interface, or matrix command changes.
- Stage a compact public `results/<date>-summary.json`; do not publish private host paths or require a local-only image.
- Update every portfolio post that repeats a changed number. Copy and embed the generated dashboard and method SVGs byte-identically; retain corrections and historical tables with clear generation labels.
