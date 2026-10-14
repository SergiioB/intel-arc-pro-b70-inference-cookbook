---
title: "B70 Correctness Validation Method"
status: living
type: method
date: 2026-09-15
---

---
id: b70-method-correctness-validation
title: B70 Correctness Validation Method
document_type: method
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/methods/benchmark-protocol.md
scope: Correctness validation for B70 engines, builds, kernels, quantized artifacts, and speculative paths
---

# B70 Correctness Validation Method

## Purpose

This method defines the correctness evidence required when adopting or changing a B70 inference engine, build, kernel, model artifact, quantization, conversion, cache mode, graph mode, or speculative-decoding path. Correctness and performance are separate result classes. A fast result does not establish correct behavior.

## Correctness scopes

Use exactly one or more of these labels and describe the tested conditions:

| Scope | Minimum evidence | Does not establish |
|---|---|---|
| `coherent_output_smoke_tested` | The target stack loads and returns syntactically usable, semantically coherent output on declared prompts | Determinism, parity, or task quality |
| `deterministic_replay_smoke_tested` | Repeated deterministic requests on the same stack produce the declared stable output or token sequence | Agreement with a reference stack |
| `token_exact_match_tested` | Candidate and reference produce identical token IDs under controlled deterministic decoding for the declared sequence | Logit closeness outside the observed path or sampled quality |
| `logit_or_KL_parity_tested` | Candidate/reference logits or distributions are compared with a defined metric, tolerance, layers/tokens, and baseline | End-task quality unless separately measured |
| `task_quality_tested` | A predeclared task set is scored with retained outputs and a defined rubric or automatic metric | General quality beyond the task set |
| `independently_reproduced` | Another operator or environment reruns the declared correctness protocol with comparable or matched coordinates | Universal correctness beyond the reproduced scope |

Do not write “correctness verified” without naming the applicable scope.

## Test design

### 1. Declare the change

Identify the candidate and the reference, including:

- engine and exact build, image digest, or commit;
- model repository, revision, files, hashes when available, and chat template/tokenizer;
- model-weight and KV quantization;
- graph/eager, kernel, scheduler, cache, context, and speculation settings;
- deterministic sampling parameters; and
- expected difference, if any.

If there is no valid reference, say so. A candidate may be smoke-tested without being parity-tested.

### 2. Select the smallest sufficient ladder

Run validation in increasing cost order:

1. **Static integrity:** configuration parses; tensor index coverage, names, shapes, dtypes, and quantization metadata are internally consistent.
2. **Load and dispatch:** the exact engine loads the artifact; logs show the intended architecture and kernel path; no silent CPU or dequantization fallback is accepted unless it is the declared candidate.
3. **Coherence smoke:** use short deterministic prompts covering plain text and any claimed structured modes such as JSON, tool calls, reasoning channels, or vision.
4. **Deterministic replay:** repeat identical requests after same-shape warmup and retain token IDs or raw responses.
5. **Candidate/reference parity:** compare token IDs or logits/KL under matched tokenizer, template, prompt, sampling, and stop conditions.
6. **Task quality:** evaluate the target workload, including long-context, tool use, coding, vision, or agent loops only when those capabilities are claimed.
7. **Stress and boundary behavior:** test the graph/eager, cache, concurrency, speculative, and context combinations that the deployment will use.

A lower rung cannot be relabeled as a higher rung.

## Controls

- Use deterministic decoding for replay and parity unless the test explicitly studies sampling.
- Preserve exact rendered input IDs. Re-verify them after any tokenizer or template change.
- Match output limits, stop conditions, reasoning settings, and visible-content handling.
- For fixed-length engine tests, require requested completion length and `finish_reason=length`; early EOS is not a valid fixed `gN` sample.
- Separate warmup/JIT behavior from measured correctness responses.
- Test cache-off and cache-on paths separately when cache behavior is part of the change.
- Test speculation disabled and enabled separately when the candidate changes draft heads, acceptance logic, or final-token boundaries.
- Preserve server logs for exceptions, fallback markers, NaN/Inf warnings, graph breaks, and kernel dispatch.

## Comparison rules

### Token exact match

Record the complete token-ID sequence or a stable hash plus a retrievable raw sequence. Define whether comparison begins at the first generated token and how EOS, stop strings, reasoning tokens, and tool-call serialization are treated. Any mismatch makes the test non-exact; report the first differing position rather than converting it to “close enough.”

### Logit or KL parity

Declare:

- reference precision and execution path;
- compared positions, vocabulary slice if any, and layer scope;
- metric and formula;
- aggregation statistic;
- numeric tolerance chosen before inspection; and
- treatment of NaN, Inf, masked tokens, and tokenizer differences.

Do not import a tolerance from another model or quantization without justification.

### Task quality

Predeclare the task set, scoring rubric, evaluator, retry policy, sampling, and failure treatment. Retain candidate and reference outputs. A task set selected after observing failures is exploratory and must remain provisional until rerun on a fixed set.

## Pass, partial, and fail

Assign a truth verdict to the stated correctness claim:

- `CONFIRMED`: all required tests for the named scope pass with retained evidence.
- `PARTIAL`: some declared capabilities pass, but a bounded mode, task class, or comparison remains unsupported.
- `REFUTED`: retained evidence contradicts the claim or a required test fails.
- `UNCERTAIN`: evidence is missing, unreconstructable, or too weak for a verdict.

A crash, malformed response, silent fallback, non-finite output, deterministic mismatch, or unaccounted early termination is a failure for the affected scope. It does not automatically invalidate unrelated scopes.

## Required output

A correctness result records:

- candidate and reference identities;
- correctness scope labels;
- exact prompts or prompt-set identity;
- rendered token counts and deterministic settings;
- repetitions and warmup;
- expected and observed behavior;
- per-test pass/fail with failure artifacts;
- truth verdict and evidence tier;
- limitations; and
- links to raw responses, token/logit data, logs, harness, and manifest.

Correctness evidence may accompany a benchmark campaign, but it must remain separately labeled from throughput, latency, capacity, and efficiency.

## Related authorities

- [B70 Benchmark Protocol](benchmark-protocol.md)
- [Quantization Evaluation](quantization-evaluation.md)
- [Observability Standard](observability-standard.md)
- [Evidence Lifecycle](../governance/evidence-lifecycle.md)
