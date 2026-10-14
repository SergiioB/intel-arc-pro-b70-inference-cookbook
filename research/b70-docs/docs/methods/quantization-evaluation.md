---
title: "B70 Quantization Evaluation Method"
status: living
type: method
date: 2026-09-15
---

---
id: b70-method-quantization-evaluation
title: B70 Quantization Evaluation Method
document_type: method
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/methods/benchmark-protocol.md
scope: Evaluation of model-weight, KV-cache, activation, and draft-model quantization on B70
---

# B70 Quantization Evaluation Method

## Purpose

This method defines how to evaluate a quantization choice for B70 inference without conflating format support, artifact integrity, correctness, capacity, performance, and task quality. It applies to model weights, KV cache, activation quantization, routed-expert packing, and speculative draft artifacts.

The method does not declare a universal best bit width or format. Decisions remain exact to the engine, build, model checkpoint, artifact, kernel path, workload, and quality scope tested.

## Independent gates

Evaluate these gates separately and in order:

| Gate | Question | Required disposition |
|---|---|---|
| Architecture support | Does the exact engine/build implement the model architecture and requested modality? | supported, blocked, or unknown |
| Checkpoint compatibility | Do metadata, packing, group size, symmetry, zero points, activation order, tensor names, and shapes match a registered loader/kernel path? | compatible, incompatible, or patch-dependent |
| Artifact integrity | Is the candidate complete, traceable to its source, and numerically sane? | pass, fail, or incomplete |
| Correctness | Does the candidate meet the declared correctness scope? | truth verdict plus scope label |
| Capacity | Does the candidate fit the declared context/concurrency with required VRAM reserve? | observed boundary; not inferred from load |
| Performance | Does it improve the declared C1, Cn, latency, or sustained metric at matched coordinates? | statistic plus comparison class |
| Quality | Does it preserve the target tasks or reference distribution to the declared tolerance? | task or parity verdict |
| Reproducibility | Can another operator obtain the same artifact and stack? | evidence tier and availability status |

A conversion completing successfully proves only that a file was produced. It does not pass the remaining gates.

## Candidate identity

Record:

- source repository, revision, files, and hashes when available;
- source precision and quantization method;
- conversion or repack tool, version/commit, exact command, calibration data, seed, and patches;
- output repository/path, files, sizes, hashes, and quantization metadata;
- whether the operation is native quantization, repack, dequantize/requantize, or requantization from an already quantized source; and
- any expected extra loss from the lineage.

Do not infer format from a repository title. Inspect `config.json`, quantization configuration, tensor dtypes, names, shapes, scales, zero points, and index coverage.

## Evaluation sequence

### 1. Static inspection

Verify configuration and tensor index coverage. Confirm that every expected matrix is present and that the candidate's group size, symmetry, activation ordering, packing, scale layout, and zero-point representation match the intended backend.

For routed MoE, inspect ordinary linear and routed-expert paths separately. Support in an ordinary linear kernel does not prove support in a fused routed-expert kernel.

### 2. Numerical artifact checks

Where source tensors are available, compare reconstructed candidate weights or dequantized samples against the source. Define the error metric and coverage before inspection. Sample-only checks remain partial; a claim of complete conversion validation requires complete matrix coverage or an explicitly justified equivalent.

A repack may be lossless only when it preserves represented values. A dequantize/requantize operation and any `--allow-requantize` path create a new lossy artifact and must be labeled accordingly.

### 3. Engine load and dispatch

Load the exact artifact in isolation after the protocol preflight. Retain logs proving the selected kernel or fallback path. Record post-load VRAM and apply the protocol's capacity guard. A silent BF16/dequantization fallback is not evidence of the intended quantized fast path.

### 4. Correctness ladder

Apply the [correctness validation method](correctness-validation.md). At minimum, a performance candidate must be `coherent_output_smoke_tested`. Adoption for structured, agentic, reasoning, long-context, or multimodal use requires tests covering those claimed modes. A changed quantized draft head requires separate speculation-on and speculation-off validation.

### 5. Capacity

Test real prompt-plus-output completions at the declared boundary. A configured context or successful server launch is not capacity evidence. Record KV type, scheduler/speculation buffers, graph mode, post-load reserve, C1/Cn scope, and completion status.

### 6. Matched performance

Use the smallest claim-sized campaign in the protocol. For tuning, screen baseline and candidates at two anchor cells with `n=3`, then rerun finalists with fresh `n=5` samples. Do not publish screening medians.

A quant comparison normally uses `matched_except_quant` only when all non-quant coordinates match. If tokenizer, checkpoint lineage, engine path, cache state, power, or timing source also changes, downgrade the comparison class or declare it not comparable.

Report model-weight speed, KV capacity, and quality as separate outcomes. Never treat lower VRAM use as a speedup or a throughput gain as quality preservation.

### 7. Quality evaluation

Choose one or both:

- `logit_or_KL_parity_tested` against an appropriate reference; or
- `task_quality_tested` on the target workload.

External KLD or perplexity evidence may inform candidate selection, but it is `E3` for the external artifact and does not establish local task quality or parity for a local conversion. State model, dataset, baseline, tool, metric, and source.

## KV-cache-specific rules

Treat model-weight quantization and KV-cache quantization as separate decisions. In particular, model-weight `Q4_0` is not KV-cache `q4_0 K`.

For llama.cpp quantized KV:

- flash attention must be enabled where required by the engine;
- evaluate K and V asymmetrically rather than assuming the same precision;
- protect K precision because it controls attention routing; and
- use the existing [KV Cache Quantization reference](../reference/kv-cache-quantization.md) for the recorded `q8_0` K + `q4_1` V fleet decision and its evidence.

Any proposal to use `q4_0` for K cache is blocked by the existing recorded catastrophic divergence evidence unless new matched evidence overturns the decision through the formal correction process.

## Decision criteria

A quantization may be recommended only when:

- exact engine/architecture/checkpoint compatibility is demonstrated;
- artifact provenance is reproducible or its local-only status is explicit;
- the required correctness and quality scopes pass;
- the target context/concurrency fits with the declared reserve;
- performance is measured at matched cells with valid statistics; and
- limitations and unavailable artifacts are stated next to the recommendation.

The decision record must include triggers for reconsideration, such as a new upstream kernel, native artifact, source revision, quality failure, context target, or reproducible image.

## Required output

Record the candidate matrix, gate verdicts, correctness scope, quality method, capacity observation, performance cells, comparison class, evidence tier, artifact availability, recommendation, and reconsideration triggers. Link large numeric evidence instead of duplicating it.

## Related authorities

- [B70 Benchmark Protocol](benchmark-protocol.md)
- [Correctness Validation](correctness-validation.md)
- [Quantization Decision Record](../reference/quantization-decision-record.md)
- [KV Cache Quantization](../reference/kv-cache-quantization.md)
- [Recommended Configurations](../operations/recommended-configs.md)
