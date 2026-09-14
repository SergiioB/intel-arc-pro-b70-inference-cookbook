---
name: b70-benchmark-visuals
description: Render and verify evidence-backed SVG dashboards, charts, method diagrams, and social or portfolio graphics for Intel Arc Pro B70 benchmark results. Use when a B70 campaign changes measured numbers, a cookbook or portfolio post needs visual explanations, an SVG must be regenerated from summary JSON, or benchmark graphics need publication qualifiers and visual QA.
---

# B70 Benchmark Visuals

Create graphics from canonical evidence instead of copying performance numbers by hand.

## Source contract

1. Read the campaign compiler output, normally `summary.json`.
2. Reject an incomplete, failed, or unreviewed campaign.
3. Pull every performance value, stack version, and model identity from the JSON.
4. Keep fixed protocol text in code only when it is not a measured value.
5. Preserve the evidence level. E2 results must say `self-reported, not independently reproduced` on-canvas.

Never merge prefill, decode, full-context, and concurrency into one speed number. Name the timing source:

- Input: `actual endpoint input tokens / client TTFT`.
- Decode: `client post-first tok/s` plus TPOT where space permits.
- Concurrency: aggregate output tok/s and request latency; never a C1 row.

## Render the standard graphics

Use the bundled dependency-free renderer:

```bash
python3 .agentic/skills/b70-benchmark-visuals/scripts/render-prefill-decode-svg.py \
  results/<run>/summary.json \
  --dashboard /path/to/cookbook/docs/assets/b70-prefill-decode-dashboard.svg \
  --method /path/to/cookbook/docs/assets/b70-benchmark-method.svg
```

The dashboard must show:

1. Exact cold input/TTFT cells: p512, p2048, p4096, p6144, p8192, and full context.
2. Fixed-output decode at p512 and p8192 for g32, g128, g256, and g512.
3. Full-context g128 and g512 results.
4. The matched p9445/g128 control when present.
5. Workload-specific winners. Never present one universal winning MTP depth.

The method diagram must preserve this sequence:

1. Pin image digest, model ID, vLLM, XPU kernel, and patch hashes.
2. Calibrate exact rendered endpoint tokens.
3. Separate the compact standard prompt from the full Pi prompt.
4. Run generic and full-output same-shape warmups.
5. Record five C1 requests per cell with unique prefixes.
6. Require exact token counts, zero cache-hit delta, and retained MTP/VRAM/failure evidence.
7. Compile medians, exclusions, and replacements.
8. Pass the claims gate before cookbook, portfolio, or social publication.

Fixed-length decode uses `ignore_eos=true`. Retain natural early-EOS output as user-behavior evidence, exclude it from the fixed gN cell, and rerun.

## Visual format

Use the existing renderer as the format authority:

- Dashboard canvas: 1600 × 1280.
- Method canvas: 1600 × 900.
- Background `#07111f`; panels `#0f1d30`; structure `#263a53`.
- Text `#f5f8fc`; muted text `#9caec4`; accent `#55d6be`.
- Mode colors: no-spec `#aeb9c7`, MTP1 `#55d6be`, MTP2 `#68a7ff`, MTP4 `#ffb45c`.
- Use system sans-serif fonts and tabular numbers. Do not depend on external fonts, scripts, raster images, or animation.
- Use rounded panels, structural borders, restrained highlights, and enough spacing for mobile scaling.
- Include `<title>`, `<desc>`, `role="img"`, and `aria-labelledby`.

Do not use a screenshot of a table when an SVG can encode the same data. Do not draw a single axis that mixes prompt processing and token generation.

## Publication workflow

1. Regenerate tables and SVGs from the same summary.
2. Copy SVGs byte-identically to cookbook and portfolio asset folders.
3. Embed the dashboard beside result tables and the method graphic beside reproduction steps.
4. Link the machine-readable summary and renderer source.
5. State the tested stack near every visual.

Verify identical copies with `cmp`. Never edit generated SVG values directly.

## Visual verification

1. Serve the output over a local HTTP server.
2. Open each SVG in a real browser at its native viewport.
3. Capture a full-page screenshot.
4. Check clipped text, overlapping labels, incorrect winners, low contrast, mobile scaling, and footer qualifiers.
5. Regenerate after fixing the renderer; do not patch generated SVG markup.

Run the skill validator after changing this skill:

```bash
python3 skill://skill-creator/scripts/quick_validate.py \
  .agentic/skills/b70-benchmark-visuals
```
