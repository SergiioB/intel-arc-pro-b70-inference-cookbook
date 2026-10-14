---
title: "Chart standard — cross-engine comparison SVGs"
type: method
status: living
date: 2026-09-15
updated: 2026-09-15
---

# Chart standard — cross-engine comparison SVGs

The house style for benchmark charts (born in the 2026-09-15 INT4-class
sweep; reference implementation:
`scripts/tmp/ov-mtp-20260914/render-ctx-sweep-svg.py`, outputs
`results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-{decode,prefill}.svg`).
Every future comparison chart MUST follow this structure — it is the
representation readers (and Sergio) expect.

## Representation rules

1. **One canvas, one question.** Decode and prefill are separate charts.
   Title states model, class, engines, host. Subtitle always carries the
   protocol AND the `PROVISIONAL — NOT FOR PUBLIC HEADLINE` banner until
   the claims audit passes.
2. **Log-x context axis** with fixed ticks 512→262144 (512, 2K, 8K, 32K,
   64K, 98K, 128K, 196K, 229K, 256K); light grid lines; y-axis is median
   tok/s with 4 gridlines.
3. **One color per engine, consistent across every chart in the repo:**
   OpenVINO `#68a7ff`, llama.cpp `#7ee787`, vLLM `#ff8a7a`, reference
   gray `#8a97a8`, warning `#ffd23f`. Dark bg `#07111f`, panel `#0f1d30`.
4. **Line style encodes protocol, not taste:**
   - solid = full-protocol measured cells;
   - dashed = lighter probes (capacity/derived) — must be labeled in the
     legend with the exact protocol (e.g. "40-tok capacity probe");
   - dotted dim = reference arm of a different class (rare; prefer omitting).
5. **Composite curves must state their switching rule** in the series
   label and legend note (e.g. "OV: MTP5 ≤64K → u4 no-MTP ≥98K"). Never
   mark a protocol boundary with a vertical "wall" line the curve crosses —
   the style change + legend note carry it.
6. **Sparse labeling: first, last, and two mid ticks per series.** Dense
   per-tick labels across ≥4 series collide. Per-series label offset
   (above/below) separates neighbors; full grids live in the evidence
   JSONs and campaign SUMMARY, never on the chart.
7. **Legend is a panel BELOW the plot** (series rows: color swatch, label,
   point count, last context), plus a ⚠ note row for protocol boundaries/
   upstream issues and an evidence-path footer. Nothing floats over data.
8. Point markers are small filled circles (r 3.5–4.5).

## Procedure (mandatory order)

1. Chart data comes **only from evidence JSONs** via a committed renderer —
   never hand-typed numbers, never hand-edited SVGs.
2. The renderer must compute medians from raw runs when a file's stored
   `median` omits a field (this exact gap dropped OV prefill to a 98K
   start once — do not repeat it).
3. Regenerate SVGs → validate XML → render PNG (`rsvg-convert -w 1600`).
4. **Vision-QA gate:** run both PNGs through image QA (labels overlap?
   clipped text? legend collisions? all series present?) and iterate to
   PASS before publishing anywhere. A chart that fails vision-QA does not
   ship, even if the data is right.
5. Mirror the exact SVGs to the cookbook assets when the comparison is
   public; keep filenames dated.

## Stale-number rules (why the 15.8 spread)

- Every engine-speed claim in any doc must cite its evidence root and
  date; when a campaign supersedes numbers, the old pages get a
  SUPERSEDED banner pointing at the new ones — the old rows stay (they
  are evidence), the banner stops them being cited.
- Engine taxonomy is explicit: OpenVINO GenAI ≠ Cascadia ≠ OVMS —
  separate rows, separate engine ids, never merged.
- Refresh sequence after any campaign: renderer → charts → engine
  dossiers → cookbook pages → release. Skipping the middle steps is how
  stale numbers survive.
