# Quality suite v2 — published

Both artifacts pass all 40 items. The suite saturates, so it is a **floor**, not a
discriminator: it proves neither model has a mechanical defect on these areas, and it cannot
rank them. Quote it as parity, never as "equal quality".

## Runs

`benchmarks/quality-suite.py`, 40 items, one sample each, temperature 0, thinking on,
`max_tokens` 1536, `llama-server -c 65536`, llama.cpp `1692f9e50` build 361, `q4_1/q4_1` KV,
`-t 4 -tb 4`, both servers up at once on separate cards.

| File | Artifact | Card | Wall | Total |
|---|---|---|---:|---:|
| `nanbeige-q4km-llamacpp.json` | Nanbeige4.3-33B Q4_K_M, rev `9beb6842`, sha256 `2cef7985…4c37` | 1 (`0000:10:00.0`, 230 W) | 331 s | **40 / 40** |
| `qwen38-27b-q4km-llamacpp.json` | Qwen3.8-27B UD-Q4_K_M | 0 (`0000:0c:00.0`, 180 W, display) | 448 s | **40 / 40** |

Areas, each item mechanically graded: math 8/8, coding 8/8 (executed against tests),
structured output 6/6, instruction following 6/6, retrieval 6/6 (4K–32K needles), tool calls
4/4, multilingual 2/2 — identical for both models.

## Grader defects fixed before this run

1. `ifev-05` was case-sensitive (`Cache is useful.` scored as a failure).
2. `ml-02` was ambiguous ("two cards" → Spanish accepts `cartas` for playing cards); replaced
   with "the model is loaded" → keywords `modelo`, `cargad`.

Both are unit-checked by running the grader against known-good and known-bad answers.

## What this does not say

- It does not contradict the earlier 8-case exact battery, where Nanbeige 6/8 and Qwen 7/8 at
  thinking-off differed: that battery probes one arithmetic slip and one verbosity case.
- It does not rank the models. Items are few, one sample each, and both are at ceiling.
  Harder material (the private 94-task v1 set, web/3D pages) is where the two separate.
- Card 0 carries the desktop display and a 180 W cap, so Qwen's longer wall time mixes model
  and host effects. Quality outcomes are unaffected by wall time.
