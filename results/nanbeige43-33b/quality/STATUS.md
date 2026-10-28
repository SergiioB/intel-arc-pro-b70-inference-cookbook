# Quality suite v2 — status

**Not published as a result yet.** These are the raw per-item outputs of the first pass and
must not be quoted as a ranking until the rerun in "Open defects" completes.

## What ran

`benchmarks/quality-suite.py`, 40 items, one sample each, temperature 0, thinking on,
`max_tokens` 1536, `llama-server -c 65536`, one Arc Pro B70 (`0000:10:00.0`, 230 W),
llama.cpp `1692f9e50` build 361 with `q4_1/q4_1` KV.

| File | Artifact | Wall | First-pass total |
|---|---|---:|---:|
| `nanbeige-q4km-llamacpp.json` | Nanbeige4.3-33B Q4_K_M, rev `9beb6842` | 330 s | 39 / 40 |
| `qwen38-27b-q4km-llamacpp.json` | Qwen3.8-27B UD-Q4_K_M | 361 s | 38 / 40 |

Same engine, same flags, same host, same harness. Grading is mechanical only (exact answers,
regex, JSON equality, executed code, needle recall, tool-call arguments, keywords).

## Open defects found after these two runs

1. `ifev-05` grader was case-sensitive, so Qwen's correct `Cache is useful.` was scored wrong.
   Fixed: per-item case flag, and the item now requires ≤6 words plus the word `cache`.
2. `ml-02` ("two cards" → Spanish) is ambiguous: both models answered `dos cartas`, which is a
   valid translation for playing cards. Replaced with "the model is loaded" → keywords
   `modelo`, `cargad`.

Both fixes are unit-checked in `benchmarks/quality-suite.py`; the files above predate them, so
`qwen38-27b-q4km-llamacpp.json` shows a failure that is a grader bug, not a model defect.

## Next action

Rerun both models with the fixed grader on the same host and flags (~6 min each), then:

1. Publish the two corrected runs here and in `docs/nanbeige43-33b/`.
2. Add the totals to `docs/MODEL-COMPARISON.md` as a **v2 suite** row, never mixed with the
   94-task v1-compare table.
3. Add a catalog record per model, and a quality panel to the figure.
