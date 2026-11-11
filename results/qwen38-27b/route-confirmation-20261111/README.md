# Qwen3.8-27B route confirmation — 2026-11-11

Supplemental matched confirmation of the two deployed one-card stacks. The
public writeup is
[`docs/qwen38-27b/ROUTE-CONFIRMATION-20261111.md`](../../../docs/qwen38-27b/ROUTE-CONFIRMATION-20261111.md).
This directory holds the derived evidence and the public harness. It is **not**
a catalog record and does not modify `data/benchmarks.v1.json`.

## Evidence

| File | Contents |
|---|---|
| `summary.json` | Counts, speed cells, paired comparisons, quality, sessions, retrieval, hard code, parity, concurrency |
| `speed-cells.csv` | Every speed cell with n, rate, TTFT, first-answer time, energy and truncation |
| `quality.json` | 40-item regression floor and 84-item supplementary subset |
| `retrieval.json` | 21 planted-code probes, re-graded from retained answers |
| `sessions.json` | 120 ten-turn session turns, re-graded from retained answers |
| `hard-code.json` | 18 executed Python outputs |
| `parity.json` | Five-prompt greedy repeatability |
| `concurrency.json` | C2/C4 aggregate batches |
| `provenance.json` | Image digest, model/binary hashes, kernel, power and sampling |
| `charts/` | Context scaling, multi-turn latency, quality by area |
| `scripts/` | Measurement client, suites and analysis used for the generation |
| `prompts.sha256` | Hash of the prompt file used to build the cells |

`prompts.json` itself is not vendored: it is a 2.9 MB derived corpus that quotes
historical cookbook text with private lab references. Rebuild it from the public
cookbook source and verify the hash in `prompts.sha256` before rerunning a cell.

## Scope

- One headless Intel Arc Pro B70 32 GB, GPU 1, 180 W cap, one kernel session.
- EXL3 4.00 bpw + MTP3, 262,144 context, fp8 KV, prefix cache on.
- TensorFold 4-bit g64 + DFlash2, 131,072 context, 2 GiB prompt cache.
- Thinking T1.0 / top_p 0.95 / top_k 20; matched non-thinking T0.7 / top_p 0.8 / top_k 20, presence 0 on both (TensorFold rejects 1.5).
- 512 generated tokens for speed; 8,192 for quality. Five paired prompts per speed cell where available.

## Corrections applied after the run

- The original retrieval and session graders called `json.loads` on the raw
  message, so fenced JSON scored as a failure. Re-grading the retained content
  after stripping one optional code fence makes all 21 retrieval probes and all
  120 session turns correct. No inference was rerun.
- Two EXL3 non-thinking `tool-04` responses returned HTTP 200 with 29 completion
  tokens and no visible content. They are excluded from paired scoring, not
  scored as model failures. The same prompt with the first seed returned a
  complete tool call.

See `HARNESS.md` for the measurement definitions and commands.
