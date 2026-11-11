# Harness and measurement definitions

This generation compares two deployed OpenAI-compatible servers on one headless
B70. It is a matched stack comparison, not an engine-isolation experiment.

## Stacks

Start each server from the recipe in the cookbook and wait for `/health`:

```sh
# EXL3 4.00 bpw + MTP3, vLLM 0.26.1 fork, 262,144 context, prefix cache on
b70 launch qwen38-27b -e exl3 --ctx 262144 --port 8102 --prefill fp16

# TensorFold 4-bit g64 + DFlash2, 131,072 context, 2 GiB prompt cache
tensorfold-native serve Qwen3.8-27B-MLX-4bit \
  --drafter Qwen3.8-27B-DFlash2 --parallel 1 --context 131072 \
  --prompt-cache-gib 2 --max-tokens 8192 --host 127.0.0.1 --port 8037
```

The EXL3 image digest, model hashes, TensorFold binary hash and kernel are in
`provenance.json`. The tested host used one GPU at a 180 W cap; restore the
original caps after the run.

## Client

`scripts/measure.py` streams `/v1/chat/completions`, keeps the raw SSE, and
records:

- client monotonic send, first generated token, first final-answer content,
  first reasoning content and completion;
- prompt/completion tokens from the server `usage`, never SSE chunk counts;
- finish reason, output text, cache counter deltas and card energy.

`first_answer_s` is the first `delta.content`; `first_reasoning_s` is the first
`delta.reasoning_content` or `delta.reasoning`. A response that spends its whole
budget on reasoning has a first-reasoning timestamp but no first-answer
timestamp. Speed probes use 512 output tokens; quality probes use 8,192.

## Cells

- **Speed:** code corpus at 0, 8K, 16K, 24K, 32K, 64K and 120K; thinking and
  matched non-thinking; a unique-prefix cold request and an identical-prompt
  replay; one discarded same-shape warmup and five fresh paired prompts where
  available.
- **Prose:** 8K, 32K and 120K non-thinking, three pairs; supplementary.
- **Quality floor:** 40 items × 3 repeats × 2 modes × 2 routes, full-answer
  grader from `benchmarks/quality-suite.py`; code executes without network or
  home in `bwrap`.
- **Broad quality:** 84-item audited non-thinking subset × 3 repeats. The raw
  item bank is not vendored; `quality.json` retains the counts, and the doc
  labels it supplementary.
- **Sessions:** 8K and 64K documents, three 10-turn sessions per route; the
  document stays in the message history and the expected three codes are
  planted at 10/50/90%.
- **Retrieval:** three planted codes at 32K, 120K, near-limit and (EXL3 only)
  240K, three repeats.
- **Hard code:** `py-xml`, `py-diff`, `py-jsonpath` × 3, executed against the
  suite's own tests.
- **Parity:** five greedy prompts × 3 repeats for TensorFold draft on/off, EXL3
  MTP3 and EXL3 serial. Internal repeatability first; cross-arm hashes are not a
  329-item parity claim.
- **Concurrency:** C2 and C4 at 32K total context, one discarded warmup plus
  five retained batches.

## Runner and analysis

The public scripts are in `scripts/`:

```sh
python3 scripts/quality.py --base http://127.0.0.1:8102 --route exl3 \
  --hwmon /sys/class/hwmon/hwmon8 --out /tmp/q/exl3/thinking --mode thinking --reps 3
python3 scripts/session.py --base http://127.0.0.1:8102 --route exl3 \
  --hwmon /sys/class/hwmon/hwmon8 --out /tmp/q/sessions/exl3
python3 scripts/retrieval.py --base http://127.0.0.1:8102 --route exl3 \
  --hwmon /sys/class/hwmon/hwmon8 --out /tmp/q/retrieval/exl3 \
  --tokenizer /path/to/EXL3/tokenizer.json
python3 scripts/parity.py --base http://127.0.0.1:8102 --route exl3 \
  --hwmon /sys/class/hwmon/hwmon8 --out /tmp/q/parity/exl3
python3 scripts/concurrency.py --base http://127.0.0.1:8102 --route exl3 \
  --hwmon /sys/class/hwmon/hwmon8 --out /tmp/q/concurrency/exl3 --concurrency 4
python3 scripts/hard_code.py --base http://127.0.0.1:8102 --route exl3 \
  --hwmon /sys/class/hwmon/hwmon8 --out /tmp/q/hard/exl3
```

`quality.py` loads `benchmarks/quality-suite.py` from the repository root.
`hard_code.py` loads the co-located `creative_suite.py`. `session.py`,
`retrieval.py` and `concurrency.py` need the exact `prompts.json` described in
`prompts.sha256`; it is not vendored because it quotes historical cookbook text
with private lab references. `analyze.py` and `report.py` are the derived-summary
and chart generators; they expect the raw evidence tree described in the source.

The broad-quality runner and its item bank are not published in this generation.
The public doc reports only its counts and labels it supplementary; the 40-item
floor is the reproducible public quality check.

## Analysis choices

- Medians for rates and latencies; n is reported per cell.
- Paired ratios use the same prompt bytes and seeds on both stacks.
- Pointwise item/repeat bootstrap intervals are descriptive, not
  multiplicity-adjusted. A 10% margin is required before a speed cell is called
  resolved.
- `usage` is mandatory for a rate; chunk counts are never tokens.
- Cache-hit deltas are retained; warm is an explicit same-prompt replay, not a
  mixed median.
- Truncation is retained as a failure, not silently treated as a complete
  answer.
