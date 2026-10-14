---
title: "Nemotron DFlash on vLLM XPU — isolated n=5 confirmation (2026-08-13)"
type: research
date: 2026-08-13
status: final
---

# Nemotron DFlash on vLLM XPU — isolated n=5 confirmation (2026-08-13)

**Status:** E2 self-report with raw evidence. Isolated C1 n=5 Lane 1 card.
Eligible as a LocalMaxxing self-report. **Not independently reproduced.**
The n=3 screen in this note's first revision is **superseded**.

Canonical evidence:
`results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/`

Superseded n=3 (do not cite medians):
`results/2026/08/nemotron-dflash-bf16-n7-screen-20260813T080234Z-1389483/`

## What worked

Local NVFP4→BF16 DFlash draft + target GPTQ-INT4 G64 + `method=dflash` n_spec=7
loaded on `vllm-openai-xpu:1da0a954-det0123` (vLLM `0.26.1rc1.dev668+g3ee2df303`),
native grouped-topk v2, SSU B8/W4, PIECEWISE+FULL XPU graphs, async scheduling.

Production `vllm-dense-profile.service` was pinned `Restart=no` for the whole
load+matrix window. `b70dense` did not respawn during measured cells. After load:
**5826 MiB** `visible_avail`.

## Isolated n=5 (publication cells)

C1, prefix cache explicitly off (0 hits / 0 queries), entropy-first exact
prompts, warmup discarded, `ignore_eos` on g128, client monotonic SSE. Cap 150 W.

| Cell | Metric | n | median | min | max | acceptance |
|------|--------|--:|-------:|----:|----:|-----------:|
| p2048/g1 | cold input (prompt/TTFT) | 5 | 6455.6 | 6262.9 | 7353.3 | — |
| p8192/g1 | cold input (prompt/TTFT) | 5 | **7160.1** | 7117.3 | 7226.3 | — |
| p512/g128 | C1 client post-first | 5 | 194.61 | 140.20 | 220.01 | 45.1% |
| p2048/g128 | C1 client post-first | 5 | **186.61** | 174.60 | 201.83 | 56.5% |
| p8192/g128 | C1 client post-first | 5 | 157.92 | 143.50 | 170.25 | 53.0% |

Window acceptance **1830/3521 = 52.0%**. Deterministic replay smoke
`exact_match=true`. Measured cell-window ~149–160 W; peak interval-average
179.3 W; pkg max 68.0 °C.

**Representative scalars:** decode **186.6** tok/s at p2048/g128; cold input
**7160** tok/s at p8192/g1. Do **not** headline p512 194.6 (family range 140–220).

Matched-except-speculation vs no-spec n=5 p8192/g128 87.25: **1.81× on that cell**.

## Why the n=3 screen is dead

n=3 p512/p8192 medians 214.6 / 185.1 ran after `vllm-dense-profile` respawned
`b70dense` during Mamba warmup. Isolated n=5 p8192 decode is 157.9, not 185.
Cite only the isolated run.

## Caveats that travel with the numbers

1. Client post-first tok/s is not llama-bench `tg`. Cold input rate is not
   isolated engine prefill.
2. Target and draft are **local conversions**, not official HF quants.
3. Deterministic replay is a smoke, not logit/KL or task-quality parity.
4. Native MTP remains 0% historically. This does not fix MTP wiring.
5. LocalMaxxing `APPROVED` means published self-report, not independent verify.

## Reproduce

```bash
# exact prompts (CPU container, no GPU needed)
# then:
bash scripts/tmp/b70-nemotron-dflash-n5-isolated.sh
```

Do not reuse `b70-nemotron-dflash-bf16-screen.sh` for public numbers (n=3, no
Restart=no pin).
