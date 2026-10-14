---
title: "B70 inference research"
type: governance
status: living
date: 2026-08-19
updated: 2026-10-09
---

# Find a useful inference improvement

```bash
python3 tools/research.py next
```

Give an agent this instruction:

> Read AGENTS.md, run `python3 tools/research.py next`, and execute the selected
> research loop. Check prior evidence and current primary sources, price the
> mechanism, then run the smallest deciding experiment if prerequisites pass.
> Preserve every sample and failure. Register the verdict. Continue to the next
> feasible question; do not reopen holds, lower gates, or publish without approval.

## Current focus (2026-10-12)

Everything learned so far, in one place:
[findings compendium](research/2026/b70-research-findings-compendium-20261012.md).

| Order | Question | Decision to earn |
|---|---|---|
| 1 | RQ-30: self-calibrated EXL3 (SC5) on the B70 | SC_5.00bpw_H6 has 0.42× (prose) / 0.16× (chat) the KL of plain EXL3 4.0 and now serves through `b70/exl3xpu-mixedk` (RL-179, RL-182). Recover its MTP speed (0.70× plain), then paired confirmation and a route decision |
| 2 | RQ-29: XTQ, our own trellis format for Xe2 | Does an Xe2-cheap trellis code (xe_ih2x2, 1.71× cheaper decode, RL-176) plus our bit-exact quantizer (RL-177, RL-181) match EXL3 KL and reach ≥ 550 GB/s at M=4? |
| 3 | RQ-17: Flash-Next quality closure | Attempt limit reached; needs an operator re-entry mechanism for the GPU0 Strata resets |

Closed in run-4: scalar INT4 with any rotation (RL-171, RL-173), E8 lattice
codebooks (RL-173), the SoA expert layout (RL-172). Stock W4A16 INT4 lost its
speed case on vLLM v0.30.0 (RL-174). The 0xsero kernel-probe program stays
closed; only mixed-K serving work on a new image tag was reopened (RL-180).

Which quant, engine and lever fits which workload on this card:
[format and lever guide](docs/reference/b70-format-and-lever-guide.md).
Invariants every campaign inherits: [standing rules](docs/governance/standing-rules.md).

## One loop, one result directory

1. **Choose:** `next` emits one question, prior outcomes, gates and a short reading set.
2. **Research and price:** `source` snapshots primary evidence; `price` evaluates explicit-unit bounds.
3. **Execute:** `run --plan PATH` freezes the plan and captures bounded commands and cleanup.
4. **Decide:** `assess --run PATH` checks paired gains, evidence and quality; a screen cannot be adopted.
5. **Retain:** `record --run PATH` appends to the existing ledger/campaign registry; `render` updates human views.

Exact commands, plan contract and promotion rules:
[operating guide](docs/governance/agent-operating-system.md).
Equations and evidence limits:
[architecture playbook](docs/methods/architecture-diagnostic-playbook.md).

## Where information belongs

| Surface | Authority |
|---|---|
| What to do next | `registry/research-program.json`; generated `research/PROGRAM.md` |
| What happened | `registry/research-ledger.json`; generated `research/LEDGER.md` |
| What currently serves | `registry/model-status.json`; generated `models/STATUS.md` |
| Raw run evidence | `results/YYYY/MM/<question>-<UTC>-<id>/` |
| Primary references and model-specific knowledge | `research/references/`, `docs/reference/`, `models/`, `engines/` |

Historical evidence remains at its original paths. `DOCUMENT-MAP.md` indexes it;
it is not the agent's starting reading list. The
[reset audit](research/2026/research-reset-audit-20261008.md) explains the faults
and the decisions behind this refactor. The previous workflow is archived under
`archive/research-reset-20261008/`.

## Work safely across machines

Radxa: `~/projects/B70-DOCS`. Desktop: `~/B70-DOCS`.
Syncthing is send-and-receive as authorized on 2026-10-08. Keep one registry
writer across both hosts. The desktop being off blocks GPU measurements, not
source research or workflow verification. Local locks do not coordinate peers.
The old Radxa `~/B70-DOCS` contains historical fragments.

## Verify changes

```bash
python3 -m unittest discover -s tools -p 'test_*.py'
python3 tools/research.py render
python3 tools/research.py check
python3 tools/build-indexes.py
```

No fresh B70 speedup is claimed by the repository refactor. A candidate becomes
a serving recommendation only after fresh matched confirmation, task-quality
review and a separate route decision. Publication requires explicit approval.
