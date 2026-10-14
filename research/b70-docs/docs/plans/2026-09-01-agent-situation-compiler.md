---
title: "Agent Situation Compiler Implementation Plan"
type: plan
date: 2026-09-01
status: final
---

# Agent Situation Compiler Implementation Plan

> **For Hermes:** Use subagent-driven-development only after Sergio says to implement. This file is the spec. Do not implement in the same turn as a docs-only request.

**Goal:** Give an agent a 2-second compile of ≤12 files for one intent so it does not reload the whole B70 library.

**Architecture:** A stdlib Python CLI (`tools/situation.py`) — implemented 2026-09-01, replacing the archived 2026-08-30 registry generator (`archive/registry-system-20260830/`, tool at git 09356e50). It reads JSON registries (not Markdown), matches `blocked_pattern` against `--goal`, and writes `docs/indexes/situation-latest.md` plus JSON stdout. It never overwrites canonical registries.

**Tech Stack:** Python 3 stdlib only. Inputs: `registry/research-ledger.json`, `registry/model-status.json`, `registry/campaigns.yaml` (JSON-subset), this repo's AGENTS.md section map (static).

---

### Task 1: CLI skeleton and intent enum

**Objective:** `python3 tools/situation.py compile --intent bench --goal "test"` prints JSON and exits 0.

**Files:**
- Create: `tools/situation.py`

**Step 1:** Add argparse with subcommands `compile` and `close` (close can be a stub that exits 2 with "not implemented").

Intents: `ops`, `bench`, `diagnose`, `research`, `publish`, `site`.

**Step 2:** Run:

```bash
python3 tools/situation.py compile --intent bench --goal "smoke"
```

Expected: JSON with keys `intent`, `goal`, `working_set`, `skill`, `blocked`, `do_not`, `budget_files`. `working_set` may still be the static table from `docs/governance/agent-operating-system.md` §3.

**Step 3:** Commit `tools/situation.py`.

---

### Task 2: Ledger + status filters

**Objective:** `blocked` and `skip_cells` are computed from JSON, not from prose.

**Files:**
- Modify: `tools/situation.py`

**Step 1:** Load `registry/research-ledger.json`. For each entry with `blocked_pattern` and `outcome` in `{blocked, failed}` and no `superseded_by` that is itself open, if `re.search(pattern, goal, re.I)` hits, append the id to `blocked`.

**Step 2:** Load `registry/model-status.json`. If `--family` or a family token in `--goal` matches, attach that row under `route` and list `validated` cells in `skip_cells`.

**Step 3:** If any `blocked` id matches and `--force` is absent, set `halt: true` and keep working_set to ledger + ADR evidence path only.

**Step 4:** Unit-style check (no pytest required):

```bash
python3 tools/situation.py compile --intent bench --goal "stock tensor-parallel-size 2 docker"
```

Expected: `blocked` contains `RL-008`, `halt` true.

---

### Task 3: Static working-set tables + 12-file cap

**Objective:** Each intent emits the paths listed in `docs/governance/agent-operating-system.md` §3, then any filter-driven extras, then fail if >12.

**Files:**
- Modify: `tools/situation.py`

Hard fail (`exit 2`) when:
- `len(working_set) > 12`
- intent is `site` or `publish` and any path starts with `results/` unless `--evidence-root` is passed
- a path does not exist (warn + drop, except kernel `AGENTS.md` which is fatal)

Write `docs/indexes/situation-latest.md` as a generated view (not numeric authority). Do not commit that file if it is gitignored; if tracked, regenerate in `research_registry.py generate` later.

---

### Task 4: `close` subcommand (accretion gate)

**Objective:** Refuse "done" unless ledger/status/registry were touched.

**Files:**
- Modify: `tools/situation.py`

```bash
python3 tools/situation.py close --campaign <id> --outcome blocked|validated|failed|provisional
```

Checks:
1. `results/<id>/` or registered campaign exists when outcome is not `blocked` without measurement
2. ledger JSON has a row for this question dated today or `--ledger-id` is passed
3. if `--route-changed`, model-status JSON mtime or explicit `--status-updated`
4. `python3 tools/situation.py compile` (all intents) returns halt=false

Does not invent ledger rows. Prints the missing step.

---

### Task 5: Wire AGENTS.md + DOCUMENT-MAP

**Objective:** Kernel says "compile first" and points at this CLI.

Already drafted in the 2026-09-01 docs pass. After the CLI exists, replace the sentence "Until `tools/situation.py compile` exists" with the exact command.

**Verify:** `python3 tools/check-links.py` still passes for new relative links.

---

### Task 6: xmxlab compile path

**Objective:** `--intent site` runs against `~/projects/xmxlab` working set only (AGENTS.md, DESIGN-AGENT-SYSTEM.md, cookbook pin). No B70-DOCS results.

Optional later: a thin wrapper `node scripts/situation.mjs` is YAGNI. Keep one Python CLI; pass `--root` if needed.

---

## Out of scope

- Rewriting `b70-inference-ops` skill numbers
- Auto-publishing to cookbook
- Changing Syncthing receiveonly
- New XeCores pages
