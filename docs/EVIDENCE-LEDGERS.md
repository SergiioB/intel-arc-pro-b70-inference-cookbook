# Evidence ledgers — how this cookbook records what worked and what did not

Two ledgers, both append-only, both validated by `scripts/check-ledgers.py` through
`python3 scripts/check-repo.py`. Their job is to stop the repository from re-running its own mistakes and from
quietly forgetting its own wins.

| Ledger | File | Holds |
|---|---|---|
| Rejected arms | [DO-NOT-REPEAT.md](DO-NOT-REPEAT.md) | Every closed attempt, with outcome, cause, scope, and the condition that would reopen it (`DNR-nnn`) |
| Accepted routes | [WHAT-WORKED.md](WHAT-WORKED.md) + [`data/benchmarks.v1.json`](../data/benchmarks.v1.json) | What to run now, per family, with the evidence boundary (`WIN-nnn` records in the catalog) |

A number in the [benchmark catalog](BENCHMARK-CATALOG.md) is a record of something that *ran*. A row in
`DO-NOT-REPEAT.md` is a record of something that ran and did not earn a place. Both are results. Neither is an
opinion, and neither may be edited into the other.

## Why this exists

The failure mode is expensive and predictable: an agent re-derives a dead end because the only trace of it is a
chat log; or an agent promotes an arm because it was fast once, because nobody wrote down that the fast cell had a
different observer. Both have already happened here. The ledgers make the repository the memory instead of the
session.

## The rules

### 1. No speed number without its shape

Every published rate states: model and revision, quantisation, engine and image/commit, patch stack, card count and
placement, context depth, concurrency, power cap, timing observer, and whether the cache was cold. A rate missing
its shape may not enter a ledger, a catalog record, or a recipe.

### 2. A parity gate before a speed claim

A faster arm is not a better arm. Before any speed result is promoted, compare its outputs with the reference
artifact on a fixed prompt set, and record how many positions matched **and how many did not**.

- Internal self-consistency (same arm twice) is not a parity check.
- A canary smoke proves the engine runs; it does not prove the quantisation is sound (see DNR-009).
- Publish disagreements. `5,959 of 5,965` with six unexplained is a usable record; "outputs matched" is not.

### 3. Unknown formats fail closed

Never infer a tensor's storage type from a filename or a friendly label. Read the per-tensor type census and refuse
the combination before allocating memory. See DNR-014: two engine ports died on this and each cost hours after a
50 GB load.

### 4. One variable per arm, or the arm proves nothing

If the drafter, the target precision, the image, and the topology all change together, no attribution is possible
and the row must say so (DNR-002). Split the arm.

### 5. Same-session control

Compare against a control measured on the same host, in the same session, under the same thermal and power
conditions. A rate from yesterday's session is context, not a control. A private screen without a matched control
does not earn a public recipe.

### 6. Record the cause, or record `unexplained`

A wrong cause is worse than no cause, because it closes a door that may be open. "Unexplained" is a valid, honest
entry and it stays valuable: it tells the next person what has already been ruled out.

### 7. History is immutable

Supersede by adding a new generation and linking the old one. Never rewrite a past measurement because today's
stack behaves differently. A failed new stack does not invalidate a valid historical result, and a valid historical
result does not justify today's recipe.

### 8. Private lab work records causes, not rates

Screens that did not run under the publication gates stay unpublished. Their structural lessons may enter the
ledger with `unpublished lab measurement` in the evidence column and the cause spelled out. Never import a private
rate into a public table, and never adopt somebody else's external result as our reproduction.

## Record shapes

### Rejected arm (`DNR-nnn`)

| Field | Required | Notes |
|---|---|---|
| ID | yes | Next free number. Never reused, never renumbered, never reordered by date |
| Arm | yes | Specific enough that a stranger could reproduce it |
| Outcome | yes | Number **and** shape, or the refusal/error that ended the arm |
| Cause | yes | Or exactly `unexplained` |
| Scope | yes | Where the conclusion stops applying |
| Revisit when | yes | A concrete condition; `Never` allowed for superseded artifacts |
| Evidence | yes | Repository path or `unpublished lab measurement` plus the cause |
| Date | yes | `YYYY-MM-DD` of the measurement, not of the write-up |

### Accepted route (`WIN-nnn`)

Accepted routes are catalog records, not prose: add a record to `data/benchmarks.v1.json` through the
[benchmark format](BENCHMARK-FORMAT.md) and the [recipe confirmation process](CONFIRM-A-RECIPE.md), then reference
it from the family route and from [WHAT-WORKED.md](WHAT-WORKED.md). Include the same parity statement required by
rule 2 — the number of matched and unmatched positions, or an explicit "output parity not yet measured" that blocks
promotion to a default.

## Workflow

```text
question
  -> read DO-NOT-REPEAT.md and the family route            (has this been closed? on what scope?)
  -> check the benchmark catalog for an existing record     (is there a valid number already?)
  -> design the arm: one variable, same-session control, parity gate
  -> run it, capture shape, cold/warm state, and outputs
  -> outcome
       accepted -> catalog record + family route + WHAT-WORKED row
       rejected -> append DNR-nnn with cause and revisit condition
       unclear  -> write it down as unclear; do not promote, do not close
  -> python3 scripts/check-repo.py
```

Nothing in this workflow requires GPU time except the arm itself. Closing an arm from a source reading — kernel
coverage tables, plugin method dispatch, memory arithmetic — is legitimate and cheap; say so in the evidence column.

## Maintenance

- Review both ledgers when a model family is added, an image or patch matrix row changes, or a driver/kernel bump
  lands: a closed arm may have reopened, and a "revisit when" may now be true.
- The `ledgers` gate fails on: a missing file, a missing required column, a duplicate or malformed ID, an empty
  required cell, an unfilled placeholder word or a bare ellipsis, an invalid date, or a markdown link to a path
  that does not exist in the repository.
- The gate cannot judge a cause. That stays a human and reviewer responsibility, which is why the cause column is
  required prose rather than a tag.
