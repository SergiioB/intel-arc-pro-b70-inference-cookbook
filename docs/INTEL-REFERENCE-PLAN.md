# Intel inference reference: from a result to a working task

Reviewed 2026-11-10. This is the delivery plan for the cookbook, XeCores, and
B70 Launcher. It adds no speed record or hardware support claim.

## What we provide

Help an Intel GPU owner choose a tested model/engine combination, reproduce a
useful task, and understand the failures and tradeoffs before changing engines.
Start with B70 hardware we have measured. Expand to another device only when
its own installation, capacity, task quality, and cleanup have been checked.

The proposed distinguishing feature is a qualification record tied to an
executable recipe: the record tells a user which deployment decision the
measurement supports. Its usefulness and difference from existing tools still
need independent users to test them. A collection of speed records alone does
not establish that people can finish their work.

## Four surfaces, four responsibilities

| Surface | Owns | Must not become |
|---|---|---|
| Private B70-DOCS | Research questions, holds, failures, raw campaign provenance, adoption decisions | An automatically published dump of lab history |
| This cookbook | Public family routes, compatible pins, reproducible evidence, correction history | A second copy of the private research library |
| XeCores | Finding a route by task/hardware, readable results, public output examples | A separate hand-maintained source of benchmark numbers |
| B70 Launcher | Inspecting and executing a qualified recipe, preflight, stop, diagnostics | Evidence that every selectable configuration was validated |

Numeric authority remains `data/benchmarks.v1.json`. Family routes own the
commands; the image/patch matrix owns compatibility. The small
`data/first-run.v1.json` selection keeps README and START aligned. It does not
replace those authorities or automatically enable a route in the launcher.

Historical pages stay in place and carry their status. A fix updates entry
points and adds a correction; it does not erase an inconvenient result.

## Closest alternatives and the remaining question

[Steve's B70 lab](https://github.com/steveseguin/b70-optimization-lab) already
publishes Intel recipes, receipts and its own engine. The
[Local AI registry](https://github.com/sybil-solutions/local-ai-registry) already
links accepted launches to checks and artifact hashes. Mia's
[recipes](https://mia-ai.net/models) show a short path from hardware to a task.
Evidence-linked recipes are an established approach. Our proposed contribution
is a narrow Intel route whose installation and useful task another owner can
complete, with diagnosis and upstream maintenance when it fails. Independent
reproduction must establish that value; no uniqueness claim follows from this audit.

## The working loop

1. Pick one user task and a failure or decision it can resolve. Inspect closed
   arms and current upstream changes before starting a new experiment.
2. Declare the artifact, engine/build, precision, context, sampling, concurrency,
   cache state, timing observer, quality gate, stop rule, and resource budget.
   Keep one active experiment. Record output differences and failed attempts.
3. Register the result privately. Prepare a compact public record with stable
   evidence, reproduction commands, limitations, and upstream credit. A passing
   screen remains a screen until confirmation and a route decision.
4. Update the selected public family route and its entry points. Run the
   cookbook check. If launcher behavior changes, test its emitted command and
   state the minimum released version that includes it.
5. Refresh XeCores from a named committed cookbook revision. Review source,
   generated assets, launcher manifests, and checksums together before a site
   deployment or application release. Source pushes and live deployments are
   separate actions. A public post links to the usable artifact and carries the
   same exactness, workload, and measurement limits.

A changed engine, draft, quant, or cache policy needs its own correctness
checks. Disable speculation for the first setup check; test the faster mode
separately. Distinguish draft-versus-serial agreement on the same build from
agreement with the stock build or a higher-precision reference.

## First two weeks

This is a proposed scope, with estimates rather than measured completion times.

| Window | Useful output | Acceptance or stop condition |
|---|---|---|
| Days 1–3 | One consistent first-run path and regression checks across its docs and launcher command | A new reader reaches the named endpoint; command preview matches the reviewed route. Source-only checks do not count as a completed installation. |
| Days 4–7 | One retained task pack for short code work and long repository questions | Record actual completion, wrong answers, setup time, first-token and total latency. Keep missing cells missing. Stop adding cells if they change no deployment decision. |
| Days 8–10 | One route card joining existing evidence, task outputs, failure notes, and a pinned launcher recipe | No copied numeric authority; the user can inspect precision, occupied context, speculation parity, cache mode, and validation scope together. |
| Days 11–14 | Three requested independent reproduction reports and one upstream issue/patch or documented upstream resolution | Count every attempt and its blocker. Three is a target, not a promised adoption result. Rework the installation path if users need private instructions. |

Keep the immediate work small: repair known entry-point defects, qualify one
useful workflow, and maintain the existing upstream contribution. New model
families, new kernel mechanisms, and a wider GPU fleet require separate sizing
and usefulness gates. A new custom engine should proceed only after a measured
removable cost and an end-to-end benefit justify its long-term maintenance.

## Proposed first task pack and continuation gate

Use one pinned repository fixture for twelve cases: four direct lookups, four
cross-file questions, two questions the fixture cannot answer, and two five-turn
change-tracking cases. These produce twenty assistant outputs. Freeze expected
facts, citation ranges and failure rules before inference. Keep development
examples separate from the held-out cases.

A proposed pilot gate is ten of twelve complete cases passing, both missing-data
questions handled correctly, and every configuration value used in the final
checklist matching the fixture. This is a narrow pilot gate, not general model
quality. Retain all outputs, setup time, interventions, actual tokens, first-token
and total task time, cache state and cleanup. Keep energy unmeasured unless aligned
counters are available. The pack and thresholds are planning proposals.

During days 15–30, repair independent-installation blockers on this same family.
During days 31–60, repeat the pack when an adopted dependency changes. At day 90,
expand only for a concrete user need with capacity and maintenance evidence.
Propose one two-hour maintenance session per week; record actual time. After two
failed attempts at the same unchanged mechanism, pause expansion and fix or
withdraw that recipe. No external attempts at day 30 means reproduction remains
untested and the scope needs review. Downloads and followers cannot substitute
for successful tasks.

## Further investigations

| Question | Smallest deciding evidence | Stop condition |
|---|---|---|
| Which engine helps a single local coding agent at depth? | Same task pack, retained answers, actual context and tokens, C1 task latency, measured energy when available | No user-visible quality/latency benefit or no comparable reference |
| When is cache-on safe for each speculative stack? | Repeated multi-turn cache-hit requests, retained outputs, cache counters, exact engine/draft pins | Any unexplained corruption; keep that route cache off |
| Does the existing TensorFold change still need our next kernel step? | Current upstream status and review feedback on [PR #2595](https://github.com/ashhart/TensorFold/pull/2595) | Equivalent upstream fix, unresolved correctness gate, or ideal benefit below the research gate |

The full local two-change attention patch and the grid-order-only upstream PR
are separate artifacts. Their measurement receipts and equality claims must
stay separate. Review the [TensorFold route](qwen38-27b/TENSORFOLD-B70.md) before
copying either result into a post or launcher note.

Public announcements and support pages need maintenance too. Update an old
hardware-funding appeal after that hardware arrives. Preserve working links
for previously announced pages. Publish corrections with the changed artifact
and scope; do not turn another developer's result into our reproduction.

Next action: follow [START.md](../START.md) with one B70 and record the first
external installation attempt.
