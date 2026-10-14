---
title: "LocalMaxxing platform — submission mechanics and B70 competitive landscape (2026-09-22)"
type: research
date: 2026-09-22
status: current
---

# LocalMaxxing — what it is, how submission works, where the B70 stands

Researched 2026-09-22 to prepare the draft-side-INT4 evaluation submission. Companion worklog with
the ready-to-send payload:
`docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/LOCALMAXXING-submission-20260922.md`.

## What the platform is

- **Site:** [localmaxxing.com](https://www.localmaxxing.com) — "Speed-test your local LLM rig.
  Compare it with the world." Community-submitted local-inference speed tests; "Every number on this
  site comes from a community-submitted run on real hardware — no vendor speed tests."
- **What it ranks:** individual runs (model + quant + engine + hardware recipe) on public
  leaderboards sortable by model, hardware, engine, and quantization; also rigs (by submission
  count), models (by approved-run count), and quality eval suites (GSM8K, HumanEval+, etc.).
  The ranking unit is the **run**, not a recipe or a campaign — each submitted record stands alone
  with its own coordinates.
- **Scale (observed 2026-09-22):** ~2,602 users, ~8,341 runs, ~543 rigs, ~929 models.
- Not to be confused with our other surfaces: **XeCores** (xecores.com, our own rendered catalog)
  and **IntelInside** (separate platform, handle `sergiiob`, open PRs). Different audiences,
  different submission paths.

## Submission mechanism (authoritative, from their docs and our prior submissions)

Three ways in; we use the API/CLI path.

1. **REST API** — `POST https://www.localmaxxing.com/api/speed-tests`, auth
   `Authorization: Bearer bhk_<40 hex>`. Returns **201 with `status: "APPROVED"` immediately** —
   approval is automatic ingestion into the self-reported dataset, **not** independent reproduction.
   Docs: [`/en/api-docs`](https://www.localmaxxing.com/en/api-docs);
   machine schema: [`/api/agent-context`](https://www.localmaxxing.com/api/agent-context),
   [`/api/openapi.json`](https://www.localmaxxing.com/api/openapi.json).
   Validation without side effects: `POST /api/speed-tests/dry-run`.
2. **`lmx` CLI** (Go, installed at `/usr/local/bin/lmx` on this host; provenance string
   `localmaxxing-go` in our archived run records):
   `lmx speed-test dry-run <file>.json` then `lmx speed-test submit <file>.json`.
   API key already configured at `~/.config/localmaxxing/config.json` (`apiKey`, manual provider).
   This is the exact path used for the 2026-08-19 Qwen3.8-27B submissions (run IDs `cmt00i05k…` et al.).
3. **Web form** — GitHub sign-in, then the site's submit UI; API keys are created in the dashboard.

**Rate limits:** 1 speed-test per user per minute (30/hour documented in agent-context); eval runs
1 per 5 min.

## Required and useful fields (from `/api/agent-context`, `engineFlagsSchema`, `hardwareSchemas`)

Required: `hfId` (resolve via `GET /api/models/search?q=`), `hardware` (discriminated union on
`hwClass`), `engineName` (enum includes `vllm`), `quantization` (free string ≤64; `GPTQ-Int4`
accepted — prior runs used it), `tokSOut`.

Important optional fields: `tokSPrefill`, `tokSTotal`, `ttftMs`, `peakVramGb`, `promptTokens`,
`outputTokens`, `contextLength`, `batchSize`, `modelRevision`, `engineVersion`, `engineRepository`,
`engineCommit` (7–64 hex), `engineBuild`, `backend` (enum includes `xpu`), `gpuPowerWatts` (array,
one per card), `notes` (≤2000), `engineFlags` (if present, `commandSnippet` required), plus
`promptSha256`/`promptSample`/`outputSha256`/`outputSample`/`engineTimingsRaw`.

`engineFlags` covers `tensorParallel`, `concurrency`, `maxRunningSeqs`, `gpuMemUtil`, `kvCacheDtype`,
`prefixCaching`, `chunkedPrefill`, `temperature`, `topP`, `topK`, `extraFlags`, and the speculative
family: `specDecoding`, `specMethod`, `specNumTokens`, `specModel`, `mtpEnabled`, `mtpDraftLayers`,
`specAcceptanceRate`, `specDraftTokens`, `specAcceptedTokens`, `specMeanAcceptedLength`.

Hardware for us: `hwClass: DISCRETE_GPU`, `gpuName: "Intel Arc Pro B70"` (canonical name in their
`hardwareOptions.discreteGpuNames`; also listed: Arc B580/B570/B60/B50, A770/A750/A380, DC Max),
`vramGb: 32`, `gpuCount`, `cpu`, `ramGb`, `os`, `powerWatts`.

## What APPROVED means (confirmed)

`POST /api/speed-tests` → `201` with `status: "APPROVED"` **immediately, on every schema-valid
submission**. There is no review step for speed tests; the platform is a self-reported dataset.
This matches our claims standard
(`.agents/skills/b70-benchmark-claims-standard/SKILL.md`): LocalMaxxing `APPROVED` = "published /
accepted into its self-reported dataset", say "approved self-reported submission", never "verified
by LocalMaxxing".

Separately, a per-run **"Verified run" badge** exists and *is* server-checked:

- prompt hash equals one of two **canonical prompts** (`reasoning-v1` ~330 tok,
  sha256 `9000edaadb16fc77ab78b39917245a03d4d7585c3ba770cb045a47a0e0683445`, min 256 output tokens;
  `code-v1` ~260 tok, sha256 `a737db738bd4e37881d1737f8dc51cae0295eff9ece6551d39826ce8ed43a690`),
  hash computed **without the cache-bust nonce line**; `promptSample` must match; non-empty
  `outputSample`; `engineTimingsRaw` present; `batchSize` 1 and `concurrency` 1; `outputTokens` ≥ 256;
  **spec/MTP runs must report `specDraftTokens` + `specAcceptedTokens`** (bare acceptance-rate is
  display-only); `tokSOut` ≤ hardware decode ceiling × 1.15 (ceiling scaled by mean accepted length
  for spec runs). Runs >10× theoretical ceiling get `SUSPICIOUS` and are hidden.
- 3 approved verified runs ⇒ the *account* shows a Verified badge automatically.
- **No B70 run on the platform currently has `verifiedRun: true`** — the badge is an open
  differentiator if tonight's re-measure harness captures the canonical-prompt fields.

No evidence-URL field exists; provenance is `commandSnippet`, `engineCommit`/`engineVersion`/
`engineBuild`, and free-text `notes`. Cache-busting for cold prompts is expected via a nonce line
stripped server-side (`promptSample` ≤2000 chars).

## Competitive landscape — Intel Arc on the platform (observed 2026-09-22)

`GET /api/speed-tests?gpuName=Intel%20Arc%20Pro%20B70` (20 visible, newest first, all APPROVED,
none verified):

| who | rig | model / quant | engine | tok/s (their cell) | date |
|---|---|---|---|---:|---|
| steveseguin | B70 ×1 | Qwen3.8-27B-FP8 / FP8, no spec | vllm | 54.22–54.32 | 2026-09-17/18 |
| steveseguin | B70 ×2 | Qwen3.8-27B-FP8 / FP8 | vllm | 88.41–90.48 | 2026-09-17 |
| JP-devv | B70 ×4 | Qwen3.8-27B-Uncensored / FP8 | vllm | 183.1 | 2026-09-17 |
| JP-devv | B70 ×4 | Qwen3.8-Flash-Next / GPTQ-Int4 | vllm | 84.8 | 2026-09-17 |
| steveseguin | B70 ×2 | devan-carlin/Qwen3.8-27B-int4-AutoRound / W4A16 g128 (gptq kernel path) | vllm | 117.53 | 2026-09-12 |
| steveseguin | B70 ×1 | RedHatAI/Qwen3.5-9B-quantized.w4a16 | vllm | 124.08 | 2026-09-12 |
| steveseguin | B70 ×1 | RedHatAI/Qwen3.5-4B-quantized.w4a16 | vllm | 191.73 | 2026-09-12 |
| steveseguin | B70 ×2 | Qwen3.5-4B W4A16 | vllm | 236.92 | 2026-09-08 |
| SergiioB (us) | B70 ×2 | AtomicChat/Qwen3.8-Flash-Next-GGUF Q4_K_M | llama.cpp | 22.96–33.25 | 2026-09-11 |
| SergiioB (us) | B70 ×1 | **same GPTQ-Int4-G128-MTP-BF16 artifact**, C1 (lmx client, short prompt/g256-ish) | vllm | **56.8** | 2026-08-19 |
| SergiioB (us) | B70 ×1 | same artifact, C5/C16/C32 aggregate | vllm | 107.8 / 200.6 / 224.2 | 2026-08-19 |
| SergiioB (us) | B70 ×1 | Nemotron-3.5-30B-A3B GPTQ-INT4-G64+DFlash | vllm | 186.6 | 2026-08-13 |
| SergiioB (us) | B70 ×1 | Qwen/Qwen3.8-27B GPTQ-Int4 (+draft-INT4-RTN variant 112.65) | vllm | 83.7 | 2026-08-15/19 |

Also on **Intel Arc B580** (closest consumer Battlemage): WizardlyBump17,
Qwen3.6-35B-A3B-MTP-GGUF Q5_K_XL llama.cpp **19.62** tok/s (acceptance 0.82) and OpenArc INT4-OV
**11.12** tok/s (2026-09-15). **Intel Arc B770: zero entries.** Nothing else Intel-family is
visible in the queried windows.

**Positioning read (all cells differ — directional only, never a matched cross-user claim):**

- Single-B70 27B-class without spec decode sits at **~54 tok/s** (steveseguin FP8, short prompt).
  Our matched A/B says: same route + draft-side INT4 = 55.61 → ~80.3 tok/s post-first at
  p8192/g128 C1 (+44.4%), pending tonight's fresh re-measure.
- Against **our own prior approved entry on the same artifact** (56.8 tok/s, 2026-08-19) the new
  recipe is the "improved recipe" story: same hfId, same single-B70 C1 class, now measured at a
  named long-context cell.
- The C32 aggregate **172.48 tok/s (2.79× over B=1)** would be the highest posted single-B70 27B
  aggregate on the platform (our own 224.2 from 2026-08-19 used a different stack/protocol and
  older numbers; steveseguin's multi-card entries are different rig classes and must never be
  averaged in).
- A canonical-prompt **verified run** would be the first verified B70 entry on the platform.

Caveat for honesty: the 2026-08-19 `cmszpqy0…` entry already published "GPTQ-Int4+draft-INT4-RTN"
at 112.65 tok/s (RTN draft quant, different/undocumented cell). The new submission should not claim
"first draft-side INT4 number on LocalMaxxing"; it claims the **matched A/B** and the named cells.

## Method fit with our protocol

Their methodology tips (warmup, fixed prompt, steady state, temp 0 for determinism, capture VRAM
and power, include `commandSnippet`, dry-run first, use canonical prompts for the badge) are a
subset of `docs/methods/benchmark-protocol.md`. Divergences to handle in wording: their `tokSOut`
is whatever we report — we must keep labeling it **client post-first rate** (request-side), and
their `gpuPowerWatts`/`powerWatts` must be filled with the **configured cap (230 W)** and the notes
must say measured draw was not recorded, never implying measured watts.

---

## Submitted 2026-09-22 — four runs, all APPROVED (self-report tier)

The publication-grade re-measure landed and the four records were submitted through
`lmx speed-test submit`, one per minute. All four returned `status: APPROVED`
(the platform's automatic ingestion — **E2 self-reported with raw evidence**, never
"independently verified"; `verifiedRun: false` on all four). Run URLs are
`https://www.localmaxxing.com/runs/<id>`.

| Run id | Record | Cell | tokSOut (client post-first median, n=5) |
|---|---|---|---:|
| `cmud9n4fp0avilq01v445nzcq` | draft-side INT4 (patched) | C1 p8192 8194 tok / g128 | 72.55 |
| `cmud9ojc70avllq0102t7zqjz` | baseline comparator | C1 p8192 8190 tok / g128 | 48.81 |
| `cmud9pxud0avolq016xd680mc` | draft-side INT4 (patched) | C1 p512 517 tok / g128 | 73.43 |
| `cmud9rcq90avrlq0198w35x7n` | baseline comparator | C1 p512 516 tok / g128 | 52.61 |

Each record carries the matched_with_tokenizer_delta classification, the 230 W
**configured cap**, the **measured average draw** (228.74 / 229.49 / 224.53 / 228.59 W
from the xe `energy1_input` counter) and the measured tokens/J delta — i.e. this
submission went further than the method-fit note above originally prescribed ("notes
must say measured draw was not recorded"), because the energy telemetry landed in
time. Absolute spec counters (`specDraftTokens`/`specAcceptedTokens`) were included in every submitted payload per the platform's MTP requirement — **but the platform's ingest dropped these fields: they read back `null` on the stored records** (verified on all 8 records via `GET /api/speed-tests`). Retain the raw counters in B70-DOCS evidence; do not rely on the platform copy.

`verificationIssues` on all four (expected — these are the `verifiedRun` badge
requirements, not validity failures): missing promptSha256, outputSample,
engineTimingsRaw; outputTokens < 256. The badge variant remains a separate
diagnostic-labeled run per §5 of the submission worklog.

Claims hygiene on these records: independently audited
(`worklogs/CLAIMS-AUDIT-LMX-20260922.md` — 0 blockers, 2 MAJOR findings, both fixed
before submit; one number corrected +31.5 % -> +31.4 % tokens/J).

One correction to the "Method fit" section above: it says measured draw "was not
recorded" — that was true of the planning draft and is no longer true; all four
submitted records carry measured draw and measured tokens/J.

**Duplicate-set incident (2026-09-22, recorded 2026-09-23):** two submission processes raced. A second set of four records with the SAME measurements was published at 22:55–22:58Z (`cmud9viep0avulq01364r8cyo` 72.55 p8192-patched, `cmud9wwv10avxlq01saixkle6` 48.81 p8192-baseline, `cmud9ybo00aw0lq01ql01bhqn` 73.43 p512-patched, `cmud9zqjp0aw3lq0160e0i44v` 52.61 p512-baseline). LocalMaxxing has **no delete path**. The 22:55–22:58Z set is canonical for citation: it carries the claims-audit residual fixes (p512-patched quotes its own cell's 73.4–88.7 sample range instead of p8192's 64.25–91.17, roundings 228.60 W / 3.90 J/token). Root cause: the pre-submit receipt check grepped `results/` and `worklogs/` but the first set's IDs had been written to `research/2026/localmaxxing-platform-20260922.md` minutes earlier — outside the checked paths.


> **Correction — 2026-09-23**
>
> **The four records above are DUPLICATED on the platform.** A parallel ZCode session on this
> host independently executed the same "publish new LocalMaxxing eval" instruction and submitted
> the same four cells seven minutes later. Both sets returned `APPROVED`. The platform has no
> delete path.
>
> **Set A (this note's table above), submitted 22:48:55–22:52:12Z:**
> `cmud9n4fp0avilq01v445nzcq` (p8192 patched, 72.55) · `cmud9ojc70avllq0102t7zqjz` (p8192
> baseline, 48.81) · `cmud9pxud0avolq016xd680mc` (p512 patched, 73.43) · `cmud9rcq90avrlq0198w35x7n`
> (p512 baseline, 52.61).
>
> **Set B (the parallel session's), submitted 22:55:27–22:58:44Z:**
> `cmud9viep0avulq01364r8cyo` (p8192 patched, 72.55) · `cmud9wwv10avxlq01saixkle6` (p8192
> baseline, 48.81) · `cmud9ybo00aw0lq01ql01bhqn` (p512 patched, 73.43) · `cmud9zqjp0aw3lq0160e0i44v`
> (p512 baseline, 52.61).
>
> Both sets carry the same audited numbers and qualifiers (verified from the two receipts dirs),
> so this is a **duplicate-ingestion defect, not a claims defect**. Remains valid: the measured
> numbers, the claims-audit verdict, and the A/B story. Withdrawn: the statement "each payload was
> submitted exactly once" (true from each session's own view, false on the platform).
>
> **Operator decision needed:** which run id is canonical for citation (suggest Set A, the earlier
> one), and whether to ask LocalMaxxing to mark the other set superseded (the API exposes
> `adminNotes`; there is no delete/withdraw path). Cause: two agents sharing one host and one goal
> both acting on the same explicit authorization without a mutual-exclusion check. The guard is now
> stated in `worklogs/CLAIMS-AUDIT-LMX-20260922.md` and the coordination memory: **before any
> outward submission from a shared host, check `results/**/lmx-*receipts*` and the worklogs for an
> already-submitted run id for the same cell.**
