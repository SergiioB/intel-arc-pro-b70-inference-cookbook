---
title: "LocalMaxxing submission — Qwen3.8-27B draft-side INT4"
type: worklog
date: 2026-09-22
status: final
---

# LocalMaxxing submission — Qwen3.8-27B draft-side INT4 (2026-09-22)

**Status: SUBMITTED — 2026-09-22T22:55–22:58Z. All four records `APPROVED` (automatic ingestion
into the self-reported dataset — NOT independent reproduction).** ⚠️ **Duplicate-set incident:** a parallel submission process published the same four measurements at 22:48–22:52Z (`cmud9n4fp0avilq01v445nzcq` / `cmud9ojc70avllq0102t7zqjz` / `cmud9pxud0avolq016xd680mc` / `cmud9rcq90avrlq0198w35x7n`, identical measurements) from pre-fix payloads; LocalMaxxing has no delete path, so 8 records exist. The set below is canonical (carries the audit's residual fixes: p512-patched quotes its own cell's 73.4–88.7 sample range; roundings 228.60 W / 3.90 J/token). Also `specDraftTokens`/`specAcceptedTokens` were sent in every payload but the platform's ingest dropped them (read back null on all 8 records); raw counters remain in B70-DOCS evidence. Run IDs:

| Record | tokSOut (median, n=5) | Run ID | createdAt |
|---|---:|---|---|
| C1 p8192/g128 **draft-side INT4 (the improvement)** | **72.55** | `cmud9viep0avulq01364r8cyo` | 2026-09-22T22:55:27Z |
| C1 p8192/g128 baseline (BF16 drafter) | 48.81 | `cmud9wwv10avxlq01saixkle6` | 2026-09-22T22:56:32Z |
| C1 p512/g128 **draft-side INT4** | **73.43** | `cmud9ybo00aw0lq01ql01bhqn` | 2026-09-22T22:57:38Z |
| C1 p512/g128 baseline | 52.61 | `cmud9zqjp0aw3lq0160e0i44v` | 2026-09-22T22:58:44Z |

Result URLs: `https://www.localmaxxing.com/runs/<id>`. Receipts (full server JSON per record):
`results/2026/09/b70-improvement-probes-20260920/lmx-submit-receipts-20260923/<name>.receipt.txt`.
Read-back verified: notes lengths 1985/1996/1984/1792 all survived; the qualifiers (metric = client
post-first NOT engine-native, `matched_with_tokenizer_delta`, graph-capture enabler, measured draw
229.50 W at a 230 W cap, greedy byte-identical correctness scope, prefix caching disabled,
`PROVISIONAL self-report`) are present in the published record text. Submitted from the four
audit-fixed payload files in `scripts/tmp/lmx-submit-20260922/` (claims audit
`worklogs/CLAIMS-AUDIT-LMX-20260922.md`: F1–F7 applied before submit, F8/F9 nits applied,
`specDraftTokens`/`specAcceptedTokens` added per the verified-run requirements). Each payload was
submitted **exactly once** (the platform has no delete path) at 65 s spacing (1/min rate limit).

Wording rule in force everywhere these numbers appear: **"approved self-reported submission on
LocalMaxxing"**, never "verified by LocalMaxxing". The optional "Verified run" badge was not targeted
(canonical prompt + g≥256 + `engineTimingsRaw` would change the cell; see §5) — `verification.verified
=false` with 5 expected issues in the dry-run is the badge status, not a rejection.

---

**Status at preparation: PREPARED — DO NOT SUBMIT UNTIL (1) tonight's fresh re-measure lands and its medians are
swapped in, and (2) the operator gives the explicit go in-session.** (both conditions met 2026-09-22/23:
the fresh n=5 re-measure landed as `…20260922T213957Z` / `…20260922T214938Z` with medians swapped in,
and the active session goal explicitly directs "PUBLISH NEW LOCALMAXXING EVAL".)

Authorization state: the operator has **explicitly authorized publishing this evaluation to
LocalMaxxing**. Posting to X/Twitter is **NOT** authorized. IntelInside and XeCores are separate
surfaces with separate paths and are out of scope here. Per the claims standard: no outward-facing
action is implied — the actual `lmx speed-test submit` is run by the operator or on an explicit
go-ahead in the session.

Platform facts and competitive landscape live in
`research/2026/localmaxxing-platform-20260922.md`. Claim source evidence:
`results/2026/09/b70-improvement-probes-20260920/p4-champion-ab-*`; narrative
`worklogs/P4-20260920.md` §11–§17; research note `research/2026/qwen38-draft-side-int4-upstream-20260921.md`.

---

## 0. NUMBERS BLOCK — PENDING FINAL RE-MEASURE (swap before submit)

All values below are **PROVISIONAL — NOT FOR PUBLIC HEADLINE** until tonight's fresh n=5 re-measure
on the champion stack completes. Replace in the payloads:

| field | current provisional value | source |
|---|---:|---|
| patched C1 p8192/g128 tokSOut (client post-first median, n=5) | **80.32** | `…T114444Z/comparison.json` |
| patched C1 p8192/g128 ttftMs (median) | **4708.5** | same |
| baseline C1 p8192/g128 tokSOut | **55.61** | same |
| baseline ttftMs (median) | **4732.2** | same |
| patched specAcceptanceRate (absolute counters, run-variable 0.55–0.69) | **0.6949** | same |
| baseline specAcceptanceRate | **0.5369** | same |
| tokSPrefill (derived = 8189 / ttft_s) | **1739.0 patched / 1730.5 baseline** | derived |
| p512/g128 patched tokSOut | **77.14 (n=3 screen only — re-measure at n=5 or drop this record)** | `P4 §13` |
| C4 aggregate | **140.30** (from 115.70 baseline; B=1 patched 61.86) | serving dossier |
| C32 aggregate | **172.48** (2.79×; per-request ~29 tok/s, TTFT ~2 s) | serving dossier |

Re-measure acceptance criteria before swap: fresh n=5 per arm, same calibrated p8192 cell
(server-tokenizer count recorded, g128 with `finish_reason=length` on all samples), cache counters
bracketed, and spec-decode draft/accepted counters captured (needed for the verified-run variant).
Do not submit any number whose n<5 or whose evidence directory is not retained.

---

## 1. What LocalMaxxing is, and the exact submission mechanism

- **Platform:** [localmaxxing.com](https://www.localmaxxing.com) — community-submitted local-LLM
  inference speed tests, ranked per **run** (model + quant + engine + hardware + coordinates) on
  public leaderboards filterable by model/hardware/engine/quant.
- **APPROVED semantics (confirmed from their docs/API):** `POST /api/speed-tests` returns
  `201 {"status":"APPROVED"}` immediately for every schema-valid submission — it is
  **automatic ingestion into a self-reported dataset, not independent reproduction**. Wording rule:
  "approved self-reported submission on LocalMaxxing", never "verified by LocalMaxxing".
  A separate server-checked **"Verified run" badge** exists (canonical prompt + samples +
  `engineTimingsRaw` + spec counters + batch 1 + ceiling check); no B70 run has it yet — optional
  stretch goal, see §5.
- **Submission paths:** REST API (`POST https://www.localmaxxing.com/api/speed-tests`, Bearer key),
  the `lmx` CLI (installed at `/usr/local/bin/lmx`, key already configured at
  `~/.config/localmaxxing/config.json` → `apiKey`), or the web form (GitHub sign-in).
  Validation-only: `POST /api/speed-tests/dry-run` (and `lmx speed-test dry-run`).
- **Docs:** `https://www.localmaxxing.com/en/api-docs` · `https://www.localmaxxing.com/api/agent-context`
  · `https://www.localmaxxing.com/api/openapi.json`.
- **Rate limit:** 1 speed-test per user per minute (30/hour). Dry-run is free of the limit.
- **Result URL after submit:** the 201 response `id` → `https://www.localmaxxing.com/runs/<id>`
  (model-page view: `https://www.localmaxxing.com/en/models/<org>/<model>?run=<id>`).
- Prior submissions from this host used exactly this CLI path (2026-08-19 Qwen3.8-27B runs
  `cmt00i05k0efims01vz3u1kl5` et al., all APPROVED).

## 2. Prepared payloads (exact JSON to send)

Save each fenced block to `scripts/tmp/lmx-submit-20260922/<name>.json` (do not hand-edit anything
except the §0 swap values). One record per submission, minimum one minute apart.

### 2.1 PRIMARY — C1 p8192/g128, champion + draft-side INT4 (the improvement)

```json
{
  "hfId": "SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16",
  "modelRevision": "main",
  "hardware": {
    "hwClass": "DISCRETE_GPU",
    "gpuName": "Intel Arc Pro B70",
    "vramGb": 32,
    "gpuCount": 1,
    "cpu": "AMD Ryzen 7 5700X3D",
    "ramGb": 64,
    "os": "Ubuntu 26.04",
    "powerWatts": 230
  },
  "engineName": "vllm",
  "engineVersion": "0.27.2rc1.dev77+gac7509e2b",
  "engineRepository": "https://github.com/vllm-project/vllm",
  "engineCommit": "ac7509e2b",
  "engineBuild": "docker vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f + local champion overlay patches applied at container start",
  "backend": "xpu",
  "quantization": "GPTQ-Int4",
  "promptTokens": 8189,
  "outputTokens": 128,
  "contextLength": 16384,
  "batchSize": 1,
  "tokSOut": 80.32,
  "tokSPrefill": 1739.0,
  "ttftMs": 4708.5,
  "engineFlags": {
    "commandSnippet": "docker run -d --name b70-q38 -p 8000:8000 --device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | sort -u | head -1) -v /dev/dri:/dev/dri:ro -v <model_dir>:/model:ro -v patch_mtp_nightly.py:/patch_mtp.py:ro -v patch_mtp_boundary.py:/patch_boundary.py:ro -v champion_overlay/patch_gdn_fuse.py:/patch_gdn_fuse.py:ro -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e B70_MTP_BF16_DRAFT=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e B70_DRAFT_LMHEAD_INT4=1 -e B70_DRAFT_MTP_INT4=1 --entrypoint bash vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f -lc 'python /patch_mtp.py && python /patch_boundary.py && python /patch_gdn_fuse.py && exec vllm serve /model --quantization gptq --dtype float16 --language-model-only --trust-remote-code --kv-cache-dtype fp8 --max-model-len 16384 --gpu-memory-utilization 0.88 --max-num-seqs 8 --max-num-batched-tokens 8192 --no-enable-prefix-caching --served-model-name qwen38 --host 0.0.0.0 --port 8000 --speculative-config \"{\\\"method\\\":\\\"mtp\\\",\\\"num_speculative_tokens\\\":4}\"'  # client: one HTTP stream, ignore_eos, g128 all finish_reason=length, unique cold prompt (8189 server-tokenizer tokens), n=5 post-warmup samples, median reported",
    "tensorParallel": 1,
    "concurrency": 1,
    "maxRunningSeqs": 8,
    "gpuMemUtil": 0.88,
    "kvCacheDtype": "fp8",
    "prefixCaching": false,
    "chunkedPrefill": true,
    "temperature": 0.7,
    "topP": 0.8,
    "topK": 20,
    "extraFlags": "--presence-penalty 1.5; one same-shape warmup discarded; cache-hit delta zero (unique prompt); weights GPTQ-INT4 g128 sym target + W4A16 g128 sym on the MTP drafter's LM head and body only",
    "specDecoding": true,
    "specMethod": "mtp",
    "specNumTokens": 4,
    "mtpEnabled": true,
    "mtpDraftLayers": 1,
    "specAcceptanceRate": 0.6949
  },
  "notes": "Client post-first streaming rate (request-side, (out_tokens-1)/(last-first token time)), median of n=5; NOT engine-native tg. C1 single-stream: batchSize=1, one in-flight request; aggregate-concurrency results are submitted as separate records, never mixed. tokSPrefill derived as promptTokens/TTFT. Matched_exact A/B vs baseline record (same image+overlay, same cell, same sampling, single variable = draft-side INT4: only the MTP drafter's LM head + MTP body quantized to W4A16 g128 sym; target weights untouched; baseline 55.61 tok/s same cell). Acceptance is run-variable in the patched config (0.55-0.69 across runs; 0.6949 absolute counters this run) vs baseline band 0.53-0.58; throughput reproduced across independent runs (+37..+46%). Correctness: greedy output byte-identical patched vs unpatched (3 prompts x 2 repeats); 30 seeded draws x 6 open-ended prompts give identical modal emitted class; every emitted token is target-verified. Power: 230 W configured cap; measured draw not recorded. Prefix caching disabled at launch; cold unique prompt. Stack includes local (not yet published) overlay patches; image digest is pullable. Evidence: results/2026/09/b70-improvement-probes-20260920/p4-champion-ab-* (B70-DOCS repo)."
}
```

### 2.2 Baseline comparator — same cell, unpatched champion (submit second; makes the A/B visible on-platform)

Identical to §2.1 except:

```json
{
  "tokSOut": 55.61,
  "tokSPrefill": 1730.5,
  "ttftMs": 4732.2,
  "engineFlags": { "…as §2.1 but without B70_DRAFT_LMHEAD_INT4=1/B70_DRAFT_MTP_INT4=1 in env or extraFlags…", "specAcceptanceRate": 0.5369 },
  "notes": "Baseline arm of the matched_exact A/B; every field as the patched record except the two draft-INT4 env patches. tokSOut 55.61 client post-first median n=5, p8192/g128 C1. Acceptance 0.5369 absolute counters (band 0.53-0.58 across runs)."
}
```

(Full JSON assembled from §2.1 at submit time; do not submit a truncated object.)

### 2.3 OPTIONAL — p512/g128 patched (only if re-measured at fresh n=5 tonight)

Same as §2.1 with `"promptTokens": 512`, `"tokSOut": <fresh n=5 median>`, `"ttftMs": <fresh median>`.
Current 77.14 is an **n=3 tuning screen — provisional, must not be a public representative number**
under the claims standard. If tonight's re-measure only covers p8192, drop this record.

### 2.4 OPTIONAL — concurrency class, kept strictly separate from C1 (submit only on explicit operator decision)

```json
{
  "hfId": "SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16",
  "hardware": { "hwClass": "DISCRETE_GPU", "gpuName": "Intel Arc Pro B70", "vramGb": 32, "gpuCount": 1, "cpu": "AMD Ryzen 7 5700X3D", "ramGb": 64, "os": "Ubuntu 26.04", "powerWatts": 230 },
  "engineName": "vllm",
  "engineVersion": "0.27.2rc1.dev77+gac7509e2b",
  "engineCommit": "ac7509e2b",
  "backend": "xpu",
  "quantization": "GPTQ-Int4",
  "batchSize": 32,
  "contextLength": 16384,
  "tokSOut": 172.48,
  "ttftMs": 2000,
  "engineFlags": {
    "commandSnippet": "<same server as §2.1>; client: 32 concurrent streaming requests over the declared campaign interval",
    "concurrency": 32,
    "maxRunningSeqs": 8,
    "gpuMemUtil": 0.88,
    "kvCacheDtype": "fp8",
    "prefixCaching": false,
    "temperature": 0.7,
    "topP": 0.8,
    "topK": 20,
    "specDecoding": true,
    "specMethod": "mtp",
    "specNumTokens": 4,
    "mtpEnabled": true
  },
  "notes": "AGGREGATE concurrency record: 172.48 tok/s = total successful generated tokens / declared campaign interval at batchSize=32 (2.79x over B=1 aggregate 61.86 patched). Per-request rate falls to ~29 tok/s and TTFT rises to ~2 s. This is a different class from any single-stream record — never compare to or average with C1 numbers. tokSOut here is aggregate, not per-request. PROVISIONAL until tonight's re-run."
}
```

Note: `--max-num-seqs 8` with client concurrency 32 means requests queue; record achieved vs
requested concurrency in notes from the fresh run and adjust (`batchSize`/`concurrency` should state
what the client drove; queueing is disclosed, not hidden).

## 3. Exact submission commands (run only on operator go-ahead)

```bash
cd ~/B70-DOCS
mkdir -p scripts/tmp/lmx-submit-20260922
# save §2.1 JSON → scripts/tmp/lmx-submit-20260922/c1-p8192-g128-draftint4-patched.json
# save §2.2 assembled JSON → scripts/tmp/lmx-submit-20260922/c1-p8192-g128-baseline.json

# 1) validate without side effects (unlimited):
lmx speed-test dry-run scripts/tmp/lmx-submit-20260922/c1-p8192-g128-draftint4-patched.json
lmx speed-test dry-run scripts/tmp/lmx-submit-20260922/c1-p8192-g128-baseline.json

# 2) submit (auth key already at ~/.config/localmaxxing/config.json; 1/min limit):
lmx speed-test submit scripts/tmp/lmx-submit-20260922/c1-p8192-g128-draftint4-patched.json
sleep 65
lmx speed-test submit scripts/tmp/lmx-submit-20260922/c1-p8192-g128-baseline.json
```

Pure-API equivalent:

```bash
KEY=$(python3 -c "import json;print(json.load(open('~/.config/localmaxxing/config.json'))['apiKey'])")
curl -sS -X POST https://www.localmaxxing.com/api/speed-tests \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  --data @scripts/tmp/lmx-submit-20260922/c1-p8192-g128-draftint4-patched.json
```

Web-form alternative: sign in with GitHub at localmaxxing.com → dashboard → submit; paste the same
fields. After the 201: record the returned run `id`, append it to
`research/2026/localmaxxing-platform-20260922.md`, and mark the §0 block "submitted <date>, run
<id>". Response includes `verificationIssues` if the verified checks would have failed — retain it.

## 4. Claims-hygiene checklist (applied to the wording above)

- [x] **No naked numbers** — every tok/s in every payload/notes carries engine, artifact, quant,
      cell (prompt/output tokens), C-class, statistic, and n.
- [x] **Metric named** — `tokSOut` declared in-notes as *client post-first streaming rate
      (request-side)*, not engine-native tg; `tokSPrefill` declared *derived = promptTokens/TTFT*.
- [x] **C1 vs Cn separated** — primary record is batchSize 1 / concurrency 1; the aggregate record
      (§2.4) is a separate submission whose notes say "different class, never compare to or average
      with C1". No aggregate appears in the C1 notes except as a pointer to a separate record.
- [x] **Statistic labeled** — medians of n=5 fresh post-warmup samples; p512 flagged n=3 screen =
      not publishable (§2.3 gate); no "up to" maxima used as headline.
- [x] **Configured cap vs measured draw** — `powerWatts: 230` declared as **configured cap**;
      notes say "measured draw not recorded". No tokens-per-watt or energy claims anywhere.
- [x] **Comparison gate** — the only multiplier-style statements are the **matched_exact** A/B
      (same image+overlay+cell+sampling, single variable) and the aggregate-vs-aggregate B-scaling;
      no cross-user multiplier is claimed anywhere (steveseguin/JP-devv cells differ).
- [x] **Acceptance honesty** — run-variable patched acceptance (0.55–0.69) disclosed in-notes
      alongside throughput reproducibility; not presented as a fixed acceptance gain.
- [x] **Correctness scope labeled** — "greedy output byte-identical (3×2)" + "30 seeded draws × 6
      prompts, identical modal emitted class" + "every emitted token target-verified"; no
      "distribution-preserving" or unqualified "correctness verified" claim.
- [x] **Cache state** — prefix caching disabled at launch; cold unique prompt; zero-hit delta noted.
- [x] **Reproducibility honesty** — pullable image **digest** recorded; local overlay patches
      explicitly declared "local (not yet published)" rather than implied public; vLLM version +
      commit recorded.
- [x] **PROVISIONAL banner retained** — §0 block and every payload carry the provisional marker
      until the fresh re-measure lands; the banner is dropped from the *notes* only at swap time,
      after the fresh evidence directory exists.
- [x] **Approval semantics** — anywhere we later cite the result: "approved self-reported
      submission on LocalMaxxing" (E1/E2 self-report tier), never "verified".
- [x] **No cross-surface leakage** — nothing here is posted to X/Twitter (not authorized),
      IntelInside, or XeCores; those have separate flows and separate claim audits.

## 5. Optional stretch (only if the re-measure harness is extended)

A **Verified-run** badge variant needs: canonical prompt `reasoning-v1`
(sha256 `9000edaadb16fc77ab78b39917245a03d4d7585c3ba770cb045a47a0e0683445`, hash computed without
the cache-bust nonce line; text via `lmx context get canonicalPrompts`), `promptSample`,
non-empty `outputSample`, `engineTimingsRaw` (≤8192 bytes), ≥256 output tokens, batch 1, and
**absolute `specDraftTokens` + `specAcceptedTokens`** (rate alone is not accepted for spec runs),
with `tokSOut` ≤ scaled decode ceiling. No B70 run currently holds the badge. This changes the cell
(longer output, fixed prompt) — treat it as a *separate diagnostic-labeled run*, not a replacement
for the p8192 representative record.

## 6. What would actually be sent

On go-ahead, the first HTTP call is `POST https://www.localmaxxing.com/api/speed-tests` with the
§2.1 JSON (Bearer `bhk_…` from the local config). Expected response:
`201 {"id":"cm…","status":"APPROVED", …}`. Nothing is sent anywhere else by this workflow; the
X/Twitter/IntelInside/XeCores surfaces stay untouched.

> **Correction — 2026-09-23.** "Each payload was submitted exactly once" is wrong **on the
> platform**, though true from this session's view. A parallel session submitted the same four
> cells seven minutes earlier (22:48:55–22:52:12Z, run ids `cmud9n4fp0…`, `cmud9ojc70…`,
> `cmud9pxud0…`, `cmud9rcq90…`), so the platform now holds **two sets of four records** for the
> same cells. Both sets are claims-audited and carry identical numbers/qualifiers — this is a
> duplicate-ingestion defect, not a claims defect. The platform has no delete path. Operator
> decision needed: which set is canonical for citation, and whether to request a superseded mark on
> the other (see `research/2026/localmaxxing-platform-20260922.md` for the full run-id table and the
> coordination guard that prevents a repeat).
