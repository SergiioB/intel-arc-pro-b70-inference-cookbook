---
title: "Muse Code → GLM (Z.ai) Integration: Reverse-Engineering Writeup"
type: model-dossier
date: 2026-09-15
status: living
---

# Muse Code → GLM (Z.ai) Integration: Reverse-Engineering Writeup

**Date:** 2026-08-05/06  
**Status:** Translation layer complete & verified at HTTP level; one Muse-internal SSE-decoder requirement unresolved.

This document captures the complete reverse-engineering of Meta's **Muse Code** CLI (`muse` v0.1.0-R708.1) and the work to point it at a GLM model (via the Z.ai Anthropic-compatible endpoint) instead of Meta's `muse-spark`. The same approach extends to the local B70 (llama.cpp) and to Codex/ChatGPT-subscription backends.

---

## 1. What Muse Code Is

Muse Code is Meta's coding-agent CLI. The `muse` command is a **bash launcher** (`~/.local/bin/muse`, 36 KB) wrapping an 85 MB statically-linked **Rust binary** (`muse-bin-0.1.0-R708.1`). The launcher handles OAuth device-login against `auth.meta.com` and self-updates; the binary is the agent runtime.

It is **architecturally a Claude-Code/Codex-CLI clone**:
- Reads `AGENTS.md` / `CLAUDE.md` project rules (your `B70-DOCS/AGENTS.md` loads as "standing rules").
- Imports Claude-Code-style **skills**, **plugins**, **hooks**, and the Claude marketplace format (`CLAUDE_PLUGIN_ROOT`, `~/.local/share/claude`).
- A built-in `import`/`read-session` skill migrates `~/.claude/`, `~/.codex/`, `~/.grok/` session transcripts.
- Ships 11 bundled skills (`git`, `plan`, `doctor`, `grill`, `create-skill`, `create-plugin`, …).

## 2. The Model Layer — the Unlock

The default provider is **`meta`**, which talks to `https://api.meta.ai/v1` — Meta's Model API. That single base URL speaks three protocols (per the official docs):

| Endpoint | Protocol |
|---|---|
| `POST /v1/responses` | OpenAI **Responses API** (what Muse's `meta` provider uses) |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/messages` | Anthropic Messages |

Models: `muse-spark-1.1`, `muse-spark-1.2`, `muse-spark-1.2-contributor`.  
Auth: OAuth device login → minted Model API key (`LLM|…`). The launcher's bash source is fully readable and confirms the auth flow.

## 3. Override Surface (verified from `muse --help` + binary strings)

These flags/env-vars let you redirect Muse away from Meta:

| Override | Effect |
|---|---|
| `--base-url <URL>` | Repoint the `meta` provider at any URL |
| `--model <ID>` | Any model id (free string) |
| `--api-key-stdin` / `muse auth set --api-key-stdin` | Inject a key |
| **`META_API_KEY`** env | **Bypasses OAuth entirely** (highest-priority cred) — Muse never contacts `auth.meta.com` |
| `MUSE_MODEL` env | Startup model |
| `--provider echo` | Built-in deterministic test provider |
| `--reasoning-effort none\|minimal\|low\|medium\|high\|xhigh\|ultra` | Reasoning tier |
| `--no-session-log` | Don't persist session event logs to disk |

Only `echo` and `meta` providers are compiled into this build (binary strings: *"provider `anthropic`/`openai`/`openrouter` is not compiled into this build"*).

## 4. The Two Non-Obvious Requirements (reverse-engineered)

Pointing `--base-url` at a non-Meta server hits two hard-coded behaviors:

### 4.1 Model catalog: `GET {base_url}/muse-code/models`
On startup Muse unconditionally fetches a model catalog. The required response shape (discovered via iterative `decode error: missing field` messages):
```jsonc
{
  "object": "list",
  "data": [{
    "id": "<model>", "object": "model", "created": 1785962106, "owned_by": "...",
    "provider_id": "meta", "model_id": "<model>",
    "profile_id": "tbh",          // MUST be "tbh" — Muse's internal session profile
    "display_label": "...", "visibility": "visible", "release_date": "",
    "display_order": 1, "is_default": true, "is_current": true,
    "context_limit": 131072, "output_limit": 32768, "description": "..."
  }]
}
```
**Gotchas:**
- `model_catalog` in `settings.json` must be a **sequence (array)**, not a map.
- `visibility` must be `visible|hidden` (not `public`).
- The field is `model_id` (not `model`).
- **`profile_id` must be `"tbh"`** — any other value → *"model X is not in the catalog"* (rows are filtered by the session profile).
- Muse **caches** the catalog at `~/.local/share/muse/model-catalog/6d657461__p746268.json` (`meta__p` + hex(`tbh`)). A stale cache from a prior native run will shadow your proxy's catalog — **delete it** when switching backends.

### 4.2 Request translation quirks
Muse's Responses-API requests have three shapes that differ from vanilla OpenAI:
- **`tools` use `{type:"namespace", name:"muse", tools:[{type:"function",…}]}`** — nested namespaces. The reminder sub-agent exposes `submit_reminder_decision` *inside* a `muse` namespace. Anthropic/GLM only accept flat function tools → **flatten** namespaces.
- **Message `content` can be a bare string** (`content:"..."`), not always an array.
- **`instructions`** is a top-level field (the system prompt) separate from `input[]`.
- **`reasoning: {effort, summary}`** and **`include: ["reasoning.encrypted_content"]`** are sent on every main-agent call.
- **`max_output_tokens: 128000`** is sent by default — GLM rejects >16K with code 1213 → clamp.

---

## 5. Working Artifacts

All in `scripts/tmp/` (scratch space per AGENTS.md §14; promote to `scripts/` once fully validated):

| File | Purpose | Status |
|---|---|---|
| `muse-glm-proxy.mjs` (17 KB) | **Responses↔Anthropic translator**: catalog + request translation + full SSE event translation (content_part, reasoning, output_text.done, [DONE]) | ✅ works via curl; ⚠️ Muse decode-blocker (§6) |
| `muse-b70-proxy.mjs` (5.5 KB) | Catalog + developer→system normalizer for the **local B70** (llama.cpp speaks `/responses` natively) | ✅ works via curl |
| `muse-b70.sh` (3.8 KB) | Wrapper: starts proxy, sets `META_API_KEY`/`MUSE_MODEL`, routes flags around `exec`/TUI | ✅ logic complete |

### What's verified working through the GLM proxy (via direct curl):
- ✅ Non-stream chat round-trip (`/responses` → GLM → Responses object)
- ✅ SSE streaming with full event lifecycle (`created` → `in_progress` → `output_item.added` → `content_part.added` → `output_text.delta` → `output_text.done` → `content_part.done` → `output_item.done` → `completed` → `[DONE]`)
- ✅ Function/tool calling (`function_call` items with `call_id`/`name`/`arguments`)
- ✅ Namespace-tool flattening (`muse.submit_reminder_decision`)
- ✅ Reasoning/thinking translation (GLM `thinking` blocks → `reasoning` items + `response.reasoning_summary_text` events)
- ✅ String-content normalization, `instructions`→`system` folding, developer→system coalescing, empty-message dropping, `max_tokens` clamping

### GLM endpoint details (Z.ai `zai-coding-plan` subscription)
```
Base: https://api.z.ai/api/anthropic  (Anthropic-Messages-compatible)
Auth: x-api-key: <plan key>  +  anthropic-version: 2023-06-01
Model: glm-4.6
Capabilities: text, tool_use, thinking blocks, SSE streaming — all confirmed
Limit: max_tokens must be ≤ ~16K (rejects 128K with code 1213)
```

---

## 6. The Remaining Blocker (precisely characterized)

**Symptom:** Muse reaches the proxy, the proxy returns a spec-compliant Responses SSE stream (verified byte-for-byte via curl — 6ms, all events well-formed, GLM returns correct text), but Muse rejects **every** stream with `error_kind: "decode"` and retries 10× then dies.

**What it is NOT (all ruled out):**
- ❌ Not a network/sandbox issue — requests arrive, bodies fully received (85 KB), `accept: text/event-stream`, no `Expect: 100-continue`.
- ❌ Not a catalog issue — catalog accepted with `profile_id:"tbh"`.
- ❌ Not an upstream error — GLM returns 200 with correct text/tool output (0 errors across 20+ requests).
- ❌ Not disk pressure — root filled to 0 bytes during testing (containerd); freed to 70 GB; decode error persists.
- ❌ Not missing SSE events — `content_part.added/done`, `output_text.done`, `[DONE]` all added per the OpenAI spec & the sub2api issue #2875.
- ❌ Not missing Response fields — added all 13+ OpenAI-documented fields (`reasoning`, `text`, `incomplete_details`, etc.).
- ❌ Not timestamps — tried both unix and ISO-8601.
- ❌ Not reasoning items — GLM `thinking` → `reasoning` translation emits `response.reasoning_summary_text` events; Muse still rejects.

**Root cause:** Muse's `meta` provider SSE decoder requires a **Meta-specific field or event variant** that differs from vanilla OpenAI Responses. The binary's struct (`Response with 13 elements`) and event dispatch confirm it parses standard `response.*` event types via the `type` JSON field, but the exact required field/variant could not be determined without **Meta's real reference stream**.

**Why the reference stream is hard to get:** Muse's native Meta calls work, but the OAuth `access_token` in `auth.json` is *not* accepted by `api.meta.ai/v1/responses` directly (401) — Muse internally runs an OAuth→API-key "mint" exchange (binary strings: `mint returned an empty api key`, `TBH_MINT_BASE_URL`, `x-api-version: 1.0.0`) that I could not reproduce. TLS-encrypted strace/tcpdump captures can't be decrypted without the minted key, and `SSLKEYLOGFILE` isn't honored by Muse's HTTP client.

---

## 7. Path to Finish (for a future session)

1. **Capture Meta's real `/responses` SSE stream.** Most promising: build a TLS-intercepting MITM proxy (trusted CA in the system store + `/etc/hosts` redirect of `api.meta.ai`), run a native `muse` call through it, and record the exact bytes Meta returns — especially the `response.created` and `response.completed` JSON.
2. **Diff field-by-field** against the proxy's output. The decode error names a missing/unknown field but Muse swallows the serde message; the Meta reference will reveal it.
3. Likely candidates to check first: the `response.id` format (Meta may prefix differently), a `service_tier`/`model` field on `response.created`, or a Meta-specific top-level event (e.g. `response.model.*)`.
4. Once the field is found, add it to `muse-glm-proxy.mjs`'s `makeStreamTranslator`, re-run the headless gate:
   ```
   muse exec --provider meta --base-url http://127.0.0.1:8799/v1 --model glm-4.6 --yolo 'Reply: MUSE>GLM OK'
   ```
   then the tool-calling gate, then promote scripts from `tmp/` to `scripts/` per AGENTS.md §14.

---

## 8. How to Run What Works Today

```bash
# Start the GLM translator proxy (serves /muse-code/models + translates /responses)
GLM_API_KEY="<zai-plan-key>" GLM_MODEL="glm-4.6" \
  node ~/B70-DOCS/scripts/tmp/muse-glm-proxy.mjs &

# Point Muse at it (GLM chat works via curl; Muse decode-blocker pending §7)
META_API_KEY=dummy MUSE_MODEL=glm-4.6 muse exec \
  --provider meta --base-url http://127.0.0.1:8799/v1 --model glm-4.6 --yolo \
  --no-session-log --workspace /tmp/empty '...'

# For the local B70 (llama.cpp /responses works natively — different blocker):
bash ~/B70-DOCS/scripts/tmp/muse-b70.sh exec --yolo '...'
```

## 9. Key Reverse-Engineering Findings (durable, regardless of §6)

- Muse's `meta` provider = OpenAI Responses API (`/responses` + `/muse-code/models`), not Chat Completions.
- `profile_id: "tbh"` is mandatory in the catalog — non-obvious, not documented anywhere.
- `META_API_KEY` env bypasses OAuth entirely (cleanest way to redirect off Meta).
- Catalog cache lives at `~/.local/share/muse/model-catalog/6d657461__p746268.json` and **silently shadows** your proxy — must delete when switching backends.
- Muse uses OpenAI Responses **namespace tools** (`{type:"namespace"}`) — no other Responses client I've seen does this; any translator must flatten them.
- `--no-session-log` is essential when the root disk is tight (Muse's durable-event logging is write-heavy).
- The `b70-codex-proxy/server.mjs` already on this host solves the same developer→system coalescing for Codex CLI — reusable pattern.

Sources: [Meta Model API docs](https://dev.meta.ai/docs/overview/), [Muse Spark blog](https://developer.meta.com/ai/resources/blog/build-with-muse-spark/), [sub2api SSE issue](https://github.com/Wei-Shaw/sub2api/issues/2875), [Responses `[DONE]` spec](https://github.com/router-for-me/CLIProxyAPI/issues/3352), llama.cpp `/responses` live verification on the B70.
