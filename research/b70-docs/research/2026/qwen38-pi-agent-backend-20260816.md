---
title: "Qwen3.8-27B as the `pi` coding-agent backend on B70 vLLM XPU (2026-08-15/16)"
type: research
date: 2026-08-16
status: final
---

# Qwen3.8-27B as the `pi` coding-agent backend on B70 vLLM XPU (2026-08-15/16)

How to run the **pi coding agent** (`/usr/bin/pi`, read/bash/edit/write tools)
with our local Qwen3.8-27B GPTQ champion as the model, with **working tool
calls**, thinking control, and full token/time accounting. This is the
"DeepSeek-harness style" agentic quality-eval path.

## 1. Server side (required flags)

The plain champion launcher is NOT enough for an agent: pi sends OpenAI
`tools` + `tool_choice:"auto"`, and vLLM rejects that unless tool parsing is
enabled. Qwen3.8 (`qwen3_5`) emits **qwen3_xml**-style tool calls — NOT hermes
JSON (verified: raw probe returned `<tool_call><function=write><parameter=…`
in `content` with `tool_calls: null` under the hermes parser).

Launch (script: `scripts/tmp/qwen38-pi-serve.sh`):

```bash
MAX_LEN=131072 GPU_UTIL=0.88 SPEC_N=4 POWER=230000000 \
GDN_OVERLAY=~/B70-DOCS/scripts/tmp/vllm-xpu-causalconv-patch/patch_gdn_mixed_split_v5.py \
EXTRA_ARGS='--enable-auto-tool-choice --tool-call-parser qwen3_xml' \
  bash scripts/tmp/b70-qw38-speed-launch.sh pi-agent-serve
```

Resulting config: 131072 ctx, fp8 KV, MTP4, XPU graphs, cache off,
`enable_auto_tool_choice: True`, `tool_call_parser: 'qwen3_xml'`,
GDN mixed-split v5 overlay (patches every `_xpu_ops.py`), ~940 MiB free
after load. Do **not** set `NOSPEC=1` for the mixed `causal_conv1d`
crash — that is the overlay's job. Sequential tool turns still sit at
agentic ~38 t/s (acceptance), not the no-spec floor. See
`research/2026/qwen38-gdn-mixed-split-v5-20260818.md` and AGENTS.md §15.14.

## 2. Register the model in pi

Edit `~/.pi/agent/models.json` (NOT `~/.pi/config.json` — that file's
`custom_models` is not the catalog pi resolves). Add under `providers`:

```json
"b70-vllm": {
  "baseUrl": "http://127.0.0.1:8000/v1",
  "api": "openai-completions",
  "apiKey": "local-b70",
  "models": [{
    "id": "qwen38",
    "name": "Qwen3.8-27B GPTQ B70 (vLLM XPU)",
    "input": ["text"],
    "supportsTools": true,
    "reasoning": true,
    "thinkingLevelMap": {
      "off": "none", "minimal": null, "low": "low", "medium": "medium",
      "high": "xhigh", "xhigh": "xhigh", "max": null
    },
    "compat": {
      "supportsDeveloperRole": false,
      "supportsReasoningEffort": true,
      "maxTokensField": "max_tokens"
    },
    "contextWindow": 131072,
    "maxTokens": 120000
  }]
}
```

Verify: `pi --list-models | grep b70-vllm` → `b70-vllm qwen38 131.1K 120K yes no`.

## 2b. Thinking levels require a small extension (CRITICAL)

pi's `openai-completions` provider sends `reasoning_effort` for all levels —
but vLLM's qwen3_5 chat template rejects `"none"` (raises, EngineCore dies)
and the qwen3_5 template gates thinking via `chat_template_kwargs.enable_thinking`,
NOT via `reasoning_effort` alone. Without extra plumbing, `--thinking off`
sends `reasoning_effort:"none"` → crash, and medium/xhigh would not be
honored by the template.

Working solution: a tiny `before_provider_request` extension that translates
the effort into the template kwargs:

`~/.pi/agent/extensions/qwen38-vllm-thinking.ts`:

```ts
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const QWEN_MODEL = "qwen38"; // match your model id

export default function (pi: ExtensionAPI) {
  pi.on("before_provider_request", (event) => {
    const payload = event.payload as Record<string, unknown>;
    const model = String(payload.model ?? "").toLowerCase();
    if (!model.includes(QWEN_MODEL)) return;

    const effort = payload.reasoning_effort;
    const enabled = typeof effort === "string" && effort !== "none";

    payload.chat_template_kwargs = {
      ...((payload.chat_template_kwargs as Record<string, unknown>) ?? {}),
      enable_thinking: enabled,
      preserve_thinking: true,
    };
    // Off is expressed exclusively via enable_thinking: false
    if (!enabled) delete payload.reasoning_effort;
  });
}
```

Load it explicitly (keeps `-ne` hermetic mode):

```bash
pi -p --provider b70-vllm --model qwen38 --thinking medium -nc -ne -ns \
  -e ~/.pi/agent/extensions/qwen38-vllm-thinking.ts "..."
```

Verified payloads (mock-echo server, 2026-08-16):

| pi --thinking | reasoning_effort | chat_template_kwargs |
|---|---|---|
| off | (absent) | `{"enable_thinking": false, "preserve_thinking": true}` |
| low | low | `{"enable_thinking": true, "preserve_thinking": true}` |
| medium | medium | `{"enable_thinking": true, "preserve_thinking": true}` |
| xhigh | xhigh | `{"enable_thinking": true, "preserve_thinking": true}` |

Do NOT use `compat.thinkingFormat: "qwen-chat-template"` on this stack: it
sends `enable_thinking`/`preserve_thinking` but never `reasoning_effort`, so
medium/xhigh cannot be expressed (on/off only). The extension above is the
correct path.

## 3. Thinking control

pi's `--thinking <level>` maps through `thinkingLevelMap` to the vLLM
top-level `reasoning_effort` request field (with the extension above
translating it into `chat_template_kwargs`). On this stack reasoning lands
inside `content` as a `<think>…</think>` block (qwen3_5 trait) — downstream
consumers must extract it from there; the API's
`completion_tokens_details.reasoning_tokens` is null on streaming usage.

## 4. Working agent invocation

```bash
pi -p --provider b70-vllm --model qwen38 --thinking medium \
   -nc -ne -ns \
   "Create a file named hello.txt containing exactly: PI-AGENT-OK."
```

Verified end-to-end: the model plans briefly, emits a qwen3_xml tool call,
vLLM parses it into `tool_calls`, pi executes `write`, file appears.

Flags used: `-p` non-interactive; `-nc` no AGENTS.md pickup; `-ne` no
extensions; `-ns` no skills (keep the eval hermetic and fast). Drop them for
a fully-featured agent session.

## 5. Games quality-eval runner

`scripts/tmp/b70-qwen38-pi-games-runner.py` + wrapper
`scripts/tmp/qwen38-pi-games-run.sh`:

- 10 ultra-detailed game specs (features + architecture; Temple Run + 9 more)
- per-game workdir under `results/2026/08/qwen38-quality-20260815/pi-games/<key>/`
- `--session session.json` for deterministic session capture
- wall time measured around the whole agent run
- token/tool stats parsed from the session file (usage objects + tool_calls)
- final artifact: `pi-games/<key>.html` with metadata header (total/prompt/
  output/reasoning tokens, agent wall time, turns, tool calls)

## Failure ledger (what it took)

1. pi with default provider → must register in `~/.pi/agent/models.json`;
   editing `~/.pi/config.json` `custom_models` does nothing.
2. First request → HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice...`.
3. `--tool-call-parser hermes` → request succeeds but `tool_calls: null`;
   the model's XML tool syntax sits unparsed in `content`.
4. `--tool-call-parser qwen3_xml` → tool calls parse and execute.
5. `/tmp` on this host is unreliable for long-lived scripts across sessions —
   keep runner scripts under `B70-DOCS/scripts/tmp/`.

Status: PROVISIONAL (quality observation, no throughput claims).
