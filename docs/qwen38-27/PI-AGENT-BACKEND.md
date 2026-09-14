# Running the `pi` Coding Agent on Qwen3.8-27B (B70 vLLM XPU)

The cookbook's Qwen3.8 recipe can also power the **pi coding agent** — the
same `pi` CLI used for the daily assistant — with fully working tool calls
(read / bash / edit / write). This is the agentic quality-eval path
("DeepSeek-harness style") and a practical local-agent serving recipe.

## 1. Launch vLLM with tool calling

The plain launch command is not enough: an agent sends OpenAI `tools` with
`tool_choice:"auto"`, and **Qwen3.8 emits qwen3_xml-style tool calls** —
`<tool_call><function=name><parameter=key>value`. The `hermes` parser leaves
them unparsed inside `content`; you must select `qwen3_xml`.

```bash
docker run -d --name qw38speed -p 8000:8000 --device /dev/dri \
  --group-add $(stat -c '%g' /dev/dri/render* | sort -u | head -1) \
  -v /dev/dri:/dev/dri:ro \
  -v /path/to/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro \
  -v patches/patch_mtp_nightly.py:/patch_mtp.py:ro \
  -v patches/patch_mtp_boundary.py:/patch_boundary.py:ro \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f -lc \
  "set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model \
     --quantization gptq --dtype float16 --max-model-len 131072 \
     --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 --port 8000 \
     --max-num-seqs 64 --max-num-batched-tokens 8192 \
     --no-enable-prefix-caching --served-model-name qwen38 --language-model-only \
     --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}' \
     --enable-auto-tool-choice --tool-call-parser qwen3_xml"
```

Differences vs the benchmark recipe: `--enable-auto-tool-choice
--tool-call-parser qwen3_xml` added; ~940 MiB free after load at U=0.88.

## 2. Register the model in pi

Add to `~/.pi/agent/models.json` under `providers` (this is the real catalog;
`~/.pi/config.json` `custom_models` is not what pi resolves):

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

## 3. Thinking levels (Qwen chat-template translation)

The Qwen3.5 chat template gates thinking via `chat_template_kwargs.enable_thinking`
and takes an effort from `reasoning_effort` (low/medium/xhigh). Add a small
`before_provider_request` extension so pi's levels reach the template
correctly — `~/.pi/agent/extensions/qwen38-vllm-thinking.ts`:

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
    if (!enabled) delete payload.reasoning_effort; // off = enable_thinking:false only
  });
}
```

## 4. Use it

```bash
pi --provider b70-vllm --model qwen38 --thinking medium
# non-interactive / scripted (load the extension explicitly):
pi -p --provider b70-vllm --model qwen38 --thinking medium -nc -ne -ns \
  -e ~/.pi/agent/extensions/qwen38-vllm-thinking.ts "task..."
```

Verified behavior (payload-level, 2026-08-16):

| `--thinking` | `reasoning_effort` | `chat_template_kwargs` |
|---|---|---|
| off | — | `enable_thinking: false, preserve_thinking: true` |
| low | `low` | `enable_thinking: true, preserve_thinking: true` |
| medium | `medium` | `enable_thinking: true, preserve_thinking: true` |
| xhigh | `xhigh` | `enable_thinking: true, preserve_thinking: true` |

The agent plans, emits qwen3_xml tool calls that vLLM parses, pi executes
`write`/`bash`/`edit`, files appear on disk.

Notes:
- Reasoning text arrives inside `content` as a `<think>…</think>` block on
  this stack; the streaming `usage` does not expose reasoning-token counts.
- Agent decode runs at ~30-50 tok/s (MTP4 acceptance ~45-60% on tool/agentic
  turns); a deep single-file game task takes minutes, not seconds.
