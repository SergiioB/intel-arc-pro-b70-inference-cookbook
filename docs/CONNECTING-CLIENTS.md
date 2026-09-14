# Connecting Pi / omp / Hermes to the B70 vLLM servers

Exact config files and blocks for the three agent harnesses, for a B70 host
serving vLLM on the bridge port. The servers expose an OpenAI-compatible
`/v1` API with **tool calling enabled**
(`--enable-auto-tool-choice --tool-call-parser qwen3_coder`).

## Ports and model names

| Server | Base URL | Served model names |
|---|---|---|
| Cookbook launcher (any model) | `http://<host>:8000/v1` | `Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4` / `Qwen3.6-27B-MTP-Preserved-GPTQ-Int4` |
| Bridge / agent endpoint | `http://<host>:8765/v1` | `active` (alias for the currently loaded model) |

Clients should use the `active` alias so a model switch does not break the
config. Replace `<host>` with `localhost` (same machine) or the LAN IP.
Replace `<INFERENCE_API_KEY>` with the key your server is started with
(`--api-key <INFERENCE_API_KEY>`; the desktop setup uses a fixed local key —
set your own for a public deployment).

## Hermes — `~/.hermes/config.yaml`

Under `providers:`, add or replace the `workstation-loaded` block:

```yaml
providers:
  workstation-loaded:
    api_key: <INFERENCE_API_KEY>
    api_mode: chat_completions
    base_url: http://<host>:8765/v1
    context_length: 131072
    default_model: active
    discover_models: false
    name: WKS Loaded Model [B70 vLLM endpoint]
```

Tool calling works automatically: Hermes sends `tool_choice: "auto"` and the
server's `qwen3_coder` parser handles the function-call format.

## Pi — `~/.pi/config.json`

Add an entry to the `custom_models` array (OpenAI-compatible provider):

```json
{
  "model_display_name": "Desktop B70 Loaded Model",
  "model": "active",
  "base_url": "http://<host>:8765/v1",
  "api_key": "<INFERENCE_API_KEY>",
  "max_tokens": 32768,
  "provider": "openai"
}
```

Use `max_tokens >= 512` for reasoning models (128 returns empty content on
the ThinkingCap/Qwen3.6 family). Document-grounded Pi sessions get the
resident-document prefix-cache speedup: a cold document costs one long TTFT,
then follow-ups reuse 89.9–94.7% of tokens (see `REAL-WORLD-PI-BENCHMARKS.md`).

## omp — `~/.omp/agent/models.yml`

Add or update the `desktop-b70` provider block (under the providers list):

```yaml
  desktop-b70:
    baseUrl: http://<host>:8765/v1
    api: openai-completions
    apiKey: <INFERENCE_API_KEY>
    compat:
      supportsDeveloperRole: false
      maxTokensField: max_tokens
    models:
      - id: active
        name: Desktop B70 Loaded Model
        reasoning: true
        input: [text, image]
        supportsTools: true
        contextWindow: 131072
        maxTokens: 32768
```

If omp reports
`400: "auto" tool choice requires --enable-auto-tool-choice and
--tool-call-parser to be set`, the server was started without the two tool
flags — restart it with the cookbook launchers (they include them), or add
`--enable-auto-tool-choice --tool-call-parser qwen3_coder` to the serve
command.

## Sanity check

```bash
curl -f http://<host>:8000/health
curl -fsS http://<host>:8000/v1/models
```

Tool-call smoke test (expect `finish_reason: tool_calls` with
`function.name: get_weather` — verified 2026-08-10 on the dense 27B MTP4
server):

```bash
curl -fsS http://<host>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <INFERENCE_API_KEY>" \
  -d '{
    "model": "active",
    "messages": [{"role": "user", "content": "What is the weather in Berlin? Use the weather tool."}],
    "tools": [{"type": "function", "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }}],
    "tool_choice": "auto",
    "max_tokens": 128
  }'
```
