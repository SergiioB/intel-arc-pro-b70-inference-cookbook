import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Qwen3.8 (qwen3_5 chat template) thinking translation for local vLLM XPU.
// pi's qwen-chat-template format is an on/off switch only (never sends
// reasoning_effort). This extension restores full levels: map pi's
// reasoning_effort (none|low|medium|xhigh) to chat_template_kwargs.
const QWEN_MODEL = "qwen38";

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

// DEBUG-MARKER: loaded
