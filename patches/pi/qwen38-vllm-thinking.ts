import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Qwen3.8 (qwen3_5 chat template) thinking translation for local vLLM XPU.
// pi's qwen-chat-template format is an on/off switch only (never sends
// reasoning_effort). This extension restores full levels: map pi's
// reasoning_effort (none|low|medium|xhigh) to chat_template_kwargs and
// apply the official model-card sampling parameters per mode:
//   thinking (low/medium/xhigh): temperature=1.0, top_p=0.95, top_k=20,
//     presence_penalty=0.0, repetition_penalty=1.0
//   non-thinking (off):          temperature=0.7, top_p=0.80, top_k=20,
//     presence_penalty=1.5, repetition_penalty=1.0
const QWEN_MODEL = "qwen38";

const THINKING_SAMPLING = {
  temperature: 1.0,
  top_p: 0.95,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 0.0,
  repetition_penalty: 1.0,
};

const INSTRUCT_SAMPLING = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 1.5,
  repetition_penalty: 1.0,
};

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

    // Official Qwen3.8-27B model-card sampling per mode.
    Object.assign(payload, enabled ? THINKING_SAMPLING : INSTRUCT_SAMPLING);
  });
}

// DEBUG-MARKER: loaded
