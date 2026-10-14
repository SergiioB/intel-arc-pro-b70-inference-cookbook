---
title: "Qwen3.6-27B ThinkingCap bounded A/B — 2026-07-19"
type: doc
date: 2026-07-19
status: final
---

# Qwen3.6-27B ThinkingCap bounded A/B — 2026-07-19

## Decision

**Keep `Qwen3.6-27B-MTP-Q5_K_M` in production. Do not promote ThinkingCap Q4.**

ThinkingCap is a valid, efficient llama.cpp candidate and it shortened one trivial reasoning trace enough to return final content. It did not solve the motivating failure on the four representative workload prompts: every one consumed the full 768-token completion in reasoning and returned empty final content. The exact production control failed those same four prompts in the same way. ThinkingCap therefore did not meet the replacement gate of superior final-content completion on the actual workload.

No production profile, catalog entry, model, projector, service unit, or active-profile file was changed. The temporary serial candidate service was removed and production was restored and verified.

## Current-source findings and selection

The primary candidate was selected from the current publisher repositories before download:

- [ThinkingCap GGUF](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF), publisher commit `0dd71bbcc88a4134cf2abc389480211d77e5a586`.
- [ThinkingCap model card](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B) reports a publisher-measured 45.8% macro reduction in thinking tokens, aggregate score 80.7 versus 81.5 for the base model, and truncation reduction from 2.9% to 0.4%. These are publisher claims, not local results.
- [Official Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) documents per-request `enable_thinking=false`; the older `/think` and `/nothink` soft switches are not the supported control.
- [bytkim MTP PI tune](https://huggingface.co/bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF) remains a separate no-thinking agent/coding candidate, with no matching projector in its repository and no published evidence for controlled reasoning. It was not downloaded.
- [natfii VLM Cascade](https://huggingface.co/natfii/Qwen3.6-27B-VLM-Cascade) remains BF16/Transformers-only in its published repository. It was rejected under the B70 conversion and storage guardrails.
- Newer [Bartowski](https://huggingface.co/bartowski/bottlecapai_ThinkingCap-Qwen3.6-27B-GGUF) and [jpetrina](https://huggingface.co/jpetrina/ThinkingCap-Qwen3.6-27B-IQ4_XS-pure-GGUF) GGUFs package the same ThinkingCap weights; neither adds model-native hard-budget behavior. No ready GGUF beta with a genuine model-native hard thinking budget was found.

Detailed discovery evidence (pre-migration `docs/research/…` file) was not carried into the typed tree.

## Artifact integrity and storage

Only one candidate was downloaded, into the approved ext4 workspace:

| Artifact | Bytes | Verified SHA-256 |
|---|---:|---|
| `ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf` | 16,810,713,056 | `b0651e28555bde7d2459ce99f091319b1a547143463e8d49f2aa7f572675fe67` |
| `mmproj-ThinkingCap-Qwen3.6-27B-f16.gguf` | 931,145,888 | `81a714ac5e8e15687371fc95a180953a29b732962f6616f791063ff127559412` |

The final-path files were reread and matched both published hashes. After download, `/mnt/<other-disk>` had 31,658,586,112 bytes free (about 29.49 GiB); the final check showed 30 GiB free. Production storage remained at about 25 GiB free. No NTFS volume was used or modified.

Embedded GGUF metadata independently confirmed:

- Qwen hybrid architecture, 65 blocks, 262,144-token declared context.
- One `nextn_predict` layer for MTP.
- An embedded Jinja template supporting `enable_thinking`, vision, and tool calls.
- Matching `qwen3vl_merger` vision projector, 768-pixel input, projection dimension 5,120.

## Test method and containment

Production control and candidate used llama.cpp b9853, one slot, full SYCL offload, flash attention, Q8 K / Q4_1 V cache, embedded MTP with a maximum draft of four, 180 W, identical deterministic request sampling (`temperature=0.2`, `top_p=0.8`, `top_k=20`) and identical seeds.

The candidate was loaded serially on port 8766 at a bounded 65,536-token server context. It ran in a transient user service with:

- `MemoryHigh=25G`
- `MemoryMax=27G`
- `MemorySwapMax=0`
- `OOMPolicy=kill`
- `Restart=no`

The transaction installed the restoration trap before stopping production. Candidate startup took about 70.3 seconds. The test process reached 13.23 GB cgroup memory peak, used zero swap, and retained 9.15 GiB visible VRAM after its 32K test. Temperature finished at 73 C. No OOM, Xe reset, or GPU fault was logged.

## Results

### Final-content completion and reasoning

Reasoning token counts below were obtained by tokenizing the returned `reasoning_content` with the shared Qwen tokenizer.

| Exact default-thinking case | Production Q5 | ThinkingCap Q4 |
|---|---:|---:|
| Arithmetic, cap 512 | 512 reasoning tokens, **no content** | 328 reasoning tokens, content at 372 total |
| Rule against perpetuities, cap 768 | 768, **no content** | 768, **no content** |
| Paged KV cache, cap 768 | 768, **no content** | 768, **no content** |
| Fault-tolerant service design, cap 768 | 768, **no content** | 768, **no content** |
| Speculation vs batching, cap 768 | 768, **no content** | 768, **no content** |
| **Final-content rate** | **0/5** | **1/5** |
| **Representative workload only** | **0/4** | **0/4** |

The candidate is demonstrably shorter-thinking on the trivial case, but the publisher aggregate claim did not translate into successful final responses for this bounded local workload.

### Per-request modes and hard-budget behavior

`chat_template_kwargs: {"enable_thinking": false}` worked on both models:

- Four of four direct text cases emitted content on both models.
- Only the law response naturally stopped within the 512-token cap; the other three hit `finish_reason=length` on both. Direct mode avoids empty final content but the application still needs workload-appropriate response caps.
- Both models emitted the required `get_weather(city="Madrid", unit="celsius")` tool call and, after a simulated tool result, returned the final answer: `The current weather in Madrid is clear with a temperature of 31°C.`

The advertised request field `reasoning_budget_tokens=256` did **not** enforce a 256-token boundary on b9853:

- Production arithmetic used 512 reasoning tokens and returned no content.
- Production systems and comparison used 768 reasoning tokens and returned no content.
- ThinkingCap arithmetic ended naturally after 410 reasoning-text tokens and returned content at 463 total tokens; this was already above the requested budget.
- ThinkingCap systems and comparison used 768 reasoning tokens and returned no content.

A final production-only diagnostic added a non-empty `reasoning_budget_message`, based on the local sampler implementation. Arithmetic still used all 512 reasoning tokens. The systems request then stopped progressing after about 338 decoded tokens, never emitted `[DONE]`, and made the server unresponsive to health requests. The client was aborted and the service was cleanly stopped and started. This path is classified as unsafe on b9853 and must not be used in production.

### Performance and MTP

| Metric | Production Q5 | ThinkingCap Q4 | Candidate delta |
|---|---:|---:|---:|
| Four direct-text mean decode | 24.62 t/s | 26.99 t/s | +9.6% |
| Four direct-text mean TTFT | 0.726 s | 0.803 s | +10.6% slower |
| Four direct-text mean MTP acceptance | 87.26% | 88.10% | +0.84 points |
| 32K prompt processing | 308.22 t/s | 303.29 t/s | -1.6% |
| 32K TTFT | 107.25 s | 108.36 s | +1.0% slower |
| 32K decode | 29.01 t/s | 31.93 t/s | +10.1% |
| Vision decode | 24.94 t/s | 27.00 t/s | +8.3% |
| Vision TTFT | 1.222 s | 1.472 s | +20.4% slower |

The Q4 candidate's decode and memory advantages are real. They do not compensate for its failure to improve representative reasoning completion.

### Quality, vision, context, and stability

- Both models returned the exact 32K final-ledger retrieval: `M00909` / `cobalt-909`.
- Both vision runs recognized the B70 chip, cyan ring, and yellow lightning bolt, but transcribed `INFERENCE` as `INFERENC` and exhausted the 256-token response cap.
- Candidate vision produced the same `non-consecutive token position` warnings already observed in the production baseline. No crash followed.
- Direct law answers were substantively similar. The candidate's direct speculation comparison incorrectly described speculative decoding as targeting TTFT as well as decode latency; the production answer correctly centered decode latency. This is a representative accuracy regression, although the sample is too small for a broad capability claim.
- Tool selection and the bounded two-step agent flow were equivalent.
- Candidate MTP was active throughout, with per-request acceptance from about 84.6% to 100%; the four direct-text mean was 88.10%.

## Production restoration evidence

After candidate testing, production restored automatically. After the later hard-budget diagnostic hang, it was restored again with a controlled manual stop/start. Final state:

- `llama-profile.service`: active/running, PID 263415 at verification, `NRestarts=0`.
- `/health`: `{"status":"ok"}`.
- Served alias/context: `active`, 131,072.
- Exactly one `llama-server` process.
- System `oom-watchdog.service`: active.
- No new OOM, Xe reset, or GPU fault in the checked kernel log.
- Profile SHA-256 unchanged: `c987ae967ec3b1e594b121d5bfc430053a313b47d9b82e9f8e4da4627355d612`.
- Active-profile SHA-256 unchanged: `93dfc8578178a396b16372268b41c6f097718cb5cf6bf6a386444735c499d257`.
- Installed unit SHA-256 unchanged: `6a8a7905d6be711bb722265d7255ac45b5f1869a01c7329db6f404fe9fa2e9e7`.

## Recommendation and next gate

Do not migrate or promote ThinkingCap Q4. Keep the candidate files only as bounded research artifacts.

The highest-impact next research action is an isolated llama.cpp hard-budget regression investigation using a newer or patched engine build against the existing production model—no new large model download is needed. A model replacement should be reconsidered only after all of these gates pass:

1. A hard 256-token reasoning budget reliably emits an end tag and final content without hanging the server.
2. Five of five default/budget completion cases return final content, including all four representative tasks.
3. Direct and reasoning quality show no representative accuracy regression.
4. Vision, tool/agent, and 32K tests pass without new stability warnings.
5. The same automatic restoration test succeeds after a forced candidate failure.

There is no migration plan to execute because the candidate did not win. Any later winning candidate must still receive a new profile/catalog entry, a serial canary switch, and an exact rollback to `qwen27-mtp-q5-256k-vision-180w`, all gated by explicit operator approval.

## Evidence locations

- Discovery: pre-migration doc, not carried (see archive/docs/pre-migration-20260819/)
- Production control: `/mnt/<other-disk>/inference-models/experiments/thinkingcap-qwen36-27b-q4km/ab/qwen27-production-control-20260719T003839+0200/`
- Exact production default control: `/mnt/<other-disk>/inference-models/experiments/thinkingcap-qwen36-27b-q4km/ab/qwen27-production-default-exact-20260719T005600+0200/`
- Candidate run: `/mnt/<other-disk>/inference-models/experiments/thinkingcap-qwen36-27b-q4km/ab/thinkingcap-candidate-20260719T005000+0200/`
- Candidate server log: `/mnt/<other-disk>/inference-models/experiments/thinkingcap-qwen36-27b-q4km/transaction-20260719T005000+0200/candidate-server.log`
- Token analysis: `/mnt/<other-disk>/inference-models/experiments/thinkingcap-qwen36-27b-q4km/ab/reasoning-token-analysis.json`
- Unsafe forced-message probe: `/mnt/<other-disk>/inference-models/experiments/thinkingcap-qwen36-27b-q4km/ab/qwen27-production-budget-message-20260719T010200+0200/`
