@ashhart, here is the combined measurement you asked for on 16 Nov: main against main with both exact changes in, one Arc Pro B70, token hashes drafted and `"draft": false`.

**Builds.** main `8cbef9777`. Combined: the same main + #2595's eight commits (they cherry-pick cleanly, including your two) + #2799's production change only: `attn_merge` in `decode.cl` and its launch grid in `forward.zig`, taken from #2799's head `d68ec445b`. In `forward.zig` the two launch lines sit next to each other; the combined tree keeps #2595's `attn_decode` grid `{T, 4, ns}` and #2799's `attn_merge` grid `{24 * 16, T}`. Both kernel bodies are byte-equal to their PR heads. #2799's contract registration conflicts with #2595's in nine contract files, so it is not in this tree and the kernel contract was not run on it; that is the rebase for whichever lands second.

**Setup.** Intel Arc Pro B70 (BMG-G31), 230 W cap. Ubuntu 26.04, kernel 7.0, compute-runtime 26.05.37020, oneAPI 2026.1, Zig 0.17.0, plus the one local build change described in #2595 (drops `-cl-intel-128-GRF-per-thread` from the Gemma 4 prompt-rows kernels, on both sides). Checkpoint `TensorFold/Qwen3.8-27B-MLX-4bit`, drafter `z-lab/Qwen3.8-27B-DFlash2`, bf16 K/V. `tensorfold-native serve --parallel 1 --context 139264 --prompt-cache-gib 0 --no-thinking`, greedy, 512 tokens, one discarded warmup a server, one server a side (main first, then combined). Binary sha256: main `d1778e0f…85b7`, combined `60a9806d…a5c5`. Prompts are the same frozen set as #2595's receipt.

**Token hashes** (`sha` from the server log). Every cell has one hash across main and combined, drafted and `"draft": false`:

| Prompt tokens | Kind | `token_sha` | Requests (main drafted / main plain / combined drafted / combined plain) |
|---|---|---|---|
| 68 | code | `ae1c3bc90a5f` | 2 / 1 / 2 / 1 |
| 52 | prose | `596a68a0fce5` | 2 / 1 / 1 / 1 |
| 7,976 | code | `46ebf41df2f3` | 2 / 1 / 1 / 1 |
| 7,963 | prose | `50324eb0292b` | 2 / 1 / 1 / 1 |
| 32,552 | code | `348fc6e61afc` | 2 / 1 / 1 / 1 |
| 32,540 | prose | `e5a276a9b09b` | 2 / 1 / 1 / 1 |
| 130,856 | code | `67b03c8a4493` | 2 / 1 / 1 / 1 |
| 130,846 | prose | `3d2fd36f54fe` | 2 / 1 / 1 / 1 |

Accepted drafts are equal too (e.g. 406/1575 at 32,552 code, 416/1485 at 130,856 code, both sides).

**Decode**, tokens/s after the first token, code / prose, one run a cell (main drafted has two requests that agree within 0.2 tok/s):

| Prompt tokens | main drafted | combined drafted | ratio | main plain | combined plain | ratio |
|---|---|---|---|---|---|---|
| 68 / 52 | 125.3 / 103.2 | 125.2 / 103.0 | 1.00 | 32.3 / 32.3 | 32.3 / 32.3 | 1.00 |
| 7,976 / 7,963 | 85.0 / 69.0 | 90.2 / 73.2 | 1.06 | 30.7 / 30.7 | 31.2 / 31.2 | 1.02 |
| 32,552 / 32,540 | 35.2 / 30.0 | 42.9 / 36.6 | 1.22 | 23.5 / 23.5 | 27.3 / 27.3 | 1.16 |
| 130,856 / 130,846 | 12.3 / 10.1 | 16.2 / 13.2 | 1.31 | 12.8 / 12.8 | 18.1 / 18.1 | 1.41 |

Time to first token is the same on both sides (21.5 s at 32K, 128.7–128.8 s at 131K). At about 131K drafted decode is still slower than plain decode on the combined build (16.2 against 18.1 on code), as on main (12.3 against 12.8).

Not run: `XPU_KV_INT8=1` (the short-context int8 question on #2595 is still open and this does not answer it), `--parallel` above 1, a second card, repeats beyond the above, resumed prompts, sampled decoding. The main→combined order was not alternated.

The per-request rows, server logs and the comparison script can go into the receipt folder of whichever PR lands second.
