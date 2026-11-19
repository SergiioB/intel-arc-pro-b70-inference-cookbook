# xpu(qwen27): shared draft trees read the prefix on XPU_ATTN_PREF too

Branch: `SergiioB:xpu-tp2-fixes-20261119`, commit `df3baa3ea` on `2286abc0c`. This change applies only to builds
that carry our prefix attention kernels (`XPU_ATTN_PREF`: tag `b70-attn-prefix-20261113`, closed PR #3181, and
`2286abc0c` on the TP runner). It does not apply to #3586 alone, which has no prefix kernel.

## Problem

With `XPU_ATTN_PREF` set and `--parallel` above 1, a stream's greedy reply depends on whether its draft tree is
verified alone or in a round shared with other streams. On two B70s with `--tp 2 --parallel 4`, the 7,963-token
`chat-8k` prompt returned a different 512-token reply in every drafted concurrent repeat than it did alone. Serial
(`"draft": false`) replies matched. Two independent one-card servers showed one differing and one matching drafted repeat;
which rounds `chat-8k` shared was not recorded there.

A lone tree took `attn_pref` + `attn_tail` + `attn_merge2`, while a shared tree took `attn_decode` + `attn_merge`.
The paths differ numerically. The prefix kernels compute scores and P·V on DPAS with bf16-rounded softmax weights
(`decode_prefix.cl:34`, `:42`). The sliced kernels are a scalar reduction with float weights (`decode.cl:125`,
`:137`). Shared serial rows already took the prefix path.

## Change

`pre = q.attn_pref` for every decode round. A tree row's prefix ends at `pref_base(pos) <= pos - 16`, which is before
its root, so the prefix chunks read only committed rows. The tree lookup stays in `attn_tail`, which the shared path
already passes the tree table.

## Measured

Two B70s, Qwen3.8-27B MLX 4-bit + DFlash2, `--tp 2 --parallel 4 --context 16384`, prompt cache off. Four prompts
(68, 52, 7,976 and 7,963 tokens) run alone, then all four at once in 3 repeats, each drafted and serial: 32 requests
per build. A request passes when its text equals the prompt's solo text and the engine logs one token hash per prompt.

| Build | XPU_ATTN_PREF | Requests matching | `chat-8k` token hash, alone / shared |
|---|---|---|---|
| `2286abc0c` | 256 | 29 of 32 (3 drafted shared `chat-8k` differ) | `5b5869d7083a` / `1167cd16ac89` |
| `df3baa3ea` | 256 | 32 of 32 | `5b5869d7083a` / `5b5869d7083a` |
| `2286abc0c` | 0 (sliced everywhere) | 32 of 32 | `1167cd16ac89` / `1167cd16ac89` |
| `78f42f240` (mitigation commit) | 256 | 32 of 32 | `5b5869d7083a` / `5b5869d7083a` |
| `15adb58cf` (head after review round 1) | 256 | 32 of 32 | `5b5869d7083a` / `5b5869d7083a` |

For this workload, the prefix-off control (third row) isolates the choice of attention path. With one path for
every row, the batched replies equal the solo replies. The runs do not isolate which numerical difference flips the
token, and no logits were kept to show a near tie. The older one-card comparison had one differing and one matching
drafted repeat; per-round co-scheduling was not recorded for it.

## Reproduce

```bash
XPU_ATTN_PREF=256 tensorfold-native serve Qwen3.8-27B-MLX-4bit --drafter Qwen3.8-27B-DFlash2 --tp 2 --parallel 4 \
  --context 16384 --prompt-cache-gib 0 --no-thinking
# send chat-8k alone, then together with three other prompts; compare the replies
```

## Revert

`git revert df3baa3ea` (one line in `zig/src/xpu/forward.zig`).

## Risk

Low. It only affects decode rounds of more than one stream with drafting on and `XPU_ATTN_PREF` set. Shared trees now
read the prefix on DPAS, the same kernel as lone trees and shared serial rows. Their speed was not measured. Not
tested: int8 K/V with `--parallel` above 1, and `--parallel 8`.
