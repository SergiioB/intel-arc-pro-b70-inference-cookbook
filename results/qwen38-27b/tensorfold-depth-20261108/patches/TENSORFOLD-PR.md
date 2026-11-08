xpu/qwen27: batch tree rows per key slice and widen slices for long rows (+49 % decode at 128K on B70)

## What

`attn_decode` on the Qwen3.8-27B Level Zero path re-streams the whole K/V cache once per tree
row: the grid is `(slices, kv heads, rows)`, so every slice for row 0 is dispatched before row 1
starts and L2 never reuses a slice. At 131K context that is about 15 × 8.6 GB of K/V plus 48 GB
of fp32 partials per verify round (~468 GB/s sustained, so the kernel is bandwidth-bound on
redundant traffic, not slow).

Two changes, both in `kernels/xpu/qwen27/decode.cl` and the launch in `src/xpu/forward.zig`:

1. Grid order `(rows, kv heads, slices)`: the up-to-16 tree rows of one 32-key slice run back
   to back and share that slice's K/V rows in L2. Arithmetic per work-group is unchanged
   (byte-identical output on its own).
2. `row_split(pos)`: keys per slice is 32 below position 8192, 128 below 65536, 256 above. It is
   a function of the row's own position, so a drafted row and the same row decoded serially
   use the same slices and the same merge order: drafted == serial stays bit-exact. The host
   sizes the grid for the narrowest row of the round; wider rows return early past their
   last slice. Multi-stream rounds keep 32.

## Measured (one Intel Arc Pro B70, 180 W cap, Qwen3.8-27B-MLX-4bit + DFlash2, C=1, greedy, 512 tokens)

Streamed decode tok/s after the first token, same card, same frozen prompts, stock v1.2.4
(`00e7ddd` + ocloc flag patch) vs patched:

| context | code stock | code patched | chat stock | chat patched |
|---|---|---|---|---|
| 68 | 123.3 | 123.2 | 98.6 | 98.5 |
| 8K | 81.7 | 87.2 (+7 %) | 65.2 | 70.0 (+7 %) |
| 32K | 36.0 | 46.6 (+29 %) | 26.6 | 34.5 (+30 %) |
| 128K | 11.0 | 16.3 (+49 %) | – | 14.6 |

`"draft": false` (serial) patched: 31.3 / 29.3 / 26.5 / 19.0 tok/s; stock serial at 128K is
12.8 (230 W), so serial gains +48 % too.

Correctness: patched drafted output == patched serial output (sha256 of the text) in all 8
cells, including the 8K cells whose generation crosses position 8192 mid-answer; repeats are
byte-identical; the 9-case exact-answer battery passes on stock and patched. Patched output
differs from stock at some depths because the merge order changes (deterministically).

## Not done

- A local-memory variant that stages each slice once for all rows (one work-group per slice,
  subgroup per row) was 5× slower here; likely register pressure at 256 GRF. Left out.
- At 128K the serial path (19.0) is still faster than drafted (16.3): the verify round costs
  more than the accepted tokens buy. A context threshold for drafting would help more.
- int8 cache (`XPU_KV_INT8=1`) and multi-stream rounds were not measured.

Evidence and harness: https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook (TensorFold page).
