# Root strays archived 2026-10-09

Moved out of the live tree during the 2026-10-09 structure audit. Nothing here is an authority.

| Item | Origin | Why archived |
|---|---|---|
| `.tmp-qw38-launch.remote.sh` | repo root, 2026-08-28 | One-off remote FP8 PP2 launch script; superseded by `docs/operations/launcher-guide.md`. |
| `owned-cid.txt` | repo root | Runtime container-id leftover written by `scripts/tmp/serve-ready-launch-20260930.py`; runtime state, not evidence. |
| `.tmp-nemotron-diag/` | repo root | Five vendored transformers/vLLM source copies used for a Nemotron diagnosis; no live references. |
| `references/` | repo root | Three undated, unsourced format notes (`native-xe2-formats-2026.md`, `nvfp4-xmx-research.md`, `sglang-dense-xe2-custom-kernel.md`). They recommended the GPTQ W4A16 route that was later removed for a weight defect (RL-087) and gave a 43 tok/s "ceiling" that ignores speculation. Superseded by `docs/reference/b70-format-and-lever-guide.md`. |
| `scripts-tmp-accidental-tree/results/` | `scripts/tmp/dflash2-objective-port/vllm/model_executor/models/results/` | A results path expanded inside a source port by mistake (one 0-byte jsonl, one 180-byte stderr). |

Deleted (0 bytes, no content): four root `b70-qwen38-flash-next-*.png|svg` renders and
`scripts/sweeps/{b70-quick-vision,qwen-vision-power}.sh`.
