# Evidence: two B70s, 250K context, TensorFold against exl3xpu (2026-11-18 to 2026-11-19)

Raw text evidence behind three pages:

- [Two cards for one stream: what scales, two fixes](../../../docs/qwen38-27b/TENSORFOLD-TWO-CARDS-20261119.md)
- [TensorFold against exl3xpu, greedy, same session](../../../docs/qwen38-27b/ROUTE-GREEDY-SAME-SESSION-20261119.md)
- [250K context and warm sessions](../../../docs/qwen38-27b/LONG-CONTEXT-250K-20261119.md)

`python3 summarize.py` prints every table on those pages from the files here (stdlib only).

Paths were rewritten before publication: `$TF` is the TensorFold work tree (source, models, toolchain),
`$WORK` the scratch folder of the campaign, `$MODELS` and `$DATA` the model disks, `~` the lab home. Server
logs and per-request text outputs are kept where they are small. Long prompt texts and streamed chunks are not
kept; the prompt files are named by sha256 in each `env.txt`.

| Folder | What | Source run |
|---|---|---|
| `routecmp/` | One card (GPU1, 230 W): TensorFold `tf1-depth`, exl3xpu `exl3-mtp3` and `exl3-serial` at 0/8K/32K/131K; `exl3-250k`; several-stream cells `tf1-conc`, `exl3-mtp3/conc.jsonl`, `tf-conc/{tf1,tf2}`; exl3xpu MTP3 329-item battery grade | 2026-11-18 07:21 UTC to 2026-11-19 01:07 UTC |
| `kv256k/` | 249,784-token prompt with int8 and bf16 K/V on one and two cards (`a`, `b`, `c`), needle retrieval grade, load refusals (`d2`, `d3`, `d3b`, `w2`), warm sessions (`d1`, `w1`) | 2026-11-18 |
| `gates/` | Two cards: resumed against fresh at 32K and 131K, `--parallel 4` against two servers, sampled and thinking cells, second 329-item battery and its per-item identity with the first | 2026-11-18 |
| `tptf/` | Fork fixes: `patches/` (three commits of branch `xpu-tp2-fixes-20261119`), `raw/*/check.jsonl`, `repro.sh`, `check.py`, standalone Level Zero copy programs in `alias/` | 2026-11-19 |
| `tpdiag/` | Dev-runner logs one card against two (`tf/`), Level Zero primitives (`prim-summary.tsv`), exchange chain (`chain-summary.tsv`), one-card round budget, unitrace crash logs, platform capture, probe sources | 2026-11-19 |
| `sweep-tp2-bf16/` | Decode by prompt length on two cards, 512 to 245,760 tokens, `samples.jsonl` (rep -1 is the warm-up) | 2026-11-19 04:35-05:28 UTC |
| `prefill8/` | int8-activation prefill flag (`XPU_PREFILL_INT8=1`): A/B timing, text parity, both batteries, the patch | 2026-11-18 |
| `upstream-receipt/` | Combined #2595 + #2799 receipt against upstream main, as posted on issue #2543 | 2026-11-18 |
| `llamacpp-tp-patch/` | llama.cpp SYCL `-sm tensor` peer-copy allreduce, local patch, measurements pending | 2026-11-19 |
| `scripts/` | Client scripts the run wrappers call (`gates.py`, `conc.py`, `warm_session.sh`) | |

The per-turn cache-off sessions of the bug-A runs (`tptf/raw/*/fresh/session-turns.jsonl`) keep the measured
fields only; the full session files hold the 165K and 250K prompt texts and are not published.
