# Evidence: Qwen3.8-27B recipes at equal caps (2026-11-19)

Raw text evidence behind [RECIPES-B70.md](../../../docs/qwen38-27b/RECIPES-B70.md) and the 2026-11-19 update of
[TENSORFOLD-TWO-CARDS-20261119.md](../../../docs/qwen38-27b/TENSORFOLD-TWO-CARDS-20261119.md).
`python3 summarize.py` prints every table from the files here (stdlib only).

Same rules as [`two-cards-250k-routes-20261119/`](../two-cards-250k-routes-20261119/README.md): private paths rewritten
(`$TF` TensorFold work tree, `$WORK` campaign scratch, `$MODELS`, `$DATA`, `~`), logs published as `*.log.txt`
because the repository ignores `*.log`, links in copied lab documents to unpublished files turned into text. The
intelinside response has the account ids removed.

| Folder | What | Caps (card 0 / card 1) |
|---|---|---|
| `rebaseline-230w/tp2/` | Served two-card cells, binary `2286abc0c`, 2026-11-19 09:45 CET | 230/230 W, 2,800/2,800 MHz |
| `tp2-prefix-180-230/tp2/` | The same cells from 2026-11-18 (published in PR #90), the "before" column | 180/230 W, 2,400/2,800 MHz |
| `tpdiag/prof/` | Dev-runner logs: `eq230-*` equal-cap budget (job 2195-c), `n40-*` runtime 26.05 against 26.40 (job 2195), `prof2-*` the 180/230 W budget; `.telemetry` files hold caps and temperatures per run | per run |
| `tpdiag/neo-repro-runs/` | Profiler-hang reproducer runs on 26.05.37020.3 and 26.40.40196.8 | – |
| `tptf-speed/` | Lever screens `l1` (`XPU_GU_WIDE64`, loss) and `l2` (`XPU_DATTN_SPLIT`, gain), 3 interleaved pairs each, `summary.json` from `speed_sum.py`; `l2-tp1` holds the one-card run's first arm only; the L2a commit as a patch; the L1 patch | 230/230 W |
| `build/ocloc-26.05-build.diff` | One-line build change for ocloc 26.05 (Gemma 4 kernels only); applies on `ac346900b` | – |
| `llamacpp-tp/` | llama.cpp `-sm tensor`: patches and APPLY, llama-bench and served cells (`base-single` one card, `probem-off` stock, `probem-on2` mode 2, `probem-diag` skip-allreduce bound), allreduce checks (`check-m2/`), crash stacks for `SYCL_CACHE_PERSISTENT` (`diag/`), the queue log with the mode-1 segfaults | 230/230 W |
| `intelinside/` | Response of the intelinside.ai submission: custom runtime 29, results 364–377 | 180/230 W |
| `routecmp-deep/` | Several streams on one card, three passes a level (also in `two-cards-250k-routes-20261119/routecmp/deep/`) | card 1 230 W |
