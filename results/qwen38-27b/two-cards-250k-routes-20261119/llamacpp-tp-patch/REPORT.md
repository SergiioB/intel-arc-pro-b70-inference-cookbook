# llama.cpp SYCL tensor split on two Arc Pro B70s: peer-copy allreduce

Status: in progress (draft written 2026-11-19 00:20 CET; measurements queued as jobs 200-215).

## 1. Result

Pending measurements.

How upstream master reduces today (llama.cpp `78e05298f`, 2026-11-19):

1. The meta backend (`ggml/src/ggml-backend-meta.cpp`) runs each subgraph on both devices, then calls the
   backend's `comm_allreduce` on the last node of the subgraph (`:2532-2561`, call at `:2551`). It looked the
   SYCL entry points up once at `:1921-1932`. If the SYCL call returns false it runs its generic butterfly
   (`:2414-2526`: `ggml_backend_tensor_copy_async` to a temp buffer, then a one-node ADD graph per device).
2. SYCL `ggml_backend_sycl_comm_allreduce_tensor` (`ggml/src/ggml-sycl/ggml-sycl.cpp:8024-8166`) handles
   N=2, F32/F16, contiguous. F32 with fewer than 32,768 elements (`:8103-8124`, every decode-time
   reduction) exchanges full F32. F32 with 32,768 elements or more (`:8126-8165`, prefill) truncates each
   partial to BF16 (`:8136`, `:8140`), exchanges it and adds it to the local F32 partial (`:8150`, `:8156`).
   Rank 0 computes `p0 + bf16(p1)` and rank 1 `p1 + bf16(p0)`, so the two ranks can hold different sums
   (RL-250 fix 5).
3. The exchange always goes through pinned host memory. With one shared SYCL context
   (`SYCL_ENABLE_DEFAULT_CONTEXTS=1`, Linux, `:7993-7996`) `ggml_sycl_comm_exchange_fast` (`:7910-7942`)
   copies both partials device-to-host, interlocks the queues with barriers, copies host-to-device, then
   blocks the CPU on both uploads (`:7940-7941`). With separate contexts `ggml_sycl_comm_exchange`
   (`:7945-7968`) blocks the CPU on both downloads (`:7962-7963`) and copies between host buffers on the
   CPU (`:7964-7965`).
4. So every reduction (two per layer) stalls the CPU until both GPUs have drained their queues and the data
   has crossed PCIe twice. The GPUs idle while the CPU then submits the next layer.

## 2. What you need

- Arc Pro B70 card 0 at `0b:00.0` (180 W cap, 2,400 MHz clock cap from the operator's cooler profile) and
  card 1 at `0f:00.0` (230 W). PCIe 4.0 x8 each, on CPU lanes. One-card cells use card 1.
- llama.cpp `78e05298f0d68abff41d08d79fa7469264b2aff0` (upstream master). Patch branch
  `sycl/tp-peer-copy-f32` = one commit `807f847` on that base. Measured build = `6b611b8`
  (that commit plus a measurement-only commit adding `GGML_SYCL_TP_DIAG_SKIP`, unset in every arm except
  `diag`).
- oneAPI DPC++ 2026.1, Level Zero v1 adapter (`UR_LOADER_USE_LEVEL_ZERO_V2=0`), compute runtime 26.05.
- Qwen3.8-27B Q4_K_M, `$DATA/sata2/gguf/qwen38-27b-Q4_K_M.gguf`, 16,810,714,432 bytes, sha256
  `2554eb0d...fbece`. WikiText-2 test text `$DATA/sata1/archive/quant-showdown-20260913/wiki.test.raw`.
- Scratch, clone, worktree and raw evidence: `$WORK/tpllama/`.

## 3. Timeline

| Local time (CET) | Event |
|---|---|
| Nov 18 23:43 | Previous `tpllama` agent cloned upstream, queued builds and runs 200-213. |
| Nov 18 23:49 | Previous agent drafted the F32 peer-copy patch (not built). Quota ran out. |
| Nov 19 00:08 | Takeover. Read briefs, job files and scripts. |
| Nov 19 00:10-00:20 | Moved the patch onto a branch; fixed peer-access enable for several contexts per process; switched K/V to q8_0 and context to 40,960 per task; added cooldown, deadline, `$CARD`/`$PORT`, a skip-allreduce diagnostic arm, a per-call latency and leak check, perplexity job; re-queued 204-215. |

## 4. Step by step

Pending.

## 5. Fix table

| # | Symptom | Cause | Fix | How verified |
|---|---|---|---|---|
| 1 | Inherited patch would call `ext_oneapi_enable_peer_access` again for each new context (llama-bench makes one per test); an exception there makes `comm_init` return null and the meta backend silently falls back to its generic butterfly. | Peer access is per device pair and per process. | Enable once per device pair (static table); a failure logs a warning and keeps host staging. | Source review; exact-check job runs one context, bench runs several. |
| 2 | Inherited run scripts used F16 K/V and 34,816 context, citing "tensor split rejects quantized K/V". | No such check found in `src/llama-context.cpp` or the meta backend; RL-250 ran q4_1 K/V under `-sm tensor`. | Use the task's q8_0/q8_0 and `-c 40960`. | Pending: base-dual job loads it. |
| 3 | Inherited scripts used `-lm dio`: every load reads 16.8 GB from SATA. | Direct I/O skips the page cache. | `-lm none` (host standard). | Load time in server logs. |

## 6. Verification

Pending.

## 7. Results and claim boundary

Pending.

## 8. If you do this again

Pending.

## 9. Adapting it

Pending.
