---
title: "Second GPU as single-stream EXL3 breakthrough — pinned-source TP feasibility audit"
type: research
date: 2026-09-29
status: provisional
author: "b70-upstream-researcher"
scope: "Static source + ledger research only: no GPU, no server, no build, no pull, no commit. Read-only extraction from the pinned exl3xpu image via docker create/export/cp on unix:///run/<exl3-docker>.sock (containers never started, created objects removed; RL-091 precedent). No shared ledger/status edits."
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
---

# Second GPU as single-stream EXL3 breakthrough — upstream feasibility (2026-09-29)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.**

Question (operator): can GPU 1 become a single-stream speed lever for the
incumbent EXL3 route — i.e. TP2 across both B70s for `exl3xpu` — and what do
the *pinned* sources actually support? Incumbent: turboderp EXL3 4bpw @
`113cf7ab`, engine image `ghcr.io/0xsero/exl3xpu@sha256:21412bdd…c870fa8`
(local tag `exl3xpu:pinned-21412bdd`, image ID `77785c51423f`), native MTP3,
fp8 KV, 63.25 tok/s champion (probe 1, RL-090).

## 1. Pinned identity (verified today)

| Object | Identity | Evidence |
|---|---|---|
| Engine image | `sha256:21412bdd7535…c870fa8`, in-image vLLM `0.26.1.dev0+g568afb3a1` (llm-scaler), torch `2.12.0+xpu` | `engines/exl3xpu/README.md:23-24`; `results/2026/09/b70-probe1-exl3-rate-20260929/RESULTS.md:51`; `/tmp/pinned-extract-20260929/opt/venv/.../torch/version.py` |
| Plugin source == upstream tip | in-image `/opt/exl3xpu/exl3xpu/*.py` and `csrc/exl3_ops.sycl` byte-identical to `github.com/0xSero/exl3xpu @ 15ded2f` (2026-09-26); only in-image difference is built `_C.so` | `diff -r` clean; clone at `/tmp/b70-k3-exl3-source-20260929` (git shallow @ `15ded2f3add…`) |
| Artifact | `Qwen3_5ForConditionalGeneration`, 64 layers (48 GDN + 16 full-attn, interval 4), hidden 5120, heads 24/4 KV (head_dim 256), GDN heads 16 key × 48 value × 128, MLP 17408, vocab 248320, MTP = 1 full-attn layer; EXL3 4.0 bpw body / head_bits 6 / mtp_bits 4, codebook mul1, calib rows 250 cols 2048 | `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926/config.json`, `quantization_config.json` |
| Runtime transport | `dist_backend: str = "xccl"  # xccl only` — in-image `platforms/xpu.py:115` | extraction; also `results/2026/09/b70-probe1-exl3-rate-20260929/RESULTS.md:90` |

Extraction artifacts for this audit: `/tmp/pinned-extract-20260929/`
(`opt/venv/.../site-packages/vllm` runtime files + `/opt/exl3xpu` plugin) and
the clone `/tmp/b70-k3-exl3-source-20260929`. Line numbers below cite these
pinned copies.

## 2. Headline verdict

**Stock TP2 on the pinned exl3xpu route is impossible today — not slow,
impossible.** The plugin's quant method hard-raises on any sharded partition
at model load:

```python
# vllm_plugin.py:157-162 (Exl3LinearMethod.create_weights)
k = input_size_per_partition
n = sum(output_partition_sizes)
if k != input_size or n != (output_size if not isinstance(layer, ParallelLMHead) else n):
    raise NotImplementedError("exl3: tensor parallel sharding not supported yet (use DP)")
```

Under `--tensor-parallel-size 2`: every `ColumnParallelLinear` (qkv, gate_up)
arrives with `n = N/2` → raise; every `RowParallelLinear` (out_proj, o_proj,
down) arrives with `k = K/2` → raise. The author's own intended multi-GPU
mode is DP (replication), per the error text. The escape hatch is deliberate
for `ParallelLMHead` (the `n != n` no-op) — the vocab-parallel head path is
shard-tolerant at creation, a design hint for the build program (or just
replicate the 6-bpw head, ~0.95 GB).

**But the EXL3 format itself does not forbid TP.** The trellis math is exact
under 128-block slicing in both dimensions (§4). The blocker is loader +
kernel plumbing, not the representation.

**And even with loaders, the transport that killed RL-075 is still the only
in-image transport** (`xccl` pinned, `platforms/xpu.py:115`). The premise
change since RL-075 is RL-059/RL-071 (copy-based P2P verified correct,
11–12 GB/s, no atomics) — i.e. the reopening condition RL-075 itself named
("a topology fix, not a rerun") is now *partially* satisfied by a custom
copy-only/flag-slot collective path that does not exist yet in any tree
(RL-091 Audit C: AMBER, greenfield).

## 3. Supported / unsupported / unknown — per component, pinned file:line

| Component | Status | Pinned evidence |
|---|---|---|
| EXL3 TP weight sharding (plugin) | **UNSUPPORTED — hard `NotImplementedError` at load** | `vllm_plugin.py:157-162` |
| EXL3 loader shard bookkeeping | Present but single-host only: contiguous-group `_range` assert, full-coverage assert | `vllm_plugin.py:184-198, 223-235` (`process_weights_after_loading`) |
| EXL3 kernel single-device assumption | `apply` calls `exl3xpu_C.linear(x, trellis, suh, svh, shard_of_nb, …)` — one device's tensors; no cross-rank op | `vllm_plugin.py:250-257`; `ops.py:130-137` |
| Trellis format under N-slice (column TP) | **SUPPORTED mathematically** — output Hadamard is per-128-block; `svh` per-element; slicing at 128 multiples exact | `ref.py:14-16,126-130,145-153`; plugin already asserts dims `% 128 == 0` (`vllm_plugin.py:163-164`) |
| Trellis format under K-slice (row TP) | **SUPPORTED mathematically** — input Hadamard per-128-block; partial sums add linearly (allreduce-compatible, fp-reassociation only) | `ref.py:14-16,126-130` |
| GDN (qwen3_5) mixer TP | **SUPPORTED in in-image vLLM** — `per_rank_hidden_size`, divisibility asserts, `redundant_tp` fallback for `n_groups % tp != 0` | in-image `mamba_mixer2.py:77-93, 106-129, 282-292`; `qwen3_5.py:547-560` (state-shape fn takes `tp_size`) |
| Shapes for this checkpoint at TP2 | ALL divisible — attn 24/4 heads, GDN 48 v-heads, hidden 5120, MLP 17408 | `config.json` values (§1) |
| MTP draft under TP2 | **NO GDN STATE IN DRAFT** (draft layer is `full_attention`) → draft can be replicated per card, zero draft collectives | in-image `qwen3_5_mtp.py:109`; RL-091 same finding at `qwen3_5_mtp.py:115-122` (0.27.2rc1 tree) |
| MTP draft head path | Custom exl3-linear draft logits (`_build_draft_head`, `compute_logits`) — needs head replication or the same loader fix | `vllm_plugin.py:274-319` |
| KV under TP2 | Splits cleanly (KV heads 4→2/rank; GDN state 48→24 v-heads/rank); capacity doubles, not needed (262,144 already native) | `config.json`; in-image `mamba_mixer2.py:282-292` |
| Collective transport (stock) | **UNSUPPORTED as a speed path** — `xccl` only; RL-075: default SK path UR40→UR20 device-lost, `SK=0` segfaults in `ze_a2a_reduce_scatter_entry`, only host-staged serves (−71.4 % p512) | in-image `platforms/xpu.py:115, 428-431`; RL-075 RESULTS.md §"Why TP2 lost" |
| Copy-based collectives | **CORRECT, measured** (exact int64/f32, both transports; 101.7 µs @4 KB, 129.2 µs @512 KB, p2p arm ≈ host-staged) — but per-op latency is the RL-075 killer class | RL-071, `results/2026/09/b70-improvement-probes-20260920/o14-collective-probe-20260925T220307Z/` |
| Flag-slot copy-only sync (B11) | **UNKNOWN — greenfield**; no spin/poll kernel in any fleet tree; graph capture not hard-blocked but reader-side visibility of peer writes unanswered; 60 µs kill threshold | RL-091 Audit C (`research/2026/b70-engine-capability-audit-20260929.md` §Audit C) |
| XPU graph capture of collectives | **UNKNOWN** — oneCCL 2022.1 headers acknowledge graph recording (`ccl.h:216-218`); torch enqueue path not statically auditable; E4 blocked by topology per RL-075 | RL-091 C1; RL-075 verdict on E4 |
| Capture-awareness of the fork's custom paths | POSITIVE SIGNAL — fp8-KV prefill path self-disables under `is_current_stream_capturing()` and under `dcp_world_size != 1` (TP ≠ DCP, so unaffected) | `vllm_patches.py:101-106, 133` |
| Kernel efficiency at half-N (n=2560/rank) | UNKNOWN — unmeasured; tiles are 16-wide → 160 n-tiles/rank, plausible; M-axis geometry unchanged | modeled only |

## 4. EXL3 packed sharding / trellis Hadamard constraints (the exact math)

From the bit-exact reference shipped in the pinned tree (`ref.py`, "Bit-exact
pure-PyTorch reference for EXL3 (exllamav3 v1.x) linear layers"):

```text
W_orig = diag(suh) . H128_blocks(rows) . W_inner . H128_blocks(cols) . diag(svh)
y      = had128(x * suh) @ W_inner  ->  had128(.) * svh          # ref.py:14-16
```

- `trellis: int16 [k/16, n/16, 16*K]` — one independent 16×16 tile per
  (i, j); `suh: fp16 [k]` (checkpoint; per-group stacked in the plugin buffer
  `(num_groups, k)`, `vllm_plugin.py:172`), `svh: fp16 [n]`.
- **Both Hadamards are block-diagonal 128×128** (`had_rows`, `ref.py:126-130`)
  — they never mix across 128-block boundaries in either dimension.
- Therefore: slicing N or K at multiples of 128 (already enforced by
  `vllm_plugin.py:163-164`) leaves every tile and every H-block intact.
  Column-sliced output = exact same arithmetic; row-sliced K = partial sums
  that add exactly (up to fp reassociation) — precisely the allreduce contract.
- The plugin's own `shard_of_nb` (output-128-block → group map,
  `vllm_plugin.py:231-235`) is a per-device host-side index — trivially
  rebuildable per rank.
- Checkpoint suh is per-group over full k — a rank holding a slice of a group
  (e.g. q 3072 of 6144) needs the group's suh row replicated (k fp16 values —
  negligible).

**Conclusion: no requantization is needed for EXL3 TP2.** The "trellis
Hadamard constraint" is an alignment constraint (multiples of 128), not a
coupling constraint. This is verifiable on CPU today (§7).

## 5. RL-075 — exact failure, and what actually changes with the new route

Prior TP2 attempt (RL-075, 2026-09-26, on the *GPTQ+MTP4 vLLM-XPU* champion,
**not** exl3xpu):

- Boot crashes: default oneCCL (`SK=1`) → UR 40 OUT_OF_RESOURCES → UR 20
  DEVICE_LOST; `SK=0` → segfault in `ze_a2a_reduce_scatter_entry` (peer-copy
  path) — both **before any measurement**; narrow per-worker
  `ZE_AFFINITY_MASK` additionally caused oneCCL uuid mismatch.
- The only serving configuration was host-staged `SK=0` + 4 GiB simple
  thresholds → per-step host dispatch for every allreduce:
  **B vs A = −71.4 % (p512), −45.5 % (p8192)**; E3 allgather-overlay could
  not rescue (−68/−70 %); E4 graph-recorded collectives BLOCKED by topology.
- Ledger reopen condition: *"Reopening requires a topology fix (remote
  atomics / peer IPC), not a rerun"* — enforced by the blocked pattern
  (`TP2.*peer.*(atomics|remote).*|reopen.*TP2.*rerun(?!.*topology)`).

Arithmetic consistency with RL-071: this checkpoint needs ~2 row-parallel
reduces per layer × 64 layers ≈ **128 allreduces of 5120-elem (10,240 B bf16)
hidden states per target forward** (+~0 if the MTP draft is replicated,
§3). At RL-071's measured host-staged ~102–105 µs/op → **≈ 13 ms/round of
pure collective latency** — enough to erase the halved-compute win; at a
flag-slot 5–15 µs/op → 0.6–1.9 ms/round — affordable. This is exactly the
B11 re-pricing in `research/2026/b70-breakthrough-architect-20260928.md`
("latency ~5–15 µs ⇒ 5–10× cheaper small collectives", with its own caveat
that "TP2 also re-opens the graphs-off penalty").

**What genuinely changes the premise now (and what does not):**

1. *Does not change:* stock oneCCL TP2 — the exl3xpu image pins the same
   transport class (`platforms/xpu.py:115`), so a stock launch would hit the
   same ze_a2a/UR failures — except it cannot even reach them: the plugin
   raises `NotImplementedError` at load (§2). A plain rerun is impossible,
   consistent with the ledger's reopen gate.
2. *Changed since RL-075:* RL-059 (2026-09-24, one day before the campaign)
   verified direct-peer ACCESS-flag copies (11.1–11.9 GB/s, checksum-clean,
   **ATOMICS absent**) and RL-071 (2026-09-25) proved copy-based 2-rank
   collectives **exact**. RL-075's arms still rode oneCCL's own transport
   selection; the newly-available option is a custom copy-only collective
   (flag-slot) that bypasses oneCCL's a2a/atomics assumptions entirely —
   this is the "topology fix" the ledger demanded, at the transport level,
   *if* the B11 unknowns resolve.
3. *Changed by the route itself:* the EXL3 decode round is byte-dominated
   and power-limited per card — 44.10 ms/round, 3.13 tok/round, one-read
   budget ≈ 16.86 GB/round, effective 348–383 GB/s = 57–63 % of the 608 GB/s
   pin floor, ccs0 ~88–100 % busy (RL-090 probe 1). Halving per-card weight
   bytes under TP2 doubles the *aggregate* power+bandwidth budget (two cards
   at their own 230 W caps) — the lever class RL-075's GPTQ route also had,
   but EXL3's round is closer to pure weight-read (44–48 ms of byte time in a
   44.1 ms round), so the ceiling is cleaner. **Arithmetic-only ceiling:
   44.1 ms → ~22–24 ms + 0.6–2 ms collectives ⇒ ~1.7–1.9× ⇒ ~107–120 tok/s
   class** — contingent on kernel efficiency at half-N (unknown), capture
   safety of the new collectives (unknown), and graphs (§3). No measurement
   claim is made here.

## 6. llama.cpp fallback — assessed accurately

- **It is not a fallback for the EXL3 route.** llama.cpp reads GGUF only;
  the incumbent EXL3 artifact (turboderp `113cf7ab`) cannot be served at all.
  A llama.cpp dual-GPU attempt is a *different quant + different engine*
  campaign: quality gates (6/6 battery, 8/10 tasks, 262,144 fp8-KV) re-open
  from zero.
- **`-sm layer` (the default) is not simultaneous-bandwidth TP.** Layers are
  placed per device and executed serially (documented on this host: GPU0 →
  host-staged copy → GPU1, AGENTS.md §16). Every layer's weights are still
  read from one card per token; per-token weight traffic is not halved in
  time. Measured same-class result: Cascadia PP2 on this topology = −0.26 %
  (RL-051). Capacity lever only.
- **`-sm row` must not be assumed.** Recorded host evidence: `-sm row`
  segfaults (AGENTS.md §18 anti-pattern, K216/MoE path); never validated for
  dense qwen35 on this host. Status: unknown/untested — and row-split is the
  only upstream mode that would give bandwidth TP.
- **`-sm tensor` (this fork's meta device) is GDN-aware and arch-enabled but
  unmeasured.** `llm_arch_supports_sm_tensor` (`src/llama-arch.cpp:1012`,
  pinned tree @ `1692f9e50`) returns true by default and QWEN35 is not in the
  false list (27 archs); the meta backend splits matmuls along axis 0/1 with
  partial-result reduce (`ggml/src/ggml-backend-meta.cpp:577-597` handle_mul_mat,
  PARTIAL-axis sync) and has explicit `GGML_OP_SSM_CONV`/`GGML_OP_SSM_SCAN`
  handlers (`:758, 972-975`); hybrid/rollback support for QWEN35 is explicit
  (`llama_arch_is_hybrid:978-979`, `supports_rs_rollback:1001-1002`). This is
  genuine TP-class splitting — but zero decode measurements exist on this
  host for any dense model, cross-device partial-reduce transport is
  unstated (host-staged copy would land in RL-071's 100 µs/op class), and it
  still serves GGUF, not EXL3. Treat as a separate, lower-priority research
  cell, not a fallback.

## 7. Cheapest deciding no-GPU test (T1) — trellis slice exactness on CPU

Decides the *format* question (§4) empirically against the real artifact, on
this host (torch `2.13.0+cpu` present), no GPU/container/build:

1. Load one EXL3 tensor quad from
   `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926` (e.g.
   `model.language_model.layers.0.mlp.gate_up_proj.{trellis,suh,svh,mul1}`
   from the safetensors shards; names per `model.safetensors.index.json`,
   2426 tensors total; GDN alternative `layers.0.linear_attn.in_proj_qkv.*`).
2. Import `ref.py` from `/tmp/b70-k3-exl3-source-20260929/exl3xpu/ref.py`
   (self-contained: numpy + torch only; `codebook_of(mcg=False, mul1=True)`
   → `CB_MUL1`).
3. Random `x [8, k]`, fp32 path:
   - **Column test:** `y_full = linear_forward(x, tre, suh, svh, K, cb)` vs
     `linear_forward(x, tre[:, :n//2//16], suh, svh[:n//2], …)` → require
     bitwise-equal columns (same accumulation order per output element).
   - **Row test:** `y_full` vs `linear_forward(x[:, :k//2], tre[:k//2//16],
     suh[:k//2], svh, …) + linear_forward(x[:, k//2:], …)` → allclose at fp32
     tolerance (reassociated summation only).
4. Run at 128-block and one non-128 boundary (the latter must FAIL the
   plugin's own `% 128` assert — negative control).

Cost: minutes, CPU-only. **PASS →** the build program (§8) is
representation-safe; **FAIL →** the route needs per-shard requantization and
stops being a "breakthrough" candidate (it becomes the closed W4A16-class
repackaging story). Note T1 tests the format, not the ESIMD kernel; kernel
half-N efficiency remains a GPU-window question.

Secondary no-GPU check (T0, already satisfied by source): the
`NotImplementedError` guard is unconditional (`vllm_plugin.py:157-162`) —
no repro needed; the code path admits no config escape.

## 8. Build program shape (draft only — nothing to submit, nothing to post)

Two prerequisites, both required before any GPU campaign:

1. **EXL3 sharded loading + per-rank kernel calls** (plugin, `vllm_plugin.py`):
   relax the :157-162 guard per partition type; per-rank trellis/suh/svh
   slice loading via the existing `_range`/`exl3_groups` machinery
   (:184-235); rebuild `shard_of_nb` per rank; kernels unchanged (they
   already operate on one device's tensors, `apply` :250-257). Replicate the
   MTP draft layer + draft head (full-attn only, §3) and the 6-bpw LM head
   (~0.95 GB) to keep the round at ~128 collectives with **zero** vocab
   allgathers.
2. **Capture-safe copy-only collectives** (new op, outside the captured graph
   initially per RL-091 guidance): flag-slot sync on the verified P2P ACCESS
   path; the two named unknowns are reader-side visibility of peer writes and
   xccl/flag-slot behavior under XPU graph capture; 60 µs/op kill threshold
   stands (B11). If collectives cannot be captured, §20 rule 1 caps the
   result at diagnostic-only until the graphs-off penalty is measured.

Expected reviewer/owner objections to pre-register: (a) "RL-075 already
killed TP2" — answered by §5 (transport-level fix + load-level blocker; not a
rerun); (b) "kernel efficiency at half-N" — unmeasured, T1 cannot answer it;
(c) "graphs-off eager penalty" — unknown on this fork, must be measured in
the same window; (d) "why not B10 draft-offload instead" — B10 (draft stack
on GPU1 with pre-drafting, architect doc) is strictly smaller, has no
collective prerequisite, and should be sequenced first if only one GPU-window
program is funded; TP2 and B10 compose (TP2 halves the verify byte time B10
parallelizes against).

## 9. Owners and handoffs (no agents spawned; for the orchestrator)

| Piece | Owner |
|---|---|
| T1 CPU slice-exactness test | b70-upstream-researcher (this note's follow-up) or b70-correctness-engineer |
| EXL3 sharded loaders + per-rank apply (plugin) | **b70-kernel-engineer** (decode-side plugin/SYCL work) |
| Flag-slot copy-only collective op + visibility microbench | b70-kernel-engineer, with **b70-isa-analyst** for the peer-write memory-model question (L0/IGC semantics) |
| Graph-capture integration / pin re-validation of exl3xpu | engine specialist (exl3xpu dossier owner) |
| TP2 matched-control campaign + energy telemetry | b70-bench-engineer (ADR-0008 gates unchanged; RL-071 band discipline) |
| Quality/bit-parity gates (row-parallel reassociation is not byte-identical) | b70-correctness-engineer |

## 10. Ledger pointers (read-only; no edits made)

RL-059 (P2P ACCESS verified, no atomics), RL-071 (copy-based collectives
exact + latency ladder), RL-075 (TP2 screen FAIL, host-staged only, reopen
gate), RL-090/probe 1 (63.25 tok/s champion, 44.10 ms round attribution),
RL-091 (engine capability audit: EP/GDN/flag-slot). This note adds the
missing piece those rows lack: the pinned exl3xpu plugin's explicit
anti-TP loader guard and the trellis-format slice-exactness argument.

## 11. Failures, unverified items, next deciding experiment

- Verified statically today: plugin TP guard, trellis math, in-image GDN/MTP
  TP wiring, xccl pin, llama.cpp arch/meta facts, image↔source identity.
- NOT verified (no GPU used, per task bounds): kernel efficiency at half-N;
  flag-slot visibility; capture of custom collectives; any speed number —
  the 1.7–1.9× ceiling in §5 is arithmetic-only and labeled as such.
- Extraction hygiene: all `docker create` objects removed; nothing started;
  isolated daemon's K3 images (built by another session ~30–60 min prior)
  were not touched.
- Next deciding experiment: **T1 (§7)** — CPU, minutes, decides
  representation safety; then the B11 visibility microbench (GPU window,
  kernel-engineer-owned) decides the transport; only both passing justifies
  opening a TP2 build/campaign window under ADR-0008.
