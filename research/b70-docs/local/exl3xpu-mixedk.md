# exl3xpu mixed-K image — build + validation  [UNVERIFIED until reviewed]

Lane: Exl3MixedK. 2026-10-12. All GPU work on desktop-b70 GPU1 (`0000:0f:00.0`,
`ONEAPI_DEVICE_SELECTOR=level_zero:1`, `/tmp/b70-gpu1.lock`,
`gpu_guard_20261011.sh` clean before/after every arm, zero new xe events in
`dmesg` delta — the one `bcs` reset in `dmesg_before.txt` predates the window
by ~56 h).

## Image

- Tag: `b70/exl3xpu-mixedk:20261012`
- Image id/digest: `sha256:14ac6f397ef92f0eccbf19e51a3506050778d1dd295fa86cc4a128e3ce9733ac`
- Base (untouched): `ghcr.io/0xsero/exl3xpu:0.1.0` id `77785c51423f`
  (`ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8`)
- Isolated daemon `unix:///run/<exl3-docker>.sock`; build log `evid/docker_build.log`.
- Layers added: patched `csrc/exl3_ops.sycl`, patched `exl3xpu/vllm_plugin.py`,
  rebuilt `exl3xpu/_C.so` (5,849,504 B, embeds `exl3xpu_build_tag()=="mixedk_20261012"`,
  verified by `strings` + in-build `torch.ops.load_library` check: `exl3_supported(k,2)`
  true for k=2..7, false for k=8).

## Kernel patch (csrc/exl3_ops.sycl, patches/exl3_ops.sycl.patch)

mul1 (cb=2) decode supports every K the SC5 artifact uses (4,5,6,7; head 6; MTP 4),
plus K=2/3 for free. K=8 not free: `Geo<8>` has V=32/gcd(8,32)=4 < 8, rejected by the
`static_assert(V>=8)` in `exl3_esimd.h:34` — left unsupported, documented.

- `:183` `nt_of` — MR=8 column-strip width: K∈{3,5,7} need NT multiple of TP=32/G=4,
  returns 4 for those (was 2 → would have failed `GemvKernel`'s `NT%TP` static_assert).
- `:264-271` fused single-kernel path (`fused_small`): added `<2,2>,<3,2>,<5,2>,<7,2>`
  (opt-in path; g_fused default off).
- `:311` small-M dispatch: `EXL3_CASE(2,2)(3,2)(5,2)(7,2)` moved out of
  `EXL3_ALL_CODEBOOKS` (cb0/cb1 stay behind the flag).
- `:541-552` new `reconstruct_q8(K,...)` switch over K=2..7 for the int8 prefill
  (`kQ8WMax=3.453125` bound is K-independent — mul1 codebook values).
- `:561-565` `exl3_reconstruct`: K=2,3,5,7 cb=2 cases.
- `:604-608` `exl3_gemm_raw` debug path: same six cases via `EXL3_RAW_CASE`.
- `:681` int8 prefill gate `(K==4||K==6)` → `K>=2 && K<=7`; `:728,:740` launch sites
  now go through `reconstruct_q8`.
- `:780` `exl3xpu_build_tag()`; `:781-787` `exl3_supported` → `cb==2 && 2<=K<=7`.
- Geo check for K=7: g=1 → D=7, V=32, G=8, WORDS=56, TP=4 — same family as K=3/5;
  GEMV lanes (32), DPAS lane map (P=1, replicate stride-1 on planarized words), and
  `ReconstructKernel` plane reads verified element-for-element vs ref. No spills
  observed in the correctness runs; K=7 kernels JIT fine under IGC.

## Plugin patch (exl3xpu/vllm_plugin.py, patches/vllm_plugin.py.patch)

- `:118-134` `_bits_for` now returns a per-checkpoint-member bits list; the old
  all-equal assert is relaxed to "every member quantized" (mixed K allowed).
- `:152-156` `Exl3LinearMethod` stores `self.Ks` (member bits).
- `:170-205` `create_weights`: member→partition mapping (`part_Ks`; the first
  checkpoint member may span leading partitions — qwen3.5 `in_proj_qkv` covers
  `in_proj_qkvz` shards 0..2, `in_proj_z` shard 3), `exl3_runs` = contiguous
  load-member runs, `exl3_mixed` flag. For mixed modules the `trellis` param is a
  flat packed int16 buffer with per-run `[k/16, n_run/16, 16*K_run]` segments
  (`exl3_tseg`); uniform modules keep the original 3-D param and code path
  byte-for-byte.
- `:238-258` `_load_trellis_mixed`: writes each checkpoint tensor into its run's
  segment; asserts exact run coverage + tile width.
- `:299-325` `process_weights_after_loading`: builds `layer.exl3_members`
  (per-run trellis view, suh row of the run's load group, svh slice, zero shard,
  bounds, per-member `cpp` flag).
- `:328-350` `apply`: mixed → per-run `torch.ops.exl3xpu_C.linear` (or the py
  fallback per member) writing into `y2[:, col0:col1]` — slice copies are
  graph-capture safe; verified under `FULL_DECODE_ONLY` serving below.

## Correctness (tests/check_mixedk.py, evid/kernels.log) — 46/46 pass

Real SC5 tensors: `layers.1.linear_attn.in_proj_qkv` K=4,
`layers.31.self_attn.{q_proj K=5, o_proj K=6, k_proj K=7}`.

- (a) reconstruct, 8/8 `torch.equal` vs `exl3xpu.ref.reconstruct_inner` (incl. K=7).
- (b) `linear` vs fp32 ref `linear_forward`, M=1,2,4,8,16,64,512:
  rel-L2 ≤ 7.6e-4 fp16 path (tol 2e-2); int8 W8A8 path M=64 ≤ 3.6e-4, M=512 1.2e-2
  (tol 5e-2 — the int8 path is the lossy route-of-record prefill).
- (c) fused mixed qkv [5,7,7] through real plugin loaders/apply:
  rel-L2 6.96e-4 (M=1), 3.59e-4 (M=8) vs unfused ref concat.

## Serve checks (route-of-record config: MTP3, FULL_DECODE_ONLY graphs, fp8 KV,
int8 prefill, oneDNN attn, KV block exact, draft vocab; `--gpu 1`, util 0.90,
max_model_len 65536)

SC5 checkpoint 18.53 GiB → 20.09 GiB resident; all 9/9 pass (`evid/canary_sc5-mtp3.log`):

```
[PASS] batt-arith   703                      prompt_tokens=73  (22 file ids +51 template)
[PASS] batt-capital Paris                    prompt_tokens=71  (20 +51)
[PASS] batt-token   B70-FLASH-OK             prompt_tokens=73  (22 +51)
[PASS] batt-sumsq   14                       prompt_tokens=79  (28 +51)
[PASS] batt-logic   No                       prompt_tokens=82  (31 +51)
[PASS] batt-json    {"status":"ok","count":3} prompt_tokens=77 (26 +51)
[PASS] long-0       43                       prompt=8161 compl=79
[PASS] long-1       Measurement              prompt=8161 compl=54
[PASS] long-2       42                       prompt=8162 compl=72
```

Note: template overhead measured +51 on this serve surface (route of record
logged +44 on plain; same jinja file — recorded, not explained). Canaries need
gen>=512 on SC5: the model emits a `</think>` preamble; at gen=64 answers were
correct but truncated (first run showed 6/6 correct prefixes, truncation-only fails).

## Smoke speed screen (NOT a claim — p512/g128 greedy C1, 1 warmup + 3 cells,
client post-first decode tok/s / wall tok/s, evid/cells_*.jsonl)

| arm                        | decode (cells 1-3)      | wall               | notes |
|----------------------------|-------------------------|--------------------|-------|
| sc5-mtp3   (new image)     | 58.1 / 55.2 / 58.1      | 50.2 / 48.7 / 50.9 |       |
| sc5-nospec (new image)     | 27.2 / 27.2 / 27.1      | 25.6 / 25.6 / 25.5 |       |
| plain-mtp3 (new image)     | 82.1 / 83.7 / 79.6      | 68.3 / 70.2 / 67.3 |       |
| plain-nospec(new image)    | 32.6 / 32.6 / 32.5      | 30.6 / 30.6 / 30.5 |       |
| plain-mtp3 (OLD image)     | 80.1 / 84.0 / 79.6      | 67.6 / 71.2 / 67.3 |       |

New-vs-old on plain-mtp3: 81.8 vs 81.2 mean decode — no regression on the new
image for the uniform path (identical code path). Prior-campaign old-image cells
(80.1/79.9/79.6) corroborate. SC5 ≈ 0.70x decode / 0.83x nospec of plain 4.0 —
expected: 5 bpw stream + K=5/7's deeper decode unroll (V=32 vs 8/16) + planarize.

## Cleanup proof

`docker ps -a`: no mixedk-* containers. `pm_asfound`/`pm_final`: power1_cap
230000000 uW unchanged, visible_avail 32615/32656 MiB restored. Guard clean
(desktop-class PIDs only). dmesg delta: zero GPU1 events.

## Commands (exact)

```
bash scripts/tmp/exl3xpu_mixedk_20261012/docker/build.sh
bash scripts/tmp/exl3xpu_mixedk_20261012/tests/lane_campaign.sh prep
bash scripts/tmp/exl3xpu_mixedk_20261012/tests/lane_campaign.sh kernels
bash scripts/tmp/exl3xpu_mixedk_20261012/tests/lane_campaign.sh sc5-mtp3
bash scripts/tmp/exl3xpu_mixedk_20261012/tests/lane_campaign.sh \
    sc5-nospec plain-mtp3 plain-nospec old-plain-mtp3
```

Serve per arm: `docker run -d --device=/dev/dri --group-add <gid> --network=host
--memory=30g -v <model>:/mnt/<disk>:ro --entrypoint python3 <img>
/opt/exl3xpu/scripts/serve.py /opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw
--gpu 1 --model-path /mnt/<disk> --port 8022 --set vllm.gpu_memory_utilization=0.90
--set vllm.max_model_len=65536 [--set vllm.speculative_config=null]`

NOTE (2026-10-12 lead safety rule): the campaign ran with `--memory=16g` per the
earlier spec; SC5 mmaps ~20 GB of weights, so replay uses `--memory=30g` (script
updated). All runs above completed under 16g without OOM/thrash failures.

## Files

`scripts/tmp/exl3xpu_mixedk_20261012/`: `src/` (patched sources),
`patches/exl3_ops.sycl.patch`, `patches/vllm_plugin.py.patch`,
`docker/{Dockerfile,build.sh}`, `tests/{check_mixedk.py,canary.py,prep_prompts.py,
client_cells.py,lane_campaign.sh,prompts_exact512.json,prompts_decoded.json}`,
`evid/` (build log, kernels.log, canary_*.log, cells_*.jsonl, server_*.log,
pm_*.txt, guard_*.txt, image_*.json).

---

## RL-182 follow-up (2026-10-12): M=4 verify-path investigation

**Image**: `b70/exl3xpu-mixedk:20261012b` `sha256:fe10536352bb1f98c29da863816032d6ed1f99b2aa854fced2bdcf75d07d0e8c` — byte-identical code to `:20261012` except the build tag string; all optimization candidates were measured and rejected (below).

**Like-for-like queued bench** (60 warmup launches + 100 queued, one sync; XeTrellisCode methodology) — logical GB/s, GPU1:

| shape | M | K4 | K5 | K6 | K7 | XTC-exl3 arm |
|---|---:|---:|---:|---:|---:|---:|
| 5120x17408 | 4 | 238 | 226 | 360 | 311 | ~500 |
| 17408x5120 | 4 | 450 | 310 | 460 | 396 | ~450 |
| 5120x12288 | 4 | 458 | 298 | 477 | 421 | ~480 |
| 5120x17408 | 1 | 318 | 377 | 446 | 478 | ~485 |
| 17408x5120 | 1 | 464 | 446 | 474 | 512 | ~475 |
| 5120x12288 | 1 | 542 | 460 | 493 | 522 | ~465 |

**Findings**:
- The initial 0.43x K5-vs-K4 M=4 gap (min-of-5 sync bench) was a cold-DVFS artifact; warm/queued K5 is ~0.65-0.95x K4 and K6/K7 meet or beat K4 (planar reads).
- XeTrellisCode's "exl3" arm is the same `DpasKernel` benched raw (no had_in/had_out, preallocated `part`, event timing). ~500 GB/s is therefore the production kernel's own ceiling — the M=4 call carries ~50% non-gemm overhead at the large shapes (had kernels, `part` alloc, 3 launches per op).
- K!=4 tensors are ~90% of M=4 linear time (SC5 byte-weighted). Recovery estimate if every call ran at raw-kernel rate: ~66 -> ~32 ms bench linear-time (model only, not an e2e claim).
- B_FAKE (decode stubbed) reached 205-345 GB/s at M=4 — decode is ~80% of in-kernel tile cost at K5 wide shapes, but kernel decode is NOT the production bottleneck.

**Tried and rejected by interleaved warm A/B**: `build_kK` VNNI-direct gathers (bit-exact vs ref, 46/46 correctness, but 0.58-0.92x baseline); `EXL3_BV_ALL` (loses steady-state); `fused_small` shape-gated dispatch (regresses on re-measure).

**Smoke on tag b**: kernels 46/46, canaries 6/6 exact, long cells 3/3 (8161-8162), MTP3 ~57.0/49.6 decode/wall tok/s, no-spec ~27.2/25.6 — all within noise of `:20261012`. Guard clean, zero new GPU1 xe events, `--memory=30g` serve cap.

**Next lever**: amortise per-call overhead in the M=4 verify path (batched/fused execution without redundant had_in), not decode rewrites.

Evidence: `evid/{micro_queued,micro_ab,micro_ab2,micro_bfake,micro_fused_a,micro_fused2_a,cells_sc5b-mtp3,cells_sc5b-nospec,canary_sc5b-mtp3,kernels}.log`, `tests/{micro_queued,micro_dbg,micro_ab,micro_ab2,micro_fused,micro_fused2}.{py,sh}`, `debug/_C_{bfake,bvall,kk}.so` under `scripts/tmp/exl3xpu_mixedk_20261012/`.
