# XTQ trellis quantizer — implementation and validation

Lane: TrellisQuantizer. Deliverable for the B70/Qwen3.8-27B XTQ sweep.
Date: 2026-10-12. All heavy work ran on `desktop-b70` inside the pinned container
(`vllm/vllm-openai-xpu:v0.30.0@sha256:fc0e...6b8d`, 12 GiB cap, GPU1 via
`ONEAPI_DEVICE_SELECTOR=level_zero:1`, `/tmp/b70-gpu1.lock`, `gpu_guard_20261011.sh`
before/after — clean).

## Code

- `scripts/tmp/xtq_quant_20261011.py` — quantizer (local mirror ~/xtq-work).
  API: `quantize_linear(W[out,in], H_x, code, K, L, V, use_ldlq, ...)`
  returns `{W_hat, trellis int16 [k/16,n/16,16K], suh, svh, Wscale, ...}`;
  harness entry `quantize(W_transformed, U_upper, code, K, seed)` per the
  agreed signature; U_upper is GPTQ-convention (U^T U = G^-1), converted
  internally to the block-LDL factor (see backend section below).
- `scripts/tmp/test_xtq_quant_20261011.py` — 12-test CPU suite.
- `scripts/tmp/xtq_real_run_20261011.py` — real-tensor + EXL3 comparison driver.
- `scripts/tmp/xtq_speed_20261012.py` — GPU1 speed ladder.
- Codebooks: `xtq_codes_20261011.py` (XeTrellisCode) is integrated —
  `get_lut()` prefers its `CODES` registry; local fallbacks for
  `exl3_mul1`/`qtip_1mad` are **bit-identical** to the module (verified
  `torch.equal` over all 2^16 states). Generic over `CODES` entries incl. V=2.

## Algorithm

Two-sided block-128 random sign-Hadamard (seed 20261011) → per-tensor scale
(Wscale = RMS(W_inner)/RMS(lut)) → 16x16 tiles, trellis L=16, K∈{3,4,5},
V∈{1,2} per the code's own metadata. Targets enter the sequence in
`ref.tensor_core_perm()` order (authoritative; QTIP's `_PERMUTE` differs —
noted, not used). Tail-biting Viterbi: pass 1 on a half-rolled sequence picks
wrap bits; pass 2 constrains s_0>>KV == s_last mod 2^(L-KV) exactly.
`margin>0` truncates pass 1 to pivot+margin steps (survivor-coalesced;
margin16 keeps quality, ~25% faster). Block LDLQ (b=16, QTIP convention,
damp=1% of mean diag) sweeps k-tiles right-to-left with buffered `prod`
updates — math matches QTIP's `ldlq` incl. the prod/buffer optimization.
Packing is the EXL3 bitstream (MSB-first circular windows, int16 lo/hi word
pairs); `pack_states`/`unpack_states`/`decode_trellis` are self-contained.

## Tests — 16/16 pass (CPU)

```
[PASS] perm==ref.tensor_core_perm
[PASS] pack/decode_states K=3,4,5   [PASS] unpack==states K=3,4,5   (ref.py-exact)
[PASS] viterbi==bruteforce          (exact DP check, 3 seqs)
[PASS] tailbite consistent          (s_0>>K == s_last mod 2^(L-K))
[PASS] quantize api K=2/K=4         (harness entry: shape, decode, seed determinism)
[PASS] gauss_mse exl3_mul1 K=2 0.06833 vs D(2)=0.0625 ratio 1.093 (2048 tiles)
[PASS] gauss_mse exl3_mul1 K=4 0.00458 vs D(4)=0.00391 ratio 1.173
[PASS] gauss_mse qtip_1mad K=2 0.06888 ratio 1.102 / K=4 0.00474 ratio 1.215
[PASS] ldlq<=viterbi Hessian metric  0.00909 vs 0.01391
```
Gaussian ratio at K=4 (1.173) is better than the deployed artifact's real-tensor
ratio (1.51x D(4)); K=2 (1.093) matches the published ~1.10.

## Real tensors — ours vs EXL3 (L2, original-basis HarnessFmt Hessians)

rel weight MSE / `tr(dW H dW^T)/tr(W H W^T)`:

| tensor | EXL3 | viterbi (no LDLQ) | LDLQ |
|---|---|---|---|
| in_proj_qkv mse | 0.00589 | **0.00466** | 0.00733 |
| in_proj_qkv herr | 0.00080 | 0.00595 | **0.00036** |
| mlp.down_proj mse | 0.00583 | **0.00463** | 0.01038 |
| mlp.down_proj herr | 0.00128 | 0.00203 | **0.00025** |

EXL3 MSE baselines reproduce FormatTheory's numbers exactly (5.890e-3 / 5.833e-3).
**Acceptance target met**: our K=4 LDLQ Hessian-weighted error beats EXL3 on both
tensors (2.2x and 5.1x lower). Plain Viterbi also beats EXL3 weight-MSE by ~21%.
The LDLQ/Viterbi pair brackets the EXL3 operating point.

## Speed (K=4, exl3_mul1, fp32 DP unless noted)

| path | s/1M weights |
|---|---|
| CPU (eager, B=512) | 657.8 |
| GPU1 eager V1 (pre-GEMM emit) | 8.9-9.7 |
| GPU1 GEMM-emit margin16 | 4.2-4.4 |
| GPU1 compiled margin16 | **1.91** (B=2048) |
| GPU1 real-tensor sweep, compiled+margin16 (incl. transform+compile) | 2.6 |
| GPU1 V=2 (xe_bs2) margin16 eager | 2.0-2.1 |

Negative results: bf16 DP cost → 26x MSE regression (rejected); compiled V=2
currently slower than eager (inductor lowering); B=32768 OOMs `from_state`.
`margin16` and `compiled` preserve quality (Gaussian 0.00471/0.00463 vs 0.00467;
real-tensor 0.00471 vs 0.00466) and tail-bite consistency holds exactly.

## Full-model projection (64 layers: 48 GDN + 16 ATT; 23.82e9 quantizable
linear weights = 93.06M tiles; Hessian factorization ~40-60s/layer CPU, overlappable)

- Best measured (compiled V1 @1.91-2.6 s/1M): **12.6-17.2 h** — misses the
  8h target by ~1.6-2.1x. torch is at ~750 GB/s effective streaming (near pin);
  the DP's ~1 MB/seq-step traffic is intrinsic.
- To meet 8h needs one of: fused SYCL Viterbi (est. ~1 s/1M → ~6.6h), or
  beam-pruned encode (traffic ~P·NC; quality unverified), or V=2 codes once
  their compiled path lands.

## Exact commands

```
# tests (CPU):  python3 test_xtq_quant_20261011.py            # -> 12/12
# real-tensor + speed (GPU1 container, flock /tmp/b70-gpu1.lock):
docker run --rm --network none --memory=12g --memory-swap=12g --cpus=6 \
  --device=/dev/dri --group-add 990 -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -v /repo -v /bf16 -v /exl3 -v /hdir -v /refsrc -v /out $IMG \
  /repo/scripts/tmp/xtq_real_run_20261011.py        # JSON + /out/xtq_real.json
docker run ... $IMG /repo/scripts/tmp/xtq_speed_20261012.py   # XTQ_BS=2048,8192
# quantize a tensor (fast path):
xq.quantize_linear(W, H, code="exl3_mul1", K=4, use_ldlq=True,
                   device="xpu", margin=16, compiled=True)
```
Raw outputs: `/mnt/<nvme-disk>/b70-run4-20261011/xtq-quant/out/`
(`xtq_real.json`, `xtq_speed.json`); logs `/tmp/xtq_*_run*.log` on desktop-b70.

## Limitations / open

- 8h projection NOT met by torch; needs the SYCL kernel (or beam) — see above.
- LDLQ lowers herr but raises weight MSE; EXL3 sits between our two points —
  percdamp/blend tuning could find a dominating point if required.
- V=2 encode works end-to-end (pack/unpack/decode generalized) but its
  Gaussian quality wasn't separately scored here.
- No real-tensor run of the compiled/margin fast path vs LDLQ yet (viterbi-only
  spot check shown above).
- Hessian convention assumes HarnessFmt's undamped original-basis Gram; a
  different H semantics changes LDLQ results but not Viterbi.

## GPU Viterbi kernel (2026-10-12) — GPU1 BMG, ESIMD

`scripts/tmp/xtq_vit_20261012/xtq_vit_esimd.cpp` (JIT path is production;
AOT spir64_gen miscompiles both ESIMD and scalar kernels on this toolchain —
documented in DESIGN.md).

Design: one work-group (64 items, ESIMD wg cap) per 256-weight sequence;
in-place fp32 cost table in the group's global slice (L2-resident);
two-phase per step (gather-min phase 1 -> barrier -> emit+store phase 2 ->
barrier); argmin via `hmin` + `merge` first-index (matches CPU tie-break);
back-pointers uint8 in global; serial backtrace inside kernel; mul1 emit =
dp4a bytesum -> 1024-entry emit LUT gather; ih2x2 = SIMD hash pair + packed
fp16 add. margin= truncates pass-1 lookahead (CPU-verified +0.5% cost @16).

Measured (JIT, X1024.bin; GPU0 = level_zero:0 = PCI 0b:00.0, display
card; GPU1 = level_zero:1 = PCI 0f:00.0; L0 index->PCI proven by a
zeDevicePciGetPropertiesExt probe):
| code  | card | m0 | m8 | m16 | 26e9-wts proj | baseline |
| mul1p*| GPU1 | 1.22 | -    | 0.95-1.05 | ~7-8h  | 8.85 (64h) |
| mul1p | GPU0 | 2.00 | 1.34 | 1.42      | 9.7h@m8 | 8.85      |
| ih2x2 | GPU1 | 0.85 | -    | 0.66-0.83 | 4.6-5.5h| 8.85      |
| ih2x2 | GPU0 | 1.03 | -    | 0.81-0.83 | 5.9-6.0h| 8.85      |

*mul1p GPU1 rows are from the pre-fix build (see correction).

CORRECTION (2026-10-12): pre-fix VitMul1P had a phase-1/phase-2 group-
ownership mismatch (minima for item*JPI+m applied to emit group
item+ITEMS*m). mul1p was never bit-exact; earlier "verified" mul1p states
decode to ~2.0 MSE. Fixed: per-group minima exchanged via SLM
(slm_scalar_store(j*4)/slm_scalar_load(jp*4)); post-fix mul1p is 64/64
bit-exact vs CPU Viterbi on GPU0 at margin 0/8/16. ih2x2 and the strided
mul1 kernel (1.71 s/1M @m16 GPU0) were and remain bit-exact on both
cards; ih2x2 meets the <=8h gate on either card, mul1 needs GPU1.

(s/1M weights; nseq 1024 for the margin sweep; GPU0 runs under
flock /tmp/b70-gpu0.lock with guard clean pre/post and no xe reset on
0b:00.0; caveat: a root VLLM::EngineCore held renderD128+129 during the
GPU0 window — timings possibly mildly pessimistic)

Reference read: Cornell-RelaxML/qtip @ e90c6688c8df (main),
lib/codebook/bitshift.py vendored at xtq-work/qtip_bitshift.py; CUDA
Viterbi (qtip-kernels) reviewed for the chunked/global-bp design.

Device-compiler notes: JIT folds (float)(half)(t)->float(t); fixed with
integer fp32->fp16 RNE (round_f16). ESIMD emit_blk32 (32-wide) produced
uniform-FINF tables — 16-wide emits shipped. Scalar SYCL fallback
(xtq_vit.cpp): 3.27 / 1.65 s/1M, also exact.

## ESIMD Viterbi as quantize() backend (2026-10-12, K5 prep)

`libxtqvit.so` (built from `xtq_vit_esimd.cpp -DXTQ_VIT_LIB`, RPATH to oneAPI
compiler libs; preloads UR loader/L0 adapter so no setvars needed) is now a
backend of `xtq_quant_20261011.py`:

    xq.quantize(W_transformed, U_upper|None, code, K, seed,
                backend="esimd", margin=16) -> (W_hat, packed, scale)

U_upper is the GPTQ convention: U^T U = G^-1 (U upper, G = rotated damped
Hessian of the TRANSFORMED basis). Internally converted via
G = U^{-1}U^{-T} -> block_LDL(G,16) in fp64, once per call — the exact
factor the right-to-left LDLQ sweep consumes (fixed 2026-10-12: was
Lfac=U.t(), wrong; now verified bit-identical states/W_hat vs feeding
block_LDL directly, Lfac reconstruction diff 9e-17).
    xq.quantize_linear(W, H, code=..., K=4, use_ldlq=..., backend="esimd",
                       margin=16)

Env: XTQ_VIT_DEVIDX (0=GPU0 0b:00.0, 1=GPU1 0f:00.0; else default_selector),
XTQ_VIT_LOCK=auto|/path (opt-in flock /tmp/b70-gpuN.lock), XTQ_VIT_MAXSEQ
(1024 default). Codes: exl3_mul1 -> mul1p kernel, xe_ih2x2 -> ih2x2 kernel.
LDLQ feedback stays interleaved per 16-col block inside the right-to-left
sweep; only the Viterbi call is replaced by one batched kernel launch.

Bit-exactness: backend='esimd' == backend='torch' on packed trellis AND W_hat
— Gaussian 128x128 smoke + real-tensor column slices (128 cols) on both L2
tensors, both codes, LDLQ on/off — all True. REQUIRED the emit fix: the torch
default was the expanded quadratic L2-2x@LUT ("gemm"); both backends now use
the canonical (f-x)^2 ("sq", default flipped; gemm remains opt-in).

Full-tensor metrics on GPU0 (ESIMD backend, margin=16, incl. transform +
LDLQ + pack/decode + metrics; matches torch-pipeline numbers):

| tensor | config | rel MSE | herr | s/1M |
| in_proj_qkv [10240,5120] | EXL3      | 0.00589 | 0.00080 | -    |
|                        | mul1 vit  | 0.00471 | 0.00588 | 2.37 |
|                        | mul1 ldlq | 0.00740 | 0.00037 | 2.44 |
|                        | ih2x2 vit | 0.00525 | 0.00667 | 1.08 |
|                        | ih2x2 ldlq| 0.00826 | 0.00041 | 1.15 |
| mlp.down_proj [5120,17408]| EXL3     | 0.00583 | 0.00128 | -    |
|                        | mul1 vit  | 0.00467 | 0.00204 | 2.47 |
|                        | mul1 ldlq | 0.01048 | 0.00025 | 2.57 |
|                        | ih2x2 vit | 0.00521 | 0.00231 | 1.09 |
|                        | ih2x2 ldlq| 0.01169 | 0.00029 | 1.38 |

LDLQ herr beats EXL3 on both tensors/codes (up to 5x); plain Viterbi beats
EXL3 MSE. Fixed mul1p on GPU1 (post-SLM-fix, 64/64 exact): m8 1.062 /
m16 1.088 s/1M -> 7.3-7.9h full-model; ih2x2 GPU1 m16 0.690 -> 5.0h;
GPU0 mul1p ~1.34-1.42 (9.7h), ih2x2 0.81-0.85 (6h).

Memory note: a transient ~24 GiB RSS spike was observed in the
pack/decode/assert tail of quantize_linear on ~90M-wt tensors (int64
bit-expansion temps in pack_states/unpack_states) — FIXED by chunking both
functions over tiles (~256 MiB temp cap; round-trip still bit-exact, 16/16
suite pass). Peak RSS now ~5 GiB. Driver: xtq_esimd_test_20261012.py;
logs /tmp/xtq_esimd_full.log, /tmp/xtq_rss.log on desktop-b70.

## Harness-API bit-exact proof (2026-10-12, K5 final)

`quantize(W_transformed, U_upper(GPTQ U^TU=G^-1), "exl3_mul1", 4, seed,
backend="esimd"|"torch"|"dual", margin=16)` — verified on both L2 real
tensors in the vllm-xpu container on GPU0 (torch backend on xpu:0 so the
shared LDLQ feedback state is identical; `dual` runs both Viterbis on the
same per-launch X and compares states):

| tensor | launches | state mismatches | packed== | W_hat== | esimd time |
|---|---|---|---|---|---|
| in_proj_qkv [10240,5120] | 320 | 0 | True | True | 128s (2.45 s/1M) |
| mlp.down_proj [5120,17408] | 1088 | 0 | True | True | 240s (2.69 s/1M) |

Root-caused divergence fix (was 78 states on 1 seq): visa mad-fusion of
`d*d + b` — single vs double rounding broke torch's lowest-index tie-break
on exact-fp ties (e.g. cand 0x3f0fe05d w8/w11). Fix:
`e = min(d*d, FINF) + b` circuit-breaker on every emit site —
`xtq_vit_esimd.cpp:103` (VitMul1 strided), `:243` (VitMul1T),
`:374`/`:392` (VitMul1P 16- and 32-wide emits), `:532-533` (VitIh2,
`min(d0*d0,FINF)+min(d1*d1,FINF)+b`); scalar `xtq_vit.cpp:30` uses
`#pragma clang fp contract(off)` (works there; the same pragma at
`xtq_vit_esimd.cpp:25` does NOT reach the visa backend — ESIMD needs the
structural fix). Verified: `XTQ_EARLY` mid-sweep cost-table dumps now
bit-identical to torch; reproducer seq + X64 + both real tensors exact.

## Proc transport + margin=0 harness proof (2026-10-12, K5 launch config)

Two transports in `_esimd_submit` (`XTQ_VIT_TRANSPORT`):
- **proc** (default when binary exists): persistent
  `xtq_vit_esimd serve` subprocess, framed stdin/stdout protocol
  (req: magic/op/margin/pivot/nseq/xbytes/code[16] + fp32 X; rep:
  magic/rc/nbytes + int32 states).  Child env gets
  `ONEAPI_DEVICE_SELECTOR=level_zero:<XTQ_VIT_DEVIDX>` explicitly — the
  kernel reaches physical GPU0 while the parent (torch) runs
  level_zero:1.  Proven: parent selector=level_zero:1 -> worker printed
  `pci 0000:0b:00.0` (via ext_intel_pci_address aspect).  One worker
  lifetime = one JIT compile for the whole tensor.
- **ctypes**: in-process libxtqvit.so (legacy path; sees only the
  parent's filtered device list).  `XTQ_VIT_LOCK=auto` opts into flock;
  the proc path NEVER locks — the launcher owns /tmp/b70-gpuN.lock.

Hard device bounds enforced in `xtq_vit_run` (both transports):
nseq<=XTQ_VIT_MAXSEQ (default 4096) and nseq*1.257MiB<=XTQ_VIT_MAXBYTES
(default 6 GiB) -> rc 5/6; printed at init.

margin=0 end-to-end proof on the two real L2 tensors (docker --memory=20g,
torch-xpu container, GPU0, dual mode = per-launch torch-vs-esimd states):

| tensor | launches | mismatched states | packed== | W_hat== | esimd s/1M |
|---|---|---|---|---|---|
| in_proj_qkv [10240,5120] | 320 | 0 | True | True | 2.62 |
| mlp.down_proj [5120,17408] | 1088 | 0 | True | True | 3.54 |

s/1M includes U^-1U^-T conversion + LDLQ + pack/decode; kernel alone ~1.4.
CAUTION: --memory=14g cgroup-OOMs on the down_proj U path (peak ~14.6 GiB,
fp64 Cholesky of 17408^2); use >=20g.  Logs: /tmp/xtq_uab_m0{,_dp}.log on
desktop-b70.  Driver: xtq_u_ab_20261012.py (XTQ_MARGIN, tensor filter arg).

## Per-channel su/sv magnitudes (2026-10-12, K5 out_proj fix)

Motivation: K5 screen found GDN `linear_attn.out_proj` L0/L1 at 28-51x
WORSE herr vs EXL3 — its inputs are head-gated norms with strong channel
heterogeneity (turboderp's svh spans 0.56-19.6, suh 0.002-0.024 there).
Our arm used sign-only suh/svh + one scalar Wscale.

EXL3 1.4.2 source (turboderp-org/exllamav3 tag v1.4.2,
`exllamav3/modules/quant/exl3_lib/quantize.py`):
- `regularize()` :1178  `out_channel_scales = block_rms(weight, dim=0)`
  (per-OUT-channel RMS over the in dim), normalized by mean (:1180-1181);
  `sv = sv_signs * scales` (:1191), `weight /= sv` (:1198), dead out
  channels stored as svh=0 (:1201); output Hadamard (:1204) applied
  AFTER the sv fold.
- :1207-1209  `in_channel_scales = block_rms(weight, dim=1)` — per-IN RMS
  computed on the ALREADY sv-scaled, output-Hadamard-transformed weight;
  `su = su_signs * in_scales / (-codebook_scale=1.24371088)` (:16,:1209);
  `weight /= su` (:1210); input Hadamard last (:1211).
- :1213-1219  global `g_scale` from a tile-sample quantize grid search;
  folded into su via `su /= g_scale` (no separate scalar in the format).
- :1579-1580  `suh = su.half()`, `svh = sv.half()` — fp16 sign*magnitude.
- `finalize_capture_H` :879-898 — Hessian sees `sigma_reg=0.025` damping
  and ONLY the su SIGNS + input Hadamard; magnitudes never enter H (so
  LDLQ/U convention is unchanged in ours either).
- skew test :1158-1169 (`out_scales:"always"` in the SC config forces
  apply_out_scales=True regardless).

Implementation: `scripts/tmp/xtq_quant_20261012b.py` (frozen 11a left
untouched for HarnessFmt's live import).  `quantize_linear(channel_scales=
True)` reproduces the exact ordering above on [k=in,n=out] weights;
`suh = fp16(sign*su_mag*Wscale)`, `svh = fp16(sign*sv_mag)` (Wscale folds
into suh like EXL3's g_scale; the `-codebook_scale` const and +1e-10 eps
are EXL3-internal bookkeeping we absorb into Wscale instead).
`quantize(..., su_mag=, sv_mag=)` folds caller-supplied mags for the
harness path; U is NOT rescaled (sign-only H, as EXL3).

CPU proof (xtq_chan_proof_20261012.py, synthetic 256x512, row norms
35x span, diag H 100x):
  scalar   rel_mse 0.01340  herr 0.01229
  perchan  rel_mse 0.00484  herr 0.00441      (2.8x both)
  ref.weight_orig(trellis, suh, svh, K=4, cb=CB_MUL1=2) == W_hat.T
  BIT-EXACT (maxdiff 0) — decode semantics verified.

GPU0 proof (xtq_chan_ab_20261012.py, docker --memory=20g, card0+renderD128
only, flock /tmp/b70-gpu0.lock, proc transport -> `pci 0000:0b:00.0`,
margin=0, dual mode = per-launch torch-xpu vs esimd state compare):

  tensor                        mode     ldlq  launches  mism  rel_mse   herr
  L0 linear_attn.out_proj       scalar   off    384        0    0.010898   -
  L0 linear_attn.out_proj       perchan  off    384        0    0.004636   -
  L2 linear_attn.in_proj_qkv    scalar   on     320        0    0.007332  0.000363
  L2 linear_attn.in_proj_qkv    perchan  on     320        0    0.007188  0.000510

The failing GDN tensor's plain-MSE drops 2.35x with per-channel scales
(herr needs the L0/L1 out_proj H dumps - none saved).  On uniform-input
in_proj_qkv the mag path is a small regression (0.00036 -> 0.00051, still
< EXL3's 0.00080) — matches EXL3's own design where the skew heuristic
(:1158-1169) can gate out-scales; our flag is "always" matching the SC
config.  container_rc=0, guard clean, no 0b:00.0 resets.
Log: /tmp/xtq_chan_ab.log on desktop-b70.

## xtq_quant_20261012c.py + EXL3 export (2026-10-12)

- `scripts/tmp/xtq_quant_20261012c.py` = 12b + `XTQ_ESIMD_DEVICES="0,1"`
  dual-worker proc transport: contiguous nseq partition across one
  `xtq_vit_esimd serve` child per device (own `ONEAPI_DEVICE_SELECTOR=
  level_zero:N` each), chunks joined in order — output identical to the
  single-worker path by construction. T2 proof queued
  (/tmp/xtq_c_proof.sh, xtq_c_proof_20261012.py): b vs c bit-exact on
  L2 in_proj_qkv at margin=0 + channel_scales.
- `scripts/tmp/xtq_export_exl3_20261012.py` writes our packed trellis +
  suh/svh/scale into EXL3's safetensors field layout (servable on
  exl3xpu, no kernel change; ref.py-verified decode order).

## prepare_incoherence split API (2026-10-12, chscale harness arm)

Harness owns damping + U build; the per-channel magnitudes CANNOT be folded
into an already-transformed weight (sv_mag applies before the output
Hadamard, su_mag before the input Hadamard), so the transform is split:

  st = xqb.prepare_incoherence(W_orig, H_undamped|None,
         su_seed=20261011, sv_seed=None, su_sign=None, sv_sign=None,
         channel_scales=True, device="cpu")
  # -> {"Wt" [out,in] fp32 (mags folded, EXL3 order, no Wscale),
  #     "G" [in,in] fp32 undamped sign-only rotated Gram (mags excluded,
  #      cf. quantize.py :879-898) or None,
  #     "suh","svh" fp32 signed mags, "su_mag","sv_mag", "zero_out"}
  U  = harness_build_U(G + harness_damping)
  W_hat_inner, packed, scale = xqb.quantize(st["Wt"], U, code, K, seed,
                                 backend="esimd", margin=0)
  suh_store = (st["suh"] * scale).half()   # fold scalar into suh (:1219)
  svh_store = st["svh"].half()
  # W_orig_hat == ref.weight_orig(packed, suh_store, svh_store, K, cb)

Verification (/tmp/xtq_prep_smoke.py, synthetic 256x512 CPU):
  ldlq=False: packed, suh_store, svh_store, W_orig all torch.equal vs
  quantize_linear(channel_scales=True) monolithic path.
  ldlq=True (harness-style U): herr 0.003491 == mono 0.003491.
Landed in xtq_quant_20261012b.py and _12c.py (append-only).

Dumps: /mnt/<nvme-disk>/b70-run4-20261011/xtq-quant/prepare/
  L0_linear_attn_out_proj.safetensors, L1_linear_attn_out_proj.safetensors
  (Wt/suh/svh/su_mag/sv_mag).  sv_mag [0.876,19.599] L0 / [0.558,15.143] L1,
  su_mag ~0.002-0.019 — the heterogeneity the scalar arm missed.  G/U for
  these tensors still need HarnessFmt's L0/L1 H captures (only L2 H exists).

## 2026-10-12 evening: channel-scale A/B, dual-worker backend, EXL3 export (T1-T3)

**T1 chan A/B** (GPU0, margin=0, dual torch-vs-esimd per launch, 0 mismatches):
- L0 linear_attn.out_proj [5120,6144] pure Viterbi: scalar rel_mse 0.01090
  -> perchan 0.00464 (2.35x better; the GDN out_proj heterogeneity family)
- L2 linear_attn.in_proj_qkv [10240,5120] LDLQ: scalar rel_mse 0.00733 /
  herr 0.000363 -> perchan rel_mse 0.00719 / herr 0.000510 (still <EXL3 0.00080)
- log /tmp/xtq_chan_ab.log; b file frozen at md5 79ff2dee

**T2 dual-worker backend** (xtq_quant_20261012c.py, md5 264a152b /
sha256 a6fe5be585c1701d59f16fe712164617e736efaf5f8d1d642de5f459bd9b77df):
- XTQ_ESIMD_DEVICES="0,1" (or "0,0") -> one serve proc per slot, contiguous
  nseq partition per launch, per-slot sequential pipes, concat in order.
- Proven bit-exact vs b on L2 in_proj_qkv (trellis/W_hat/suh/svh torch.equal)
  with XTQ_ESIMD_DEVICES=0,0 (HarnessFmt K5 held gpu1; 0,1 unproven but
  identical plumbing; selector->PCI verified separately).  Re-proven on the
  post-fix build: /tmp/xtq_c_proof2.log (b 1.811 vs c 3.154 s/1M, margin=0).
- Speed: b=1.811 s/1M vs c(0,0)=3.154 s/1M -- same-card workers contend;
  dual-card speedup unmeasured (needs GPU1 free).
- log /tmp/xtq_c_proof2.log

**Scalar-path suh fold fix — unified-path rewrite** (HarnessFmt flag +
Main spec, 2026-10-12 ~22:1x): quantize_linear(channel_scales=False)
previously returned sign-only suh/svh and folded Wscale only into W_hat ->
ref.weight_orig(trellis,suh,svh) could not reproduce W_hat.  Final form
(md5 264a152b): scalar mode runs the IDENTICAL transform+store code path
as channel_scales=True with unit magnitudes (r_out=r_in=1, zero_out none)
so Wscale folds into suh exactly as per-channel (quantize.py:1219
su /= g_scale); W_hat built from_inner on the STORED fp16 suh/svh.
Pre-rewrite copy kept as xtq_quant_20261012c_prevfix.py (md5 00ae88be)
for the bit-identity assert.
Proofs (xtq_c_unified_proof_20261012.py):
- cpu: synthetic 512x512 (35x row-norm span, SPD H 100x diag span),
  margin=16 torch backend -- chan=False ref_eq=true maxabs 0
  (herr 0.02728), chan=True ref_eq=true (herr 0.00323), and channel path
  bit-identical to prevfix on trellis/W_hat/suh/svh.  UNIFIED_CPU PASS.
- gpu (GPU0 pci 0000:0b:00.0, esimd, margin=0, ldlq, real
  L2 in_proj_qkv + captured H): chan=False ref_eq=true maxabs 0,
  herr 0.000363 / rel_mse 0.00733, 102.5s (1.96 s/1M);
  chan=True ref_eq=true, herr 0.000510 / rel_mse 0.00719, 100.6s.
  Scalar herr < perchan here, as expected on this tensor.  UNIFIED_GPU
  PASS; guard clean pre/post.
Note: sweep normalizes Wq by fp32 Wscale in both modes (the fp16 rounding
lives in the stored suh only) -- identical to the per-channel convention.

**T3 EXL3 artifact writer** scripts/tmp/xtq_export_exl3_20261012.py md5 34a8e5d7:
- Header-preserving blob patch: input <root>/tensors/L{nn}/{module}.{safetensors,
  json} (trellis I16, suh F16, svh F16) -> template keys layers.{nn}.{module}.
  {trellis,suh,svh}; .mul1 scalars keep template MUL1_MULT bytes; all
  non-safetensors files + index + quantization_config copied verbatim;
  ARTIFACTS.sha256 regenerated. Shape/dtype asserted against template.
  selftest_a skips non-decoder-layer keys (lm_head/mtp.* -- they stay
  verbatim; HarnessFmt only emits model.language_model.layers.*).
- (a) RAN: 400 modules extracted -> rebuilt model shards byte-identical
  to template (all 2 shards + every file). container xtq-export-selfa,
  docker --memory=24g, SATA scratch.
- (b) RAN: real L2 mlp.down_proj (b, perchan+LDLQ+esimd, 171s,
  rel_mse=0.01029, herr=0.000264) -> ref.weight_orig(written) == saved
  W_hat torch.equal (maxabs 0.0).
- (c) RAN: check-keys on the real export: 0 problems (all 2426 template
  keys present, same shape/dtype).
- dump input at /mnt/<nvme-disk>/b70-run4-20261011/xtq-quant/
  exporttest/; outputs /mnt/<sata-disk>/xtq-export-20261012/
  {out-a,out-real}; log /tmp/xtq_dump.log

## Post-rewrite re-verification on final c (md5 264a152b, 2026-10-12 ~23:0x)

The T1 A/B above ran on pre-unified-rewrite b; re-proven on the FINAL c:
- CPU (xtq_chan_proof_c_20261012.py, host torch): scalar rel_mse 0.01340 /
  herr 0.01229 -> perchan 0.00484 / 0.00441; ref.weight_orig == W_hat.T
  maxdiff 0.  scalar |suh| = 13.1875 (Wscale now in suh — the unified fix
  visible; b had stored 1.0).
- GPU0 (xtq_chan_ab_c_20261012.py, same isolation: docker --memory=20g,
  card0+renderD128, flock b70-gpu0.lock, proc transport pci 0000:0b:00.0,
  margin=0, dual torch-xpu vs esimd): L0 linear_attn.out_proj 384 launches,
  0 mismatched states BOTH arms; scalar rel_mse 0.01091 -> perchan 0.00464
  (340s/338s).  rc=0, guard clean post.  Log /tmp/xtq_chan_ab_c.log.
- K5 campaign outcome (HarnessFmt, 2026-10-12): arm ran on frozen b
  channel-scale code and FAILED the gate — XTQ KL 1.15854x plain-Wiki /
  1.41236x chat vs criterion <=0.90 both.  Per-channel scaling landed and
  fixed the target family but the arm still missed; c (unified path,
  scalar-suh fix, dual-worker) was never run (deadline).

## Beat-turboderp variant sweep (xtq_quant_20261013d.py, 2026-10-13)

Task (Main): A-F variants scored per tensor on cal Gram + held-out Gram vs
turboderp's own plain-4.00bpw fields (B), in d only (b/c frozen by live jobs).
Q2 baseline (325 tensors, layers.jsonl xtq-full-chscale-cal512k): ours median
1.0264x B, all module types 1.013-1.034 (worst self_attn.q_proj 1.034; max
1.0599 L19 k_proj).  Win gate for requant: <=0.95x B held-out every type and
<=0.90x median.

d adds (additive recipe keys, running-sweep-safe):
- recipe={sigma_reg, cb_fold, gscale:"rms"|"search"|"ldlq"|float, gscale_sub}
  = EXL3 1.4.2 recipe copy (quantize.py:880 sigma*meandiag orig-basis damp,
  :442 +2*sigma retry, :1210 su/=(-codebook_scale 1.24371088),
  :979-1042 g_scale tile-search via _exl3_sample_tiles+g_scale_search).
- sv_refine / scale_refine spec: post-hoc scale solves on STORED fields vs
  tr(dW H dW^T).  "sv" = per-output-channel multiplier closed form
  (row-separable).  "su" = JOINT input-channel LS: W_hat = Y diag(suh),
  Y = W_hat|suh=1 -> s* = A^-1 c, A = (Y^T Y) (hadamard) H, c = diag(Y^T W H),
  fp64 chol with ridge+retry.  su_damp solves on sigma-damped Gram.
  su_vec/sv_vec = signed overrides for requant passes (su_vec also sets the
  rotation-basis sign vector).
- All variants rebuild W_hat from the STORED fp16 suh/svh so
  ref.weight_orig == W_hat (ref_equal asserted in every run).

CPU validation (512x512 synthetic, heterogeneous channels): e_t none
0.003485 -> sv 0.003455 -> su,sv 0.003447 -> su,sv,su,sv 0.003447 ->
requant2 0.003438.  ref_eq True on all.

Beat sweep (xtq_beat_sweep_20261013.py on GPU0, docker --memory=20g, 15
tensors L02 GDN+mlp, L23 attn qkvo, L24 GDN; H_cal/H_ho/H_cal_half0/1):
held-out e_t / B so far: F_svrefine best on qkv 0.9678x, z 0.9744x;
D_cvsigma best on out_proj 0.980x (sigma*=0.1).  C reproduces B within ~1%
held-out (recipe faithful).  G sweep (xtq_beat_g_20261013.py: G_su, G_susv,
G_susv2, G_sud(su_damp 0.05), G_req2 requant, G_combo CV-sigma+su,sv+requant)
queued on same lock.

### Slice findings (L02 512x512 slices, torch CPU margin=0, 2026-10-13)
- down_proj slice: B ho 0.005408; C 0.003997; F 0.003986; G_susv 0.003978
  (best; joint su solve edges sv alone on cal AND ho).
- out_proj slice: same ordering, G_susv 0.002937 < F 0.002947 < C 0.002958.
- Seed variance is real: seed 12345 beat default on cal AND ho on the
  out_proj slice (0.002909 vs 0.002937); cal ranking predicts ho ranking.
- Requant: adopting refined suh SIGNS regressed (0.002889->0.002912 cal);
  SIGN-PRESERVING requant (keep sign_vec, adopt |s*|/|sv*|) improved
  held-out 0.003978->0.003819 (~4%).  G driver uses sign-preserving.

### Full-tensor + incident (2026-10-13 ~12:45)
- Beat sweep killed mid-tensor-9 by another lane's launcher cleanup (sudo
  kill -TERM, journal 12:45/13:26, before labeled XTQNativeScore GPU1 job).
  Relaunched beat+G as ONE lock-held docker with b70.owner=TrellisQuantizer
  label; resume skips done-markers.
- L02 tally (held-out e_t/B): D_cvsigma wins mlp+out (down 0.9851, up 0.9892,
  gate 0.9917, out 0.980); F_svrefine wins in_proj (qkv 0.9678, z 0.9744).
- G_susv on L02 in_proj_qkv FULL tensor: e_t_ho 0.0007546 vs F 0.0007580
  = 0.9636x B — joint su solve > sv-alone on real full tensor. ref_eq True.

### A-F full table (beat_report.json, 15 tensors, held-out e_t/B)
Per-variant medians: A 1.0371 | C 0.9946 | E 0.9948 | F 0.9939 | EF 0.9914 |
D_cvsigma 0.9858 (best; wins 8/15).  All tensors <1.0x B.  C reproduces B
within ~1% -> exl3 recipe copy verified faithful.
D's CV sigma picks: 0.1 on all mlp + out_proj + L24 in_proj; 0.025-0.05 on
attention + L02 in_proj -> higher damping generalizes on the heterogeneous
tensors (calibration mildly overfit at sigma=0.025 there).

### G sweep interim (beat_g_report.json, joint-suh LS + requant + seeds)
L02_qkv: G_combo 0.9549 | req2 0.9594 | susv 0.9635
L02_z:   G_combo 0.9678 | req2 0.9678
L02_out: G_combo 0.9844
L02_gate:G_combo 0.9907
G_combo = CV sigma + scale_refine "su,sv" + sign-preserving requant wins all.

### Full A-F table (held-out e_t / B_turbo, beat_report.json — SN770 copy lost in disk event 18:45; numbers preserved here)
per-tensor best-2 (variant=ratio):
  L02 qkv     F=0.9678  EF=0.9725        L02 z      F=0.9744  EF=0.9749
  L02 out     D=0.9800  F=0.9914         L02 down   D=0.9851  EF=0.9990
  L02 gate    D=0.9917  F=0.9944         L02 up     D=0.9892  F=0.9942
  L23 k       F=0.9641  EF=0.9726        L23 o      EF=0.9945 E=0.9947
  L23 q       EF=0.9789 F=0.9860         L23 v      D=0.9858  F=0.9861
  L24 qkv     D=0.9872  EF=0.9893        L24 z      D=0.9849  C=0.9944
  L24 out     D=0.9758  F=0.9901         L24 gate   D=0.9817  EF=0.9914
  L24 up      D=0.9827  F=0.9944
medians: D 0.9858 | EF 0.9914 | F 0.9939 | C 0.9946 | E 0.9948 | A 1.0371.
D CV sigma*: 0.1 all mlp+out+L24in; 0.025-0.05 attn+L02in.

### G sweep partial (5 tensors before disk loss):
  L02 qkv G_combo=0.9549 req2=0.9594 susv=0.9635   <- best
  L02 z   G_combo=0.9678 req2=0.9678
  L02 out G_combo=0.9844
  L02 gate G_combo=0.9907
  L02 up  G_combo=0.9904
G_combo (CV-sigma + su,sv + sign-preserve requant) best on all 5.

### STORAGE EVENT 2026-10-13 ~18:45
SN770 (/mnt/<nvme-disk>) vanished from bus; a new
"flashnext_nvme" drive occupies the slot (operator swap/reformat?).
beat-tbd-inputs dump + report JSONs lived there — gone.  All scripts survive
in ~/B70-DOCS/scripts/tmp on root fs.  Numbers above preserved from logs.
My processes cleaned; no locks held.
