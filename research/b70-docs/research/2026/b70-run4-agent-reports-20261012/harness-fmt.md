---
title: "Run-4 subagent report: harness-fmt (2026-10-12)"
type: research
date: 2026-10-12
status: provisional
---

# Harness format screen evidence

## Implementation
- `scripts/tmp/fmt_screen_20261011.py` copied from the sealed r4 source; only the fmt copy is edited.
- Declarative default arm registry includes bf16, exl3, r4_g64, int4_g64_rht, e8rvq4_rht, e8rvq4_fold. Optional old diagnostic arms remain selectable through FMT_ARMS.
- RHT per-linear transform is T=block-H128 @ random sign D. Weight transform T_out W T_in^T, Hessian transform T_in H T_in^T, inverse reconstructed weight T_out^T Q T_in. Seed 20260925+n, shared between scalar/E8 RHT arms.
- Classic inverse-factor chain replaced with reverse-order Cholesky: J H J=L L^T; U=J L^-1 J; U^T U=H^-1. No second Cholesky of the rounded inverse Gram. Tile Cholesky/solve uses CPU fp64, device Schur updates remain strict fp32. Fallback executes classic fp64 CPU chain in-place on one matrix with `out=H64`. Fixed percdamp 0.01; no adaptive damping.
- Two-run exact equality and finite/positive-diagonal proofs embedded; layers 2/11 additionally capture H/U and compare classic fp64 factors/codes.
- EXL3 reconstruct code and ref pins unchanged. fp32 fold probe every layer, conditional gptq_plain defect control, EXL3 r01 sign, NLL gather bounds retained.
- Launcher uses shared flock, GPU1 selector, pinned image, 14g cap default, optional capped 24g, pre/post guard, ownership-safe cleanup and live MemAvailable/reset monitor.

## Precision root-cause evidence
CPU-only image inspection command (no GPU devices mounted): `sudo -n docker -H unix:///run/<exl3-docker>.sock run --rm --network none --memory=2g --memory-swap=2g --cpus=2 --entrypoint python3 vllm/vllm-openai-xpu:v0.30.0@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d -c ...` observed torch 2.13.0+xpu, fp32 precision highest, ONEDNN_DEFAULT_FPMATH_MODE unset, DNNL_DEFAULT_FPMATH_MODE unset, deterministic algorithms False. No evidence supports TF32/bf16x3 as the default cause.
Exact installed torch source commit cf30153c4c131c8164ee7798e5022d810682e2cb:
- https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/mkldnn/xpu/detail/Matmul.cpp : fp32 matmul explicitly selects tf32 iff allowTF32OneDNN, strict otherwise; deterministic attribute enabled from global deterministicAlgorithms.
- https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/mkldnn/xpu/Blas.cpp : addmm routes to oneDNN.
[INFERENCE] The old second Cholesky of X^T X rounds an inverse before refactoring, creating an avoidable numerical failure point; nondeterministic oneDNN reductions can alter threshold crossings. Strict/deterministic math and reverse-H route remove those causes, subject to real smoke evidence below.
CPU fp64 256x256 randomized SPD reverse-factor proof: rel Frobenius vs in-place classic 8.082837251794116e-16; normalized identity residual 1.6059543649778261e-15; exactly upper-triangular True.

## Commands
First full-calibration determinism launch (unsealed; max six allowed):
```sh
env B70_RUN_DIR=/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/proof13 FMT_ARMS=bf16,exl3,r4_g64 FMT_HINV_VERIFY=1 R4_SMOKE=1 R4_SMOKE_FULLCALIB=1 R4_SMOKE_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12 R4_DUMP_H=/run_dir/hessians bash scripts/tmp/fmt_screen_launch_20261011.sh run
```
Raw root: `/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/proof13`.

Additional CPU-only import/synthetic smoke (same pinned container, --memory=2g --cpus=2 --network none, repo ro, no /dev/dri): `import fmt_screen_20261011 as f` succeeded; random W [256,384] RHT forward+inverse rel Fro 4.142990235322941e-7; transformed Hessian vs transformed-weight Gram rel Fro 4.964348931935092e-7; random H [512,512] new fp32 upper factor vs classic fp64 rel Fro 1.4659759858144735e-7, no fallback. These are synthetic CPU checks, not real-GPU acceptance.

## Completed GPU factorization proof (proof13)
- 52/52 identical damped-H factor pairs torch.equal=True, finite=True and positive_diagonal=True; no fallback. Full 64x512 calibration, layers 0–12.
- Classic comparison (fp32 U vs classic-fp64 U cast fp32):

|Layer|Group|Rel Fro|
|---|---|---|
|2|linear_attn.in_proj_qkv|0.00247873575|
|2|linear_attn.out_proj|8.93049037e-06|
|2|mlp.gate_proj|0.00229643821|
|2|mlp.down_proj|7.75652952e-05|
|11|self_attn.q_proj|0.00101062981|
|11|self_attn.o_proj|3.19519095e-05|
|11|mlp.gate_proj|0.00111925916|
|11|mlp.down_proj|2.38982902e-05|

|Layer|Linear|INT4 code fraction differing vs classic fp64|
|---|---|---|
|2|linear_attn.in_proj_qkv|0.0485468842|
|2|linear_attn.in_proj_z|0.0492381454|
|2|linear_attn.out_proj|0.0119992578|
|2|mlp.gate_proj|0.050314568|
|2|mlp.up_proj|0.0504639037|
|2|mlp.down_proj|0.0806892216|
|11|self_attn.q_proj|0.0428731292|
|11|self_attn.k_proj|0.042473603|
|11|self_attn.v_proj|0.0407169349|
|11|self_attn.o_proj|0.018057283|
|11|mlp.gate_proj|0.043196097|
|11|mlp.up_proj|0.0437664166|
|11|mlp.down_proj|0.0360892117|

Lead decision: retain deterministic device fp32 route; these arithmetic differences are disclosed, not hidden. Classic AutoGPTQ is itself fp32; lead requests Hessian-weighted error comparison on L2 int4_g128_rht/down before interpreting code flips. No <=1e-5 classic-closeness claim.
- RSS peak 13.2103 GiB, fits 14g cap; no need to raise to 24g. XPU allocated peak 15.9 GiB (not physical VRAM total).
- proof13 layer loop + final metrics 992.0 s. Layers 2/11 include expensive classic-factor/code proofs; this is not a normal all-format 64-layer wall projection.
- Fold probe max rel L2 1.5301264966183226e-6. EXL3 r01 correct, skipped gptq_plain defect control explicitly because that arm was inactive. Other prefix canaries already fail in bf16, so final full-depth correctness gate is not meaningful at 13 layers.

## Old-path real-Hessian replay
```sh
env B70_RUN_DIR=/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/replay13 FMT_CAPTURE_DIR=/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/proof13/hessians bash scripts/tmp/fmt_screen_launch_20261011.sh hinv-replay
```
Raw `replay13/raw/hinv_replay.json`: both L2 and L11 down_proj identical-H old fp32 chain results were NOT torch.equal for the deterministic=False pair and WERE torch.equal for deterministic=True pair. All 8 trials PD succeeded; inverse-Gram asymmetry max 0.0. This proves reduction-order nondeterminism in the old path; it does NOT reproduce the historical PD failure on these particular captures. TF32 getter False, default precision highest, source strict: TF32/bf16x3 root-cause hypothesis disproven for these defaults. Historical PD-loss cause remains [UNVERIFIED] beyond the numerically fragile second inverse-Gram Cholesky plus proven nondeterminism.

CPU-only real L2/down reverse-fp64 probe: 42.9061 s, peak RSS 7.4044 GiB, rel Fro vs classic fp64 cast fp32 5.60367334401235e-6. Main path deliberately not changed to CPU fp64.

## Subsequent scalar-only screen commands
E8 GPU work was explicitly cancelled by lead; E8 arms remain selectable code paths only. Default arms changed to bf16/exl3/r4_g64/int4_g64_rht/int4_g128_rht. True act-order is not a one-flag change in this GPTQ implementation and was not added under the lead's conditional request.

```sh
env B70_RUN_DIR=/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/scalar13 FMT_ARMS=bf16,exl3,r4_g64,int4_g64_rht,int4_g128_rht R4_SMOKE=1 R4_SMOKE_FULLCALIB=1 R4_SMOKE_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12 bash scripts/tmp/fmt_screen_launch_20261011.sh run

env B70_RUN_DIR=/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/scalar13-weighted FMT_ARMS=bf16,exl3,r4_g64,int4_g64_rht,int4_g128_rht FMT_FP64_WEIGHTED_COMPARE=1 R4_SMOKE=1 R4_SMOKE_FULLCALIB=1 R4_SMOKE_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12 bash scripts/tmp/fmt_screen_launch_20261011.sh run
```
Second scalar launch provides per-tensor Hessian-weighted original-basis error and numel plus the L2/G128/down classic-fp64 weighted comparison requested later by lead. Weighted e_t uses undamped raw calibration Gram; its denominator is tr(W H Wt), not a normalized H. Fold readers undo Q and norm gain; their H is mapped into that original-coordinate linear input basis. Statistics use actual bf16-stored reconstructed weights, so include storage rounding. No nearly-free bf16 output-KL proxy exists without extra lm_head logits; omitted as permitted.

The launch-time script/codec hashes are captured in raw/identity_start.json. Sources were extended between separate runs, so do not assume proof13 and scalar13 share identical harness hashes.

## proof13 timing table (seconds, wall-clock phases)

|L|Total|Plain build|Fold build|EXL3 wait/build|Calib|Hinv+quant+proof|Fold probe|Propagate|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|62.3|3.82|4.78|18.23|3.85|23.68|5.97|1.3|
|1|40.3|2.77|2.53|3.48|1.75|21.69|6.28|1.16|
|2|262.2|3.11|2.91|3.53|2.06|243.07|5.69|1.07|
|3|39.0|2.65|2.62|3.56|1.72|21.71|5.64|0.59|
|4|37.3|2.85|2.71|3.34|2.06|18.80|5.74|1.05|
|5|38.4|2.81|2.7|3.53|1.72|19.88|6.07|1.08|
|6|38.6|2.94|2.81|3.66|2.2|19.90|5.33|1.04|
|7|41.5|2.91|2.59|3.37|1.48|23.61|6.33|0.44|
|8|43.3|2.98|3.02|3.70|2.02|23.00|6.73|1.14|
|9|40.5|2.99|3.02|3.46|2.01|21.13|5.95|1.07|
|10|41.0|2.67|2.66|3.54|1.94|22.24|6.2|1.1|
|11|265.4|2.61|2.66|3.27|1.51|248.75|5.57|0.43|
|12|38.4|2.89|2.74|2.66|1.48|20.86|6.09|1.1|

Raw phase timings include per-input-group arm hinv/loop breakdown in layers.jsonl. XTX is a subphase of calibration (do not sum it twice). Weighted-error timings in later runs are a subphase of quantizer-loop timing.

## Scalar13 decisive quality result

|Arm|KL mean (nats)|KL / EXL3|Top1 agreement (%)|r01 margin (logit)|
|---|---:|---:|---:|---:|
|bf16|0|0.000000|100.000000|1.19048834|
|exl3|0.00122329092119|1.000000|96.754813|1.24599791|
|r4_g64|0.00358332926407|2.929254|94.471157|1.13503528|
|int4_g64_rht|0.00300992117263|2.460511|95.673078|1.17801404|
|int4_g128_rht|0.00347804231569|2.843185|94.951928|1.20393515|

Raw `scalar13/measurements.json`, `raw/arms_all.json`, `raw/layers.jsonl`. 1664 eval positions, 64x512 calibration. Incoherence improves G64 versus residual-fold comparator but remains 2.4605x EXL3 KL; stock-native G128 RHT remains 2.8432x. This is quality of reconstructed layers, not stock runtime speed.
XPU allocated peak 19.6 GiB; allocator reserved peak 22.7090 GiB; CPU RSS 10.8409 GiB; no fallback; 14g cap sufficient. Normal per-layer projection (layer-type weighted to 48 GDN +16 full-attention): 5166.56 s = 1.435 h; x1.25 timeout 6458.20 s = 1.794 h, below 3 h target. Actual layer loop + final metrics 1055.5 s. Startup manifest verification/first-decode cost retained in raw wall time (command 1278.59 s includes earlier shared-lock wait).

## Scalar13 per-layer time table (seconds)
|L|Total|Build plain|Fold build|EXL3 wait/build|Calib sum|Hinv+quant|Fold probe|Propagate|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|112.7|3.8|5.27|25.52|8.41|60.01|6.12|2.61|
|1|82.0|3.04|2.8|3.81|6.17|57.50|5.56|2.45|
|2|75.8|2.95|2.72|3.41|6.03|52.27|5.42|2.39|
|3|77.1|2.66|2.77|3.14|5.67|55.63|5.49|1.25|
|4|75.8|2.9|2.91|3.46|6.22|50.94|6.14|2.64|
|5|78.9|3.32|2.67|3.50|6.27|54.21|5.79|2.45|
|6|77.3|2.8|2.78|3.69|6.10|53.10|5.82|2.4|
|7|81.4|2.9|2.8|3.38|5.48|59.56|5.35|1.13|
|8|78.8|2.99|2.75|3.46|6.39|53.49|6.38|2.59|
|9|79.7|3.08|2.85|3.60|6.26|54.53|6.15|2.44|
|10|78.3|2.95|2.79|3.84|6.02|54.10|5.51|2.38|
|11|78.8|2.82|2.54|3.34|5.51|57.66|5.2|1.13|
|12|73.4|2.85|2.71|2.74|5.80|50.64|5.68|2.39|

## Implementation line anchors (current source before XTQ integration)
- Arm declarations/plugin classification: `scripts/tmp/fmt_screen_20261011.py:117-148`.
- Exact RHT coordinate transforms: `scripts/tmp/fmt_screen_20261011.py:150-169`. Input signs seed=20260925+n, output signs seed=20260925+n+1000003; independent and shared by scalar/E8 comparators.
- CPU fp64 tile Cholesky and device bordered blocked factorization: `scripts/tmp/fmt_screen_20261011.py:337-395`.
- Reverse-H upper factor/classic in-place fp64 fallback and damping/proofs: `scripts/tmp/fmt_screen_20261011.py:400-527`.
- Original-basis weighted error: `scripts/tmp/fmt_screen_20261011.py:922-940`.
- Captured-H old-path replay: `scripts/tmp/fmt_screen_20261011.py:943-992`.
- Strict/deterministic math: `scripts/tmp/fmt_screen_20261011.py:1019-1024`.
- Weighted-fp64 comparison: `scripts/tmp/fmt_screen_20261011.py:1312-1329`.
- Fold sanity probe: `scripts/tmp/fmt_screen_20261011.py:1367-1370`; NLL bounds `:1465-1473`; conditional controls `:1510-1539`; per-format KL gate `:1544-1547`; RSS/projection `:1623-1626`.
- GPU lock/guard/live monitor/memory caps: `scripts/tmp/fmt_screen_launch_20261011.sh:7,24-53`; selector/env `:55-64`; pre/post evidence and owned cleanup `:69-111`.

Checks actually exercised: CPU-only synthetic reverse-factor/RHT/weighted-error smokes; GPU proof13, captured-H replay, scalar13. `bash -n scripts/tmp/fmt_screen_launch_20261011.sh` passed once. No sealed run, no registry/research edits, no pushes. GPU0 computation not used; guards only inspect its holders.

## Weighted follow-up: failure retained and exact correction
First weighted run stopped after layers 0–6 at L7 fold MLP reader: exact zero RMSNorm gains make the naive inverse gain map undefined, and quantization can leak into the zero-gain input directions. Failure raw root `scalar13-weighted/raw/run_stdout.log`, clean post-guard.
Permanent exact fix uses pre-gain RMSNorm inputs in original coordinates for all folded readers, not only zero-gain special cases: W_eff=W_original*gain, W_hat_eff=W_fold_hat Q.T, H_eff=Q H_fold Q.T. This preserves the exact linear output weighted-error objective, includes nullspace leakage, and does not suppress it via pseudoinverse/zeroing. Explicit basis metadata distinguishes effective norm-gain-folded weights from unfused original weights. Other tensors use original linear input basis.

L2 INT4 G128 RHT/down weighted objective (completed before failure): fp32 e_t 0.00046333138412749244; classic-fp64 e_t 0.00046322908276758646; ratio 1.00022084 (+0.0221%, within1%). Code difference fraction 0.14200595021247864 (14.2006%) does not imply objective drift. Both tr(W H Wt)=476103.6875, numel=89128960. Raw `scalar13-weighted/raw/fp64_weighted_comparison.json`.

Corrected final prefix command (no repeated expensive fp64 proof):
```sh
env B70_RUN_DIR=/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/scalar13-weighted-fixed FMT_ARMS=bf16,exl3,r4_g64,int4_g64_rht,int4_g128_rht FMT_FP64_WEIGHTED_COMPARE=0 R4_SMOKE=1 R4_SMOKE_FULLCALIB=1 R4_SMOKE_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12 bash scripts/tmp/fmt_screen_launch_20261011.sh run
```
Host-crash correction from lead: weighted13-fixed never reached execution; the queued job died when the session was terminated by systemd-oomd. Its earlier command above is an attempted queue, NOT completed GPU evidence. Completed runs remain proof13, old-H replay, scalar13, and failed weighted13. The resumed scalar-only command below supersedes the queued five-arm rerun.
CPU-only nullspace regression smoke with 8/128 exact-zero gains: weighted helper numerator 427.3432922363281, direct squared linear-output error 427.3432922363281, relative difference exactly 0.0. This checks leakage is retained rather than discarded.

Pre-XTQ source hashes (weighted first run launch identity): harness 0272bb0375473ec6052c88e28c7cb38a9400bae9659cf7d5755a13cc116cab5a; launcher f49ab4b8bfcdd7d7509cf44577ca41f582ded6add59399e67157c33eddd15dae. Sealed original r4 Python 2caf5e719dd2966522f6921530e9b5c76a1d4be5695fed0955681220482c4577; original launcher b1142036d5369976a7fe38f3c96c8bc8345ad002c5fe274cc0a464d7cf58d172.

## Host-crash resume
Container cap stays14g; launch requires MemAvailable>=20GiB and no other heavy concurrent work. No-code disk floor4GiB, code-save floor15GiB: live SN770 free9GiB cannot meet previous arbitrary15GiB no-code gate but safely fits two original-H captures (~1.23GiB).

```sh
env B70_RUN_DIR=/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/weighted-resume13 FMT_ARMS=bf16,exl3,int4_g64_rht FMT_CAPTURE_ORIGINAL_H=/original_h R4_SMOKE=1 R4_SMOKE_FULLCALIB=1 R4_SMOKE_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12 bash scripts/tmp/fmt_screen_launch_20261011.sh run
```
Reference BF16 and EXL3 now have own full calibration streams/Hessians for per-tensor weighted-error measurements; neither is quantized by our loop. Layer2 original BF16 input Gram captures key H (undamped fp32): `/mnt/<nvme-disk>/b70-run4-20261011/xtq-quant/H/L2_mlp.down_proj.safetensors` and `L2_linear_attn.in_proj_qkv.safetensors`.


## Resumed weighted screen COMPLETE
- bf16/exl3/int4_g64_rht each has81 linear records over all13 layers (243 total records), full64x512 calibration; scalar KL0.003009921172633767, EXL3 KL0.001223290921188891, ratio2.46051133095, same as earlier scalar13.
- Peak RSS10.3003273GiB; XPU allocated16.5GiB/reserved21.4589844GiB; minimum host MemAvailable45269MiB.14g cap sufficient; clean pre/post guard.
- 64-layer projection2202.613333s=36.7102min; timeout x1.25=2753.266667s=45.8878min.

|L|Total s|Build s|EXL3 wait/build s|Calib all s|Quant s|Weighted subphase s|Fold probe s|Propagate s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|49.4|3.38|14.34|7.04|16.29|1.14|3.61|2.09|
|1|32.9|1.85|1.99|6.00|14.71|1.16|3.6|2.04|
|2|33.9|1.94|2.18|5.99|14.50|1.16|3.6|2.04|
|3|33.2|1.87|2.03|5.56|16.37|1.15|3.55|1.1|
|4|33.4|1.89|1.97|6.18|14.96|1.19|3.63|2.02|
|5|33.2|1.84|1.99|6.19|14.83|1.18|3.57|1.99|
|6|33.4|1.89|2.20|6.20|14.75|1.18|3.6|2.0|
|7|32.8|1.81|2.00|5.47|16.37|1.16|3.49|1.05|
|8|33.4|1.92|2.05|6.19|14.80|1.19|3.63|2.02|
|9|33.4|1.89|2.12|6.19|14.75|1.17|3.66|2.02|
|10|33.6|1.93|2.25|6.19|14.77|1.18|3.65|2.02|
|11|32.8|1.82|2.04|5.47|16.21|1.14|3.52|1.04|
|12|32.5|1.92|1.73|5.71|14.75|1.19|3.66|2.02|

## K1 allocation proxy (not measured KL gain)
Exact commands executed sequentially, no other heavy concurrent work:
```sh
/usr/bin/python3 scripts/tmp/format_theory_20261011/waterfill_k1.py --layers /mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/weighted-resume13/raw/layers.jsonl --arm int4_g64_rht --out /mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/weighted-resume13/raw/k1_int4_g64_rht.json
/usr/bin/python3 scripts/tmp/format_theory_20261011/waterfill_k1.py --layers /mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/weighted-resume13/raw/layers.jsonl --arm exl3 --out /mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/weighted-resume13/raw/k1_exl3.json
```
81 tensors,4944035840 weights/arm. Scalar weightedAM/GM5.854083005660292; EXL3 weightedAM/GM5.329903756895977; both pass1.10 proxy gate. Continuous rates scalar[-1.024292469,7.087944031],EXL3[-.351885319,7.033769608]bpw are unconstrained; negative rates not physically implementable. Integer-curve allocation predicted gains4.436348637 and4.560174007 at4.000bpw; only a per-tensor exponential-error proxy, not empirical allocation or model KL. Per-tensor R* and rounded rates retained in respective JSON per_tensor lists.

## Original Hessian capture identities
- `L2_mlp.down_proj.safetensors`: keyH,fp32[17408,17408],undamped original BF16 input Gram; SHA2566eb9708520a6abdaf81c6ccbf55c0a8e8ceb7bc4ffa98cbd3d3244874d190d88.
- `L2_linear_attn.in_proj_qkv.safetensors`: keyH,fp32[5120,5120],undamped original BF16 norm-output input Gram; SHA256d6d09441ffe2504526b056aa4d874f0fc8881c01ac048e81e3a69bc699f17655.
- Both at `/mnt/<nvme-disk>/b70-run4-20261011/xtq-quant/H/`, delivered to TrellisQuantizer. Apply .01 damping once, not twice.
- Resumed run harnessSHA256fd031f6ef165b31bbb068592f0443415ad4bdfb31dd5ea58df81e9f6d254d1ec; launcherSHA2562ad944fe0ad697163b31b3370eb4cdd44794a1aba0eabfcb490650cdc9ba9155.

## K1 follow-up qualification from FormatTheory
Independent lane reproduced AM/GM and2..7 integer proxies exactly. Rate floorR>=0 gives physically valid proxy gains5.781(scalar)/5.320(EXL3); restrictedXTQ{3,4,5} allocation at4.0bpw gives2.461/2.495,31/12/38 tensors at3/4/5bits. This downgrades41%weights and increases some tensor error up to3.24x; a layer-local summed proxy gain may worsen true modelKL. Allocation must be screened, not adopted by proxy. Evidence `local://format-theory.md` and FormatTheory/k1_real_analysis.json.

## Priority pivot: self-calibrated EXL3 full-depth evaluation
Lead redirected K5 until a real XTQ quantizer exists. Generalized decoder now takes artifact_root and infersK from trellis shape[2]/16, with exact XPUinteger+CPUHadamard checks per artifact/K. SC4/SC5 arms read-only mounted; farms require lead-owned ARTIFACTS.sha256 andREVISION.txt finalization. FMT_CALIB_FREE=1 removes all calibration/Hessian streams; no-fold probesN/A. NLLbounds retained. Cheap per-layer eval hidden-state relativeMSE logged; not mislabeled as KL contribution.
SC4 quant config codebookmul1, out_scalesalways, head_bits5; plain EXL3 head_bits6. Both apply outputscales throughsvh, same pinnedref.py representation. Full artifact comparison must disclose head scope (head allocation differs); pending lead decision on body-only commonBF16head versus nativeheads/both.


## SC evaluator implementation details
- Generalized decode root and per(root,K) exact selfchecks; K inferred from packed16K shape.
- Calibration-free EXL3 arms load native artifact embedding, unquantized layer tensors and finalnorm.
- Optional FMT_NATIVE_HEAD=1 emits nativeheads primary plus <arm>_bf16head diagnostic metrics. Head reconstruction streamed8192 outputcolumns with128boundary alignment; random K5 CPU full-versus-chunk reconstruction=torch.equalTrue,maxdiff0. Native decodedweights castBF16 then FP32matmul consistently with body quality harness, not native serving kernel.
- Per-layer cumulative hiddenstateMSE versusBF16 included; no additiveKL attribution claim. Full final-logit metrics unchanged and NLLbounds retained.
- Root manifests/revisions are lead-owned download finalization; launcher fails closed until present; no downloadfarm writes by this lane.
- SC quant config calibration250x2048,head5,plainhead6; in-domain publishedKL not comparable to ourWiki eval until measured.
- K5 genericXTQ registration removed after prioritypivot to avoid incomplete callable arms while quantizer absent.

SC4 metadata rate histogram (allstoredtrellis inclMTP/head): K3=34,K4=297,K5=66,K6=4, noK7. SC run adds tiny synthetic128x128 mul1 exact integer/fullweight proofs atK3,5,6,7 so K7decode is tested even absent artifactK7. Actual artifact proofs remain separate perroot/K; synthetic does not standinfor whole-artifact integrity.

SC5 metadata trellis histogram(allinclMTP/head):K4=41,K5=272,K6=66,K7=22; actualK7proof therefore included when SC5decode encountersit. Both downloads reportedrc0; scheduled single4arm readiness watcher, noGPU until manifest+revision finalized. Previous waiting3arm watcher cancelledbefore launch. Planned command:
```sh
env B70_RUN_DIR=/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/sc45-full64 FMT_ARMS=bf16,exl3,exl3_sc4,exl3_sc5 FMT_CALIB_FREE=1 FMT_NATIVE_HEAD=1 R4_SMOKE=0 bash scripts/tmp/fmt_screen_launch_20261011.sh run
```

SC-ready source hashes beforelaunch: harness447abb127e1000dbdaab3ac1ed0cc1d8514c9bce73f8e3fa083f569ccddaad97; launchera19998e18319501d79ccb6db5ac837fee762ed18fb7524b67ab8ca96c606ed25. Launchidentity capturesactualhash binding.

## SC full64 SAFETY BLOCKED (new GPU1 reset)
Finalized SC4/SC5 manifest checks22/22 and11/11 passed, along withBF16/states/plain manifests. SyntheticK3,5,6,7 integer+fullweight torch.equalTrue onGPU1. Run stopped BEFOREL0 completion by live monitor onNEWGPU1 xe Engine memoryCATerror[18] followed byBCSenginereset.
Raw root `/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/sc45-full64/raw/`. dmesg_after.txt:2018-2022: monotonic202589.512959CATerror,202589.513200Engine reset engine_class=bcs,guc_id6,state0x249, timedoutpython3PID3209552,devcoredumpcreated. Notinbeforelog. HostMemAvailable47734MiB atstop; containerExit137,OOMKilledFalse. Ourmonitor stoppedOWNcontainerandremovedit; postguardclean. Nootherprocesskilled.
**Hardrule obeyed: no furtherGPUwork after reset.** No64-layerSCmetrics/ratios exist; do not substitute prefix/publishednumbers. Publishedin-domainKLDclaims remain unverifiedbyourharness. Rootartifactverificationandsyntheticdecodeproofs are the only completedSCevidence.

## Held-out SC trace CPU preparation complete
`FMT_HELDOUT_TRACE=/exl3_sc4/qbench_prompts_gen.json` selects storedalready-templatedinput_ids+response_ids; doesnotusecal_trace. Exactresponsepredictorpositions mask[promptlen-1:sequencelen-1], excludesprompt/pad/lastposition. First2fittingrows0/1:prompt84/49,response748/783,2x832tokens,1531scoredresponsepositions. TraceSHA2566070481d810633355e264a618f8a3727af0fc135b3c3671c917928dc6dc215ed,templateenable_thinkingTrue/reasoning_effortmedium. CPUonlybuild_inputs testpassed,maskshape2x832. RuntimeGPUevaluationblockedbyreset; noheldoutKLclaim. Thisisasmallslicediagnostic,notreproductionofhisfull45930response-tokenmetric.

## Final source references/identities
FinalharnessSHA25645db24ff46f6d5dee7064c86b9591a717cb101be25665cbea32d23922f3953ab; finalauncher2499ff606d5df9d487be5915aa41c3e575a73f4c3ef589cb755700daa9a70d92. Originalsealedr4fileshashesunchanged.
- `scripts/tmp/fmt_screen_20261011.py:699-750`: artifactroot/mixedKdecoder,actualexactproofs.
- `:753-789`: streamednativeheaddecode; `:791-810`: nativeartifacttensorloading.
- `:839-866`: heldouttracesliceresponse-positionmaskandprovenance; `:1625-1649`: maskedKL/top1/NLLwithbounds.
- `:1026-1051`: fold-awareweightederrorobjective; `:1173-1189`: syntheticmixedKproofs.
- `:1607-1616,1669-1670,1752-1756`: nativehead+sharedheaddiagnosticsandscope.
- `scripts/tmp/fmt_screen_launch_20261011.sh:7,36-51,54,76-80`: lock,livereset/RAMmonitor,containercap,launchmemory/diskgates; `:68-70`: SCevaluationflags.
Mainverificationnext(nonGPUonlyuntilresetcleared): inspectincidentdmesg+containerexit+postguard; CPUheldoutmasktestalreadyexercised. FurtherGPUrunrequireslead-approvedsafetyresolution; thislanestoppedexactlyathardrule.

## Authorized safety-fixed rerun
CPU-onlyRCAandexhaustivepre-fixdel/cache/workerinventory:`local://harness-sc-copy-rca.md`. WorkeronlyloadspackedCPUtensors; integerXPUdecodeonmainafter.result; all11cachefreesfollowglobalsync; CPU/GPUcopysourceskeptuntilsync; embeddingsserialized. Coredumpsaved508712bytesSHA96759ad73457841f3a041f8f6e33cd14cac7dea8e04ffe37f5faa9552a7bfaf5. Cross-threadallocatorracehypothesisnotproven,removablebysafeconstruction.
MainGOONErerun; exactcommand:
```sh
env B70_RUN_DIR=/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/sc45-full64-safe FMT_ARMS=bf16,exl3,exl3_sc4,exl3_sc5 FMT_CALIB_FREE=1 FMT_NATIVE_HEAD=1 FMT_HELDOUT_TRACE=/exl3_sc4/qbench_prompts_gen.json R4_SMOKE=0 bash scripts/tmp/fmt_screen_launch_20261011.sh run
```
BothWiki1664positionsandheldout1531responsepositionsincombined4x832batch,samepass,separatemasks/metrics. CPUmasktestpassed. Logitsmaterializedonemetricarmatatime(not7fullvocabtensorresident)tosaveVRAM. AnynewGPU1reset=>stop,no2ndrerun.

## FINAL SC FULL64 SUCCESS (authorized safety-fixed rerun)
Wiki1664positions; heldout1531responsepredictorpositions(first2conversationrows,reasoningtrue/medium); nativeartifactheadsprimary+commonBF16headsdiagnostic.

|Arm/head|Wiki KL nats|Wiki top1 %|Heldout KL nats|Heldout top1 %|r01 margin|
|---|---:|---:|---:|---:|---:|
|bf16|0|100.000000|0|99.999994|0.134922028|
|exl3|0.01390727051|94.651449|0.008758048527|96.407574|1.22377396|
|exl3_sc4|0.01602494344|94.290870|0.006531513762|96.799475|0.43432045|
|exl3_sc5|0.005815584213|96.574521|0.001413123566|98.105812|0.110237122|
|exl3_bf16head|0.01382460352|94.891828|0.008692796342|96.472889|1.18030357|
|exl3_sc4_bf16head|0.01536106691|94.411063|0.006349782459|97.060740|0.414009094|
|exl3_sc5_bf16head|0.005655915942|96.754813|0.001327212201|98.040491|0.0967941284|

|Arm/head|r01|h03|n01|n02|n03|
|---|---:|---:|---:|---:|---:|
|bf16|0.134922028|0.348146439|7.12586212|3.4269619|0.692676544|
|exl3|1.22377396|-0.061296463|6.29910278|3.15303612|0.338745117|
|exl3_sc4|0.43432045|-0.237369537|7.37737274|0.894439697|0.745185852|
|exl3_sc5|0.110237122|0.759746552|7.48317337|3.44925117|0.0415439606|
|exl3_bf16head|1.18030357|-0.125516891|6.29631424|3.0965004|0.342035294|
|exl3_sc4_bf16head|0.414009094|-0.181009293|7.36112595|0.912340164|0.751501083|
|exl3_sc5_bf16head|0.0967941284|0.708047867|7.50044632|3.42640305|0.0599746704|

Native Wiki SC4/plain ratio 1.15227092435; SC5/plain 0.41816862698. Shared-BF16-head Wiki SC4/plain 1.11113977953; SC5/plain 0.40911957683. Held-out native SC4/plain 0.74577273026, SC5/plain 0.16135141994; shared-head held-out ratios 0.73046488260 / 0.15267954628. SC4 improves the in-domain slice but worsens raw Wiki; SC5 improves both. Do NOT generalize the 1531-token slice to the published full 45930-response-token metric. Native held-out plain 0.008758, SC4 0.006532, SC5 0.001413 are broadly near published 0.0082/0.0062/0.0016 but not an exact reproduction.
Wall command 1666.36 s = 27.7727 min; layer loop + heads/metrics 1544.8 s = 25.7467 min; summed rounded 64-layer body times 1477.9 s. Peak RSS 8.677017 GiB, host MemAvailable minimum 30092 MiB. XPU allocator logical allocated 35.0 GiB/reserved 44.476563 GiB exceeds physical 32 GiB: these are logical allocator statistics, NOT physical-resident VRAM. Possible managed oversubscription [INFERENCE]; physical residency not measured. No OOM, zero new GPU1 reset lines by before/after comparison, owned container removed, postguard clean, rc=0.
ActualmixedKartifactchecksalltorch.equal:plainK4(+headK6),SC4K3/4/5/6,SC5K4/5/6/7;syntheticK3/5/6/7alsoPASS. Bitexactdecodeprovesreconstruction,nottheSCallocationqualityitself.
Rawroot`/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/sc45-full64-safe/`; fullarmmetrics/canarymarginsinmeasurements.json+raw/arms_all.json;all64perlayerphasesandhiddenMSEinraw/layers.jsonl.

Final successful run/current code SHA256: Python f4d915df8c22a8ee22ba5f70f6ab03a1f566992fee0a33ec51f90a6cef051aaa; launcher 2499ff606d5df9d487be5915aa41c3e575a73f4c3ef589cb755700daa9a70d92.

## Full64 per-layer wall table (seconds)
|Layer|Total|plain decode/build|SC4 decode/build|SC5 decode/build|propagate|
|---|---:|---:|---:|---:|---:|
|0|81.5|15.22|30.50|31.38|1.93|
|1|31.0|6.14|15.46|6.63|0.54|
|2|27.9|6.12|6.49|12.39|0.56|
|3|21.3|5.94|6.30|6.40|0.4|
|4|22.2|6.12|6.53|6.73|0.54|
|5|22.1|6.14|6.53|6.57|0.54|
|6|22.0|6.10|6.47|6.58|0.55|
|7|21.2|5.95|6.24|6.59|0.22|
|8|22.1|6.13|6.50|6.64|0.54|
|9|22.0|6.13|6.43|6.61|0.55|
|10|22.0|6.14|6.49|6.52|0.53|
|11|21.1|5.90|6.45|6.30|0.23|
|12|21.9|6.16|6.39|6.52|0.54|
|13|21.8|6.10|6.40|6.50|0.53|
|14|22.0|6.20|6.40|6.54|0.53|
|15|22.9|5.90|7.21|7.28|0.22|
|16|24.4|6.62|6.42|7.57|0.6|
|17|22.0|6.11|6.44|6.54|0.55|
|18|22.3|6.24|6.48|6.73|0.54|
|19|20.9|5.96|6.19|6.35|0.22|
|20|22.1|6.19|6.46|6.69|0.53|
|21|22.1|6.09|6.46|6.64|0.53|
|22|22.1|6.13|6.45|6.73|0.55|
|23|21.1|5.90|6.33|6.46|0.22|
|24|22.4|6.14|6.54|6.90|0.53|
|25|22.1|6.11|6.51|6.64|0.53|
|26|22.1|6.13|6.50|6.68|0.53|
|27|20.9|5.92|6.21|6.36|0.22|
|28|22.1|6.11|6.47|6.64|0.55|
|29|22.1|6.14|6.43|6.67|0.53|
|30|22.1|6.16|6.46|6.63|0.54|
|31|21.0|5.95|6.19|6.45|0.22|
|32|22.0|6.18|6.42|6.60|0.55|
|33|22.1|6.17|6.44|6.60|0.54|
|34|23.1|6.35|6.70|7.10|0.54|
|35|21.3|6.02|6.36|6.49|0.22|
|36|22.2|6.22|6.49|6.60|0.55|
|37|22.0|6.08|6.41|6.64|0.53|
|38|21.9|6.12|6.35|6.56|0.53|
|39|20.9|5.84|6.15|6.43|0.23|
|40|22.1|6.21|6.38|6.66|0.55|
|41|22.1|6.14|6.39|6.69|0.54|
|42|22.3|6.21|6.62|6.62|0.54|
|43|21.0|5.93|6.28|6.39|0.22|
|44|22.0|6.12|6.47|6.56|0.55|
|45|22.0|6.10|6.41|6.58|0.55|
|46|22.0|6.11|6.48|6.61|0.53|
|47|21.0|5.98|6.17|6.42|0.21|
|48|22.3|6.17|6.52|6.73|0.54|
|49|22.0|6.13|6.46|6.58|0.54|
|50|22.2|6.15|6.53|6.64|0.54|
|51|21.0|5.88|6.18|6.49|0.22|
|52|22.1|6.13|6.45|6.67|0.55|
|53|22.0|6.11|6.37|6.69|0.53|
|54|22.3|6.14|6.52|6.86|0.53|
|55|21.1|5.89|6.29|6.47|0.22|
|56|22.5|6.16|6.72|6.68|0.54|
|57|22.4|6.20|6.59|6.69|0.53|
|58|22.3|6.12|6.54|6.76|0.54|
|59|21.2|5.89|6.39|6.42|0.22|
|60|22.2|6.15|6.55|6.61|0.55|
|61|22.2|6.13|6.55|6.63|0.54|
|62|22.1|6.09|6.48|6.69|0.54|
|63|21.2|6.05|6.27|6.44|0.21|

## New RQ-29 K5 XTQ assignment
Two registered arms: xtq_exl3code (exl3_mul1,V1,K4) and xtq_ih2x2 (xe_ih2x2,V2,K4), same harness two-sided RHT with recorded input/output sign seeds; references BF16/plainEXL3. Exact full Viterbi(no margin) planned. BothWiki/heldout chat masks and per-tensor weighted errors reused.
Launch prerequisites: TrellisQuantizer backend='esimd' integration and bit-exact Torch/ESIMD proof on2real tensors. Current quantize API treats U as unit-upper Gram factor, while harness uses U.TU=H_damped^-1; this mismatch sent to owner for correct LDLQ conversion. No silent transpose or launch with wrong factor.
Projected prefix using4.94403584B linear weights per arm: fullmul1 1.05-1.22s/M =>86.5-100.5min;IH2~.85s/M=>70min; factor/LDLQ/calibration overhead10-30min => roughly2h50-3h20 [INFERENCE]. Mainaccepted~3h10. SerialGPU0chosen for both Viterbi arms to avoid unproven GPU1VRAM28GiB ceiling or reintroducing cross-threadXPU lifetime hazards. Main allows GPU0 under whole-runlock<=8GiB; all other harness workGPU1. Serving SC5 checks first if Exl3MixedK ready; coordination messages sent.
Safety updated: no heavy systemd-run--user scopes; Docker/system.slice only with>=30% memory headroom. Harness14gcap versus expected~10GiB peak. Optional dual-lock launcher currently acquiresGPU1thenGPU0, noGPU0activity until backendproof.

