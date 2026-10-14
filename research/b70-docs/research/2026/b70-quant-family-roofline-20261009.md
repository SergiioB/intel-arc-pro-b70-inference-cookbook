# B70 quant-family pricing — one card, M=1–4 and prefill

**2026-10-09 · E0 INFERRED bounds, not benchmark predictions or an adopted route.** CPU/static work only. Quality is **UNKNOWN for every new artifact** until separately tested. Arithmetic feasibility does not imply acceptable quality or kernel availability.

## Answer

**INFERRED:** There is no proved universal minimizer. The most economical *simple representation* remains symmetric affine INT4 G128/G64, with W4A16 for inexpensive activation handling and a separate W4A8 implementation for the INT8 compute class. **IQ4_XS** is the first compact nonlinear challenger; **IQ3_XXS/S** are the missing intermediate-bit challengers. IQ2/IQ1 and ternary win only the byte table, not necessarily issue time or quality. MXFP4 is feasible storage, especially for an already-MXFP4 checkpoint such as the user-specified gpt-oss case; lack of native FP4 does **not** prohibit execution, but does prohibit assuming Blackwell-like FP4 throughput. MXFP8 is a quality/capacity option, not an arithmetic decode-speed winner versus comparable INT4.

**MEASURED/current authority:** This is not a recommendation to replace Qwen3.8-27B EXL3 with GPTQ. The [typed route registry](../../registry/model-status.json#L7-L18) records failed W4A16 artifacts and the retained EXL3 quality route. [RL-122](../../registry/research-ledger.json#L1244-L1256) holds EXL3 kernel/image probes; this pricing does not reopen them.

## 1. Hardware, law and corrected evidence

- **SPEC:** 32 Xe2 cores, 256 XMX, 32 GB, **608 decimal GB/s pin**: [hardware reference](../../docs/reference/hardware-software-stack.md#L117-L123). Native DPAS has no FP8/FP4 datatype; W4A16 dequantizes to FP16/BF16 operands, not a native INT4×FP16 operation. See [architecture geometry](b70-xe2-architecture-prefill-analysis-20260807.md#L45-L75), subject to its correction banner.
- **MEASURED, operator microbenchmark / not registered protocol:** INT8 ≈363 TOPS, BF16 ≈170 TFLOPS, counting MAC=2 operations: [rate-class authority](xmx-dpas-rate-classes-20260915.md#L25-L61). Sub-8 weights with A8 do not multiply TOPS. Pure A4×W4 instruction-efficiency results there are not a quality-viable application peak.
- **MEASURED, historical diagnostic:** [route lessons §3](b70-format-route-lessons-20260930.md#L53-L67) reports ~336 instructions/512 B tile group for trellis state/codebook/moves. This is **0.65625 counted instructions/logical packed byte**, not a calibrated universal issue-slot rate. Branches, SIMD width, instruction dependencies and sends matter. At idealized 13.5 GB: **8.859 G counted instructions/pass**; ~8.6 G in the lesson uses a different actual footprint.
- **MEASURED, correction of scope:** The refreshed raw qkv control is **379.713 GB/s**, versus B_FAKE **636.446 GB/s**, −40.345% latency; down/gate B_FAKE are 474.14/521.27 GB/s. All B_FAKE numerical checks fail. See raw ablation (`results/2026/09/b70-m4-flag-ablation-20260930T143631Z/ABLATION-RESULT.json`). The rounded **381 GB/s** from [lessons §0](b70-format-route-lessons-20260930.md#L13-L24) is an illustrative route anchor, not whole-model sustained DRAM bandwidth. **636 cannot be used as a physical pin ceiling**: numerator is logical weight bytes and the invalid ablation changes work; cache/traffic/timing need counters. 381/608=62.66%; 37.34% is the gap to pin, not a measured whole-request removable share. 1−381/636=40.09% is the rounded ablation latency difference.
- **INFERRED correction:** The lessons' 27B/4bpw/608/yield3 table says ~111 tok/s, but **3×608/13.5=135.1**. 111 requires 16.43 GB/round or499.5 GB/s. The [breakthrough note's conditional EXL3 round budget](b70-breakthrough-architect-20260928.md#L26-L37) uses **16.68 GB/round**, including its assumed head/MTP/embedding traffic: at yield3 this gives **109.4 tok/s**, at yield2.65 **96.6≈97**. This explains the scale of111, but is a **different byte convention**, not the stated13.5GB weights-only arithmetic. The older note's chunk/round timings are explicitly unverified and not reused here. Its approximate component prose is not an exact traffic census.

| INFERRED convention, 608GB/s | GB/step | y1 ceiling | y3 ceiling | What it means |
|---|---:|---:|---:|---|
| Uniform27B, exactly4bit, weights only | 13.500 | 45.0 | 135.1 | Main table convention; no draft/head exceptions/KV |
| Historical EXL3 conditional round | 16.680 | 36.5 | 109.4 | Same fixed round cost at hypothetical yields1/3, not a prediction for actual M=1; named head/MTP/embedding assumptions, no added KV here |

For new formats **do not multiply16.68 by bpw/4 indiscriminately**. If only the historical12.230GB body changes4→b bits and4.450GB remaining traffic stays fixed, `Dround(b)=12.230*(b/4)+4.450`. At b=3,2.5,2: D=13.6225,12.09375,10.565GB; pin ceilings y1/y3=44.6/133.9,50.3/150.8,57.5/172.6. These are **conditional round sensitivities**, separate from the all-weight rows below; actual group metadata/mixed precision and changed draft acceptance must be re-censused. Body-only4→3 ideal speedup=16.68/13.6225=1.224, not4/3.

**INFERRED model (all omitted work is optimistic):**

```
D_step = W_target_union + k*W_draft_reread + D_KV + D_activations + D_temporaries
I_step = sum(dynamic instructions weighted by relevant issue-resource cost)
t_step >= max(D_step/B_effective, I_step/R_issue, 2*P_active*M/P_rate) + serial
rate <= yield/t_step; yield = 1 + sum(per-position prefix survivals)
```

The proposed `bytes/token × issue-slots/byte` equals `issue-slots/token`: it discards the independent memory floor. **Optimize the maximum of the byte and issue floors**, under quality/capacity constraints. R_issue is unmeasured for the candidate mixtures, so a numeric dequant-adjusted ceiling would be fabricated. [Diagnostic playbook](../../docs/methods/architecture-diagnostic-playbook.md#L61-L135) supplies the accounting.

## 2. Compact weight table

**INFERRED assumptions for every numeric cell:** exactly 27.0B dense active weights, exactly 3.0B active of a fully resident 30.0B MoE; uniform listed storage including block scales where defined; no embedding/head exceptions, padding, KV, drafts, launch, activation or output traffic. GB=10^9 B. Values are **GB per target pass at M=1**, hence GB/token at yield1. With ideal verify reuse, divide GB/pass by yield for emitted-token bytes. Dense resident=listed dense GB; **MoE resident=10×active GB**, never 3B-sized capacity.

Rate columns are **pin-only upper bounds**, tok/s **yield1 / yield3**. Yield3 requires M≥3 (usually M=4 with three drafts); impossible at M=1 or M=2 in this accounting. M=4 expert union may multiply the MoE active bytes by up to 4 (shared weights reduce this extreme), lowering yield3 ceilings up to 4×. Generic shared+expert accounting is `D(M)=Dshared+Dunion(experts across M rows)`. The lessons' 1.5–2 GB MoE estimate rather than ideal 1.5 GB at 4bpw is another 1–1.333× byte multiplier.

**Issue/path legend — INFERRED, not measured counts:**
- **L:** affine unpack/sign/scale; lowest-complexity candidates. **B:** bitplane/packed-scale assembly. **S:** scale-vector/conversion work. **U:** codebook LUT + signs/shuffle (IQ4 small scalar LUT; IQ1/2/3 vector grids). **T:** trellis/state/moves/rotation, observed high cost. **X:** radix conversion or sparse residual/transform overhead.
- **D:** dp4a-feedable with quantized A8 and suitable integer values, not with A16; an XVE/vector route, not automatically XMX. **I:** INT8-class DPAS after required decode and A8 quantization, group scaling/corrections. **F:** FP16/BF16 DPAS after dequantization. `I?` means a designed path, not proven installed dispatch. All XMX-feedable entries still require actual M/shape dispatch proof. No format name alone guarantees XMX at M=1.
- **F/Q?** physical representation feasible; exact quality/loader/performance NEEDS-MEASUREMENT. **P** payload-only lower bound; additional metadata/residuals increase bytes. **Mix** not a single tensor format.

| Format | bpw | GB/pass dense / MoE-active | Dense ceiling y1 / y3 | MoE ceiling y1 / y3 | Issue; decode feed → prefill class | Physical verdict |
|---|---:|---:|---:|---:|---|---|
| GPTQ sym INT4 G128, W4A16 | 4.125 | 13.922 / 1.547 | 43.7 / 131.0 | 393 / 1179 | L; F → F | F/Q?; known 27B artifact quality rejection |
| GPTQ sym INT4 G64, W4A16 | 4.25 | 14.344 / 1.594 | 42.4 / 127.2 | 381 / 1144 | L; F → F | F/Q?; MoE historical route anchor |
| GPTQ G128/G64, **W4A8** | 4.125 / 4.25 | as above | as above | as above | L+activation quant/group partial scales; D/I → I | F/Q?; A8 error additional; no extra INT4 TOPS |
| IQ4_XS | 4.25 | 14.344 / 1.594 | 42.4 / 127.2 | 381 / 1144 | U-small+B+S; D/I? or F → I?/F | F/Q?; compact nonlinear first challenger |
| IQ2_XXS | 2.0625 | 6.961 / 0.773 | 87.3 / 262.0 | 786 / 2358 | U-grid+sign+S; D/I? or F → I?/F | F/Q?; quality unknown |
| IQ2_XS | 2.3125 | 7.805 / 0.867 | 77.9 / 233.7 | 701 / 2103 | U-grid+B+S; D/I? or F → I?/F | F/Q? |
| IQ2_S | 2.5625 | 8.648 / 0.961 | 70.3 / 210.9 | 633 / 1898 | U-grid+B+S; D/I? or F → I?/F | F/Q? |
| IQ2_M recipe | **b_mix** | 3.375b / 0.375b | 180.15/b / 540.44/b | 1621.33/b / 4864/b | mixed tensor dispatch | Mix: header census required; not an IQ2_M block ABI |
| IQ1_S | 1.5625 | 5.273 / 0.586 | 115.3 / 345.9 | 1038 / 3113 | U-grid+shift/correction+S; D/I? or F → I?/F | F/Q?; not simple binary signs |
| IQ1_M | 1.75 | 5.906 / 0.656 | 102.9 / 308.8 | 926 / 2779 | U-grid+B+shift/correction+S; D/I? or F | F/Q? |
| TQ1_0 ternary | 1.6875 | 5.695 / 0.633 | 106.8 / 320.3 | 961 / 2882 | X-radix3+S; D/I? or F → I?/F | F/Q?; ternary-trained/QAT checkpoint preferred hypothesis, not proof |
| TQ2_0 ternary | 2.0625 | 6.961 / 0.773 | 87.3 / 262.0 | 786 / 2358 | L/B two-bit decode+S; D/I? or F → I?/F | F/Q?; no automatic sparsity TOPS |
| MXFP4 E2M1, E8M0 G32 | 4.25 | 14.344 / 1.594 | 42.4 / 127.2 | 381 / 1144 | U-small/S float expand; F, or D/I? exact integer recode → F/I? | F/Q?; **native FP4 acceleration rejected** |
| MXFP8 E4M3/E5M2, E8M0 G32 | 8.25 | 27.844 / 3.094 | 21.8 / 65.5 | 197 / 590 | S software FP8 expand → F | F/Q?; native FP8/INT8-rate assumption rejected |
| NVFP4 E2M1, 8-bit G16 scale | 4.5+tensor scale | ≥15.188 / ≥1.688 | ≤40.0 / ≤120.1 | ≤360 / ≤1081 | U-small+S; F, or D/I? integer recode → F/I? | F/Q?; native FP4 rejected |
| Q4_K block | 4.5 | 15.188 / 1.688 | 40.0 / 120.1 | 360 / 1081 | B+affine min+S; D/I? or F → I?/F | F/Q? |
| Q4_K_XL recipe | **b_mix** | 3.375b / 0.375b | 180.15/b / 540.44/b | 1621.33/b / 4864/b | mixed Q4/Q5/Q6/Q8 etc. | Mix: cannot price XL as exactly 4bpw |
| Q5_K block (not Q5_K_M recipe) | 5.5 | 18.562 / 2.062 | 32.8 / 98.3 | 295 / 884 | B high-bit merge+S; D/I? or F → I?/F | F/Q?; spends bytes for quality |
| Q6_K | 6.5625 | 22.148 / 2.461 | 27.5 / 82.4 | 247 / 741 | B planes+S; D/I? or F → I?/F | F/Q?; spends bytes for quality |
| Q8_0 | 8.5 | 28.688 / 3.188 | 21.2 / 63.6 | 191 / 572 | L scale only; D/I or F → I/F | F/Q?; MoE resident 31.875 GB, tight capacity |
| EXL3 4bpw nominal | 4+overhead | ≥13.500 / ≥1.500 | ≤45.0 / ≤135.1 | ≤405 / ≤1216 | T≈336/512B anchor; F → F (I prefill requires extra route) | P; retained 27B quality evidence, not all models |
| EXL3 3bpw nominal | 3+overhead | ≥10.125 / ≥1.125 | ≤60.0 / ≤180.1 | ≤540 / ≤1621 | T, cost at 3bpw unpriced → F | P; measured prior 3bpw route slower, not a byte-law violation |
| EXL3 2.5bpw nominal | 2.5+overhead | ≥8.438 / ≥0.938 | ≤72.1 / ≤216.2 | ≤649 / ≤1946 | T + mixed allocation possible → F | P; quality/issue unknown |
| EXL3 2bpw nominal | 2+overhead | ≥6.750 / ≥0.750 | ≤90.1 / ≤270.2 | ≤811 / ≤2432 | T, not same cost/byte assumed → F | P; quality/issue unknown |
| QuIP#-class lattice 2 / 3 bit | 2 / 3 +overhead | ≥6.750 / ≥0.750; ≥10.125 / ≥1.125 | ≤90.1/270.2; ≤60.0/180.1 | ≤811/2432; ≤540/1621 | U-lattice+transforms; F → F, I requires separate recode | P; no proof LUT is cheaper than trellis |
| SqueezeLLM-class scalar nonuniform 3 bit | 3+LUT+outliers | ≥10.125 / ≥1.125 | ≤60.0 / ≤180.1 | ≤540 / ≤1621 | U-scalar + X-sparse residual; F → F | P; **not** same codec as QuIP#; 2bit only hypothetical here |
| **IQ3_XXS** (added) | 3.0625 | 10.336 / 1.148 | 58.8 / 176.5 | 529 / 1588 | U-grid+sign+S; D/I? or F → I?/F | F/Q?; useful intermediate-bit challenger |
| **IQ3_S** (added) | 3.4375 | 11.602 / 1.289 | 52.4 / 157.2 | 472 / 1415 | U-grid+B+S; D/I? or F → I?/F | F/Q? |
| Affine W2 G128 sym (added) | 2.125 | 7.172 / 0.797 | 84.8 / 254.3 | 763 / 2289 | L; D/I? with A8, F with A16 | F/Q?; low-bit quality unknown |
| Affine W3 G128 sym (added) | 3.125 | 10.547 / 1.172 | 57.6 / 172.9 | 519 / 1556 | B, no native INT3; unpack to I/F | F/Q?; existing W3 escalation remains closed |

**SPEC/source arithmetic:** GGUF tensor layouts from the [ggml primary header](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h), inspected 2026-10-09 (mutable source, not a pinned deployment). Examples: IQ4_XS=136 B/256; IQ2_XXS=66/256; IQ2_XS=74/256; IQ2_S=82/256; IQ1_S=50/256; IQ1_M=56/256; TQ1=54/256; TQ2=66/256; Q4_K=144/256; Q5_K=176/256; Q6_K=210/256; Q8_0=34/32. [Retained prior architecture review](quant-format-showdown-20260913/ARCHITECT-VERDICT.md#L35-L46) corroborates the higher-bit bpw. IQ2_M and XL are tensor allocation recipes, so use `sum(n_tensor*bpw_tensor)/sum(n_tensor)`, separately weighted for active experts. Q5_0 is 5.5bpw, Q5_1 is 6bpw; do not conflate Q5 variants.

**INFERRED metadata assumptions:** GPTQ sym uses one FP16 scale/group: bpw=4+16/G. Packed four-bit zero-points add 4/G bpw when actually stored (G128 becomes 4.15625, G64 4.3125); g_idx, padding, FP16 exclusions and export layout are extra. AWQ/HQQ/AutoRound can export similar affine storage; calibration name is not an issue-cost class. Rotated affine W4 adds transform overhead but can improve quality at near-iso-bytes; it does not automatically shrink bytes. MXFP8 uses one E8M0 byte per32 payload bytes. NVFP4 assumes one 8-bit scale/16 plus any tensor/global scale; exact layout is checkpoint-dependent.

## 3. Instruction economics and M regimes

**INFERRED cheap-path ranking, not measured instruction totals:** Q8/affine integer unpack has no codebook gather; IQ4 needs a 16-entry nonlinear LUT, scale assembly, then integer or FP operands; IQ1/2/3 need vector-grid decode, signs and sometimes shift corrections; EXL3 adds trellis state and transforms. However LUT placement in GRF versus SLM/L1, integer recoding and vectorization can reverse close rankings. No numerical `instructions/512 B` is assigned to uncompiled codecs.

**INFERRED DPAS counting:** one SG16 DPAS at RC=M≤4 covers 16×K weights reused over M rows. K=16 for F and K=32 for W4A8. Per 1024 weights, ideal **4 F-DPAS vs 2 I-DPAS instructions**, independent of M while RC encodes the rows; **this is an instruction count, not equal execution cycles across repeat counts**. At raw4bit: 512B/1024 weights, hence F=1/128 DPAS instructions/B and I=1/256; add unpack, group-scale partial accumulation, activation quantization, operand moves, loads/stores and padding. The 336-instruction trellis group is a particular kernel census; its denominator does not establish a format-independent count. dp4a counts require matching SIMD width too.

**INFERRED overlooked MXFP4 path:** E2M1 magnitudes `{0,.5,1,1.5,2,3,4,6}` ×2 give `{0,1,2,3,4,6,8,12}`. Signed INT8 can represent them **exactly**. Decode 4-bit code→signed8 via tiny LUT, perform A8×W8 DPAS, apply block scale/2 to each group partial. Same for NVFP4's E2M1 values. This avoids native FP4 dependency but does **not** permit signed4 values (12 exceeds7), and group scaling prevents a single unscaled full-K accumulation. A8 quantization, LUT/sign work, per-group scale costs, saturation and kernel availability remain open. For gpt-oss MXFP4 weights this avoids an additional weight requantization in principle; quality of changed activations/rounding is still unknown.

**INFERRED M=1–4:** arithmetic intensity for exactly4bit weights is ~4M operations/B =4–16, well below ~280 BF16 or ~597 INT8 operations/B ridge at the cited rate anchors. Thus theoretical peak compute does not decide small-M winner; XVE GEMV may beat DPAS setup at M=1–2. M=3–4 reuse is valuable only when actual kernels reuse weights, scales and decoded tiles. MoE can lose that reuse to routing divergence, dispatch, host sync and insufficient per-expert M.

**INFERRED prefill:** full weight reuse makes F-class compute ceilings ~3148 dense / 28,333 ideal A3B tokens/s, versus I-class ~6722 / 60,500, before attention, routing, unpack and occupancy. These are compatible-class operator-anchor extrapolations, not measured prefill or guaranteed peaks. At4bit the ridge is M≈70 for F and≈149 for I; in MoE this is **rows per expert**, not global batch. All listed formats can feed XMX *after decoding*, but only affine integer plus A8 directly targets the integer operand class; codebook/float storage is not itself XMX-native. Prefill dequantize-once staging can amortize issue cost but adds writes/reads/VRAM. Expanding a whole27B/30B model to BF16 would require54/60GB, rejected on one card; tiled expansion is feasible.

**INFERRED contention:** affine and microscale G16/G32 incur more scale partials than G128; LUTs consume GRF/SLM/L1 and extra sends; trellis/rotations add live registers and synchronization. 256-GRF mode trades occupancy for fewer spills. Unified L1/SLM192KB is not a blanket192KB user-SLM allocation budget. Decode tables/predecode buffers must fit queried allocation limits and leave occupancy. 18MB L2 cannot hold billions of packed weights. Persistent32-WG grids do not prove occupancy or safe global barriers.

**INFERRED break-even versus idealized EXL3@381, same quality/yield:** candidate must achieve `B_c >381*(bpw_c/4)` to beat EXL3's weight-pass time. IQ4_XS/G64/MXFP4: **404.81 GB/s**; G128: **392.91**; Q4_K/NVFP4: **428.63**; IQ3_XXS: **291.70**; IQ2_XXS: **196.45**. To clear a10% rate gate multiply these by1.10. This is a deciding threshold, not a predicted bandwidth. Q8_0 would require809.63 GB/s, beyond pin: **rejected as a same-yield weight-only speed replacement** under this baseline; it may still win quality or remove other overhead.

## 4. KV is a separate axis

**INFERRED geometry-normalized table:** Let `E=2*L_full_attention*n_kv_heads*head_dim` cached elements per occupied token (K+V equal here), `C=context`, `r=number of physical cache scans/verify round`. Bytes added per round ≈`r*C*E*b_element` plus writes, scale metadata and block slack. For illustration only set E=32768, so8bit payload is32KiB/context-token, as in the user-specified qwen3_5-class approximation; **not a universal27B/MoE geometry**, and hybrid recurrent state is separate. The [ledger's RL-085 excerpt through situation compiler](../../registry/research-ledger.json) reports larger actual engine pool accounting; use engine metadata for capacity.

| KV format | Stored B/element | Example KiB/context-token, K+V | Issue class / execution | Physical verdict |
|---|---:|---:|---|---|
| q8_0 both K/V | 34/32=1.0625 | 34 | integer scale+convert; I attention needs A8; otherwise F | feasible if attention kernel supports; quality unknown on new artifact |
| q4_1 both | 20/32=.625 | 20 | nibble/min/scale + conversion/correction | feasible; K quality **not established** by weight results |
| established q8_0 K / q4_1 V | .84375 average | 27 | separate K/V dequant | fleet policy, not universal parity proof |
| fp8 | 1+scales | ≥32 | software unpack/convert to F; **no FP8 DPAS** | feasible in supported engine, quality/scale dependent |
| int8 | 1+scales | ≥32 | I only with compatible quantized query/probabilities, else F | not interchangeable with fp8 despite same payload bytes |
| iQ4 KV (if IQ4_XS layout) | .53125 | 17 | LUT+scale+attention integration | hypothetical layout/support, quality unknown |
| iQ4 KV (if IQ4_NL layout) | .5625 | 18 | LUT+scale | distinct from XS; “iQ4 KV” alone underspecified |

**INFERRED example:** C=32768 gives1.074GB fp8 payload per scan, 0.906GB mixed q8/q4_1, 0.570GB IQ4_XS. C=131072 gives4.295/3.624/2.282GB. Add to each weight row before using608/(W+KV), or3×608/(W_union+rKV+kDdraft). If M rows independently scan KV, r can approach M; only a fused shared read justifies r=1. KV quantization reduces resident bytes too but does not guarantee those savings convert to bandwidth or quality gains. At long context, this term can dominate the A3B MoE's ~1.5GB weights.

## 5. Physical verdict and handoff

**FEASIBLE / NEEDS-MEASUREMENT:** affine INT4, IQ4/IQ3, microscaled FP4 software decode, all other listed codecs as representations. Their byte floors fit the law; no arithmetic proves their quality. IQ1/2/ternary are not rejected solely for low bits. TQ1 has18.18% fewer bytes than TQ2 but can lose on radix decode; measured effective bandwidth must retain >81.82% of TQ2's for a weight-floor speed win. Q8 MoE31.875GB weights is tight, not a mathematical OOM proof: 32GB marketing versus recorded32,656MiB visible differs; runtime/KV/live-free budgets decide. No deployment authorized.

**INFEASIBLE claims:** native FP4 or FP8 DPAS acceleration; INT4×A8 gives2×INT8 TOPS;636 as sustained pin bandwidth; unconditional3× MoE speculation; arbitrary quality preservation; model-wide BF16 expansion on32GB; comparing quant formats by nominal bits alone.

**INFERRED cheapest falsifier, handed off not run:**
1. **b70-isa-analyst:** source/header census of one exact artifact per shortlisted family, active/resident tensor bytes, export layout, generated ISA already available (no new build in this task). Match logical512B and decoded1024-weight denominators; count DPAS separately from ALU/shuffle/send; record GRF/spills, activation quant and scale partials. Stop if missing SYCL/model/attention dispatch or load-time expansion erases byte saving.
2. **b70-bench-engineer / b70-profiler, only authorized future window:** same-source affine4 G128 vs IQ4_XS first; IQ3_XXS second only if quality passes. Use dense N,K representative matrices and actual MoE expert shapes at M=1,2,3,4 and prefill M=128/512; keep artifact/layout/cap/clock regime fixed per comparison, graph replay where supported, one shape warmup, interleaved three-pair screen, five fresh confirmation pairs. Count physical DRAM traffic, total pass time, instruction issues/spills and MoE distinct expert unions. A candidate must clear **≥10% whole-task improvement**, not just operator speed, with exact-format correctness and separately frozen held-out quality budget. At equal quality/yield, use the break-even bandwidths above only as preliminary floors; if removed wall-time share prices below10%, stop.
3. **Quality gate before speed adoption:** canonical BF16/retained quality route versus candidate, held-out logit/KL + representative exact-answer, multi-turn and long-context checks; W4A8 separate A8 ablation. Numeric quality tolerance must be predeclared by lead/model owner, not invented here. No new candidate is quality-approved by this report. For ternary/1bit, require a compatible trained/QAT artifact or stop at quality before any kernel investment.
4. **b70-speculation-engineer:** measure per-position survivals and draft reread bytes jointly; use y=3 only as sensitivity until actual round yield/work is known. Do not maximize acceptance independently of round time.

### Specialist Finding: B70 Hardware Architect
- **Verdict:** FEASIBLE — shortlist is physically viable; performance/quality winner remains NEEDS-MEASUREMENT.
- **Hardware Bottleneck:** GDDR6 bandwidth versus dequant instruction issue; small-M MoE adds launch/host synchronization floors, not necessarily DPAS saturation.
- **Proof & Calculation:** G128 affine4=13.922GB dense/1.547GB active MoE; IQ4_XS=14.344/1.594; IQ3_XXS=10.336/1.148. Pin ceilings y1/y3=43.7/131.0,42.4/127.2,58.8/176.5 dense respectively. These exclude draft/KV/union/issue costs.336/512=.65625 counted instructions/B; unknown issue throughput prevents invented realized-rate columns.
- **Evidence Artifacts:** this report; [CPU checker](b70-quant-family-roofline-20261009.py); linked authorities and raw diagnostic above.
- **Gate Recommendation:** source/ISA/capacity + held-out quality first, then one authorized paired IQ4/affine experiment; no held EXL3 probe restart.

**Checks/outcome:** research next and corrected situation compile return halt=false; selected portfolio question remains RQ-10 and execution_authorized=false. Static price example and custom Python assertions pass, shell exit0. One initial compile invocation failed with invalid intent `architecture` (exit2), corrected to supported `research`; no provider/tool outage. No GPU, server, build, model download, delete, commit or publication. Only this report and CPU checker created. Unverified: pinned new-format dispatch, dynamic issue counts, sustainable bandwidth, expert-union reuse, exact model sizes, new-artifact quality, live VRAM and energy. External header was inspected as mutable primary-source data, not a pinned compatibility claim.

## Addendum — ISA/layout census cross-check (2026-10-09, later; isa-analyst session)

- **TQ1_0 dispute RESOLVED in favor of the table:** the isa-analyst census first reported 56 B (qs[50]) but its own macro arithmetic and the pinned header disagree with that. Authoritative ggml-common.h @ `50e3e3e480c80b9ee0ea067e475602d540dc62c8` defines `qs[(QK_K - 4*QK_K/64)/5]` = 48 B + `qh[4]` + `d(2)` = **54 B = 1.6875 bpw** (upstream comment: "// 1.6875 bpw"; static_assert matches). The 54 B/1.6875 table entry above is CONFIRMED; the interim 56 B correction is withdrawn. TQ1_0 remains SYCL-blocked (CPU fallback) regardless. Same fetch re-confirms: IQ4_XS 136 B/256, IQ3_XXS 98 B/256, IQ3_S 110 B/256, Q4_K 144 B/256, Q8_0 34 B/32, TQ2_0 66 B/256, Q6_K 210 B/256, IQ2_XXS 66 B/256, IQ2_XS 74 B/256, IQ2_S 82 B/256, IQ1_S 50 B/256, IQ1_M 56 B/256, MXFP4 17 B/32 = 4.25, NVFP4 36 B/64 = 4.5 + scales.
- **Integer-DPAS absence confirmed at ISA level:** all emitted DPAS in the retained EXL3 M=4 ablation disassembly operate on `:hf` (FP16) operands (e.g. `dpas.8x4 (16|M0) r88:f r88:f r51:hf r33.0:hf`); integer DPAS (`:s4`/`:s8`) count = 0. llama.cpp SYCL runs MMVQ `__dp4a` integer dots vs Q8_1 activations (MMQ hard-disabled for accuracy); Strata dequants to FP16 buffers then oneMKL FP16 GEMM (`dpas.8x4:hf`). No engine on this stack feeds XMX integer DPAS with quantized weights today — the §1 `I`-class rows are designed paths, none installed.
