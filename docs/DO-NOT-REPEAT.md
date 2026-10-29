# Do not repeat — closed arms on Arc Pro B60/B70

The canonical ledger of attempts that are **closed**: measured, and not to be re-run without a new reason.
Read this before starting any optimisation, port, or engine swap. The companion rulebook is
[Evidence ledgers](EVIDENCE-LEDGERS.md); accepted wins live in [What worked](WHAT-WORKED.md) and the
[benchmark catalog](BENCHMARK-CATALOG.md).

**A row here is not a claim that the idea is impossible.** It is a record that *this* attempt, on *this* stack,
under *these* limits, produced this outcome — with the cause, so the next person can tell whether their situation
differs.

## How to read a row

| Field | Meaning |
|---|---|
| ID | Stable `DNR-nnn`. Never reuse or renumber. |
| Arm | What was actually tried, specific enough to reproduce. |
| Outcome | What the measurement showed, with the number **and its shape**. |
| Cause | Why it behaved that way. If unknown, write "unexplained" — never guess. |
| Scope | Where the result stops applying. |
| Revisit when | The concrete condition that makes a retry worthwhile. |

## Closed arms

| ID | Arm | Outcome | Cause | Scope | Revisit when | Evidence | Date |
|---|---|---|---|---|---|---|---|
| DNR-001 | Native MTP4 speculative decoding on the later Qwen3.6-35B-A3B MoE stack | Speculation engaged with zero net gain and one net loss at short prompts | Draft cost cancels acceptance at short prompt lengths on this stack | Short-prompt cell only; long loaded context behaves differently | A matched context-by-context sweep shows a positive net cell | [ceiling analysis](qwen36-35a3/MOE-INT4-THREE-ENGINE-CEILING-20260917.md) | 2026-09-17 |
| DNR-002 | DFlash2 as a replacement drafter for the dense GPTQ route | Not ready; target precision and topology changed with the drafter, so any gain is unattributable | Two variables changed in one arm | Dense GPTQ route | A matched quality-and-speed pair with identical target precision exists | [DFlash2 comparison](qwen38-27b/MTP-DEPTH-DFLASH2-KNORM-20260914.md) | 2026-09-14 |
| DNR-003 | Verify-head quantization or mixed GGUF precision as a free speed win | Quality or decode tradeoff in every arm; no free win | Verification precision is part of the arithmetic contract, not a tunable | All routes | Never as a speed forecast; only as a measured quality trade with a parity gate | [drafters review](QUANTIZATION-AND-DRAFTERS.md), [format showdown](qwen38-27b/QUANT-FORMAT-SHOWDOWN-B70.md) | 2026-09-14 |
| DNR-004 | Tensor parallelism as an automatic single-request decode gain (FP8 TP2 and follow-ons) | Break-even at best for one conversation; independent servers win for users | Per-token collectives cost what the split saves at decode-sized messages | B70 pairs over PCIe | A collective path costs well under ~5 µs per exchange at decode size. **2026-10-29 update — external evidence, our measurement pending:** a kernel-direct peer store adds 1.4–3.3 µs per exchange at 4–40 KB and about 6.1 µs inside a real engine, on two B70s, reported by steveseguin/b70-optimization-lab and not reproduced on our hardware | [FP8 TP2](qwen38-27b/FP8-TP2-W8A16.md), [TP2 guide](DUAL-B70-TP2.md), [host notes](HOST-STABILITY-AND-HARDWARE-NOTES.md#h16-peer-access-without-atomics-is-why-cross-device-polling-fails) | 2026-09-14 |
| DNR-005 | llama.cpp RPC split across two cards | Retained only 54% of one card's rate | RPC transport becomes the step's critical path | llama.cpp RPC backend | Never for this stack; re-test only if the transport changes | [FP8 TP2 comparison](qwen38-27b/FP8-TP2-W8A16.md) | 2026-09-20 |
| DNR-006 | Cascadia 0.2.4 pipeline across two cards | Retained ~100% of one card — no gain | Pipeline bubbles absorb the second card's compute | Two-card pipeline on this workload | Independent per-GPU servers are displaced by a measured positive net cell | [what worked](WHAT-WORKED.md) | 2026-09-26 |
| DNR-007 | Intel `0.26.0-b2` image as a speed upgrade | No change: our pinned `0.27.2rc1` already carries the relevant fixes, and the kernels bump measured as a loss | Version number is not a capability | XPU vLLM images | A release note names a fix we actually need | [what worked](WHAT-WORKED.md) | 2026-09-26 |
| DNR-008 | vLLM `0.17` MXFP4 image with 7 local patches | 10.4 tok/s decode, far behind the llama.cpp baseline of the day | Old image plus a Triton path leaving ~40% unused | Superseded image | Never; the image is superseded | [campaign log](CAMPAIGN-LOG.md) | 2026-08-06 |
| DNR-009 | INT4 W4A16 target conversion (arext, local AutoRound, GPTQ) as a quality-preserving speed route | Every arm failed the exact-answer battery (answered 30, correct 14) | Target-weight quantisation changed answers; the drafter was never the problem | 27B class on one card | A W4A16 artifact passes the exact-answer battery first | [what worked](WHAT-WORKED.md), [quantization review](QUANTIZATION-AND-DRAFTERS.md) | 2026-09-26 |
| DNR-010 | Native exllamav3 build on XPU for a 122B MoE checkpoint | Loads and answers correctly but decodes ~2 orders of magnitude below the serving stacks; expert cache unreachable (`--moe_cpu_split` absent on XPU) | Resident fused-MoE kernels are CUDA-only; XPU path streams experts with no cache | exllamav3 native on XPU, any MoE | An XPU fused-MoE kernel with an expert cache exists and is measured | unpublished lab measurement; cause reproducible from the engine source | 2026-10-22 |
| DNR-011 | vLLM EXL3 plugin (`exl3xpu`) for a MoE checkpoint | Cannot load: `get_quant_method` handles only `LinearBase` and `ParallelLMHead`, so `FusedMoE` receives `None` | The plugin implements dense linear only; MoE needs `Exl3FusedMoEMethod` plus a grouped trellis GEMM | Any MoE under the exl3 plugin | A released plugin version registers a MoE quant method | unpublished lab measurement; plugin source inspection | 2026-10-22 |
| DNR-012 | XeStrata serving of a 122B Unsloth IQ3_XXS GGUF on two cards | Refused at MoE layer 0: no expert kernel for the tensor types present; `--layer-split` also unavailable for the model geometry | Two independent gaps: the IQ kernel table covers types 16/17/18/20/21/22/23/29 but not 19 (IQ1_S), and layer split is ported only for the Flash-Next geometry | XeStrata free build, Qwen-MoE geometry | Both an IQ1_S expert kernel and a split path for the geometry ship | unpublished lab measurement; kernel coverage readable in the engine source | 2026-10-19 |
| DNR-013 | Repacking a 122B to W4A16 for vLLM instead of serving the existing GGUF | Do not start: projected ~66.7 GB of weights against ~63 GB usable on two cards leaves no KV, and the conversion needs ~245 GB of BF16 source against 44 GB free | Arithmetic, not judgement: the artifact does not fit the cards it was bought for | Two-card 122B on 32 GB cards | Free space for the source plus a ≤3.5 bpw target artifact exists | [ornith 122B route](ornith15-122b/ORNITH15-122B-LLAMACPP.md) | 2026-10-22 |
| DNR-014 | Reading a model's tensor formats from its filename label (for example "IQ3_XXS") | Two engine ports died on this: the geometry table rejected a type the file actually contains, and the kernel table then had no implementation for it | Mixed quantisation is per tensor; Unsloth UD files carry types the label does not name | Every GGUF consumer, loader, or engine port | Never. Read the tensor-type census and fail closed instead | [benchmark format](BENCHMARK-FORMAT.md) | 2026-10-19 |
| DNR-015 | Quoting a concurrency aggregate as single-stream speed ("150 t/s") | Wrong by an order of magnitude for one user: 153 tok/s was C16 aggregate | Measurement definition error, not a hardware result | Any throughput claim | Never. State prompt length, output length, concurrency, and observer with every rate | [campaign log](CAMPAIGN-LOG.md) | 2026-08-08 |
| DNR-016 | Fusing post-attention RMSNorm with a hand-written W4A16 projection on the 27B decode path (M=4, K=5120, N=17408), third-party: steveseguin/b70-optimization-lab, not reproduced here | The custom two-kernel fused prototype measured 0.2064 ms against 0.1195 ms for production native fused-add RMSNorm plus oneDNN at the same shape, 72.8% slower; it only beat an eager allocating reference at 0.2890 ms | The hand-written W4A16 projection cannot match oneDNN/DPAS, so the boundary fused is too small to pay for losing that GEMM | That shape and design on one B70, ESIMD two-kernel prototype; says nothing about fusing a larger boundary | The projection stays oneDNN/DPAS-backed while a larger boundary is fused without adding a graph split | [Prototype README](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen27_fused_postattn_rms_w4a16/README.md) | 2026-07-11 |
| DNR-017 | Replaying a recorded launch list or command list as a decode speedup (own Xe runtime, one B70, layers 24-47, synthetic finite data), third-party: steveseguin/b70-optimization-lab, not reproduced here | Parity, not a win: 19.86 ms per half token replayed against 19.82 ms submitted one by one, with 1,142 native submissions over 878 logical sites | Submission is not the critical path on that host: his own per-launch host cost measured 0.1-0.2 µs and the half token is device kernel time; serialising work inside one kernel to merge launches measured 5-10 times slower | Command-list replay on a host whose launch cost is already sub-microsecond; graph capture on a slow host is a different mechanism with a different result | A host whose measured launch cost is 5 µs or more, where submission, not kernels, sets the pace | [Scoreboard](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/own-xpu-runtime/SCOREBOARD.md), [packet 34](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/own-xpu-runtime/stage2/packet34-replay-bench/README.md) | 2026-10-12 |

## Recording a new closed arm

1. Reproduce or measure enough to state the outcome **and** its shape (model, quant, engine, image, patch stack,
   cards, context, concurrency, power cap, observer).
2. Identify the cause, or write `unexplained`. A guessed cause is worse than none: it closes a door that may be open.
3. Append the next `DNR-nnn` — never insert in the middle, never renumber.
4. Fill **Revisit when** honestly. "Never" is acceptable for superseded artifacts.
5. Run `python3 scripts/check-repo.py`. The `ledgers` gate validates this table.
6. Link evidence that exists in the repository. If the evidence is a private lab record, say so and publish the
   *cause*, never a private rate.
