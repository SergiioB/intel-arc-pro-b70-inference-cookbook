# Situation compile

Generated view — not numeric authority. Do not copy tok/s from this file.

intent: research
goal: E0 invention of byte-level B70 quantization formats; CPU static only
halt: false
changed_variable: —
skill: —
budget_files: 12

blocked:
- RL-058
- RL-059
- RL-062
- RL-063
- RL-064
- RL-072
- RL-095

prior_evidence (history, not a replication exemption):
- RL-103 [blocked]: Does the pinned exl3xpu image support stock TP2 for the incumbent EXL3 4bpw Qwen3.8-27B route, and which transport would it use?
  evidence: research/2026/b70-exl3-tp-feasibility-upstream-20260929.md
  evidence: research/2026/b70-exl3-tp-decision-20260929.md
- RL-013 [failed]: Quantization power sweep — FP8/INT4 engine sweep 150–230W
- RL-121 [validated]: External route assessment: do Strata v0.1.40.2 (SYCL engine, official Arc support) and the sybil-solutions Flash-Next B70 offload build beat our Flash-Next placement, and what transfers to the dense champion?
  evidence: research/2026/flashnext-external-routes-assessment-20261007.md
- RL-107 [provisional]: TaskB Gate0: does the W4A16 30-defect flip class generalize to heldout low-margin decision positions (step-0 census), and does the ADVANCE verdict survive sequence-level (whole-answer-string) scoring?
  evidence: results/2026/09/b70-margin-requant-gate0-20260929T204352Z/RESULTS.md
  evidence: results/2026/09/b70-margin-requant-gate0-seqsupp-20260929T212625Z/RESULTS.md
  evidence: research/2026/b70-margin-requant-gate0-decision-20260929.md
  evidence: research/2026/b70-margin-requant-gate0-census-20260929.md
  evidence: research/2026/b70-margin-requant-splice-design-20260929.md
- RL-106 [provisional]: Does the pinned plugin's opt-in pruned draft head (EXL3_DRAFT_VOCAB + _build_draft_head + _patch_mtp_draft_logits) cut the 2.86 GB/round shared-lm_head draft re-read, and is it quality/output-preserving?
  evidence: research/2026/b70-exl3-round-attribution-audit-20260929.md
  evidence: research/2026/b70-exl3-tp-decision-20260929.md
- RL-099 [validated]: K=3 P3 excursion root-cause: why did one dense cell (k=6144->5120, M=8, bf16, seed 1701) fail at abs 0.125 vs ~0.07 bound while 50+ 'identical' re-executions ran clean?
  evidence: research/2026/b70-k3-exl3-extension-20260929/PATCH-SERIES.txt
- RL-095 [validated]: K=3 EXL3 kernel extension: author the complete reviewable patch series (build enablement, template repair, strip selection, owned correctness test, A/B spec, K-aware loader guard) with static validation only?
  evidence: research/2026/b70-k3-exl3-extension-20260929
- RL-091 [validated]: What do the pinned engine trees actually support for the three new-topology candidates (expert-parallel placement, GDN constant-state exploitation, flag-slot copy-based sync), per file:line?
  evidence: research/2026/b70-engine-capability-audit-20260929.md
- RL-072 [validated]: O13: what does the 63-67% body bucket contain seen from Level Zero under graph replay, and does the thermal fix work in practice?
- RL-064 [validated]: Is max-num-batched-tokens / chunked-prefill size a C1 cold-input (TTFT) lever for Qwen3.8-27B on vllm-xpu at p8192/g1?
  evidence: results/2026/09/b70-improvement-probes-20260920/p4-qwen27-prefill-knob-screen-20260925T093105Z
- RL-063 [validated]: Does the merged vLLM #56096 fused QK-norm+RoPE+gate dispatch apply to the Qwen3.8-27B champion route (qwen3_5) on XPU?
  evidence: results/2026/09/b70-improvement-probes-20260920/P3-56096-DISPATCH-GATE-20260925.md
- RL-062 [validated]: Do L-B (weight-stationary draft fusion) or L-C (exact verify-head kernel) clear the >=5% Amdahl gate against MEASURED decode shares on the retained champion stack?
  evidence: results/2026/09/b70-improvement-probes-20260920/p2-qwen27-eager-composition-20260925T085704Z/RESULTS.md
- RL-059 [validated]: Does the ASUS ROG STRIX X570-F dual Intel Arc Pro B70 topology support Level Zero direct peer memory access and bidirectional peer transfers?
  evidence: results/2026/09/b70-improvement-probes-20260920/p2-levelzero-p2p-probe-20260924T194500Z/probe_output.txt
- RL-058 [provisional]: After D11, which Qwen3.8-27B C1 improvement deserves the next bounded probe, and which proposed gains or mechanisms are unsupported?
  evidence: research/2026/exl3-analysis-and-next-inference-levers-20260924.md
  evidence: results/2026/09/b70-improvement-probes-20260920/d11-stitched-fastmtp1500-probe-20260924T172339Z

working_set:
- role: situation
  tree: b70-docs
  path: registry/research-ledger.json
- role: situation
  tree: b70-docs
  path: registry/model-status.json
- role: situation
  tree: b70-docs
  path: registry/research-program.json
- role: orientation
  tree: b70-docs
  path: research/README.md
- role: kernel
  tree: b70-docs
  path: DOCUMENT-MAP.md
- role: system
  tree: b70-docs
  path: docs/governance/agent-operating-system.md
- role: decision
  tree: b70-docs
  path: research/2026/b70-k3-exl3-extension-20260929
- role: decision
  tree: b70-docs
  path: results/2026/09/b70-improvement-probes-20260920/p4-qwen27-prefill-knob-screen-20260925T093105Z
- role: decision
  tree: b70-docs
  path: results/2026/09/b70-improvement-probes-20260920/P3-56096-DISPATCH-GATE-20260925.md
- role: decision
  tree: b70-docs
  path: results/2026/09/b70-improvement-probes-20260920/p2-qwen27-eager-composition-20260925T085704Z/RESULTS.md
- role: decision
  tree: b70-docs
  path: results/2026/09/b70-improvement-probes-20260920/p2-levelzero-p2p-probe-20260924T194500Z/probe_output.txt
- role: decision
  tree: b70-docs
  path: research/2026/exl3-analysis-and-next-inference-levers-20260924.md

do_not:
- RL-058 Do not retry D11/D12/D14 or unchanged failed depth/verify-head/DFlash2/wheel hypotheses; do not promote inferred 538.7 GB/s utilization, QuaRot-to-140 tok/s, IQ3-MoE-to-dense-Qwen prefill, or IPC-to-160-190 tok/s claims. No INT3 conversion without executable packing/kernel and quality gates; no TP2 or peer-force settings from literature alone. A negative result requires a changed mechanism to reopen.
- RL-059 Do not assume hardware remote atomics exist across PCIe; ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS is absent. Any oneCCL collective implementation must use copy-based synchronization. Do not enable CCL_TOPO_P2P_ACCESS=1 or launch TP2 servers without separate authorization under ADR-0008.
- RL-062 Kernel hunting closed for this stack: do not commission L-B/L-C or any target GEMM rewrite without unitrace (or equivalent Level Zero) evidence of real redundant draft re-reads or off-floor kernels (RL-060). Do not use eager-mode kernel shares as champion headroom or as speed claims.
- RL-063 Do not re-screen #53989/#56096 fused QK-norm-RoPE-gate on the qwen3_5 XPU route; reopening requires an upstream qwen3_5 XPU implementation of the fused path.
- RL-064 Do not re-sweep max-num-batched-tokens for C1 p8192 TTFT without a different question (concurrency, longer context, or a new engine); the three-way screen is a documented non-lever.
- RL-072 Single-card decode kernel hunting is CLOSED at every level (bytes floor + rate ceiling + L0-visible composition). For tracing: use unitrace launcher mode (never LD_PRELOAD into vllm); expect ~35% overhead and docker-stop truncation (repair the JSON tail).
- RL-095 D3 decision pending (architect); image build + P3 GPU window gated behind it

warnings:
- RL-058: descriptive adoption guard requires manual review
- RL-059: descriptive adoption guard requires manual review
- RL-062: descriptive adoption guard requires manual review
- RL-063: descriptive adoption guard requires manual review
- RL-064: descriptive adoption guard requires manual review
- RL-072: descriptive adoption guard requires manual review
- RL-095: descriptive adoption guard requires manual review
- Prior evidence is scoped history, not an automatic replication exemption or an adopted route.
- Working set limited to 12 files; remaining sources are listed in prior_evidence.
