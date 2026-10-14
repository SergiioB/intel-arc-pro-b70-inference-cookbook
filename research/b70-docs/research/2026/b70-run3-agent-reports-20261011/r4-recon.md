---
title: "Run-3 subagent report: r4-recon (2026-10-11)"
type: research
date: 2026-10-11
status: provisional
---

# R4 reconnaissance — offline residual-stream Hadamard fold into Qwen3.8-27B BF16, then GPTQ INT4 G128

Date: 2026-10-11. Scope: read-only facts for the "fold an offline QuaRot-style residual-stream
Hadamard rotation into the BF16 weights, then GPTQ-quantize to a standard INT4 G128 auto_gptq
checkpoint served by the existing fast vLLM-XPU W4A16 path" design. No GPU, no repo writes, no
downloads. Every claim cites a file:line, command output, or primary URL.

Artifact identities used below:
- BF16 source: `/mnt/<nvme-disk>/b70-bf16-qwen38-20260926/` (18 shards + index + `ARTIFACTS.sha256`;
  `ls` output), twin of `/mnt/<sata-disk>/Qwen3.8-27B-BF16/`.
- HF reference code: `transformers 5.15.0` at `~/.local/lib/python3.14/site-packages/transformers`
  (`python3 -c 'import transformers,os;print(transformers.__version__);print(os.path.dirname(transformers.__file__))'` → `5.15.0`, `…/site-packages/transformers`).
- Serving code reference: `vllm-project/vllm` tag `v0.30.0` (primary URL) and the local fork
  `~/vllm-src/vllm` @ `066ae99e24ef` (2026-08-14).

---

## Q1 — Architecture: every residual-stream reader/writer, norms, and what the fold touches

### 1.1 Config facts (`…/b70-bf16-qwen38-20260926/config.json`, `cat`)
- `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5`, `text_config.model_type: qwen3_5_text`.
- `hidden_size 5120`, `intermediate_size 17408`, `num_hidden_layers 64`, `full_attention_interval 4`,
  `layer_types` = the pattern `[linear_attention ×3, full_attention ×1]` repeated 16× → **48 Gated-DeltaNet
  (linear_attention) blocks + 16 full-attention blocks**.
- `head_dim 256`, `num_attention_heads 24`, `num_key_value_heads 4`, `partial_rotary_factor 0.25`,
  `rms_norm_eps 1e-6`, `attention_bias false`, `attn_output_gate true`, `tie_word_embeddings false`.
- `mtp_num_hidden_layers 1`, `mtp_use_dedicated_embeddings false`, `linear_conv_kernel_dim 4`.

### 1.2 Residual-stream topology (HF `models/qwen3_5/modeling_qwen3_5.py`)
`Qwen3_5DecoderLayer.forward` (lines 753–788):
```
residual = hidden_states
hidden_states = input_layernorm(hidden_states)
<token mixer>                      # linear_attn (GatedDeltaNet) OR self_attn
hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = residual + hidden_states
```
So, per block:
- **Readers of the residual** (consume `input_layernorm` output):
  - GDN block: `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`
    (all `nn.Linear(hidden,·,bias=False)`, lines 465–468).
  - full-attention block: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj` (lines 645–651, `bias=attention_bias=False`).
  - Both block types: `mlp.gate_proj`, `mlp.up_proj` (consume `post_attention_layernorm` output; MLP lines 704–717).
- **Writers into the residual** (their output is added back):
  - GDN: `linear_attn.out_proj` (line 470, bias=False).
  - attention: `self_attn.o_proj` (line 701, bias=False).
  - both: `mlp.down_proj`.

Confirmed by the actual tensor list (`python3` over `model.safetensors.index.json`): layer 0 (linear_attention)
= `input_layernorm, linear_attn.{A_log,conv1d,dt_bias,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,norm,out_proj},
mlp.{down,gate,up},post_attention_layernorm`; layer 3 (full_attention) replaces `linear_attn.*` with
`self_attn.{k_norm,k_proj,o_proj,q_norm,q_proj,v_proj}`. **No bias tensors anywhere** in the language model.

Two off-residual norms exist and are NOT residual readers:
- `Qwen3_5RMSNormGated` (lines 168–186) normalizes the GDN per-value-head output (dim 128) with a SiLU gate — internal.
- `self_attn.q_norm` / `k_norm` (lines 654–655, `Qwen3_5RMSNorm(head_dim=256)`) — applied per head, after projection,
  before RoPE.

### 1.3 Embedding / final norm / lm_head (tensor list + code)
- Embeddings: `model.language_model.embed_tokens` (`Qwen3_5TextModel.__init__`, line 1132ff) — the first writer into the residual stream.
- Final norm: `model.language_model.norm` (`Qwen3_5TextModel`, RMSNorm over hidden).
- Head: `lm_head.weight` is a **separate tensor** (`tie_word_embeddings false`). In vLLM it is `ParallelLMHead`
  with `quant_config` = the model quant config (v0.30.0 `qwen3_5.py`), i.e. the head could be quantized, but the
  champion recipe set `lm_head=False` (see Q4).

### 1.4 MTP head (`mtp.*`)
Checkpoint tensors (`python3` over the index): `mtp.fc.weight`, `mtp.pre_fc_norm_hidden.weight`,
`mtp.pre_fc_norm_embedding.weight`, `mtp.norm.weight`, and one full-attention decoder layer
`mtp.layers.0.{input_layernorm,post_attention_layernorm,self_attn.{q,k,v,o,q_norm,k_norm},mlp.{gate,up,down}}`.
transformers **ignores** `mtp.*` on load (`modeling_qwen3_5.py:804,1577`: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", …]`).
The implementation of record is vLLM `model_executor/models/qwen3_5_mtp.py`. Its `Qwen3_5MultiTokenPredictor.forward`
(v0.30.0, primary URL):
```
inputs_embeds = self.pre_fc_norm_embedding(embed_input_ids(input_ids))
hidden_states = self.pre_fc_norm_hidden(hidden_states)
hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))   # fc: Linear(2*5120 -> 5120)
hidden_states, residual = self.layers[0] (hidden_states)                      # full_attention DecoderLayer
hidden_states, _ = self.norm(hidden_states, residual)                        # mtp.norm
```
Answering the specific questions:
- **What feeds `mtp.fc`:** the concatenation along the feature dim of `pre_fc_norm_embedding(token_embedding)`
  (columns 0:5120 of `mtp.fc.weight`) and `pre_fc_norm_hidden(target_hidden_state)` (columns 5120:10240). This is a
  **mixed-basis** input — the two halves live in different spaces (see Q1.6).
- **Which norms:** `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden` (RMSNorm-5120 with `1+weight` gain), then
  `mtp.layers.0.{input,post_attention}` (RMSNorm-5120) and per-head `q_norm`/`k_norm` (RMSNorm-256), then `mtp.norm`
  before the draft LM head.
- **`hidden_states` basis:** vLLM's `Qwen3NextModel.forward` applies `self.norm` and returns the **post-`model.norm`
  hidden** (`qwen3_next.py:666–680` local fork: `hidden_states, _ = self.norm(hidden_states, residual) … return hidden_states`),
  and `compute_logits` merely applies `lm_head` with no further norm (`qwen3_5.py:416–420`, `qwen3_next.py:843–847`).
  The MTP is fed exactly that tensor (`v1/worker/gpu_model_runner.py:5318`: `target_hidden_states = hidden_states[:num_scheduled_tokens]`;
  the only override hook `get_mtp_target_hidden_states` exists for DeepSeek-V4 only, `gpu_model_runner.py:5301`).
  ⇒ **MTP's `pre_fc_norm_hidden` input is the post-final-norm hidden state**, computed from the *rotated* residual
  if the residual stream is globally rotated.
- **MTP embedding table:** `qwen3_5_mtp.py::load_weights` remaps `mtp.` → `model.` and strips `language_model.`, so the
  MTP `embed_tokens` is loaded from the **same** checkpoint tensor as the main `embed_tokens`. There is no distinct
  `mtp.embed_tokens` tensor.

### 1.5 RMSNorm formula and gain foldability
`Qwen3_5RMSNorm` (`modeling_qwen3_5.py:720–737`):
```
self.weight = nn.Parameter(torch.zeros(dim))          # zero-init
def forward(self, x):
    output = x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)   # in fp32
    output = output * (1.0 + self.weight.float())               # gain = 1 + weight
    return output.type_as(x)
```
vLLM uses the same convention: `Qwen3_5RMSNorm` is aliased to `GemmaRMSNorm`
(`vllm/model_executor/models/qwen3_5.py:39: from …layernorm import GemmaRMSNorm as Qwen3_5RMSNorm`), whose
docstring is explicit: "`x * (1 + w)` instead of `x * w`" and the weight is zero-init
(`layers/layernorm.py:132–149`, v0.30.0 same).

**Foldability.** The gain `diag(1+w)` is applied to the normalized vector immediately before the following
linear(s), so it is a per-input-channel diagonal and folds exactly into those linears' input columns
(`W ← W·diag(1+w)`), after which the norm weight is zeroed. The fold for the *rotated* case is exact only because
H is orthogonal over the **full 5120 hidden dim** and RMSNorm normalizes over that same full dim:
`rmsnorm(Hᵀx) = Hᵀ·rmsnorm(x)` (orthogonal maps preserve the L2 norm, hence the mean-square). The reader then
computes `W·diag(1+w)·H · Hᵀrmsnorm(x) = W·diag(1+w)·rmsnorm(x)`. This is why the harness folds gain **and** zeroes
the norm weight — keeping the gain in place does not commute with H. (Per-head norms `q_norm`/`k_norm` normalize over
`head_dim=256`, not the residual; they do not participate in the residual fold.)

### 1.6 Which linears need rotation, and what blocks the fold
Convention (matches the harness, Q2): rotated basis `x̂ = Hᵀx`; **readers** `W ← W·H`; **writers** `W ← Hᵀ·W`;
embeddings and the head are readers of nothing but write/read the stream ⇒ `E ← E·H` (row space) and `W_lm ← W_lm·H`.

| Module | Role on residual | Fold |
|---|---|---|
| `linear_attn.in_proj_qkv / in_proj_z / in_proj_a / in_proj_b` | reader | `W·H` |
| `self_attn.q_proj / k_proj / v_proj` | reader | `W·H` |
| `mlp.gate_proj / up_proj` | reader | `W·H` |
| `linear_attn.out_proj`, `self_attn.o_proj`, `mlp.down_proj` | writer | `Hᵀ·W` |
| `model.language_model.embed_tokens` | stream source | `E·H` |
| `model.language_model.norm` | residual norm | gain fold + zero (see caveat) |
| `lm_head` | stream reader | `W·H` (+ gain fold) |
| `mtp.*` | see Q1.4 | hidden half only — **not** a uniform fold |
| `model.visual.*` merger | injects into the embedding stream | output-side `Hᵀ` (not covered) |

Because q and k read the **same** residual vector, `QKᵀ = (W_q H)(Hᵀx)·((W_k H)(Hᵀx))ᵀ = W_q x xᵀ W_kᵀ` is exactly
preserved — no head-dim Hadamard and no RoPE change are needed for a **weight-only** W4A16 export (R2/R3/R4 rotations
would only be needed to quantize activations, which this route does not do).

**Blockers / hazards to the fold:**
1. **No biases** (`attention_bias=false`, MLP `bias=False`, GDN `bias=False`) and **no residual/layer scaling**
   (`config.layer_scale` absent; vLLM reads `getattr(config,"layer_scale",False)` at `qwen3_5.py:~190`), so nothing
   else needs a compensating term. Good.
2. **All residual norms are RMSNorm with a foldable `1+w` gain**; the only non-RMS norm is `Qwen3_5RMSNormGated`
   and it is off-stream. Good.
3. **`mtp.fc` is a mixed-basis linear.** Its input is `cat([embedding-normed, hidden-normed])`; only the hidden half
   lives on the rotated residual. A uniform fold is impossible: the matrix must be split at column 5120, the
   embedding half folded with the (rotated) embedding basis and the hidden half with the rotated-residual basis.
4. **`model.norm` fans out to two consumers in different subsequent nonlinearities** — the target `lm_head` and
   `mtp.pre_fc_norm_hidden` — and the MTP consumes the **post-norm** hidden (Q1.4). Because `diag(1+w)` does not
   commute with H, no single weight-only fold preserves both consumers exactly:
   - Folding the final norm's gain into `lm_head` (and zeroing `model.norm`) makes the target head exact but hands
     MTP `Hᵀ·rmsnorm(x)` instead of `diag(1+w)·rmsnorm(x)`.
   - Keeping `model.norm` intact forces `lm_head' = W_lm·diag(g)·H·diag(g)⁻¹` (exact, but numerically nasty and
     requires `g_i ≠ 0` for every channel) and still leaves MTP receiving `diag(g)·Hᵀrmsnorm(x)` ≠ `Hᵀ·diag(g)·rmsnorm(x)`.
   Practical exact routes: (a) insert **one** explicit un-rotation GEMM (`x_unrot = H·x_rot`, 5120×5120 BF16, ~0.05 MFLOP/token,
   a stock `linear` — no new kernel) before `model.norm`, leaving `model.norm`, `lm_head` and all of MTP unrotated;
   or (b) feed MTP the **pre-norm** residual instead (vLLM already exposes the `get_mtp_target_hidden_states()`
   extension point at `gpu_model_runner.py:5301`, currently used only by DeepSeek-V4), which makes both the head fold
   and the MTP hidden-half fold exact. Either is a small engine change; neither is covered by the harness.
5. **Vision tower** (`model.visual.*`, present in the checkpoint) writes into the embedding stream through the
   multimodal merge; if vision is ever enabled the merger output must be rotated too (`Hᵀ` on the final projection).
   The reference route is `--language-model-only`, so this is a latent hazard rather than an active blocker.

---

## Q2 — Does `scripts/tmp/rotation_t1t2_common.py` already implement the fold? What is missing?

Yes, for the **per-decoder-block** case, and no for the head/MTP/embedding case.

Implemented:
- `build_H_5120(seed=SEED=20260925)` (line 97): `H = D · (Paley(19) ⊗ Sylvester(256)) / √5120`, with `assert_hadamard`
  exact-orthogonality checks on both factors and `‖HHᵀ−I‖∞ < 1e-3` on the product; fp32 `[5120,5120]`; Rademacher row signs.
  (A companion `build_R_17408_blockdiag` exists for the optional online FFN-intermediate arm.)
- Reader/writer lists (lines 173–179): readers `in_proj_qkv/in_proj_z/in_proj_a/in_proj_b`, `q/k/v_proj`, `gate/up_proj`;
  writers `out_proj`, `o_proj`, `down_proj`.
- `h_fold(sd,H)` (line 201): readers `W·H`, writers `Hᵀ·W`, done in fp32 then cast to BF16.
- `norm_gain_fold(layer,cfg)` (line 182): folds `(1+weight)` of `input_layernorm` and `post_attention_layernorm` into the
  respective readers and **zeroes** those two norm weights (exactly the correct rotated-basis treatment from Q1.5).
- `QUANTIZED_TENSORS_GDN/ATT` (lines 359–362) reproduce the champion's 400-qweight scope; the comment at 363–365 records
  that `in_proj_a/b` were **not** quantized in the champion artifact (`48·6 + 16·7 = 400`) yet are still rotated for equivalence.
- Deployment-arithmetic helpers `rot_input`/`unrot_output` and a classic GPTQ `gptq_quant` matching the champion contract
  (`BITS,SYM,GROUP,DESC_ACT,PERCDAMP = 4,True,128,False,0.01`, line 55).

**Not covered** (confirmed by reading the whole file):
- MTP in any form — `mtp.fc` (incl. the mixed-basis split), `mtp.pre_fc_norm_hidden/embedding`, `mtp.norm`, `mtp.layers.0`.
- `model.language_model.embed_tokens` (the stream source) and `lm_head`.
- The final `model.language_model.norm` (its gain-fold/MTP fan-out problem of Q1.6).
- The vision merger.
- Whole-model orchestration: the harness builds one `Qwen3_5DecoderLayer` from a shard and runs a block-level
  forward; there is no exporter, no per-shard streamer, no output checkpoint writer.

---

## Q3 — Toolchain on this host

Command outputs (all read-only):
- System: `python3 --version` → **3.14.4**; `python3 -m pip list` → only `torch 2.13.0+cpu`, `transformers 5.15.0`,
  `safetensors 0.8.0` (plus `autocommand`) among relevant packages.
- `python3 -c 'import torch;print(torch.__version__, torch.xpu.is_available())'` → **`2.13.0+cpu False`**.

Venvs / interpreters with torch (each probed with its own interpreter):
| interpreter | torch | `torch.xpu.is_available()` | transformers / optimum |
|---|---|---|---|
| `/mnt/<nvme-disk>/shard-venv` | 2.14.0+cpu | **False** | transformers 5.4.0, optimum 2.3.0, optimum_intel 2.2.0 |
| `~/cascadia-venv` | 2.14.1+cpu | **False** | transformers 5.5.4, optimum 2.3.0, optimum_intel 2.2.0 |
| `/mnt/<sata-disk>/cascadia-venv` | 2.13.0+cpu | — | transformers 5.4.0 |
| `/mnt/<sata-disk>/d11-venv` (py3.12) | 2.9.0+cpu | **False** | transformers 5.16.1 |
| `/mnt/<sata-disk>/.ov-genai-venv` | 2.14.0+cpu | — | transformers 5.2.0, optimum_intel 2.1.0 |
| `~/.venvs/hf` | 2.14.1+cpu | — | — |

**No torch with XPU exists on this host.** `find / -xdev -name 'libtorch_xpu.so'` → empty;
`find / -xdev -name 'torch-*+xpu*'` → empty; `find … -name 'intel_extension_for_pytorch'` → empty. Every torch is a
`+cpu` build (`torch.xpu` is importable but `is_available()` is False — the module exists, the backend does not load).

Quantizers / converters (host-wide `find / -xdev -name '<pkg>*dist-info'`):
- **Absent:** `gptqmodel`, `auto_gptq`, `auto_round`, `llm_compressor`, `compressed_tensors`, `exllamav3`, `accelerate`.
  This matches `results/2026/09/b70-probe2-30defect-20260929/RESULTS.md` ("No `compressed_tensors` / `auto_gptq` /
  `gptqmodel` / `accelerate` on host").
- **Present:** `optimum 2.3.0` + `optimum_intel 2.2.0` (cascadia-venv, shard-venv), `optimum_intel 2.1.0` (.ov-genai-venv).

Docker, **main daemon** (`docker images`; data-root `/var/lib/docker`):
```
strata-sycl-serve:0.1.40.3   13.3GB
strata-sycl-dev:latest       13.3GB   (llama.cpp SYCL for B70 — image labels: org.opencontainers.image.title=llama.cpp)
pixel-sandbox-candidate:4.3.29-uid-1000 / ods-* / openclaw-sandbox:bookworm-slim
ghcr.io/berriai/litellm / ghcr.io/ggml-org/llama.cpp
```
**None of these contain torch-xpu.** There is no vLLM image in the main daemon at all.

Docker, **isolated daemon** (not running; store inspected read-only at
`/mnt/<nvme-disk>/<exl3-dockerd>/image/overlay2/repositories.json`):
```
ghcr.io/0xsero/exl3xpu:0.1.0  ==  @sha256:21412bdd7535…  (image id 77785c51423f…)
vllm/vllm-openai-xpu:v0.30.0  ==  @sha256:fc0e112afb64…  (image id 8def9aae5c58…)
```
(`docs/plans/2026-10-09-zcode-workflow-prompt.md:59` documents the same two digests.)

---

## Q4 — How the removed champion `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` was produced

Recipe of record: `/mnt/<models-disk>/qw38-quant/quantize.py` (still present; `cat`). Key facts:
- Tool: **GPTQModel** (`from gptqmodel import GPTQModel, QuantizeConfig`), run inside container `qw38-gptq-quant:latest`
  on an XPU torch (`assert torch.xpu.is_available()`), with `device="xpu:0"`.
- Quantizer settings:
  `QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False, lm_head=False, checkpoint_format="gptq",
  dynamic={"-:.*mtp.*": {}}, offload_to_disk=False)`. `checkpoint_format="gptq"` ⇒ **auto_gptq packing** (not
  compressed-tensors). `dynamic={"-:.*mtp.*": {}}` excludes **all** `mtp.*` tensors; `lm_head=False` keeps the head BF16.
- Calibration corpus: `Salesforce/wikitext`, `wikitext-2-raw-v1`, `split="train"`, first 128 texts with `len(strip())>200`,
  `batch_size=1`, sequences truncated to 2048 tokens (`tok(..., max_length=2048)`).
- Export/verify: writes `…-GPTQ-Int4-sym-G128-MTP-BF16/` under `/qw38-gptq-out/…`; the script asserts `qweight` present,
  `mtp.*` present with **zero** `qweight/qzeros/g_idx` leaks, and `quantization_config` `bits=4, sym=True, group_size=128`.
- Provenance line in the removal manifest: "converter_provenance=GPTQModel, wikitext-2-raw 128-text calibration, mtp.*
  BF16 preserved" (`results/2026/09/autoround-vs-exl3-20260926/gptq-removal-manifest.txt`).

Tensor scope (independent packing audit, `results/2026/09/b70-probe2-30defect-20260929/RESULTS.md:57–63`):
> "400 quantized linears each; **lm_head/embed/norms/conv/in_proj_a/b/mtp.\*/visual.\* unquantized**"
and for the auto_gptq flavor: "**ARMC = auto_gptq (`qweight` I32 [in/8,out], `scales` F16 [in/128,out], `qzeros` I32
storing z−1 → zp=8 sym, `g_idx` sequential → desc_act off)**". (The compressed-tensors sister export uses
`weight_packed I32 [out,in/8]`, `weight_scale F16 [out,in/128]`, no stored zp.)
- 400 = 48 GDN blocks × 6 (in_proj_qkv, in_proj_z, out_proj, gate, up, down) + 16 attention blocks × 7
  (q,k,v,o,gate,up,down); `in_proj_a/b` (48-wide) **stayed BF16** — matching the harness comment at
  `rotation_t1t2_common.py:363–365`.
- Size/contents: "400 qweight tensors, 15 BF16 MTP tensors, zero MTP leaks, and 19.56 GB"
  (`models/qwen/qwen3.8-27b/artifacts.md`).
- Artifact fate: local copies deleted per operator order 2026-09-26 (`gptq-removal-manifest.txt`; the `-nogidx` dir is
  confirmed empty in `results/2026/09/b70-inventory-20260929/INVENTORY.md`); the HF repo is still live
  (`curl -sI` → HTTP 200) and is marked "recoverable".

---

## Q5 — vLLM v0.30.0 on XPU: GPTQ/wNa16 path, packing, and BF16 `mtp.*`

Primary source, tag `v0.30.0` (raw.githubusercontent.com/vllm-project/vllm/v0.30.0/…):
- **XPU quantization support** — `vllm/platforms/xpu.py`, `XPUPlatform.supported_quantization` includes
  `"awq", "gptq", "moe_wna16", "auto_awq", "auto_gptq", "inc", "fp8", …, "compressed-tensors"`. GPTQ/auto_gptq are
  first-class on XPU.
- **Kernel selection** — `vllm/model_executor/kernels/linear/__init__.py`, `_POSSIBLE_KERNELS[PlatformEnum.XPU] =
  [XPUW4A8IntLinearKernel, XPUwNa16LinearKernel]`. `auto_gptq.py` builds an `MPLinearLayerConfig` and calls
  `choose_mp_linear_kernel(...)`; on XPU a 4-bit `uint4b8` W4A16 layer lands on **`XPUwNa16LinearKernel`**.
  `kernels/linear/mixed_precision/xpu.py` (`XPUwNa16LinearKernel.can_implement`): only XPU; activation must be
  **BF16 or FP16** (⇒ W4A16, not W4A8); weight type ∈ `{uint4, uint4b8}`; `group_size % 32 == 0`; input size `% 32 == 0`.
  For symmetric checkpoints it supplies `weight_zero_point = 8`. **No new kernel is needed** for a standard
  sym-INT4 G128 export.
- **Packing expected** — `auto_gptq.py::AutoGPTQLinearMethod.create_weights` allocates
  `qweight` = `PackedvLLMParameter(shape=[input_per_partition // pack_factor, output_per_partition], int32, packed_dim=0)`
  (`pack_factor = 32 // bits = 8` for 4-bit), `scales` = `GroupQuantScaleParameter[input/group, output]`, `qzeros` int32;
  for sym it passes `zero_points=False` to the kernel. The XPU kernel's `process_weights_after_loading` detects the
  on-disk layout by comparing `qweight.shape[0]` with `scale.shape[0]` and transposes only when needed, so it accepts
  **both** the auto_gptq layout (`[in/8, out]`, the champion's) and the compressed-tensors `[out, in/8]` layout.
- **Qwen3.8 / Qwen3_5 + MTP** — `models/registry.py` registers `"Qwen3_5ForConditionalGeneration"`, `"Qwen3_5MTP":
  ("qwen3_5_mtp","Qwen3_5MTP")`. `models/qwen3_5_mtp.py` contains an explicit GPTQ provision:

  > "GPTQ: quantized checkpoints may exclude MTP from quantization via `quantization_config.dynamic` with `-:pattern`
  > entries. When detected, **disable quantization for MTP layers so they use unquantized params**." — and the code
  > reads `hf_qc["dynamic"]`, and when any key starts with `-:` and contains `mtp`, sets `vllm_config.quant_config = None`
  > while constructing the MTP decoder layer(s).

  Additionally `auto_gptq.py::AutoGPTQConfig.get_quant_method` uses `get_dynamic_override` / `get_linear_quant_method`
  so a `-:.*mtp.*` rule returns an unquantized method per linear (this is what keeps `mtp.fc` and the MTP body BF16
  even though `fc` is constructed with the GPTQ config for non-NVFP4 checkpoints). **This is exactly the champion
  layout**, so the checkpoint serves correctly with `--quantization gptq`/`auto_gptq` + `method=mtp`.
- **RL-087 packing-speed note.** The registry record states: "arext compressed-tensors packing is 4-5x slower than
  champion's auto_gptq packing on XPUwNa16 — checkpoint FORMAT dominates converter choice"
  (`registry/research-ledger.json` RL-087, 2026-09-27). The campaign's own numbers
  (`results/2026/09/autoround-vs-exl3-20260926/RESULTS.md:58–60`) are `18.12 GiB / 83.3 ms ≈ 222 GB/s` for
  compressed-tensors vs "~430–495 GB/s effective" for the removed GPTQ champion (`19.56 GiB / ~40 ms step`) — i.e. the
  measurable bandwidth ratio is ≈2×, and the "4-5×" appears to fold in the overall wall-rate gap of the AutoRound arms
  (~15 vs 73.4 post-first). **The two figures disagree; treat "4-5×" as the ledger's aggregate wording, "≈2×"
  load-bandwidth as the measured one.** `[UNVERIFIED]` which is the intended metric.

---

## Q6 — Champion speed record, image/patches, and whether the image still exists

Speed record (repo records):
- `archive/docs/pre-migration-20260819/benchmark-history.md:2655–2676`, "Run 48 — Qwen3.8 at Qwen-recommended sampling":
  C1 p512/g128 median **102.61 tok/s** (range 95.7–106.7, acceptance 91.8 %), n=5, non-thinking preset
  (temp 0.7 / top_p 0.80 / top_k 20 / presence_penalty 1.5). Same stack as Run 47: image `f01e24f6`, MTP4, fp8 KV,
  prefix cache **ON**, 230 W. (Greedy diagnostic 106.7 tok/s.)
- Draft-side INT4 cells (`models/qwen/qwen3.8-27b/serving.md`, 2026-09-22 re-measure): **73.43 tok/s** p512/g128 and
  **72.55 tok/s** p8192/g128 client post-first, n=5+5, cold prefix, against 52.61 / 48.81 baseline (+39.6 % / +48.6 %).
  Enabler = XPU graph capture (`VLLM_XPU_ENABLE_XPU_GRAPH=1`).
- `engines/vllm-xpu/README.md:50` records GPTQ-Int4 + MTP4 (TP1) as "serving leader, 102.61 tok/s p512/g128 client
  post-first (Run 48 …)"; the route was removed 2026-09-27 for RL-087/RL-092.

Image + patches it needed:
- Image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
  (in-container vLLM `0.27.2rc1.dev77`, `vllm-xpu-kernels 0.1.12.3`), plus a 3-patch source overlay
  (#53997 + #53990 + #53989) and `VLLM_XPU_ENABLE_XPU_GRAPH=1`, MTP4, fp8 KV, cache on 230 W cap
  (`docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/RUNTIME-patches-assessment-20260923.md`;
  `models/qwen/qwen3.8-27b/serving.md`). A derived image with kernels 0.1.15.4 was also built (`b70-vllm-kernels-0.1.15.4:f01e24f6`).
- Patch categories: sampler/graph fixes (`torch/xpu/graphs.py` recording), a fused `fused_qk_rmsnorm_rope_gate` path,
  and (for the draft-INT4 lever) `patch_draft_lmhead_int4.py` / `patch_draft_mtp_int4.py`, env-gated
  `B70_DRAFT_LMHEAD_INT4`/`B70_DRAFT_MTP_INT4` (also a reviewable vLLM tree change `VLLM_DRAFT_INT4=1`).

Is the image still present anywhere?
- **No.** Main daemon `docker images` lists no `vllm*` image at all. The isolated daemon's on-disk store
  (`…/<exl3-dockerd>/image/overlay2/repositories.json`, read-only) contains only `ghcr.io/0xsero/exl3xpu` and
  `vllm/vllm-openai-xpu:v0.30.0@sha256:fc0e112afb64…` — **not** `f01e24f6`. This is corroborated by
  `docs/plans/2026-09-20-b70-improvement-roadmap/worklogs/P4-20260920.md:107` ("The champion image
  `…@sha256:f01e24f6…` is not present locally (only `…@sha256:2c427ef4…` is)"). The `qw38-gptq-quant` conversion image
  is likewise absent from both stores.
- The checkpoints themselves: the GPTQ local copy is removed (recoverable on HF, HTTP 200); EXL3 4bpw is present at
  `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926/`; the BF16 source is present in two copies.

---

## Blockers and missing pieces (ranked by severity)

1. **The R4 quality premise is unproven at whole-model scale.** The only rotation evidence is two-block, offline:
   rotated+GPTQ W4-G128 beat unrotated by KL 0.40× (GDN L30) / 0.58× (ATT L31) at equal bytes, with every tensor's
   recon error reduced, top-1 +0.11/+0.25 pts (`results/2026/09/qwen27-rotation-t1t2-20260925T1812Z/RESULTS.md`;
   RL-067). There is **no** whole-model canary/margin result. Meanwhile RL-092 proved the 30-defect is a **weights**
   effect — a ~**−3.3-logit systematic margin inversion** under exact fp32 accumulation, established mid-network
   (~−7.1 at layer ~60) — and the champion's own BF16 margin was only **+0.40**. A 2× block-KL reduction is not
   evidence that a −3.3-logit inversion is recovered. The one program question aimed exactly here (RQ-26) is
   `blocked`: RL-150 and RL-157 both failed their guards, RL-151 showed the output-KL metric degenerate on 4/6 blocks,
   the RTN-INT3 sanity arm reads ~0 block-delta KL on `att_l03` (weights perturbed, rel_fro 0.24–0.26) and the
   homemade AutoRound loss rises over 100 iters with every AR tensor escaping to INT8. **The measuring instrument
   must be fixed and the margin question answered before any R4 claim.**
2. **The MTP unit is not foldable by the same transform** (Q1.4/Q1.6). `mtp.fc` is a mixed-basis concat linear, and
   MTP's hidden input is the **post-`model.norm`** hidden (vLLM `gpu_model_runner.py:5318`), which fans out the final
   norm to two consumers. Options are all extra work: (a) one explicit un-rotation GEMM before `model.norm`
   (stock linear, no new kernel); (b) a small vLLM change to feed MTP the pre-norm residual (the
   `get_mtp_target_hidden_states()` hook already exists); or (c) declare the drafter approximate and prove acceptance
   + spec-canary. The harness covers none of this.
3. **Embeddings, `lm_head`, final norm, vision merger are outside the harness.** A whole-model converter must add
   `E ← E·H` (main + MTP-shared table), `W_lm ← W_lm·H` (+ gain fold), the final-norm decision, and the vision-merger
   output rotation. None exist in `rotation_t1t2_common.py` (Q2).
4. **No quantizer toolchain on the host.** No XPU torch anywhere (`torch.xpu.is_available()` False for every
   interpreter; no `libtorch_xpu.so`), and no `gptqmodel` / `auto_gptq` / `auto_round` / `llm-compressor` /
   `compressed_tensors`. The champion's `qw38-gptq-quant` image is gone from both daemons. The GPTQ step must run in
   a container (the isolated daemon holds `vllm/vllm-openai-xpu:v0.30.0`, digest `fc0e112a…`) and GPTQModel must be
   installed/derived there. The fold itself is pure CPU (safetensors ⇄ fp32 matmul) and is feasible on the host.
5. **Storage / RAM for a 54 GB fold-and-quant pass.** The BF16 source is ~54 GB across 18 shards; the known host
   failure mode is the gptqmodel OOM/"54 GB materialization" (`research/2026/qwen38-27b-day0-adoption-20260814.md`).
   Budget disk for source + folded intermediate + ~20 GB output, and stream layers.
6. **Fold arithmetic precision.** The harness folds in fp32 then rounds to BF16. For a real export the built-in
   GPTQ chain must consume the **fp32 folded** weights directly; a BF16 safetensors round-trip would erode the KL
   gain (RL-067's own T1 supplement shows a lossless gain-fold-only refactor already breaches the 5e-2 max-abs gate
   3.5–4.6× on real activations, i.e. the BF16-rewrite floor is real).
7. **The champion serving image `f01e24f6` (+3 patches, kernels 0.1.12.3) is gone** from both docker stores, so the
   102.61 / 73.43 tok/s numbers are not reproducible as-is; the only present XPU serving image is
   `vllm/vllm-openai-xpu:v0.30.0` (digest `fc0e112a…`), which is untested for this model in-repo. Any new speed claim
   must be made on a re-established, digest-pinned stack.
8. **The removed champion checkpoint is gone locally** (recoverable on HF), and `compressed-tensors` packing is the
   measurably slower loader on `XPUwNa16` (222 vs ~430–495 GB/s per RESULTS.md; the ledger says "4-5×" — the two
   figures conflict). The R4 export should stay **auto_gptq-packed** to inherit the fast loader.
9. **Vision path latent hazard**: if the multimodal tower is enabled, its merger output must be rotated or quality
   silently degrades; the reference route is `--language-model-only`, so this is out of the immediate critical path
   but must be handled or explicitly excluded.
10. **Minor / bookkeeping:** `in_proj_a/b` stay BF16 in the champion scope (still must be rotated for equivalence);
    RL-087's "4-5×" vs RESULTS.md's "≈2×" packing-speed discrepancy is unresolved; `desc_act=False` ⇒ sequential
    `g_idx` (the champion also ran a `-nogidx` local variant).
