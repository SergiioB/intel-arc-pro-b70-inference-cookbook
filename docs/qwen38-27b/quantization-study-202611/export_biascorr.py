#!/usr/bin/env python3
"""Standalone Pos0-Excluded Nagel BiasCorr Sidecar Exporter for MLX-4bit Checkpoints.

Computes exact FP32 per-channel output bias corrections for all 496 transformer
linear layers of a Qwen3.5 / Qwen3.8 MLX-4bit checkpoint (group_size=64, affine
uint4) relative to its BF16 reference weights:

    delta_b = (W_bf16 - dequant(W_mlx4)) @ E[x_{t >= 1}]

Excluding position 0 (the attention-sink BOS token) avoids saturating E[x] with
the 100x+ initial-token outlier spike. The resulting ~15 MiB safetensors sidecar
(`*.bias_corr`, F32) is loaded by TensorFold via `XPU_BIASCORR_FILE` and fused
directly into the W4 GEMM/GEMV epilogue at ~0% runtime overhead.

Also supports `--zero-bias` to export an exact IEEE-754 negative-zero (`-0.0f`)
control sidecar for proving bit-exact runtime neutrality (`acc + (-0.0f) == acc`
in round-to-nearest-even for all finite values including `+0.0f` and `-0.0f`).

KL gate (on by default when both --eval-prose and --eval-code are given). The
saved sidecar is reloaded from disk and scored with a 64-layer streamed causal
KL against BF16 on held-out windows, as three arms: no correction, +b and -b.
It is marked PUBLISHABLE only if all of these hold:
  1. prose KL(+b) - KL(base) has a 95% blk32 bootstrap CI entirely below 0,
  2. code  KL(+b) - KL(base) has a 95% blk32 bootstrap CI entirely below 0,
  3. the sign-flipped -b control raises mean KL on both prose and code.
Otherwise the .meta.json says REJECTED_BY_KL_GATE and the exit code is 3.
A sidecar exported without the gate is marked UNGATED. gate_sidecar() takes
per-position KL arrays, so other test scripts can apply the same rule.

Zero private paths; supports Intel XPU, CUDA, and CPU (Triton unpack on GPU,
vectorized PyTorch fallback on CPU).
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
import sys
import time
from typing import List, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

if _HAS_TRITON:
    @triton.jit
    def _unpack_mlx_u32_bf16_k(W_ptr, S_ptr, B_ptr, O_ptr, K_words, ng, BLOCK_KW: tl.constexpr):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        offs = blk * BLOCK_KW + tl.arange(0, BLOCK_KW)
        m = offs < K_words
        w = tl.load(W_ptr + row * K_words + offs, mask=m, other=0)
        g = offs >> 3
        s = tl.load(S_ptr + row * ng + g, mask=m, other=0).to(tl.float32)
        b = tl.load(B_ptr + row * ng + g, mask=m, other=0).to(tl.float32)
        base = row * (K_words * 8) + offs * 8
        for j in tl.static_range(0, 8):
            q = ((w >> (j * 4)) & 15).to(tl.float32)
            val = (q * s + b).to(tl.bfloat16)
            tl.store(O_ptr + base + j, val, mask=m)


def select_device(req: str) -> torch.device:
    if req == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(req)


def device_sync(dev: torch.device) -> None:
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def device_empty_cache(dev: torch.device) -> None:
    if dev.type == "xpu":
        torch.xpu.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def unpack_mlx_u32_bf16(W_i32: torch.Tensor, S_bf16: torch.Tensor, B_bf16: torch.Tensor, scratch: torch.Tensor = None) -> torch.Tensor:
    """Dequantize MLX-4bit (group_size=64, packed u32) weights to BF16."""
    N, Kw = W_i32.shape
    K = Kw * 8
    ng = S_bf16.shape[1]
    assert K == ng * 64, f"Expected group_size=64 (K={K}, ng={ng})"
    if _HAS_TRITON and W_i32.device.type in ("xpu", "cuda"):
        if scratch is not None and N * K <= scratch.numel():
            out = scratch[: N * K].view(N, K)
        else:
            out = torch.empty((N, K), device=W_i32.device, dtype=torch.bfloat16)
        _unpack_mlx_u32_bf16_k[(N, triton.cdiv(Kw, 128))](W_i32, S_bf16, B_bf16, out, Kw, ng, BLOCK_KW=128)
        return out
    shifts = torch.arange(0, 32, 4, device=W_i32.device, dtype=torch.int32)
    q = ((W_i32.unsqueeze(-1) >> shifts) & 15).to(torch.float32).reshape(N, ng, 64)
    out = (q * S_bf16.float().unsqueeze(-1) + B_bf16.float().unsqueeze(-1)).reshape(N, K)
    return out.to(torch.bfloat16)


def load_text_source(path: str) -> str:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        tbl = pq.read_table(path)
        col = "text" if "text" in tbl.column_names else tbl.column_names[0]
        return "\n".join(x for x in tbl[col].to_pylist() if x and str(x).strip())
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def tokenize_windows(tok, paths: List[str], seq_len: int, num_seqs: int, from_tail: bool = False, tail_offset: int = 4096) -> List[List[int]]:
    txt = "\n\n".join(load_text_source(p) for p in paths)
    ids = tok.encode(txt, add_special_tokens=False)
    need = seq_len * num_seqs
    if len(ids) < need:
        raise ValueError(f"Calibration source has only {len(ids)} tokens, need {need} ({num_seqs}x{seq_len})")
    if from_tail:
        start = max(0, len(ids) - max(tail_offset, need))
    else:
        start = 0
    return [ids[start + i * seq_len : start + (i + 1) * seq_len] for i in range(num_seqs)]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(rows: List[List[int]]) -> str:
    return hashlib.sha256(np.asarray(rows, dtype=np.int32).tobytes()).hexdigest()


GATE_PASS = "PUBLISHABLE"
GATE_FAIL = "REJECTED_BY_KL_GATE"
GATE_PENDING = "GATE_PENDING"
GATE_SKIPPED = "UNGATED"
GATE_EXIT_CODE = 3
GATE_DOMAINS = ("prose", "code")
GATE_ARMS = ("base", "plus", "minus")
BLOCK = 32


def blk32_ci(delta: np.ndarray, n_boot: int = 10000, seed: int = 20261125):
    """Mean and 95% CI of a paired per-position difference, shape (windows, positions).

    32-position blocks (plus one shorter tail block) are resampled within each
    window, so no block crosses a window boundary and no position is dropped.
    """
    n_win, n_pos = delta.shape
    starts = np.arange(0, n_pos, BLOCK)
    n_blk = len(starts)
    block_sums = np.add.reduceat(delta, starts, axis=1).reshape(-1)
    idx = np.random.default_rng(seed).integers(0, n_blk, size=(n_boot, n_win, n_blk))
    boots = block_sums[np.arange(n_win)[None, :, None] * n_blk + idx].sum(axis=2).mean(axis=1) / n_pos
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(delta.mean()), float(lo), float(hi)


def gate_sidecar(kl: Mapping[str, Mapping[str, np.ndarray]], n_boot: int = 10000, seed: int = 20261125) -> dict:
    """Decide whether a bias sidecar may be marked PUBLISHABLE.

    kl[domain][arm] is per-position KL vs BF16 with shape (windows, positions),
    for domain in ("prose", "code") and arm in ("base", "plus", "minus").
    Raises ValueError on missing, misshaped, too short or non-finite input.
    """
    checks, domains = {}, {}
    for dom in GATE_DOMAINS:
        if dom not in kl or any(arm not in kl[dom] for arm in GATE_ARMS):
            raise ValueError(f"KL gate needs arms {GATE_ARMS} for domain {dom!r}")
        arr = {arm: np.asarray(kl[dom][arm], dtype=np.float64) for arm in GATE_ARMS}
        shape = arr["base"].shape
        if any(a.shape != shape for a in arr.values()) or len(shape) != 2:
            raise ValueError(f"KL gate: {dom} arms must share one (windows, positions) shape")
        # With a single block per window every resample is identical and the CI has zero width.
        if shape[0] < 1 or shape[1] <= BLOCK:
            raise ValueError(f"KL gate: {dom} needs more than {BLOCK} scored positions per window, got shape {shape}")
        if not all(np.isfinite(a).all() for a in arr.values()):
            raise ValueError(f"KL gate: non-finite KL in {dom}")
        d_plus, lo, hi = blk32_ci(arr["plus"] - arr["base"], n_boot, seed)
        d_minus, mlo, mhi = blk32_ci(arr["minus"] - arr["base"], n_boot, seed)
        base = float(arr["base"].mean())
        domains[dom] = {
            "kl_base": base,
            "kl_plus": float(arr["plus"].mean()),
            "kl_minus": float(arr["minus"].mean()),
            "delta_plus": d_plus,
            "delta_plus_pct": 100.0 * d_plus / base if base > 0 else None,
            "delta_plus_ci95_blk32": [lo, hi],
            "delta_minus": d_minus,
            "delta_minus_ci95_blk32": [mlo, mhi],
            "shape": list(shape),
        }
        checks[f"{dom}_kl_drops_ci95_below_0"] = hi < 0.0
        checks[f"{dom}_minus_b_worsens_kl"] = d_minus > 0.0
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "status": GATE_FAIL if failed else GATE_PASS,
        "checks": checks,
        "failed_checks": failed,
        "domains": domains,
        "bootstrap": {"block": BLOCK, "n_boot": n_boot, "seed": seed, "resampling": "within window"},
    }


def load_arch(bf16_dir: str):
    """Text config and the transformers classes needed to stream one decoder layer at a time."""
    cfg_raw = json.load(open(os.path.join(bf16_dir, "config.json")))
    text_cfg = Qwen3_5TextConfig(**cfg_raw["text_config"])
    # Standalone layers must use a causal attention path.
    text_cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        dummy = Qwen3_5ForCausalLM(text_cfg)
    classes = {
        "layer": type(dummy.model.layers[0]),
        "modeling": inspect.getmodule(type(dummy.model)),
        "rotary": type(dummy.model.rotary_emb),
        "norm": type(dummy.model.norm),
    }
    return text_cfg, classes


def make_rotary(classes, text_cfg, dev: torch.device) -> nn.Module:
    # Only some transformers releases let the rotary embedding take device=.
    kw = {"device": dev} if "device" in inspect.signature(classes["rotary"]).parameters else {}
    return classes["rotary"](config=text_cfg, **kw).to(dev)


def shard_reader(model_dir: str):
    """get(key) over a sharded safetensors checkpoint, one shard open at a time."""
    wmap = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    cache = {}

    def get(key: str) -> torch.Tensor:
        fn = wmap[key]
        if fn not in cache:
            cache.clear()
            cache[fn] = safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu")
        t = cache[fn].get_tensor(key)
        return t.view(torch.int32) if t.dtype == torch.uint32 else t

    get.keys = wmap.keys()
    return get


def load_bf16_layer(get_bf, classes, text_cfg, lidx: int, dev: torch.device) -> nn.Module:
    with torch.device("meta"):
        layer = classes["layer"](text_cfg, lidx)
    layer = layer.to_empty(device="cpu")
    prefix = f"model.language_model.layers.{lidx}."
    layer.load_state_dict({name: get_bf(prefix + name) for name in layer.state_dict().keys()}, strict=True)
    return layer.to(device=dev, dtype=torch.bfloat16).eval()


def layer_runner(classes, text_cfg, rotary_emb, hidden: torch.Tensor):
    """run(layer, layer_index, h) for a batch shaped like hidden, with causal masks."""
    B, S, _ = hidden.shape
    pos4 = torch.arange(S, device=hidden.device).view(1, 1, -1).expand(4, B, -1)
    text_pos = pos4[0]
    pos_emb = rotary_emb(hidden, pos4[1:])
    mk = {"config": text_cfg, "inputs_embeds": hidden, "attention_mask": None, "past_key_values": None, "position_ids": text_pos}
    masks = {
        "full_attention": classes["modeling"].create_causal_mask(**mk),
        "linear_attention": classes["modeling"].create_recurrent_attention_mask(**mk),
    }

    def run(layer, lidx: int, h: torch.Tensor) -> torch.Tensor:
        out = layer(h, position_embeddings=pos_emb, attention_mask=masks[text_cfg.layer_types[lidx]], position_ids=text_pos)
        return out[0] if isinstance(out, tuple) else out

    return run


def mlx_dense(get_mlx, name: str, dev: torch.device, rows: torch.Tensor = None, row_chunk: int = 4096) -> torch.Tensor:
    """BF16 weight of an MLX tensor (quantized or dense); rows selects embedding rows."""
    def part(suffix):
        t = get_mlx(f"{name}.{suffix}")
        return t if rows is None else t[rows]

    if f"{name}.scales" not in get_mlx.keys:
        return part("weight").to(dev, dtype=torch.bfloat16)
    w, s, b = part("weight"), part("scales"), part("biases")
    return torch.cat([
        unpack_mlx_u32_bf16(w[i:i + row_chunk].to(dev), s[i:i + row_chunk].to(dev), b[i:i + row_chunk].to(dev))
        for i in range(0, w.shape[0], row_chunk)
    ])


@torch.no_grad()
def eval_causal_kl(bf16_dir: str, mlx_dir: str, sidecar_path: str, windows: Mapping[str, List[List[int]]], dev: torch.device) -> dict:
    """Per-position KL vs BF16 for the MLX checkpoint with no correction, +b and -b.

    All four streams (BF16 reference and three MLX arms) advance together, one
    decoder layer at a time. The MLX arms use the MLX embedding and lm_head.
    Returns kl[domain][arm] as (windows, seq_len - 1) float64 arrays.
    """
    text_cfg, classes = load_arch(bf16_dir)
    get_bf, get_mlx = shard_reader(bf16_dir), shard_reader(mlx_dir)
    with safe_open(sidecar_path, framework="pt", device="cpu") as f:
        sidecar = {k: f.get_tensor(k) for k in f.keys()}
    rotary_emb = make_rotary(classes, text_cfg, dev)
    ids = {dom: torch.tensor(windows[dom], dtype=torch.long) for dom in GATE_DOMAINS}

    emb = get_bf("model.language_model.embed_tokens.weight")
    ref = {dom: emb[ids[dom]].to(dev, dtype=torch.bfloat16) for dom in GATE_DOMAINS}
    del emb
    arms = {}
    for dom in GATE_DOMAINS:
        flat = ids[dom].reshape(-1)
        h0 = mlx_dense(get_mlx, "language_model.model.embed_tokens", dev, rows=flat).reshape(ref[dom].shape)
        arms[dom] = {arm: h0 for arm in GATE_ARMS}
    runners = {dom: layer_runner(classes, text_cfg, rotary_emb, ref[dom]) for dom in GATE_DOMAINS}

    used = set()
    for lidx in range(text_cfg.num_hidden_layers):
        layer = load_bf16_layer(get_bf, classes, text_cfg, lidx, dev)
        for dom in GATE_DOMAINS:
            ref[dom] = runners[dom](layer, lidx, ref[dom])
        corr = {}
        for n, m in [(n, m) for n, m in layer.named_modules() if isinstance(m, nn.Linear)]:
            name = f"language_model.model.layers.{lidx}.{n}"
            key = f"{name}.bias_corr"
            if key not in sidecar:
                raise ValueError(f"KL gate: sidecar has no tensor {key}")
            used.add(key)
            m.weight.data = mlx_dense(get_mlx, name, dev)
            corr[m] = sidecar[key].to(dev, dtype=torch.float32)
        for arm, sign in (("base", 0), ("plus", 1), ("minus", -1)):
            hooks = [] if sign == 0 else [
                m.register_forward_hook(lambda _m, _i, out, b=sign * bc: (out.float() + b).to(out.dtype))
                for m, bc in corr.items()
            ]
            for dom in GATE_DOMAINS:
                arms[dom][arm] = runners[dom](layer, lidx, arms[dom][arm])
            for h in hooks:
                h.remove()
        del layer, corr
        if lidx % 8 == 7 or lidx == text_cfg.num_hidden_layers - 1:
            gc.collect()
            device_empty_cache(dev)
            print(f"  KL eval L{lidx+1:02d}/{text_cfg.num_hidden_layers}", flush=True)
    if used != set(sidecar):
        raise ValueError(f"KL gate: {len(set(sidecar) - used)} sidecar tensors match no decoder linear")

    with torch.device("meta"):
        norm = classes["norm"](text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)
    norm = norm.to_empty(device="cpu")
    norm.load_state_dict({"weight": get_bf("model.language_model.norm.weight")})
    norm = norm.to(device=dev, dtype=torch.bfloat16).eval()
    # Transformers checkpoints omit lm_head.weight when it is tied to the embedding.
    head_key = "lm_head.weight" if "lm_head.weight" in get_bf.keys else "model.language_model.embed_tokens.weight"
    head_ref = get_bf(head_key).to(dev, dtype=torch.bfloat16)
    mlx_head = "language_model.lm_head" if "language_model.lm_head.weight" in get_mlx.keys else "language_model.model.embed_tokens"
    head_q = mlx_dense(get_mlx, mlx_head, dev)

    kl = {}
    for dom in GATE_DOMAINS:
        def last(h):
            return norm(h)[:, :-1, :].reshape(-1, text_cfg.hidden_size)

        h_ref = last(ref[dom])
        h_arm = {arm: last(arms[dom][arm]) for arm in GATE_ARMS}
        out = {arm: [] for arm in GATE_ARMS}
        for i0 in range(0, h_ref.shape[0], 64):
            ref_logp = F.log_softmax(F.linear(h_ref[i0:i0 + 64], head_ref).float(), dim=-1)
            ref_p = ref_logp.exp()
            for arm in GATE_ARMS:
                q_logp = F.log_softmax(F.linear(h_arm[arm][i0:i0 + 64], head_q).float(), dim=-1)
                out[arm].append((ref_p * (ref_logp - q_logp)).sum(dim=-1).clamp(min=0.0).cpu())
        n_win, n_pos = ids[dom].shape[0], ids[dom].shape[1] - 1
        kl[dom] = {arm: torch.cat(v).double().numpy().reshape(n_win, n_pos) for arm, v in out.items()}
    return kl


def ngrams(rows: List[List[int]], n: int = 6) -> set:
    return {tuple(row[i:i + n]) for row in rows for i in range(len(row) - n + 1)}


def write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description="Export Pos0-excluded FP32 BiasCorr safetensors sidecar for MLX-4bit models.")
    ap.add_argument("--bf16-dir", required=True, help="Path to BF16 reference checkpoint directory.")
    ap.add_argument("--mlx-dir", required=True, help="Path to shipped MLX-4bit checkpoint directory.")
    ap.add_argument("--out", required=True, help="Output .safetensors sidecar path.")
    ap.add_argument("--meta-out", default="", help="Optional JSON metadata output path.")
    ap.add_argument("--device", default="auto", choices=["auto", "xpu", "cuda", "cpu"], help="Execution device.")
    ap.add_argument("--seq-len", type=int, default=512, help="Tokens per calibration window (default: 512).")
    ap.add_argument("--prose-files", nargs="*", default=[], help="Prose calibration file(s) (.txt/.md/.parquet).")
    ap.add_argument("--prose-seqs", type=int, default=2, help="Number of seq-len windows drawn from --prose-files.")
    ap.add_argument("--prose-from-tail", action="store_true", help="Take prose windows from tail offset (for disjoint split).")
    ap.add_argument("--prose-tail-offset", type=int, default=4096, help="Tail token offset when --prose-from-tail is set.")
    ap.add_argument("--code-files", nargs="*", default=[], help="Code calibration file(s) (.py/.c/.rs/etc.).")
    ap.add_argument("--code-seqs", type=int, default=2, help="Number of seq-len windows drawn from --code-files.")
    ap.add_argument("--include-pos0", action="store_true", help="Disable pos0 exclusion (not recommended; ablation only).")
    ap.add_argument("--zero-bias", action="store_true", help="Export exact IEEE-754 -0.0f control sidecar (bit-identity audit).")
    ap.add_argument("--cal-label", default="custom", help="Short label stored in safetensors metadata.")
    ap.add_argument("--eval-prose", nargs="+", default=[], help="Held-out prose file(s) for the KL gate (head of the text).")
    ap.add_argument("--eval-code", nargs="+", default=[], help="Held-out code file(s) for the KL gate (head of the text).")
    ap.add_argument("--eval-seqs", type=int, default=4, help="Number of seq-len windows per eval domain (default: 4).")
    ap.add_argument("--kl-gate", action=argparse.BooleanOptionalAction, default=None,
                    help="Gate the sidecar on causal KL (default: on when --eval-prose and --eval-code are given).")
    args = ap.parse_args(argv)

    has_eval = (bool(args.eval_prose), bool(args.eval_code))
    gate_on = all(has_eval) if args.kl_gate is None else args.kl_gate
    if any(has_eval) and not all(has_eval):
        ap.error("the KL gate needs both --eval-prose and --eval-code")
    if gate_on and not all(has_eval):
        ap.error("--kl-gate needs --eval-prose and --eval-code")
    if args.zero_bias and (gate_on or any(has_eval)):
        ap.error("--zero-bias is a null control and is never gated or published")
    if gate_on and (args.seq_len <= BLOCK + 1 or args.eval_seqs < 1):
        ap.error(f"the KL gate needs --seq-len > {BLOCK + 1} and --eval-seqs >= 1")
    meta_out = args.meta_out or (os.path.splitext(args.out)[0] + ".meta.json" if gate_on else "")

    t0 = time.perf_counter()
    dev = select_device(args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    text_cfg, classes = load_arch(args.bf16_dir)
    get_mlx = shard_reader(args.mlx_dir)

    if args.zero_bias:
        tensors_out = {}
        for lidx in range(text_cfg.num_hidden_layers):
            with torch.device("meta"):
                layer = classes["layer"](text_cfg, lidx)
            mlx_prefix = f"language_model.model.layers.{lidx}."
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    n_out = mod.out_features
                    tensors_out[f"{mlx_prefix}{name}.bias_corr"] = torch.full((n_out,), -0.0, dtype=torch.float32)
        save_file(tensors_out, args.out, metadata={
            "format": "tensorfold-biascorr-v1",
            "control": "ieee754_neg_zero",
            "num_tensors": str(len(tensors_out)),
        })
        sidecar_sha = sha256_file(args.out)
        print(f"Exported -0.0f control sidecar: {args.out} ({len(tensors_out)} tensors, sha256={sidecar_sha})")
        return 0

    tok = AutoTokenizer.from_pretrained(args.bf16_dir)
    cal_rows: List[List[int]] = []
    if args.prose_files and args.prose_seqs > 0:
        cal_rows.extend(tokenize_windows(tok, args.prose_files, args.seq_len, args.prose_seqs, args.prose_from_tail, args.prose_tail_offset))
    if args.code_files and args.code_seqs > 0:
        cal_rows.extend(tokenize_windows(tok, args.code_files, args.seq_len, args.code_seqs, from_tail=False))
    if not cal_rows:
        raise ValueError("Provide at least one of --prose-files or --code-files with non-zero seq count.")

    # Tokenize eval windows before the expensive passes so bad inputs fail early.
    eval_windows = {}
    if gate_on:
        for dom, files in zip(GATE_DOMAINS, (args.eval_prose, args.eval_code)):
            eval_windows[dom] = tokenize_windows(tok, files, args.seq_len, args.eval_seqs)
            if any(w in cal_rows for w in eval_windows[dom]):
                ap.error(f"a {dom} eval window is also a calibration window; eval text must be held out")
    cal_sha = sha256_ids(cal_rows)
    cal_tensor = torch.tensor(cal_rows, dtype=torch.long, device=dev)
    B, S = cal_tensor.shape
    scored_toks = B * (S - (0 if args.include_pos0 else 1))
    print(f"[export_biascorr] device={dev} | cal={args.cal_label} | shape={B}x{S} | scored_toks={scored_toks} | pos0_excl={not args.include_pos0} | cal_sha={cal_sha[:16]}", flush=True)

    get_bf = shard_reader(args.bf16_dir)

    scratch = torch.empty(17408 * 5120, device=dev, dtype=torch.bfloat16) if dev.type in ("xpu", "cuda") else None
    rotary_emb = make_rotary(classes, text_cfg, dev)

    with torch.no_grad():
        emb_bf = get_bf("model.language_model.embed_tokens.weight").to(device=dev, dtype=torch.bfloat16)
        cal_h = F.embedding(cal_tensor, emb_bf)
        del emb_bf
        device_empty_cache(dev)

    run_layer = layer_runner(classes, text_cfg, rotary_emb, cal_h)

    tensors_out = {}
    norms = []
    compute_s = 0.0

    for lidx in range(text_cfg.num_hidden_layers):
        bf_layer = load_bf16_layer(get_bf, classes, text_cfg, lidx, dev)

        linears_bf = [(n, m) for n, m in bf_layer.named_modules() if isinstance(m, nn.Linear)]

        device_sync(dev)
        t_bc0 = time.perf_counter()
        mean_x = {}
        hooks = []
        for n, m in linears_bf:
            def _mk(nm):
                def _h(mod, inp, out):
                    x0 = inp[0].detach().float()
                    if (not args.include_pos0) and x0.ndim == 3 and x0.shape[1] > 1:
                        x0 = x0[:, 1:, :]
                    mean_x[nm] = x0.reshape(-1, x0.shape[-1]).mean(dim=0)
                return _h
            hooks.append(m.register_forward_hook(_mk(n)))
        with torch.no_grad():
            cal_h = run_layer(bf_layer, lidx, cal_h)
        for h in hooks:
            h.remove()

        mlx_prefix = f"language_model.model.layers.{lidx}."
        with torch.no_grad():
            for n, m in linears_bf:
                y_bf = F.linear(mean_x[n].unsqueeze(0), m.weight.data.float()).squeeze(0)
                w_i32 = get_mlx(f"{mlx_prefix}{n}.weight").to(dev)
                s_bf16 = get_mlx(f"{mlx_prefix}{n}.scales").to(dev, dtype=torch.bfloat16)
                b_bf16 = get_mlx(f"{mlx_prefix}{n}.biases").to(dev, dtype=torch.bfloat16)
                w_q = unpack_mlx_u32_bf16(w_i32, s_bf16, b_bf16, scratch=scratch).float()
                y_q = F.linear(mean_x[n].unsqueeze(0), w_q).squeeze(0)
                db = (y_bf - y_q).contiguous().cpu().to(torch.float32)
                if not torch.isfinite(db).all().item():
                    raise RuntimeError(f"Non-finite bias at layer {lidx}.{n}")
                tensors_out[f"{mlx_prefix}{n}.bias_corr"] = db
                norms.append(float(db.norm().item()))
                del w_i32, s_bf16, b_bf16, w_q
        device_sync(dev)
        compute_s += time.perf_counter() - t_bc0

        del bf_layer, mean_x
        if lidx % 8 == 7 or lidx == text_cfg.num_hidden_layers - 1:
            gc.collect()
            device_empty_cache(dev)
            print(f"  L{lidx+1:02d}/{text_cfg.num_hidden_layers} | tensors={len(tensors_out)} | mean_l2={sum(norms)/len(norms):.4f} | wall={time.perf_counter()-t0:.1f}s", flush=True)

    save_file(tensors_out, args.out, metadata={
        "format": "tensorfold-biascorr-v1",
        "cal_label": args.cal_label,
        "cal_shape": f"{B}x{S}",
        "cal_scored_tokens": str(scored_toks),
        "cal_token_ids_sha256": cal_sha,
        "pos0_excluded": str(not args.include_pos0).lower(),
        "num_tensors": str(len(tensors_out)),
    })
    os.chmod(args.out, 0o644)
    sidecar_sha = sha256_file(args.out)
    sz = os.path.getsize(args.out)
    meta = {
        "cal_label": args.cal_label,
        "cal_shape": [B, S],
        "cal_scored_tokens": scored_toks,
        "cal_token_ids_sha256": cal_sha,
        "pos0_excluded": not args.include_pos0,
        "num_tensors": len(tensors_out),
        "size_bytes": sz,
        "size_mib": round(sz / (1024 * 1024), 3),
        "sidecar_sha256": sidecar_sha,
        "compute_s": round(compute_s, 3),
        "wall_s": round(time.perf_counter() - t0, 2),
        "mean_l2": round(sum(norms) / len(norms), 5),
        "max_l2": round(max(norms), 5),
        "status": GATE_PENDING if gate_on else GATE_SKIPPED,
    }
    # Written before the gate runs: a crashed gate must not leave an older verdict beside this sidecar.
    if meta_out:
        write_json(meta_out, meta)
    print(f"\nDONE: {args.out} ({meta['size_mib']} MiB, {meta['num_tensors']} F32 tensors, sha256={sidecar_sha[:16]}, wall={meta['wall_s']}s)")
    if not gate_on:
        print(f"status={GATE_SKIPPED}: pass --eval-prose and --eval-code to run the KL gate", flush=True)
        return 0

    del cal_h, cal_tensor, scratch, run_layer
    gc.collect()
    device_empty_cache(dev)
    t_gate = time.perf_counter()
    gate = gate_sidecar(eval_causal_kl(args.bf16_dir, args.mlx_dir, args.out, eval_windows, dev))
    cal_grams = ngrams(cal_rows)
    gate["eval"] = {
        dom: {"token_ids_sha256": sha256_ids(rows), "cal_6gram_overlap": len(ngrams(rows) & cal_grams)}
        for dom, rows in eval_windows.items()
    }
    gate["wall_s"] = round(time.perf_counter() - t_gate, 2)
    meta.update(status=gate["status"], kl_gate=gate)
    write_json(meta_out, meta)
    for dom, d in gate["domains"].items():
        lo, hi = d["delta_plus_ci95_blk32"]
        print(f"  {dom}: KL {d['kl_base']:.5f} -> {d['kl_plus']:.5f} (delta {d['delta_plus']:+.5f}, 95% CI [{lo:+.5f}, {hi:+.5f}]) | -b {d['kl_minus']:.5f}")
        if gate["eval"][dom]["cal_6gram_overlap"]:
            print(f"  WARNING: {dom} eval shares {gate['eval'][dom]['cal_6gram_overlap']} token 6-grams with calibration")
    if gate["status"] != GATE_PASS:
        print(f"status={GATE_FAIL}: failed {', '.join(gate['failed_checks'])} (see {meta_out})", file=sys.stderr, flush=True)
        return GATE_EXIT_CODE
    print(f"status={GATE_PASS} ({meta_out})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
