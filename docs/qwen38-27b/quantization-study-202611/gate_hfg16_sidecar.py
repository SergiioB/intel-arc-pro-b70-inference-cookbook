#!/usr/bin/env python3
"""Extract the BiasCorr tensors stored in an HF-INT4-G16 checkpoint and KL-gate them.

The sidecar holds the checkpoint's own `*.bias_corr` tensors, byte for byte,
under the checkpoint's own key names. The saved file is reloaded from disk and
scored with the same rule as export_biascorr.py (`gate_sidecar`): BF16 against
the HF-INT4-G16 W4A16 weights with no correction, +b and -b, streamed one
decoder layer at a time with causal attention.

Scope: weights are decoded with the plugin's reference decoder and run in a BF16
causal stream. This measures the checkpoint's weights and corrections. It does
not run the vLLM kernels.

Needs export_biascorr.py beside this file and the hf_int4_g16_xpu package on
PYTHONPATH. Exit code 0 is PUBLISHABLE, 3 is REJECTED_BY_KL_GATE.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

import export_biascorr as eb
from hf_int4_g16_xpu.format import HADAMARD_BLOCK, make_had128, signs_for_k, unpack_hf_int4_reference

PACKED_SUFFIXES = (".q_pack", ".u_pack", ".d_row", ".bias_corr")


def checkpoint_index(root: str) -> dict:
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def payload_sha256(root: str, wmap: dict, key: str) -> str:
    """SHA-256 of one tensor's raw bytes, read straight from its shard."""
    with open(os.path.join(root, wmap[key]), "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        start, end = json.loads(f.read(header_len))[key]["data_offsets"]
        f.seek(8 + header_len + start)
        digest, left = hashlib.sha256(), end - start
        while left:
            chunk = f.read(min(left, 1 << 22))
            if not chunk:
                raise ValueError(f"short read for {key}")
            digest.update(chunk)
            left -= len(chunk)
    return digest.hexdigest()


def verify_unquantized_tensors(bf16_dir: str, ckpt_dir: str) -> int:
    """Every tensor the converter copied must equal BF16, so the BF16 layer can host the arms."""
    bf_map, q_map = checkpoint_index(bf16_dir), checkpoint_index(ckpt_dir)
    copied = [k for k in q_map if not k.endswith(PACKED_SUFFIXES)]
    bad = [k for k in copied if k not in bf_map or payload_sha256(ckpt_dir, q_map, k) != payload_sha256(bf16_dir, bf_map, k)]
    if bad:
        raise ValueError(f"{len(bad)} copied tensors differ from BF16, first: {bad[0]}")
    return len(copied)


def extract_sidecar(ckpt_dir: str, out: str, calibration: dict) -> int:
    get = eb.shard_reader(ckpt_dir)
    keys = sorted(k for k in get.keys if k.endswith(".bias_corr"))
    tensors = {k: get(k).clone() for k in keys}
    for key, t in tensors.items():
        if t.dtype != torch.float32 or t.ndim != 1 or not torch.isfinite(t).all():
            raise ValueError(f"{key}: expected a finite 1D F32 tensor")
    save_file(tensors, out, metadata={
        "format": "hf_int4_g16-biascorr-v1",
        "source": "extracted from the HF-INT4-G16-BC checkpoint",
        "cal_token_ids_sha256": calibration["cal_token_sha256"],
        "pos0_excluded": str(calibration["streaming"]["pos0_excluded"]).lower(),
        "num_tensors": str(len(tensors)),
    })
    os.chmod(out, 0o644)
    # Reload and compare raw bytes against the checkpoint shards.
    q_map = checkpoint_index(ckpt_dir)
    with safe_open(out, framework="pt", device="cpu") as f:
        for key in keys:
            if hashlib.sha256(f.get_tensor(key).numpy().tobytes()).hexdigest() != payload_sha256(ckpt_dir, q_map, key):
                raise ValueError(f"{key}: sidecar bytes differ from the checkpoint")
    return len(keys)


def calibration_rows(tok, calibration: dict, train_parquet: str) -> list:
    """Rebuild the converter's calibration tokens and require its recorded hash."""
    import pyarrow.parquet as pq

    width = calibration["tokens_per_split"]
    train = "\n".join(pq.read_table(train_parquet)["text"].to_pylist())
    code = "\n\n".join(open(p, encoding="utf-8", errors="ignore").read() for p in calibration["code_sources"])
    rows = []
    for text, start in ((train, calibration["prose_token_start"]), (code, calibration["code_token_start"])):
        rows.append(tok.encode(text, add_special_tokens=False)[start:start + width])
    got = eb.sha256_ids(rows)
    if got != calibration["cal_token_sha256"]:
        raise ValueError(f"rebuilt calibration hash {got} != recorded {calibration['cal_token_sha256']}")
    return rows


@torch.no_grad()
def eval_causal_kl(bf16_dir: str, ckpt_dir: str, sidecar_path: str, windows: dict, dev: torch.device):
    """Per-position KL vs BF16 for the HF-INT4-G16 weights with no correction, +b and -b."""
    text_cfg, classes = eb.load_arch(bf16_dir)
    get_bf, get_q = eb.shard_reader(bf16_dir), eb.shard_reader(ckpt_dir)
    with safe_open(sidecar_path, framework="pt", device="cpu") as f:
        sidecar = {k: f.get_tensor(k) for k in f.keys()}
    rotary_emb = eb.make_rotary(classes, text_cfg, dev)
    had = make_had128(dev)
    signs = {}
    ids = {dom: torch.tensor(windows[dom], dtype=torch.long) for dom in eb.GATE_DOMAINS}

    def decode(name: str) -> torch.Tensor:
        # Rotated weight is d_row * (q * u); H128 is its own inverse, signs are +-1.
        wi8 = unpack_hf_int4_reference(get_q(name + ".q_pack").to(dev), get_q(name + ".u_pack").to(dev))
        k = wi8.shape[1]
        if k not in signs:
            signs[k] = signs_for_k(k, device=dev).float()
        rotated = wi8.float() * get_q(name + ".d_row").to(dev)[:, None]
        return ((rotated.reshape(-1, HADAMARD_BLOCK) @ had).reshape(rotated.shape) * signs[k]).to(torch.bfloat16)

    # Embedding, norms and lm_head are verified byte-identical to BF16 by verify_unquantized_tensors.
    emb = get_bf("model.language_model.embed_tokens.weight")
    ref = {dom: emb[ids[dom]].to(dev, dtype=torch.bfloat16) for dom in eb.GATE_DOMAINS}
    del emb
    arms = {dom: {arm: ref[dom] for arm in eb.GATE_ARMS} for dom in eb.GATE_DOMAINS}
    runners = {dom: eb.layer_runner(classes, text_cfg, rotary_emb, ref[dom]) for dom in eb.GATE_DOMAINS}

    used, l0_sqnr_db = set(), None
    for lidx in range(text_cfg.num_hidden_layers):
        layer = eb.load_bf16_layer(get_bf, classes, text_cfg, lidx, dev)
        for dom in eb.GATE_DOMAINS:
            ref[dom] = runners[dom](layer, lidx, ref[dom])
        corr = {}
        for n, m in [(n, m) for n, m in layer.named_modules() if isinstance(m, nn.Linear)]:
            name = f"model.language_model.layers.{lidx}.{n}"
            key = f"{name}.bias_corr"
            if key not in sidecar:
                raise ValueError(f"KL gate: sidecar has no tensor {key}")
            used.add(key)
            w = decode(name)
            if lidx == 0 and n == "mlp.gate_proj":
                exact = m.weight.data.float()
                l0_sqnr_db = float(10 * torch.log10(exact.pow(2).sum() / (exact - w.float()).pow(2).sum()))
                if not l0_sqnr_db > 10.0:
                    raise ValueError(f"decoded layer 0 SQNR {l0_sqnr_db:.2f} dB: decoder or layout is wrong")
                print(f"  decode check: layer 0 gate_proj SQNR {l0_sqnr_db:.4f} dB", flush=True)
            m.weight.data = w
            corr[m] = sidecar[key].to(dev, dtype=torch.float32)
        for arm, sign in (("base", 0), ("plus", 1), ("minus", -1)):
            hooks = [] if sign == 0 else [
                m.register_forward_hook(lambda _m, _i, out, b=sign * bc: (out.float() + b).to(out.dtype))
                for m, bc in corr.items()
            ]
            for dom in eb.GATE_DOMAINS:
                arms[dom][arm] = runners[dom](layer, lidx, arms[dom][arm])
            for h in hooks:
                h.remove()
        del layer, corr
        if lidx % 8 == 7 or lidx == text_cfg.num_hidden_layers - 1:
            gc.collect()
            eb.device_empty_cache(dev)
            print(f"  KL eval L{lidx+1:02d}/{text_cfg.num_hidden_layers}", flush=True)
    if used != set(sidecar):
        raise ValueError(f"KL gate: {len(set(sidecar) - used)} sidecar tensors match no decoder linear")

    with torch.device("meta"):
        norm = classes["norm"](text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)
    norm = norm.to_empty(device="cpu")
    norm.load_state_dict({"weight": get_bf("model.language_model.norm.weight")})
    norm = norm.to(device=dev, dtype=torch.bfloat16).eval()
    head = get_bf("lm_head.weight").to(dev, dtype=torch.bfloat16)

    kl = {}
    for dom in eb.GATE_DOMAINS:
        def last(h):
            return norm(h)[:, :-1, :].reshape(-1, text_cfg.hidden_size)

        h_ref = last(ref[dom])
        h_arm = {arm: last(arms[dom][arm]) for arm in eb.GATE_ARMS}
        out = {arm: [] for arm in eb.GATE_ARMS}
        for i0 in range(0, h_ref.shape[0], 64):
            ref_logp = F.log_softmax(F.linear(h_ref[i0:i0 + 64], head).float(), dim=-1)
            ref_p = ref_logp.exp()
            for arm in eb.GATE_ARMS:
                q_logp = F.log_softmax(F.linear(h_arm[arm][i0:i0 + 64], head).float(), dim=-1)
                out[arm].append((ref_p * (ref_logp - q_logp)).sum(dim=-1).clamp(min=0.0).cpu())
        n_win, n_pos = ids[dom].shape[0], ids[dom].shape[1] - 1
        kl[dom] = {arm: torch.cat(v).double().numpy().reshape(n_win, n_pos) for arm, v in out.items()}
    return kl, l0_sqnr_db


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bf16-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True, help="HF-INT4-G16-BC checkpoint directory.")
    ap.add_argument("--out", required=True, help="Output .safetensors sidecar path.")
    ap.add_argument("--train-parquet", required=True, help="WikiText-2 raw train parquet (rebuilds calibration tokens).")
    ap.add_argument("--eval-prose", nargs="+", required=True)
    ap.add_argument("--eval-code", nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--eval-seqs", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "xpu", "cuda", "cpu"])
    args = ap.parse_args()

    t0 = time.perf_counter()
    dev = eb.select_device(args.device)
    meta_out = os.path.splitext(args.out)[0] + ".meta.json"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(os.path.join(args.ckpt_dir, "hf_int4_g16_conversion_report.json")) as f:
        calibration = json.load(f)["calibration"]

    tok = AutoTokenizer.from_pretrained(args.bf16_dir)
    cal = calibration_rows(tok, calibration, args.train_parquet)
    windows = {
        dom: eb.tokenize_windows(tok, files, args.seq_len, args.eval_seqs)
        for dom, files in zip(eb.GATE_DOMAINS, (args.eval_prose, args.eval_code))
    }
    print(f"calibration tokens rebuilt, sha256={calibration['cal_token_sha256'][:16]}", flush=True)

    copied = verify_unquantized_tensors(args.bf16_dir, args.ckpt_dir)
    print(f"{copied} copied tensors are byte-identical to BF16", flush=True)
    count = extract_sidecar(args.ckpt_dir, args.out, calibration)
    sidecar_sha = eb.sha256_file(args.out)
    size = os.path.getsize(args.out)
    meta = {
        "cal_label": f"converter_{calibration['mode']}",
        "cal_shape": calibration["streaming"]["batch_shape"],
        "cal_scored_tokens": calibration["rows"] * (calibration["tokens_per_split"] - 1),
        "cal_token_ids_sha256": calibration["cal_token_sha256"],
        "pos0_excluded": calibration["streaming"]["pos0_excluded"],
        "num_tensors": count,
        "size_bytes": size,
        "size_mib": round(size / (1024 * 1024), 3),
        "sidecar_sha256": sidecar_sha,
        "tensor_bytes_match_checkpoint": True,
        "copied_tensors_identical_to_bf16": copied,
        "status": eb.GATE_PENDING,
    }
    eb.write_json(meta_out, meta)
    print(f"sidecar: {count} tensors, {meta['size_mib']} MiB, sha256={sidecar_sha}", flush=True)

    t_gate = time.perf_counter()
    kl, l0_sqnr_db = eval_causal_kl(args.bf16_dir, args.ckpt_dir, args.out, windows, dev)
    gate = eb.gate_sidecar(kl)
    cal_grams = eb.ngrams(cal)
    gate["eval"] = {
        dom: {"token_ids_sha256": eb.sha256_ids(rows), "cal_6gram_overlap": len(eb.ngrams(rows) & cal_grams)}
        for dom, rows in windows.items()
    }
    gate["wall_s"] = round(time.perf_counter() - t_gate, 2)
    meta.update(status=gate["status"], kl_gate=gate, decode_check_l0_gate_proj_sqnr_db=round(l0_sqnr_db, 4),
                wall_s=round(time.perf_counter() - t0, 2))
    eb.write_json(meta_out, meta)
    for dom, d in gate["domains"].items():
        lo, hi = d["delta_plus_ci95_blk32"]
        print(f"  {dom}: KL {d['kl_base']:.5f} -> {d['kl_plus']:.5f} (delta {d['delta_plus']:+.5f}, 95% CI [{lo:+.5f}, {hi:+.5f}]) | -b {d['kl_minus']:.5f}")
        print(f"  {dom}: calibration/eval shared 6-grams = {gate['eval'][dom]['cal_6gram_overlap']}")
    if gate["status"] != eb.GATE_PASS:
        print(f"status={eb.GATE_FAIL}: failed {', '.join(gate['failed_checks'])}", file=sys.stderr, flush=True)
        return eb.GATE_EXIT_CODE
    print(f"status={eb.GATE_PASS} ({meta_out})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
