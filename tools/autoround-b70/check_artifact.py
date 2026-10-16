#!/usr/bin/env python3
"""Static (CPU-only) checker for the Bolt 74B-A9B AutoRound W4A16 vLLM export.

Compares the exported artifact in --artifact against the BF16 source in --src:

  A. quantized-module inventory, predicted from the source index + the
     config's llm_config.block_configs (routed / shared / latent / mamba /
     attention), with exact layer-set coverage;
  B. scale sanity: no NaN, no Inf, no zero scale anywhere;
  C. fraction of mamba `out_proj` scales below 6.1e-5 (reported, not failed);
  D. every `e_score_correction_bias` present, float32, byte-identical to source;
  E. non-quantized tensors (A_log / D / dt_bias / conv1d / every norm / router
     gate / vision tower / mlp1 projector / embeddings / lm_head)
     byte-identical to source.

`--identity-only` runs only D+E (that is what is used to prove the reader walks
every shard; it is valid against `--artifact == --src`).

Read-only: never writes into --src or --artifact. Only safetensors + torch are
required (no GPU, no CUDA). Exit status = number of failed checks (capped 99).

Usage:
  check_artifact.py --src $SRC --artifact $OUT/bolt74b-bf16-w4g128 --json check.json
  check_artifact.py --identity-only --src $SRC --artifact $SRC     # reader self-test
  (--src / --artifact default to the SRC / ARTIFACT environment variables)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys

DEFAULT_SRC = os.environ.get("SRC")
DEFAULT_ART = os.environ.get("ARTIFACT")

OUT_PROJ_SCALE_THRESHOLD = 6.1e-5
CHUNK_ROWS = 256

# A module is "quantized" iff its source name matches this and it is a Linear
# that convert_nemotron_omni.py hands to AutoRound (ignore_layers =
# vision_model,vision_projector,lm_head,embeddings).
QUANT_RE = re.compile(
    r"^language_model\.backbone\.layers\.(?P<layer>\d+)\.mixer\."
    r"(?P<body>"
    r"experts\.\d+\.(?:up|down)_proj"
    r"|shared_experts\.(?:up|down)_proj"
    r"|fc[12]_latent_proj"
    r"|in_proj"
    r"|out_proj"
    r"|q_proj|k_proj|v_proj|o_proj"
    r")\.weight$"
)

QPACK_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx", ".weight_scale",
                  ".weight_shape", ".weight_zero_point")
# suffix that marks a module as carrying quantized weights
QUANT_MARKERS = (".qweight", ".weight_scale")

E_SCORE_RE = re.compile(r"\.mixer\.gate\.e_score_correction_bias$")
GATE_RE = re.compile(r"\.mixer\.gate\.weight$")
MAMBA_CORE_RE = re.compile(r"\.mixer\.(?:A_log|D|dt_bias|conv1d\.weight|conv1d\.bias)$")
NORM_RE = re.compile(r"\.norm\.weight$|\.norm_f\.weight$")


def log(msg=""):
    print(msg, flush=True)


def fail(state, name, detail):
    state["checks"].append({"name": name, "ok": False, "detail": detail})
    log(f"  FAIL  {name}: {detail}")
    return False


def ok(state, name, detail):
    state["checks"].append({"name": name, "ok": True, "detail": detail})
    log(f"  ok    {name}: {detail}")
    return True


def info(state, name, detail):
    state["notes"].append({"name": name, "detail": detail})
    log(f"  ----  {name}: {detail}")


# ---------------------------------------------------------------- shard utils

# The export renames the nested LM prefix (AutoRound's nemotron_h_puzzle rule); vLLM's
# nano_nemotron_vl mapper produces the same target names. Compare under the source names.
ART_ALIASES = (("language_model.model.", "language_model.backbone."),)
REAL_NAME = {}  # source-style name -> name stored in the artifact shard


def to_source_name(k):
    for new, old in ART_ALIASES:
        if k.startswith(new):
            return old + k[len(new):]
    return k


def scan_dir(d, alias=False):
    """name -> absolute shard path, plus the ordered shard list."""
    from safetensors import safe_open

    mapping, dupes, shards, unreadable = {}, [], [], []
    if not os.path.isdir(d):
        return mapping, dupes, shards, unreadable
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".safetensors"):
            continue
        p = os.path.join(d, fn)
        shards.append(p)
        try:
            with safe_open(p, framework="pt", device="cpu") as h:
                for k in h.keys():
                    n = to_source_name(k) if alias else k
                    if n in mapping:
                        dupes.append((n, mapping[n], p))
                    mapping[n] = p
                    if n != k:
                        REAL_NAME[n] = k
        except Exception as exc:  # partial download / truncated file
            shards.pop()
            unreadable.append((p, f"{type(exc).__name__}: {exc}"))
    return mapping, dupes, shards, unreadable


def to_bytes(t):
    import torch
    return t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def same_values(src_path, art_path, name):
    """True when the artifact tensor holds exactly the source values (e.g. BF16 widened to F32)."""
    from safetensors import safe_open
    import torch

    with safe_open(src_path, framework="pt", device="cpu") as f:
        s = f.get_tensor(name)
    with safe_open(art_path, framework="pt", device="cpu") as f:
        a = f.get_tensor(REAL_NAME.get(name, name))
    return s.dtype != a.dtype and torch.equal(s.to(a.dtype), a)


def hash_tensor(path, name, chunk_rows=CHUNK_ROWS):
    """sha256 over the raw little-endian bytes of a tensor, read in row chunks."""
    from safetensors import safe_open
    import torch  # noqa: F401  (needed by to_bytes)

    h = hashlib.sha256()
    with safe_open(path, framework="pt", device="cpu") as f:
        name = name if name in f.keys() else REAL_NAME.get(name, name)
        sl = f.get_slice(name)
        shape = list(sl.get_shape())
        dtype = str(sl.get_dtype())
        if not shape:
            h.update(to_bytes(f.get_tensor(name)))
            return h.hexdigest(), shape, dtype
        n = shape[0]
        for i in range(0, max(n, 1), chunk_rows):
            h.update(to_bytes(sl[i:i + chunk_rows]))
    return h.hexdigest(), shape, dtype


# ---------------------------------------------------------------- predictions

def predict(src_index, src_config):
    """Expected quantized-module sets, derived from the source index + config."""
    block_configs = src_config["llm_config"]["block_configs"]
    moe_layers = [i for i, b in enumerate(block_configs) if b["block_type"] == "moe"]
    mamba_layers = [i for i, b in enumerate(block_configs) if b["block_type"] == "mamba"]
    attn_layers = [i for i, b in enumerate(block_configs) if b["block_type"] == "attention"]
    routed_experts = sum(b["n_routed_experts"] for b in block_configs if b["block_type"] == "moe")

    expected = {"routed": set(), "shared": set(), "latent": set(), "mamba": set(), "attn": set()}
    for name in src_index:
        m = QUANT_RE.match(name)
        if not m:
            continue
        base = name[: -len(".weight")]
        body = m.group("body")
        if body.startswith("experts."):
            kind = "routed"
        elif body.startswith("shared_experts."):
            kind = "shared"
        elif body.startswith("fc") and body.endswith("_latent_proj"):
            kind = "latent"
        elif body in ("in_proj", "out_proj"):
            kind = "mamba"
        else:
            kind = "attn"
        expected[kind].add(base)

    counted = {
        "routed": routed_experts * 2,          # up_proj + down_proj per expert
        "shared": len(moe_layers) * 2,
        "latent": len(moe_layers) * 2,         # fc1_latent_proj + fc2_latent_proj
        "mamba": len(mamba_layers) * 2,        # in_proj + out_proj
        "attn": len(attn_layers) * 4,          # q/k/v/o
    }
    return expected, counted, {
        "moe_layers": sorted(moe_layers),
        "mamba_layers": sorted(mamba_layers),
        "attn_layers": sorted(attn_layers),
        "routed_experts": routed_experts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--artifact", default=DEFAULT_ART)
    ap.add_argument("--identity-only", action="store_true",
                    help="run only the byte-identity checks (valid with --artifact == --src)")
    ap.add_argument("--json", default=None, help="also write the full report as JSON here")
    ap.add_argument("--quiet-scale-hist", action="store_true")
    args = ap.parse_args()
    if not args.src or not args.artifact:
        ap.error("--src and --artifact (or SRC and ARTIFACT) are required")

    state = {"checks": [], "notes": [], "src": args.src, "artifact": args.artifact}

    if not os.path.isdir(args.src):
        log(f"FATAL: source dir missing: {args.src}")
        return 99
    src_idx_path = os.path.join(args.src, "model.safetensors.index.json")
    if not os.path.exists(src_idx_path):
        log(f"FATAL: source index missing: {src_idx_path}")
        return 99
    src_index = json.load(open(src_idx_path))["weight_map"]
    src_config = json.load(open(os.path.join(args.src, "config.json")))

    log(f"source   : {args.src}  ({len(src_index)} indexed tensors)")
    log(f"artifact : {args.artifact}")

    src_map, src_dupes, src_shards, src_unreadable = scan_dir(args.src)
    missing_shards = sorted({s for s in set(src_index.values()) if not os.path.exists(os.path.join(args.src, s))})
    log(f"source shards readable: {len(src_shards)}  missing: {len(missing_shards)}  partial/unreadable: {len(src_unreadable)}"
        + (f"  e.g. {[os.path.basename(p) for p, _ in src_unreadable[:3]]}" if src_unreadable else ""))
    if src_dupes:
        log(f"  WARN duplicate source tensor names: {src_dupes[:3]}")

    art_map, art_dupes, art_shards, art_unreadable = scan_dir(args.artifact, alias=True)
    log(f"artifact shards readable: {len(art_shards)}  tensors: {len(art_map)}"
        + (f"  partial/unreadable: {len(art_unreadable)}" if art_unreadable else ""))
    if art_dupes:
        fail(state, "no-duplicate-artifact-names", f"{len(art_dupes)} duplicated names, e.g. {art_dupes[:3]}")
    same_dir = os.path.realpath(args.artifact) == os.path.realpath(args.src)

    # artifact-side metadata: complete shard set + int4 quantization_config
    art_idx_path = os.path.join(args.artifact, "model.safetensors.index.json")
    if os.path.exists(art_idx_path) and not same_dir:
        art_idx = json.load(open(art_idx_path))["weight_map"]
        miss = sorted({s for s in set(art_idx.values())
                       if s.endswith(".safetensors") and not os.path.exists(os.path.join(args.artifact, s))})
        if miss:
            fail(state, "artifact-shards-complete", f"{len(miss)} shards in the artifact index are absent: {miss[:3]}")
        else:
            ok(state, "artifact-shards-complete", f"all {len(set(art_idx.values()))} indexed shards present")
    elif not same_dir:
        info(state, "artifact-shards-complete", "no model.safetensors.index.json in the artifact")
    art_cfg_path = os.path.join(args.artifact, "config.json")
    if os.path.exists(art_cfg_path) and not same_dir:
        qc = json.load(open(art_cfg_path)).get("quantization_config")
        if not qc:
            fail(state, "artifact-quant-config", "config.json carries no quantization_config")
        else:
            bits = qc.get("bits") or qc.get("weight_bits")
            gs = qc.get("group_size")
            detail = (f"quant_method={qc.get('quant_method')} bits={bits} group_size={gs} "
                      f"sym={qc.get('sym', qc.get('is_sym'))} desc_act={qc.get('desc_act')} "
                      f"modules_to_not_convert={qc.get('modules_to_not_convert')}")
            if bits == 4:
                ok(state, "artifact-quant-config", detail)
            else:
                fail(state, "artifact-quant-config", "bits must be 4 (XPU WNA16 supports int2/int4 only): " + detail)
    elif not same_dir:
        fail(state, "artifact-quant-config", "artifact config.json absent")

    # sidecar files vLLM needs to build the model (remote code + tokenizer + template).
    # auto-round does not guarantee they are copied into the export directory.
    if not same_dir:
        needed = [f for f in sorted(os.listdir(args.src))
                  if f.endswith(".py") or f in ("config.json", "generation_config.json",
                                                "tokenizer.json", "tokenizer_config.json",
                                                "chat_template.jinja", "preprocessor_config.json",
                                                "processor_config.json")]
        missing_side = [f for f in needed if not os.path.exists(os.path.join(args.artifact, f))]
        if missing_side:
            fail(state, "artifact-sidecar-files",
                 f"{len(missing_side)} required file(s) absent from the artifact "
                 f"(copy them from {args.src}): {missing_side}")
        else:
            ok(state, "artifact-sidecar-files", f"all {len(needed)} remote-code/tokenizer/template files present")

    # ------------------------------------------------------------ A. inventory
    expected, counted, layers = predict(src_index, src_config)
    total_expected = sum(counted.values())
    log("")
    log(f"[A] predicted quantized modules from BF16 index + block_configs")
    log(f"    {len(layers['moe_layers'])} moe + {len(layers['mamba_layers'])} mamba "
        f"+ {len(layers['attn_layers'])} attention layers, {layers['routed_experts']} routed experts")
    log(f"    routed={counted['routed']} shared={counted['shared']} latent={counted['latent']} "
        f"mamba={counted['mamba']} attn={counted['attn']}  TOTAL={total_expected}")
    state["expected_count"] = total_expected
    state["expected_breakdown"] = counted
    for kind in expected:
        if len(expected[kind]) != counted[kind]:
            fail(state, f"prediction-self-consistency-{kind}",
                 f"index yields {len(expected[kind])} modules but formula says {counted[kind]}")
    if total_expected == 27920:
        ok(state, "expected-count", f"{total_expected} == reviewer estimate 27920")
    else:
        info(state, "expected-count", f"{total_expected} (reviewer estimate was 27920)")

    if not args.identity_only:
        quant_bases = collections.defaultdict(set)
        for name in art_map:
            for suf in QPACK_SUFFIXES:
                if name.endswith(suf):
                    base = name[: -len(suf)]
                    quant_bases[base].add(suf)
                    break
        quantized = {b for b, s in quant_bases.items()
                     if any(x in s for x in QUANT_MARKERS)}
        # classify artifact modules with the source predicate (names are preserved)
        art_by_kind = collections.defaultdict(set)
        unknown_bases = []
        for b in sorted(quantized):
            m = QUANT_RE.match(b + ".weight")
            if not m:
                unknown_bases.append(b)
                continue
            body = m.group("body")
            kind = ("routed" if body.startswith("experts.") else
                    "shared" if body.startswith("shared_experts.") else
                    "latent" if body.startswith("fc") and body.endswith("_latent_proj") else
                    "mamba" if body in ("in_proj", "out_proj") else "attn")
            art_by_kind[kind].add(b)

        log("")
        log(f"[A] artifact quantized modules: {len(quantized)}  (expected {total_expected})")
        for kind in ("routed", "shared", "latent", "mamba", "attn"):
            got, want = len(art_by_kind[kind]), len(expected[kind])
            detail = f"{got} found, {want} expected"
            if got == want and art_by_kind[kind] == expected[kind]:
                ok(state, f"count-{kind}", detail + ", layer set identical")
            else:
                miss = sorted(expected[kind] - art_by_kind[kind])
                extra = sorted(art_by_kind[kind] - expected[kind])
                fail(state, f"count-{kind}",
                     f"{detail}; missing={len(miss)} {miss[:3]} unexpected={len(extra)} {extra[:3]}")
        if unknown_bases:
            fail(state, "no-unexpected-quantized",
                 f"{len(unknown_bases)} quantized modules outside the planned scope: {unknown_bases[:5]}")
        else:
            ok(state, "no-unexpected-quantized", "every quantized module is inside the planned Linear scope")

        # unexpected keys that are not quant params and not in the source
        src_and_art_extra = [k for k in art_map
                             if k not in src_map and not k.endswith(QPACK_SUFFIXES)
                             and k != "model_extra_tensors.safetensors"]
        if src_and_art_extra:
            info(state, "artifact-only-keys",
                 f"{len(src_and_art_extra)} keys absent from source: {src_and_art_extra[:5]}")
        else:
            ok(state, "no-stray-keys", "no artifact-only non-quant keys")

        # ------------------------------------------------- B/C. scale sanity
        log("")
        log("[B] scales: NaN / Inf / zero")
        scales = {}
        for base in sorted(quantized):
            for suf in (".scales", ".weight_scale"):
                n = base + suf
                if n in art_map:
                    scales[base] = (n, art_map[n])
                    break
        if len(scales) != len(quantized):
            fail(state, "scales-present",
                 f"{len(quantized) - len(scales)} quantized modules have no .scales/.weight_scale")
        else:
            ok(state, "scales-present", f"all {len(scales)} quantized modules carry a scale tensor")

        import torch
        bad_nan = bad_inf = bad_zero = 0
        worst = []
        out_proj_small = out_proj_total = 0
        out_proj_hist = collections.Counter()
        with torch.no_grad():
            for base, (name, path) in scales.items():
                from safetensors import safe_open
                with safe_open(path, framework="pt", device="cpu") as f:
                    t = f.get_tensor(REAL_NAME.get(name, name))
                t = t.to(torch.float32)
                n_nan = int(torch.isnan(t).sum())
                n_inf = int(torch.isinf(t).sum())
                a = t[~torch.isnan(t) & ~torch.isinf(t)].abs()
                n_zero = int((a == 0).sum())
                bad_nan += n_nan
                bad_inf += n_inf
                bad_zero += n_zero
                if n_nan or n_inf or n_zero:
                    worst.append((base, n_nan, n_inf, n_zero))
                if base.endswith(".mixer.out_proj"):
                    out_proj_total += a.numel()
                    small = int((a < OUT_PROJ_SCALE_THRESHOLD).sum())
                    out_proj_small += small
                    for e in (-20, -18, -17, -16, -15, -14, -13, -12):
                        out_proj_hist[f"<1e{e}"] = int((a < 10.0 ** e).sum())
                    out_proj_hist["min"] = float(a.min()) if a.numel() else None
                    out_proj_hist["max"] = float(a.max()) if a.numel() else None
        if bad_nan == 0 and bad_inf == 0:
            ok(state, "scales-finite", f"no NaN, no Inf across {len(scales)} scale tensors")
        else:
            fail(state, "scales-finite", f"{bad_nan} NaN and {bad_inf} Inf scale values; e.g. {worst[:5]}")
        if bad_zero == 0:
            ok(state, "scales-nonzero", "no zero scale values")
        else:
            fail(state, "scales-nonzero", f"{bad_zero} zero scale values; e.g. {worst[:5]}")

        log("")
        log("[C] mamba out_proj scale distribution")
        if out_proj_total == 0:
            fail(state, "out_proj-scales", "no .mixer.out_proj scale tensors found")
        else:
            frac = out_proj_small / out_proj_total
            info(state, "out_proj-scales-below-6.1e-5",
                 f"{out_proj_small}/{out_proj_total} = {frac:.6f} below {OUT_PROJ_SCALE_THRESHOLD:g} "
                 f"(min={out_proj_hist.get('min')}, max={out_proj_hist.get('max')})")
            info(state, "out_proj-scale-histogram", json.dumps({k: v for k, v in out_proj_hist.items()
                                                               if not isinstance(v, float)}))

    # ------------------------------------------------------- D/E. byte identity
    log("")
    log("[D/E] byte identity vs source" + ("  (SELF-TEST: artifact == source)" if same_dir else ""))

    groups = collections.OrderedDict()
    groups["e_score_correction_bias (F32 router bias)"] = (E_SCORE_RE, True)
    groups["mamba core: A_log/D/dt_bias/conv1d"] = (MAMBA_CORE_RE, False)
    groups["norms: layer norm + norm_f"] = (NORM_RE, False)
    groups["router gate.weight (must stay unquantized)"] = (GATE_RE, False)
    groups["vision tower (vision_model.*)"] = (re.compile(r"^vision_model\."), False)
    groups["projector (mlp1.*)"] = (re.compile(r"^mlp1\."), False)
    groups["embeddings"] = (re.compile(r"^language_model\.backbone\.embeddings\.weight$"), False)
    groups["lm_head"] = (re.compile(r"^language_model\.lm_head\.weight$"), False)

    covered = set()
    for label, (rx, must_f32) in groups.items():
        names = sorted(k for k in src_index if rx.search(k))
        if must_f32 and len(names) != len(layers["moe_layers"]):
            fail(state, f"e_score_correction_bias-count",
                 f"{len(names)} found in source, {len(layers['moe_layers'])} moe layers expected")
        identical = differ = unavailable = missing = upcast = 0
        details = []
        for name in names:
            covered.add(name)
            sp = src_map.get(name)
            if sp is None or not os.path.exists(sp):
                unavailable += 1
                continue
            if name not in art_map:
                missing += 1
                details.append(f"missing {name}")
                continue
            ap = art_map[name]
            try:
                hs, shape_s, dtype_s = hash_tensor(sp, name)
                ha, shape_a, dtype_a = hash_tensor(ap, name)
            except Exception as exc:
                unavailable += 1
                details.append(f"unreadable {name} ({type(exc).__name__})")
                continue
            if must_f32 and dtype_a != "F32":
                fail(state, "e_score_correction_bias-dtype", f"{name} is {dtype_a}, must be F32")
                must_f32 = False
            if hs == ha and shape_s == shape_a and dtype_s == dtype_a:
                identical += 1
            elif shape_s == shape_a and dtype_a == "F32" and same_values(sp, ap, name):
                # AutoRound's Nemotron-H post-load fixup restores A_log/D/dt_bias in F32:
                # a lossless widening of the BF16 source, not a change.
                identical += 1
                upcast += 1
            else:
                differ += 1
                details.append(f"MISMATCH {name} src={shape_s}/{dtype_s} art={shape_a}/{dtype_a}")
        detail = (f"{len(names)} tensors: {identical} identical"
                  + (f" ({upcast} as exact F32 upcast)" if upcast else "")
                  + f", {differ} differ, "
                  f"{missing} missing in artifact, {unavailable} unreadable (source shard absent)")
        if differ or missing:
            fail(state, label, detail + "; " + "; ".join(details[:3]))
        elif identical == 0 and names and unavailable == 0:
            fail(state, label, detail + " -- nothing could actually be verified")
        elif identical == 0 and names:
            state["notes"].append({"name": label, "detail": "NOT VERIFIED YET: " + detail})
            log(f"  WARN  {label}: {detail}")
        else:
            ok(state, label, detail)

    # every source tensor the identity half did not cover, for the record
    uncovered = [k for k in src_index if k not in covered and not QUANT_RE.match(k)]
    if uncovered:
        info(state, "not-covered-by-identity-half",
             f"{len(uncovered)} non-quantized source tensors outside the required list, e.g. {uncovered[:4]}")

    # ---------------------------------------------------------------- summary
    nfail = sum(1 for c in state["checks"] if not c["ok"])
    log("")
    log(f"SUMMARY: {len(state['checks'])} checks, {nfail} failed, {len(state['notes'])} informational")
    if missing_shards:
        log(f"NOTE: source download is incomplete ({len(missing_shards)} shards absent); "
            f"identity coverage is partial, re-run when complete.")
    if args.json:
        state["failed"] = nfail
        state["missing_source_shards"] = missing_shards
        with open(args.json, "w") as f:
            json.dump(state, f, indent=1, default=str)
        log(f"report written to {args.json}")
    return min(nfail, 99)


if __name__ == "__main__":
    sys.exit(main())
