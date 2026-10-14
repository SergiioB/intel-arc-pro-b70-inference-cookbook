"""Divergence attribution at the first divergent module (CPU only, no GPU, no extra pass).

Inputs are the two already-captured sides of the SAME forward:
  * reference raw per-layer outputs (batch 1, exact prompt ids),
  * native per-module outputs from the engine capture,
  * the enumerated module map (class/key/layer_idx), so the pairing is by identity, never by fit.

What it reports, per layer (and for the layer the caller names):
  1. rfn / max|d| / frac<1e-3 (the same numbers the comparator prints, for continuity),
  2. a linear fit d ~ a*x + b on the reference value: a near 0 with a small b is consistent with
     a value-independent perturbation (rounding-like); a systematic a != 0 is a scale-type
     difference; a near-zero a with rare huge d is localized math, not precision,
  3. BIT-LEVEL match counts against two deterministic encodings of the reference:
     native == fp16(ref) and native == bf16(ref). A high fp16(bf16) match count is strong
     evidence the native path is the reference rounded at that width; a low count is evidence
     it is not,
  4. channel-localized structure: the last-dim channel with the largest mean|d| and the share of
     total squared error it carries (a small number of channels dominating = structural,
     a flat distribution = generic precision).

A high fp16/bf16 match count is evidence, not proof: a true same-precision attribution still
needs an fp16-compute CPU reference of the named sub-op on the same inputs, which this tool
tells you where to point at (channel/step) but does not itself compute.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1)


def _load_map(p: Path) -> dict:
    m = json.loads(Path(p).read_text())
    out = {}
    for e in m.get("entries", []):
        if e.get("class", "").endswith("TransformerBlock") and e.get("layer_idx") is not None:
            li = int(e["layer_idx"])
            if li >= 0:
                out[li] = e
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-map", required=True)
    ap.add_argument("--native", required=True, help="native per-module capture (safetensors)")
    ap.add_argument("--reference", required=True, help="reference raw per-layer outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--focus", type=int, default=None, help="layer to attribute in detail")
    ap.add_argument("--channels", type=int, default=3, help="worst channels to list for --focus")
    args = ap.parse_args()

    nats = load_file(args.native)
    refs = load_file(args.reference)
    # Duplicate-tensor FLAG (never an unconditional refusal): identical native tensors can be
    # legitimate (e.g. an exactly-zero residual in a degenerate case), so this reports the pair
    # for review and marks the run; it does not stop the analysis and does not special-case
    # anything away. Observed once for real: m_49 == m_50 (reference layers 46 and 47), where a
    # duplicated capture is the likely cause and the pair is flagged in the output.
    dup_flags: list = []
    seen: dict = {}
    for k in sorted(nats, key=lambda s: (len(s), s)):
        t = nats[k]
        if t.numel() <= 1:
            continue
        h = (tuple(t.shape), float(t.float().sum()), float(t.float().std()))
        for k2, h2 in seen.items():
            if h == h2:
                dup_flags.append({"a": k2, "b": k, "shape": list(t.shape),
                                  "note": "identical values; verify capture reuse before trusting either"})
        seen[k] = h
    if dup_flags:
        print(f"FLAG: {len(dup_flags)} duplicate native tensor pair(s) - review capture reuse; "
              f"analysis continues: {dup_flags[:2]}")
    by_layer = _load_map(Path(args.native_map))
    if not by_layer:
        raise SystemExit("native map has no TransformerBlock entries: refusing to guess")

    keys = sorted((int(k.split("_")[-1]) for k in refs), key=int)
    rows = []
    for j in keys:
        e = by_layer.get(j)
        if e is None:
            raise SystemExit(f"reference layer {j} has no mapped native module: refusing to guess")
        nk = f"m_{int(e['index'])}"
        if nk not in nats:
            raise SystemExit(f"native capture lacks {nk} for reference layer {j}")
        a = _flat(nats[nk])
        b = _flat(refs[f"layer_{j}"])
        if a.numel() != b.numel():
            raise SystemExit(f"layer {j}: element count differs ({a.numel()} vs {b.numel()})")
        d = a - b
        ref_norm = float(b.norm())
        rfn = float(d.norm() / ref_norm) if ref_norm else float("nan")
        rec = {
            "layer": j, "native_key": nk, "native_class": e["class"], "native_layer_idx": e["layer_idx"],
            "rfn": rfn, "max_abs": float(d.abs().max()),
            "frac_within_1e-3": float((d.abs() <= 1e-3 * b.abs().clamp_min(1e-6)).float().mean()),
        }
        # linear fit d ~ a*x + b
        x = b.double()
        y = d.double()
        xm, ym = x.mean(), y.mean()
        vx = ((x - xm) ** 2).sum()
        slope = float(((x - xm) * (y - ym)).sum() / vx) if vx > 0 else float("nan")
        rec["fit_slope_d_per_x"] = slope
        rec["fit_intercept"] = float(ym - slope * xm)
        denom = float((y ** 2).sum())
        rec["resid_frac"] = (float(((y - (slope * x + (ym - slope * xm))) ** 2).sum()) / denom) if denom > 0 else None
        # bit-level match counts against deterministic encodings of the reference
        r16 = b.half().float()
        rb = b.bfloat16().float()
        rec["bitmatch_fp16_of_ref"] = float((a.view(torch.int32) == r16.view(torch.int32)).float().mean())
        rec["bitmatch_bf16_of_ref"] = float((a.view(torch.int32) == rb.view(torch.int32)).float().mean())
        rows.append(rec)

    focus = args.focus if args.focus is not None else min(rows, key=lambda r: -r["rfn"])["layer"]
    row = next(r for r in rows if r["layer"] == focus)
    a = nats[f"m_{by_layer[focus]['index']}"].detach().float()
    b = refs[f"layer_{focus}"].detach().float()
    d = (a - b).abs()
    ch = d.reshape(-1, d.shape[-1]).mean(0)
    total = float((d ** 2).sum())
    worst = torch.argsort(ch, descending=True)[: args.channels]
    chan = [{"channel": int(c), "mean_abs": float(ch[c]),
             "share_of_sq_error": float((d[..., c] ** 2).sum() / total) if total else None} for c in worst]

    out = {"method": "identity-mapped reference-vs-native comparison, CPU only",
           "per_layer": rows, "focus_layer": focus, "focus": row, "focus_worst_channels": chan,
           "duplicate_native_tensors": dup_flags,
           "note": "bitmatch counts are evidence of a width-related encoding, not proof; "
                   "proof needs an fp16-compute CPU reference of the named sub-op"}
    Path(args.out).write_text(json.dumps(out, indent=1))
    order = sorted(rows, key=lambda r: -r["rfn"])
    print(f"{'layer':>5} {'rfn':>9} {'max|d|':>9} {'slope':>9} {'resid':>7} {'fp16match':>10} {'bf16match':>10}")
    rf = lambda v: "n/a" if v is None else f"{v:.4f}"
    for r in rows:
        print(f"{r['layer']:>5} {r['rfn']:>9.6f} {r['max_abs']:>9.5f} {r['fit_slope_d_per_x']:>9.5f} "
              f"{rf(r['resid_frac']):>7} {r['bitmatch_fp16_of_ref']:>10.5f} {r['bitmatch_bf16_of_ref']:>10.5f}")
    print("worst layers by rfn:", [r["layer"] for r in order[:5]])
    print(f"FOCUS layer {focus}: {json.dumps(row, indent=1)}")
    print("worst channels:", json.dumps(chan))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
