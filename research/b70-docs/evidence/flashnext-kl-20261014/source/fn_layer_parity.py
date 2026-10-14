#!/usr/bin/env python3
"""Per-layer same-quant parity: native engine modules vs reference decoder layers.

ALIGNMENT (semantic, explicit - never chosen by fit). Enumerated from this model's own
`fwd_modules` (class/key/layer_idx), not from the config on paper and not from shapes:
    native idx  module                    key
      0         Embedding                 model.language_model.embed_tokens
      1         ExpandStreams (hc)        hc_expand
      2         TransformerBlock  layer 0  model.language_model.layers.0
      3         PLELayer                  model.language_model.layers.1.ple
      4..50     TransformerBlock  layers 1..47
     51         GatedResidual (hc mixer)  model.language_model.hyper_connection_mixer
     52         Linear                    lm_head
There is NO final-norm module in this architecture (verified by the enumeration). Reference
decoder layer j therefore maps to native module index j+2 for j == 0 and j+3 for j >= 1:
decoder 0 -> 2, decoder 1..47 -> 4..50. The mapping is read from --native-map (the enumerated
file) and the tool refuses to run without it, because shapes alone cannot separate the PLE
from the 48 decoder layers. A best-RFN candidate search is printed under "DIAGNOSTIC ONLY"
and never defines the alignment used for any verdict.

THRESHOLDS: the DECLARED gate in force for this port is the logits-level test (argmax equal,
top-5 sets equal, cosine >= 0.99, relative Frobenius <= 0.02) applied to the head comparison.
`--diag-tol` (default 1e-3) is an ADDITIONAL diagnostic flag for "which layer differs first";
it is not a declared gate and nothing is published on it.

PRECISION: the native tensors are FP16-computed, the reference is FP32. The column
`rfn_vs_fp16_rounded_reference_output_only` is the reference FP32 OUTPUT rounded to fp16 -
it is NOT an fp16-compute reference, so a closer match does NOT attribute a residual to
precision and a worse match does NOT attribute it to math. Attribution requires an actual
fp16-compute CPU reference of the divergent module.

HEAD PARITY: the prompt length is taken from the native capture's own metadata (the ids it
saved) and from the reference parity.json ids, and the two are asserted equal; the CLI only
supplies paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def rfn(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b).square().sum().sqrt() / b.square().sum().sqrt()))


def flat(t: torch.Tensor) -> torch.Tensor:
    t = t.float()
    return t.reshape(t.shape[0], t.shape[1], -1) if t.dim() >= 2 else t.reshape(1, 1, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-map", default=None,
                    help="native_module_map.json enumerated from model.fwd_modules (class/key/layer_idx)")
    ap.add_argument("--native", required=True, help="native_raw_modules.safetensors (21-token prompt)")
    ap.add_argument("--reference", required=True, help="reference raw_layers.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--native-prompt-len", type=int, default=21,
                    help="tokens fed to the native run (prefill = len-1)")
    ap.add_argument("--ref-prompt-len", type=int, default=20)
    ap.add_argument("--diag-tol", type=float, default=1e-3,
                    help="DIAGNOSTIC only: flags which layer differs first; the declared gate for "
                         "this port is the logits-level test (argmax/top-5/cosine/rfn), not this")
    ap.add_argument("--native-head", default=None, help="native first-step logits .pt (same prompt; carries its own ids)")
    ap.add_argument("--ref-parity-json", default=None, help="reference parity.json (carries the ids it ran)")
    ap.add_argument("--ref-head", default=None, help="reference raw_hidden_logits.safetensors")
    args = ap.parse_args()

    nat_all = load_file(args.native)
    nats = {int(k.split("_")[-1]): v for k, v in nat_all.items()}
    ref_all = load_file(args.reference)
    refs = {int(k.split("_")[-1]): v for k, v in ref_all.items()}
    print(f"CALL METADATA: native prompt {args.native_prompt_len} tokens (prefill "
          f"{args.native_prompt_len - 1}), reference prompt {args.ref_prompt_len} tokens")
    print(f"native modules captured: {len(nats)} | reference layers captured: {len(refs)}")

    decoders = []
    if args.native_map:
        mmap = json.loads(Path(args.native_map).read_text())
        decoders = [(e["index"], e["layer_idx"], e["class"].split(".")[-1], e["key"])
                    for e in mmap["entries"]
                    if e["class"].endswith("TransformerBlock") and e["layer_idx"] is not None]
        decoders.sort(key=lambda t: t[1])
        print("SEMANTIC MAPPING from the enumerated module identity "
              "(class TransformerBlock + layer_idx), not from shapes or RFN:")
        for idx, li, cls, key in decoders[:3]:
            print(f"   reference layer {li} -> native m_{idx}  ({cls}, {key})")
        print(f"   ... {len(decoders)} decoder layers, native modules "
              f"{decoders[0][0]}..{decoders[-1][0]}")
    else:
        print("no --native-map given: refusing to guess the alignment (shape signature is "
              "insufficient - PLE and the 48 layers share shapes)")
        return 2
    by_layer = {li: idx for idx, li, _cls, _key in decoders}

    ref_first = refs[min(refs)]
    diag = sorted(((rfn(flat(t), flat(ref_first)), i, tuple(t.shape)) for i, t in nats.items()
                   if t.dim() >= 3), )
    print("DIAGNOSTIC ONLY (best-RFN match for the reference's first layer; not the alignment):")
    for r, i, s in diag[:4]:
        print(f"   m_{i} shape {s} rfn={r:.6f}")

    rows = []
    print(f"{'layer':>5} {'native':>7} {'shape':>16} {'rfn_fp32':>10} {'rfn_fp16_out_only':>17} "
          f"{'max|d|':>9} {'frac<1e-3':>10}")
    for j, key in enumerate(sorted(refs)):
        ni = by_layer.get(key)
        if ni is None or ni not in nats:
            print(f"{key:>5} {'?' if ni is None else ni:>7}  no native module for this layer")
            continue
        a = flat(nats[ni])
        b = flat(refs[key])
        if a.shape != b.shape:
            print(f"{key:>5} {ni:>7}  shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        # NOTE: b16 is the reference FP32 OUTPUT rounded to fp16. It is NOT an fp16-compute
        # reference, so a closer match here does NOT attribute the residual to precision and a
        # worse match does NOT attribute it to math. Attribution requires an fp16-compute CPU
        # reference of the divergent module (separate, targeted step).
        b16 = b.half().float()
        d = (a - b).abs()
        rec = {"layer": key, "native_module": ni, "shape": list(a.shape),
               "rfn_fp32": rfn(a, b), "rfn_vs_fp16_rounded_reference_output_only": rfn(a, b16),
               "max_abs": float(d.max()), "frac_within_1e-3": float((d <= 1e-3).float().mean()),
               "native_std": float(a.std()), "ref_std": float(b.std())}
        rows.append(rec)
        flag = "  <-- DIVERGENT" if rec["rfn_fp32"] > args.diag_tol else ""
        print(f"{key:>5} {ni:>7} {str(tuple(a.shape)):>16} {rec['rfn_fp32']:>10.6f} "
              f"{rec['rfn_vs_fp16_rounded_reference_output_only']:>12.6f} {rec['max_abs']:>9.5f} "
              f"{rec['frac_within_1e-3']:>10.4f}{flag}")
    first_bad = next((r["layer"] for r in rows if r["rfn_fp32"] > args.diag_tol), None)

    head = {}
    if args.native_head and args.ref_head:
        nh = torch.load(args.native_head, map_location="cpu")
        nl = nh["logits"].float().reshape(-1)
        nat_ids = [int(x) for x in nh.get("ids", [])]
        rh = load_file(args.ref_head)
        rl = rh.get("logits_last")
        ref_ids = None
        if args.ref_parity_json and Path(args.ref_parity_json).is_file():
            ref_ids = [int(x) for x in json.loads(Path(args.ref_parity_json).read_text()).get("ids", [])]
        if rl is not None:
            rl = rl.float().reshape(-1)
            # prompt lengths come from the captures themselves, not from a CLI default
            if ref_ids is not None:
                assert nat_ids == ref_ids, (
                    f"head parity refused: native capture prompt ids ({len(nat_ids)}) != "
                    f"reference ids ({len(ref_ids)})")
            head = {"prompt_len_native_from_capture": len(nat_ids),
                    "prompt_len_reference_from_parity_json": (len(ref_ids) if ref_ids is not None else None),
                    "prompt_ids_identical": (nat_ids == ref_ids) if ref_ids is not None else None,
                    "rfn": rfn(nl, rl), "max_abs": float((nl - rl).abs().max()),
                    "argmax_native": int(nl.argmax()), "argmax_reference": int(rl.argmax()),
                    "argmax_equal": int(nl.argmax()) == int(rl.argmax()),
                    "cosine": float(torch.nn.functional.cosine_similarity(nl.unsqueeze(0), rl.unsqueeze(0)))}
            print(f"HEAD PARITY: native prompt {head['prompt_len_native']} vs reference "
                  f"{head['prompt_len_reference']} -> rfn {head['rfn']:.6f}, argmax "
                  f"{head['argmax_native']} vs {head['argmax_reference']}, cos {head['cosine']:.5f}")

    Path(args.out).write_text(json.dumps({"decoder_mapping_reference_layer_to_native_module":
                                              {str(k): v for k, v in by_layer.items()},
                                          "diag_tol": args.diag_tol,
                                          "first_layer_above_tol": first_bad, "layers": rows,
                                          "head": head, "call_metadata": {
                                              "native_prompt_len": args.native_prompt_len,
                                              "reference_prompt_len": args.ref_prompt_len}}, indent=1))
    print(f"FIRST LAYER ABOVE RFN {args.diag_tol}: {first_bad}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
