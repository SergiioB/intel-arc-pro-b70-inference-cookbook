#!/usr/bin/env python3
"""Intel Arc Pro B60/B70 (Battlemage Xe2) roofline & SWA-KV ceiling calculator.

Stdlib-only CLI. Computes practical decode, prefill, speculative, and KV-cache
ceilings using measured B70 constants (sustained BW ~525 GB/s, practical BF16
XMX ~78 TFLOPS, no native FP4/FP8 DPAS on Xe2).

Usage:
  python3 b70_ceiling.py --spec examples/inkling2-mini-b70.json
  python3 b70_ceiling.py --spec examples/inkling2-mini-b70.json --json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def compute_ceilings(spec: dict[str, Any]) -> dict[str, Any]:
    hw = spec.get("hardware", {})
    bw_gb_s = float(hw.get("sustained_bw_gb_s", 525.0))
    tflops_practical = float(hw.get("practical_xmx_tflops", 78.0))
    vram_gib = float(hw.get("vram_gib", 31.85))
    cards = int(hw.get("gpu_count", 1))
    effective_bw = bw_gb_s * cards
    effective_tflops = tflops_practical * cards

    m = spec["model"]
    dense_gb = float(m.get("dense_active_gb", 0.0))
    expert_active_gb = float(m.get("expert_active_gb", 0.0))
    lm_head_gb = float(m.get("lm_head_gb", 0.0))
    weight_total_gib = float(m.get("resident_weights_gib", 0.0))
    runtime_overhead_gib = float(m.get("runtime_overhead_gib", 3.0))
    active_params_b = float(m.get("effective_active_params_b", 0.0))

    full_visits = int(m.get("full_attn_visits", 0))
    swa_visits = int(m.get("swa_attn_visits", 0))
    swa_window = int(m.get("swa_window", 1024))
    kv_heads = int(m.get("kv_heads", 0))
    head_dim = int(m.get("head_dim", 128))
    kv_bytes_elem = float(m.get("kv_bytes_per_elem", 2.0))
    attn_hidden = int(m.get("attn_hidden_size", kv_heads * head_dim))

    # KV bytes per token for full-attention visits (K + V)
    kv_bytes_per_full_tok = 2.0 * full_visits * kv_heads * head_dim * kv_bytes_elem
    kv_bytes_per_swa_tok = 2.0 * swa_visits * kv_heads * head_dim * kv_bytes_elem
    swa_fixed_read_gb = (kv_bytes_per_swa_tok * swa_window) / 1e9

    base_decode_gb = dense_gb + expert_active_gb + lm_head_gb

    depths = [int(d) for d in spec.get("depths", [0, 8192, 32768, 65536, 131072, 200000, 262144])]
    decode_rows = []
    prefill_rows = []

    for depth in depths:
        kv_read_gb = (kv_bytes_per_full_tok * depth) / 1e9 + (
            (kv_bytes_per_swa_tok * min(depth, swa_window)) / 1e9 if depth > 0 else 0.0
        )
        total_gb = base_decode_gb + kv_read_gb
        dec_ceiling = effective_bw / total_gb if total_gb > 0 else 0.0
        decode_rows.append(
            {
                "depth": depth,
                "weight_gb": round(base_decode_gb, 3),
                "kv_read_gb": round(kv_read_gb, 3),
                "total_read_gb": round(total_gb, 3),
                "decode_ceiling_tok_s": round(dec_ceiling, 1),
            }
        )

        if depth >= 2048:
            linear_flops = 2.0 * (active_params_b * 1e9) * depth
            full_attn_flops = 2.0 * full_visits * attn_hidden * (depth**2)
            swa_attn_flops = 2.0 * swa_visits * attn_hidden * depth * min(depth, swa_window)
            total_tflops = (linear_flops + full_attn_flops + swa_attn_flops) / 1e12
            ttft_s = total_tflops / effective_tflops if effective_tflops > 0 else 0.0
            prefill_tok_s = depth / ttft_s if ttft_s > 0 else 0.0
            prefill_rows.append(
                {
                    "prompt_tokens": depth,
                    "total_tflops": round(total_tflops, 2),
                    "ttft_floor_s": round(ttft_s, 2),
                    "prefill_ceiling_tok_s": round(prefill_tok_s, 0),
                }
            )

    # Speculative decode estimate
    spec_cfg = spec.get("speculation")
    spec_rows = []
    if spec_cfg:
        k = int(spec_cfg.get("draft_k", 7))
        draft_step_ms = float(spec_cfg.get("draft_overhead_ms", 4.0))
        verify_extra_ms = float(spec_cfg.get("verify_compute_overhead_ms", 0.0))
        for acc in spec_cfg.get("acceptance_lens", [2.4, 3.0, 3.5]):
            base_step_ms = (base_decode_gb / effective_bw) * 1000.0
            step_ms = base_step_ms + verify_extra_ms + draft_step_ms
            spec_rows.append(
                {
                    "draft_k": k,
                    "acceptance_len": float(acc),
                    "step_ms_floor": round(step_ms, 2),
                    "spec_decode_ceiling_tok_s": round((float(acc) / step_ms) * 1000.0, 1),
                }
            )

    # KV capacity estimate across max_num_batched_tokens (mbt)
    avail_kv_gib = max(0.0, vram_gib * cards - weight_total_gib - runtime_overhead_gib)
    avail_kv_bytes = avail_kv_gib * (1024**3)
    swa_mult = float(m.get("vllm_swa_batch_multiplier", 6.0))
    kv_profiles = []
    for mbt in spec.get("mbt_sweep", [512, 1024, 2048, 4096]):
        # vLLM SlidingWindowManager reserves SWA window + multi-batch in-flight slots across all SWA visits
        swa_reserve_bytes = kv_bytes_per_swa_tok * (swa_window + swa_mult * int(mbt))
        rem_bytes = max(0.0, avail_kv_bytes - swa_reserve_bytes)
        est_full_ctx = int(rem_bytes / kv_bytes_per_full_tok) if kv_bytes_per_full_tok > 0 else 0
        kv_profiles.append(
            {
                "max_num_batched_tokens": int(mbt),
                "avail_kv_gib": round(avail_kv_gib, 2),
                "swa_reserve_gib": round(swa_reserve_bytes / (1024**3), 2),
                "est_max_context_tokens": est_full_ctx,
            }
        )

    return {
        "name": spec.get("name", "unnamed-model"),
        "hardware": {
            "gpu_count": cards,
            "vram_gib_per_gpu": vram_gib,
            "effective_bw_gb_s": effective_bw,
            "effective_xmx_tflops": effective_tflops,
        },
        "base_decode_gb_per_tok": round(base_decode_gb, 3),
        "swa_fixed_read_gb": round(swa_fixed_read_gb, 4),
        "kv_bytes_per_full_token": int(kv_bytes_per_full_tok),
        "decode_ceilings": decode_rows,
        "prefill_ceilings": prefill_rows,
        "speculative_ceilings_0k": spec_rows,
        "kv_mbt_profiles": kv_profiles,
    }


def print_report(res: dict[str, Any]) -> None:
    hw = res["hardware"]
    print(f"=== Intel XPU Roofline: {res['name']} ===")
    print(
        f"Hardware: {hw['gpu_count']}x GPU ({hw['vram_gib_per_gpu']:.2f} GiB each) | "
        f"BW={hw['effective_bw_gb_s']:.0f} GB/s | XMX={hw['effective_xmx_tflops']:.0f} TFLOPS practical"
    )
    print(
        f"Active decode weights: {res['base_decode_gb_per_tok']:.3f} GB/tok | "
        f"Full-attn KV: {res['kv_bytes_per_full_token']:,} B/tok"
    )
    print("\n-- Decode Ceiling (Memory-Bound, No Speculation) --")
    print(f"{'Depth':>8s} | {'Weights GB':>10s} | {'KV Read GB':>10s} | {'Total GB':>9s} | {'Ceiling tok/s':>13s}")
    for r in res["decode_ceilings"]:
        print(
            f"{r['depth']:8d} | {r['weight_gb']:10.3f} | {r['kv_read_gb']:10.3f} | "
            f"{r['total_read_gb']:9.3f} | {r['decode_ceiling_tok_s']:13.1f}"
        )

    if res["speculative_ceilings_0k"]:
        print("\n-- Speculative Decode Ceiling (0K) --")
        for r in res["speculative_ceilings_0k"]:
            print(
                f"  k={r['draft_k']} acc={r['acceptance_len']:.2f}: "
                f"step_floor={r['step_ms_floor']:.2f} ms -> ceiling={r['spec_decode_ceiling_tok_s']:.1f} tok/s"
            )

    print("\n-- Cold Prefill Ceiling (Compute-Bound, Practical XMX MFU) --")
    print(f"{'Prompt':>8s} | {'Total TFLOPs':>12s} | {'TTFT Floor s':>12s} | {'Ceiling tok/s':>13s}")
    for r in res["prefill_ceilings"]:
        print(
            f"{r['prompt_tokens']:8d} | {r['total_tflops']:12.2f} | "
            f"{r['ttft_floor_s']:12.2f} | {r['prefill_ceiling_tok_s']:13.0f}"
        )

    print("\n-- KV Capacity vs --max-num-batched-tokens (SWA Sensitivity) --")
    print(f"{'mbt':>6s} | {'KV Pool GiB':>11s} | {'SWA Reserve GiB':>15s} | {'Est Max Ctx':>12s}")
    for r in res["kv_mbt_profiles"]:
        print(
            f"{r['max_num_batched_tokens']:6d} | {r['avail_kv_gib']:11.2f} | "
            f"{r['swa_reserve_gib']:15.2f} | {r['est_max_context_tokens']:12,d}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="Path to model/hardware ceiling spec JSON")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = ap.parse_args()

    spec = json.loads(pathlib.Path(args.spec).read_text())
    res = compute_ceilings(spec)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print_report(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
