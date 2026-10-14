#!/usr/bin/env python3
"""RQ-11 one-round attribution parser: split ONE DFlash decode round into named buckets.

Input:  a Kineto chrome trace from the pinned v0.30.0 server profiled over exactly
        one decode round (torch profiler via --profiler-config + /start_profile).
Output: bucket shares over the round wall window (sum == 1.0 by construction):
          expert_gemm   - routed-expert fused MoE GEMM kernels (XPUExpertsWNA16 ->
                          vllm_xpu_kernels xpu_fused_moe(is_int4=True)), target-side
          dispatch      - router/topk/align/permute/gather-scatter dispatch kernels,
                          target-side
          draft         - device kernels inside DFlash draft-model windows (CPU-track
                          module/python frames matching DFLASH_PAT)
          host          - round wall window minus total device-busy time (launch gaps,
                          sync waits, scheduler time on the host track)
          residual      - device-busy time not in the three buckets: target attention,
                          mamba/SSM scan, dense/shared GEMMs, norms, memcpy - NAMED via
                          the emitted per-kernel-name table.

Usage: python3 parse_attribution.py TRACE.json[.gz] [--rounds LAST] [--dump-names]
The classification table below is provisional until first run; --dump-names prints the
full device-kernel name histogram so the final bucketing is name-grounded, never guessed.
"""
from __future__ import annotations

import collections
import gzip
import json
import re
import sys
from pathlib import Path

EXPERT_GEMM_PAT = re.compile(
    r"fused_moe|xpu_fused_moe|moe.*gemm|gemm.*moe|wna16", re.I)
DISPATCH_PAT = re.compile(
    r"grouped_topk|topk_softmax|moe_align|moe_sum|permute|unpermute|"
    r"token_sort|count_and_sort|expert.*index|index_select|moe.*route|router", re.I)
DRAFT_PAT = re.compile(r"dflash|draft", re.I)
DEVICE_CATS = {"kernel", "Kernel", "gpu_memcpy", "gpu_memset", "Memcpy"}
HOST_CATS = {"cuda_runtime", "cuda_driver", "xpu_runtime", "xpu_driver",
             "cpu_op", "user_annotation", "python_function"}


def load(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    return json.loads(path.read_text())


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dump_names = "--dump-names" in sys.argv
    if not args:
        print(__doc__)
        return 2
    trace = load(Path(args[0]))
    events = trace.get("traceEvents", [])
    dev = [e for e in events
           if e.get("ph") == "X" and e.get("cat") in DEVICE_CATS and e.get("dur")]
    if not dev:
        print("FATAL: no device-kernel events found in trace; cats present: "
              f"{sorted({e.get('cat') for e in events if e.get('ph') == 'X'})}")
        return 2

    # Draft windows from the CPU track (module/user-annotation frames naming the draft).
    draft_spans: list[tuple[float, float]] = []
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in HOST_CATS \
                and DRAFT_PAT.search(str(e.get("name", ""))) and e.get("dur"):
            draft_spans.append((e["ts"], e["ts"] + e["dur"]))
    draft_spans.sort()

    def in_draft(ts: float, te: float) -> bool:
        for ds, de in draft_spans:
            if te <= ds:
                break
            if ts < de and te > ds:
                return True
        return False

    rounds = int(next((a.split("=")[1] for a in sys.argv[1:]
                       if a.startswith("--rounds=")), "1"))
    # Round window: union of all device time; keep only the LAST `rounds` unit by
    # splitting the device timeline at the largest gaps if the capture holds more.
    dev.sort(key=lambda e: e["ts"])
    if rounds == 1 and len(dev) > 1:
        gaps = [(b["ts"] - (a["ts"] + a["dur"]), i)
                for i, (a, b) in enumerate(zip(dev, dev[1:])) if b["ts"] > a["ts"] + a["dur"]]
        if gaps:
            gaps.sort(reverse=True)
            cut = gaps[0][1]  # largest internal gap separates capture pieces; keep tail
            if gaps[0][0] > 3 * (dev[-1]["ts"] + dev[-1]["dur"] - dev[0]["ts"]) / max(len(dev), 1) * 50:
                dev = dev[cut + 1:]
    w0 = dev[0]["ts"]
    w1 = max(e["ts"] + e["dur"] for e in dev)
    wall = w1 - w0

    hist: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0])
    buckets = {"expert_gemm": 0.0, "dispatch": 0.0, "draft": 0.0}
    for e in dev:
        ts, te, name = e["ts"], e["ts"] + e["dur"], str(e.get("name", ""))
        h = hist[name]
        h[0] += e["dur"]
        h[1] += 1
        if in_draft(ts, te):
            buckets["draft"] += e["dur"]
        elif EXPERT_GEMM_PAT.search(name):
            buckets["expert_gemm"] += e["dur"]
        elif DISPATCH_PAT.search(name):
            buckets["dispatch"] += e["dur"]
    device_busy = sum(e["dur"] for e in dev)
    buckets["host_waits_launch"] = max(wall - device_busy, 0.0)
    buckets["residual_device_other"] = max(
        device_busy - buckets["expert_gemm"] - buckets["dispatch"] - buckets["draft"], 0.0)

    print(f"round_window_us {wall:.0f} device_events {len(dev)} device_busy_us {device_busy:.0f} "
          f"draft_spans {len(draft_spans)}")
    total = sum(buckets.values())
    for k in ("expert_gemm", "dispatch", "draft", "host_waits_launch",
              "residual_device_other"):
        print(f"bucket {k} {buckets[k]:.0f}us share {buckets[k] / total:.4f}")
    print(f"share_sum {sum(buckets[k] / total for k in buckets):.6f}")
    if dump_names:
        print("-- device kernel name histogram (us, count, share, in_draft?) --")
        for name, (us, n) in sorted(hist.items(), key=lambda kv: -kv[1][0]):
            print(f"{us:12.0f} {n:6d} {us / device_busy:7.4f}  {name[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
