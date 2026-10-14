#!/usr/bin/env python3
"""Owned K3 correctness gate; draft for tests/test_k3_real_geometry.py upstream.

Default mode reads only checkpoint JSON/safetensors headers (stdlib, no torch).
GPU mode is FUTURE execution, not a benchmark; it loads one real module at a time,
compares decode exactly and dense outputs within frozen tolerances, then replays
small-M graphs. Requires an architect D3 disposition AND an assigned GPU window.
The external fail-closed runner must enforce ownership, storage/VRAM, timeout and
cleanup gates from PATCH-SERIES.txt. No server, build, power or device management.
Output is exclusive-create JSONL; only the final PASS denotes complete coverage.
Example (CPU): python3 tests/test_k3_real_geometry.py --model /path/to/3bpw --output census.jsonl
GPU use: same command plus --mode gpu --authorization /path/to/approved-window.json
Capture status: D3 HOLD until architect approval; replay is not allocator proof.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

BASE = "15ded2f3add148c4db3c900cba7de878238f53bc"
DEFAULT_MODEL = "/mnt/<nvme-disk>/b70-exl3-qwen38-3bpw-20260929"
EXPECTED = {(5120, 1024): 32, (5120, 6144): 48, (5120, 10240): 48,
            (5120, 12288): 16, (5120, 17408): 128, (6144, 5120): 64,
            (17408, 5120): 64}
MS = (1, 2, 3, 4, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 64, 65, 127, 128, 129, 256, 512)
SEEDS = (1701, 2909)
# Proposed conservative gate, frozen BEFORE first GPU observation. No automatic relaxation.
# For each row: rel-L2 <= 0.01 AND max abs error <= 0.01 + 0.05 * RMS(reference row).
REL_L2 = 0.01
ABS_FLOOR = 0.01
RMS_FACTOR = 0.05


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def census(model):
    """No payload reads. Validate all indexed header entries and real K3 geometry."""
    model = model.resolve(strict=True)
    qb = (model / "quantization_config.json").read_bytes()
    ib = (model / "model.safetensors.index.json").read_bytes()
    q = json.loads(qb)
    for key, value in {"quant_method": "exl3", "version": "1.4.2", "bits": 3.0,
                       "head_bits": 6, "mtp_bits": 4, "codebook": "mul1"}.items():
        require(q.get(key) == value, f"artifact identity mismatch: {key}")
    index = json.loads(ib)["weight_map"]
    headers, identities = {}, {}
    for name in sorted(set(index.values())):
        path = (model / name).resolve(strict=True)
        require(path.is_relative_to(model), f"shard outside artifact: {name}")
        with path.open("rb") as f:
            prefix = f.read(8)
            require(len(prefix) == 8, f"short header: {name}")
            size = struct.unpack("<Q", prefix)[0]
            require(2 <= size <= 64 * 1024 * 1024, f"invalid header length: {name}")
            raw = f.read(size)
            require(len(raw) == size, f"truncated header: {name}")
        header = json.loads(raw)
        identities[name] = {"bytes": path.stat().st_size, "header_sha256": digest(prefix + raw)}
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            require(key not in headers, f"duplicate tensor: {key}")
            require(index.get(key) == name, f"index/header mismatch: {key}")
            shape = entry["shape"]
            require(all(isinstance(x, int) and x >= 0 for x in shape), f"bad shape: {key}")
            a, b = entry["data_offsets"]
            require(0 <= a <= b <= path.stat().st_size - 8 - size, f"bad offsets: {key}")
            headers[key] = dict(entry, file=name)
    require(set(headers) == set(index), "index does not cover headers exactly")
    modules, widths, shapes = [], Counter(), Counter()
    non_k3 = []
    for key, entry in sorted(headers.items()):
        if not key.endswith(".trellis"):
            continue
        shape = entry["shape"]
        require(len(shape) == 3 and entry["dtype"] == "I16", f"bad trellis: {key}")
        require(entry["data_offsets"][1] - entry["data_offsets"][0] == 2 * math.prod(shape),
                f"trellis byte count: {key}")
        widths[shape[-1]] += 1
        if shape[-1] != 48:
            non_k3.append({"key": key, "shape": shape})
            continue
        root = key.removesuffix(".trellis")
        k, n = shape[0] * 16, shape[1] * 16
        require(k % 128 == n % 128 == 0, f"unaligned module: {key}")
        for suffix, expected in (("suh", [k]), ("svh", [n])):
            h = headers[root + "." + suffix]
            require(h["shape"] == expected and h["dtype"] == "F16", f"bad {suffix}: {key}")
        require(root + ".mul1" in headers and root + ".mcg" not in headers, f"wrong codebook: {key}")
        modules.append({"key": root, "k": k, "n": n, "trellis_shape": shape,
                        "tensors": {s: headers[root + "." + s] for s in ("trellis", "suh", "svh", "mul1")}})
        shapes[k, n] += 1
    require(len(modules) == 400 and dict(shapes) == EXPECTED, "incomplete or changed K3 geometry census")
    require(dict(widths) == {48: 400, 64: 8, 96: 1}, "unexpected mixed-precision tensor census")
    return {"artifact": str(model), "source_base": BASE, "config_sha256": digest(qb),
            "index_sha256": digest(ib), "shards": identities,
            "width_counts": dict(widths), "modules": modules, "non_k3": non_k3,
            "payload_verified": False, "scope": "header-only; not checkpoint checksum verification"}


def gpu_gate(args, manifest, emit):
    require(args.authorization is not None, "GPU mode needs explicit authorization JSON")
    auth = json.loads(args.authorization.read_text())
    require(auth.get("source_base") == BASE and auth.get("window_id"), "missing pinned assigned window")
    require(auth.get("d3_status") == "architect-approved" and auth.get("d3_review_record"), "D3 HOLD")
    require(auth.get("d3_option") in ("graph-private-pre-reservation", "caller-owned-workspace"), "D3 option missing")
    require(auth.get("empty_gpu_gate_passed") is True and auth.get("preflight_evidence"), "external preflight missing")
    require(os.environ.get("ZE_AFFINITY_MASK") in ("0", "1"), "pin one assigned card")
    require(os.environ.get("EXL3_INT8_PREFILL", "0") == "0", "int8 prefill must be off")
    require(os.environ.get("EXL3_BACKEND", "auto") == "auto", "test requires default auto backend")
    # Import ONLY after header census and explicit external gates. No implicit builds.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from safetensors import safe_open
    from exl3xpu import ops, triton_kernels as tk
    require(torch.xpu.is_available(), "XPU unavailable; never skip")
    require(ops.SMALL_M_MAX == 128 and ops.RECON_SLICE_N == 16384, "non-baseline dispatch thresholds")
    E = ops._get_esimd()
    require(E and E.exl3_supported(3, 2), "all-codebooks K3 extension missing")
    E.exl3_set_int8(0)
    E.exl3_set_fused(0)
    E.exl3_set_vec_max_m(2)
    dev = torch.device("xpu:0")

    def exact(a, b, label):
        require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), f"nonfinite {label}")
        # Compare representation for reconstruction, numerical exactness for fp32 one-hot.
        aa = a.view(torch.int16) if a.dtype == torch.float16 else a
        bb = b.view(torch.int16) if b.dtype == torch.float16 else b
        require(torch.equal(aa, bb), f"zero-tolerance decode mismatch: {label}")

    def close(a, b, label):
        a, b = a.float(), b.float()
        require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), f"nonfinite {label}")
        err = a - b
        rel = err.norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)
        rms = b.square().mean(dim=1).sqrt()
        mx = err.abs().amax(dim=1)
        require(bool(((rel <= REL_L2) & (mx <= ABS_FLOOR + RMS_FACTOR * rms)).all().item()),
                f"dense tolerance failed {label}: rel={rel.max().item()} abs={mx.max().item()}")
        return {"rel_l2_max": rel.max().item(), "max_abs": mx.max().item()}

    checked = 0
    for module in manifest["modules"]:
        key, k, n = module["key"], module["k"], module["n"]
        emit({"event": "module_begin", "key": key, "k": k, "n": n})
        tensors = {}
        for suffix in ("trellis", "suh", "svh"):
            info = module["tensors"][suffix]
            with safe_open(str(args.model / info["file"]), framework="pt", device="cpu") as f:
                cpu = f.get_tensor(key + "." + suffix)
                require(list(cpu.shape) == info["shape"], "payload shape changed after census")
                tensors[suffix] = cpu.to(dev)
        tr, su, sv = (tensors[x] for x in ("trellis", "suh", "svh"))
        su = su.unsqueeze(0)
        shard = torch.zeros(n // 128, dtype=torch.int32, device=dev)
        w = torch.empty((k, n), dtype=torch.float16, device=dev)
        # Exercise production slice offsets/tails, including N=17408 > slice_n.
        for n0 in range(0, n, 16384):
            nn = min(16384, n - n0)
            wc = torch.empty((k, nn), dtype=torch.float16, device=dev)
            E.exl3_reconstruct(tr, wc, n0, 3, 2)
            wt = tk.reconstruct(tr, 3, 2, n0, nn)
            exact(wc, wt, key + f" reconstruct@{n0}")
            w[:, n0:n0 + nn] = wc
            del wc, wt
        # The P3 raw K3 entry forces all MR instantiations, including the D1 MR8 fix.
        positions = (0, 15, 16, 127, 128, k // 2, k - 16, k - 1)
        for mr in (1, 2, 4, 8):
            xb = torch.zeros((1, k // 16, mr, 16), dtype=torch.float16, device=dev)
            for row, pos in enumerate(positions[:mr]):
                xb[0, pos // 16, row, pos % 16] = 1
            reference = w[list(positions[:mr])].float()
            for path in (0, 1):
                raw = E.exl3_gemm_raw(xb, tr, shard, n, 3, 2, path)
                exact(raw, reference, key + f" onehot MR={mr} path={path}")
            del xb, reference, raw
        for dtype in (torch.float16, torch.bfloat16):
            for seed in SEEDS:
                gen = torch.Generator(device="cpu").manual_seed(seed)
                for m in MS:
                    x = torch.randn((m, k), generator=gen, dtype=torch.float32).to(device=dev, dtype=dtype)
                    # (a) Independent Triton fallback, explicitly selected, no ESIMD routing.
                    old = ops._backend
                    try:
                        ops._backend = "triton"
                        yt = ops.exl3_linear_impl(x, tr, su, sv, shard, [0, n], 3, 2)
                    finally:
                        ops._backend = old
                    # (b) Reconstruct reference: same input transform as Triton, fp32 dense dot,
                    # final output cast only after the output transform. NOT a dense equality gate.
                    xh = tk.had_in(x, su)
                    yr = torch.empty((m, n), device=dev, dtype=dtype)
                    tk.had_out(xh[0].float() @ w.float(), sv, yr)
                    yc = E.linear(x, tr, su, sv, shard, [0, n], 3, 2, 128, 16384)
                    public = ops.exl3_linear(x, tr, su, sv, shard, [0, n], 3, 2)
                    stats = {"vs_triton": close(yc, yt, key), "vs_reconstruct": close(yc, yr, key),
                             "public_vs_cpp": close(public, yc, key)}
                    if m > 128:
                        # Exercise D5's Python large-M branch even when C++ supports K3.
                        yp = ops.exl3_linear_impl(x, tr, su, sv, shard, [0, n], 3, 2)
                        stats["python_large_vs_reference"] = close(yp, yr, key)
                        del yp
                    if m <= 128:
                        # Warm outside capture; allocator safety is separately owned by D3.
                        # Only one live graph at a time; no shared-global-scratch concurrency claim.
                        stream = torch.xpu.Stream()
                        stream.wait_stream(torch.xpu.current_stream())
                        with torch.xpu.stream(stream):
                            for _ in range(3):
                                warm = ops.exl3_linear(x, tr, su, sv, shard, [0, n], 3, 2)
                        torch.xpu.current_stream().wait_stream(stream)
                        torch.xpu.synchronize()
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph):
                            yg = ops.exl3_linear(x, tr, su, sv, shard, [0, n], 3, 2)
                        for _ in range(3):
                            graph.replay()
                            close(yg, yc, key + " graph-original")
                        # Changed input prevents a cached/stale output from passing replay.
                        x.neg_()
                        changed = E.linear(x, tr, su, sv, shard, [0, n], 3, 2, 128, 16384)
                        graph.replay()
                        close(yg, changed, key + " graph-changed-input")
                        del graph, yg, warm, changed, stream
                    torch.xpu.synchronize()
                    emit({"event": "cell_pass", "key": key, "M": m, "dtype": str(dtype),
                          "seed": seed, "graph_replay": m <= 128, **stats})
                    checked += 1
                    del x, yt, yr, yc, public, xh
        del w, tr, su, sv, shard, tensors
        emit({"event": "module_pass", "key": key})
    require(checked == 400 * len(MS) * len(SEEDS) * 2, "incomplete cell coverage")
    emit({"event": "PASS", "scope": "K3 numerics and small-M replay only; not D3 allocator certification",
          "modules": 400, "cells": checked, "dense_tolerance": [REL_L2, ABS_FLOOR, RMS_FACTOR]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("census", "gpu"), default="census")
    p.add_argument("--authorization", type=Path)
    args = p.parse_args()
    # No report, cache or output can be written through the read-only artifact root.
    require(not args.output.resolve().is_relative_to(args.model.resolve()), "output must be outside artifact")
    with args.output.open("x") as out:
        def emit(item):
            out.write(json.dumps(item, sort_keys=True) + "\n")
            out.flush()
        try:
            manifest = census(args.model)
            emit({"event": "census", **manifest})
            if args.mode == "census":
                emit({"event": "PASS", "scope": "header census ONLY; GPU not run", "modules": 400})
            else:
                gpu_gate(args, manifest, emit)
        except Exception as e:
            emit({"event": "FAIL", "error": repr(e)})
            raise


if __name__ == "__main__":
    main()
