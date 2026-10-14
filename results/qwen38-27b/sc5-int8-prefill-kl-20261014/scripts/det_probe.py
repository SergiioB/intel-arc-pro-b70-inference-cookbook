#!/usr/bin/env python3
"""Run-to-run determinism probes for exl3xpu prefill paths on B70 (GPU1).

For each probe: fixed-seed inputs, one warmup, N=30 repeats; per repeat compare to
repeat 0 with torch.equal; record #non-equal repeats, max abs diff, #diff elements.
Usage: det_probe.py --model <exl3-checkpoint-dir> --out probe_results.json
"""
import argparse, json, os, sys, time

import torch
from safetensors import safe_open

sys.path.insert(0, "/opt/exl3xpu")
torch.ops.load_library("/opt/exl3xpu/exl3xpu/_C.so")
E = torch.ops.exl3xpu_C
DEV = "xpu"
N = 30

CASES = [
    ("model.language_model.layers.1.linear_attn.in_proj_qkv", 4),   # [320, 640, 64]  n=10240
    ("model.language_model.layers.31.self_attn.q_proj", 5),         # [320, 768, 80]  n=12288
    ("model.language_model.layers.31.self_attn.o_proj", 6),         # [384, 320, 96]  n=5120
    ("model.language_model.layers.31.self_attn.k_proj", 7),         # [320, 64, 112]  n=1024
]

RESULTS = {}


def bits(t):
    """Bitwise view for equality (NaN payloads compare equal bitwise)."""
    dt = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32}
    return t.view(dt[t.dtype]) if t.dtype in dt else t


def rec(group, name, fn, n=N):
    """fn() -> tensor. Returns stats dict."""
    y0 = fn()
    torch.xpu.synchronize()
    b0 = bits(y0).clone()          # clone: fn may write into a reused out-param next call
    y0c = y0.float().clone()
    neq = 0
    max_abs = 0.0
    max_ndiff = 0
    nan0 = int(torch.isnan(y0.float()).sum().item())
    for _ in range(n - 1):
        y = fn()
        torch.xpu.synchronize()
        ne = bits(y) != b0
        if ne.any().item():
            neq += 1
            d = (y.float() - y0c).abs()
            d = torch.nan_to_num(d, nan=0.0)
            max_abs = max(max_abs, d.max().item())
            max_ndiff = max(max_ndiff, int(ne.sum().item()))
    st = {"deterministic": neq == 0, "n_repeats": n, "n_unequal": neq,
          "max_abs_diff": max_abs, "max_diff_elems": max_ndiff, "n_nan_ref": nan0}
    RESULTS.setdefault(group, {})[name] = st
    print(f"[{group}] {name}: det={st['deterministic']} neq={neq}/{n-1} maxabs={max_abs:.3e} ndiff={max_ndiff}", flush=True)
    return st


def load_member(idx, model, name):
    f = idx[name + ".trellis"]
    with safe_open(os.path.join(model, f), framework="pt") as sf:
        return (sf.get_tensor(name + ".trellis").to(DEV),
                sf.get_tensor(name + ".suh").to(DEV),
                sf.get_tensor(name + ".svh").to(DEV))


def linear_call(x, tr, suh, svh, K):
    n = tr.shape[1] * 16
    shard = torch.zeros(n // 128, dtype=torch.int32, device=DEV)
    return E.linear(x, tr, suh.unsqueeze(0).contiguous(), svh, shard, [0, n], K, 2, 128, 16384)


def probe_A(model, idx):
    for name, K in CASES:
        tr, suh, svh = load_member(idx, model, name)
        k = tr.shape[0] * 16
        for int8 in (0, 1):
            E.exl3_set_int8(int8)
            tag = "int8" if int8 else "fp16"
            for M in (1024, 600):
                g = torch.Generator(device="cpu").manual_seed(20261014 + M + int8 * 7 + K)
                x = torch.randn((M, k), generator=g).to(DEV).half().contiguous()
                rec("A_linear", f"{name.split('.')[-1]} K={K} M={M} {tag}",
                    lambda x=x, tr=tr, suh=suh, svh=svh, K=K: linear_call(x, tr, suh, svh, K))
    E.exl3_set_int8(0)


def probe_B(model, idx):
    for name, K in CASES:
        tr, suh, svh = load_member(idx, model, name)
        k = tr.shape[0] * 16
        M = 1024
        g = torch.Generator(device="cpu").manual_seed(999 + K)
        x = torch.randn((M, k), generator=g).to(DEV).half().contiguous()
        # 0.1% of elements nudged one fp16 ulp toward +inf
        nn = max(1, x.numel() // 1000)
        gi = torch.Generator(device="cpu").manual_seed(4242 + K)
        sel = torch.randperm(x.numel(), generator=gi)[:nn].to(DEV)
        x2 = x.clone()
        flat = x2.view(-1)
        flat[sel] = torch.nextafter(flat[sel], torch.tensor(float("inf"), device=DEV, dtype=torch.float16))
        dx = (x2.float() - x.float()).norm() / x.float().norm().clamp_min(1e-30)
        for int8 in (0, 1):
            E.exl3_set_int8(int8)
            tag = "int8" if int8 else "fp16"
            y1 = linear_call(x, tr, suh, svh, K)
            y2 = linear_call(x2, tr, suh, svh, K)
            torch.xpu.synchronize()
            dy = (y2.float() - y1.float()).norm() / y1.float().norm().clamp_min(1e-30)
            frac = float((y2 != y1).sum().item()) / y1.numel()
            amp = (dy / dx).item()
            RESULTS.setdefault("B_amplification", {})[f"{name.split('.')[-1]} K={K} {tag}"] = {
                "relL2_dx": dx.item(), "relL2_dy": dy.item(), "amplification": amp,
                "frac_out_changed": frac}
            print(f"[B] {name.split('.')[-1]} K={K} {tag}: amp={amp:.2f} dy={dy.item():.3e} "
                  f"dx={dx.item():.3e} frac_changed={frac:.4f}", flush=True)
    E.exl3_set_int8(0)


def probe_C():
    for k, n in ((5120, 17408), (17408, 5120)):
        g = torch.Generator(device="cpu").manual_seed(k + n)
        a = torch.randn((1024, k), generator=g).to(DEV).half()
        b = torch.randn((k, n), generator=g).to(DEV).half()
        rec("C_torch_matmul", f"[1024,{k}]x[{k},{n}] fp16",
            lambda a=a, b=b: torch.matmul(a, b))


def probe_D():
    D = 256
    G, Hk = 6, 4        # Hq=24, Hk=4 -> G=6
    scale = D ** -0.5
    for seq_len, Q in ((1024, 1024), (2048, 1024)):
        g = torch.Generator(device="cpu").manual_seed(seq_len)
        Qp = (Q + 255) // 256 * 256
        Lp = seq_len
        qg = torch.zeros((G, Qp, D), dtype=torch.float16, device=DEV)
        qg[:, Qp - Q:] = torch.randn((G, Q, D), generator=g).half().to(DEV)
        kb = torch.randn((1, Lp, D), generator=g).half().to(DEV).contiguous()
        vb = torch.randn((1, Lp, D), generator=g).half().to(DEV).contiguous()
        og = torch.empty((G, Qp, D), dtype=torch.float16, device=DEV)
        none = torch.empty(0, device=DEV)
        rec("D_sdpa_onednn", f"Q={Q} seq={seq_len}",
            lambda: (E.exl3_sdpa_len(qg, kb, vb, og, none, True, scale, seq_len, Qp), og)[1])
        # comparison: vllm_xpu_kernels FA2 varlen path (the _fa fallback in fp8kv_prefill)
        try:
            from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
            Hq = G * Hk
            qf = torch.randn((Q, Hq, D), generator=g).half().to(DEV).contiguous()
            kf = torch.randn((seq_len, Hk, D), generator=g).half().to(DEV).contiguous()
            vf = torch.randn((seq_len, Hk, D), generator=g).half().to(DEV).contiguous()
            cq = torch.tensor([0, Q], dtype=torch.int32, device=DEV)
            ck = torch.tensor([0, seq_len], dtype=torch.int32, device=DEV)
            rec("D_fa_varlen", f"Q={Q} seq={seq_len}",
                lambda: flash_attn_varlen_func(qf, kf, vf, Q, cq, seq_len, cu_seqlens_k=ck,
                                               causal=True, softmax_scale=scale,
                                               return_softmax_lse=True)[0])
        except Exception as e:
            RESULTS.setdefault("D_fa_varlen", {})[f"Q={Q} seq={seq_len}"] = {"error": repr(e)}
            print(f"[D_fa_varlen] Q={Q} seq={seq_len}: ERROR {e!r}", flush=True)


def probe_E():
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
    Hk, Hv, Dk, Dv = 16, 48, 128, 128
    T = 1024
    for warm in (False, True):
        g = torch.Generator(device="cpu").manual_seed(31337 + int(warm))
        # serving applies l2norm to q,k upstream (fused_post_conv_prep apply_l2norm=True);
        # unnormalized keys produce NaN (verified: sweep 2026-10-14, nan=0 only when normalized)
        q = torch.nn.functional.normalize(
            torch.randn((1, T, Hk, Dk), generator=g).float(), dim=-1).to(DEV).half()
        k = torch.nn.functional.normalize(
            torch.randn((1, T, Hk, Dk), generator=g).float(), dim=-1).to(DEV).half()
        v = torch.randn((1, T, Hv, Dv), generator=g).to(DEV).half()
        gg = (-torch.rand((1, T, Hv), generator=g) * 0.1).to(DEV).float()   # log-space decay
        beta = torch.rand((1, T, Hv), generator=g).to(DEV).float()
        st = (torch.randn((1, Hv, Dv, Dk), generator=g).to(DEV).float()
              if warm else torch.zeros((1, Hv, Dv, Dk), device=DEV).float())
        cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)
        tag = "init_state" if warm else "cold"
        try:
            rec("E_gdn_chunk", f"T={T} {tag}",
                lambda: chunk_gated_delta_rule(q, k, v, gg, beta, initial_state=st,
                                               output_final_state=True, cu_seqlens=cu)[0])
        except Exception as e:
            RESULTS.setdefault("E_gdn_chunk", {})[f"T={T} {tag}"] = {"error": repr(e)}
            print(f"[E_gdn_chunk] T={T} {tag}: ERROR {e!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    print(f"device={torch.xpu.get_device_name(0)} torch={torch.__version__}", flush=True)
    print(f"env: ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')} "
          f"EXL3_ONEDNN_ATTN={os.environ.get('EXL3_ONEDNN_ATTN')} "
          f"UR_L0_USE_COPY_ENGINE={os.environ.get('UR_L0_USE_COPY_ENGINE')} "
          f"ONEDNN_DEFAULT_FPMATH_MODE={os.environ.get('ONEDNN_DEFAULT_FPMATH_MODE')}", flush=True)
    print(f"build_tag={E.exl3xpu_build_tag() if hasattr(E,'exl3xpu_build_tag') else '?'}", flush=True)
    t0 = time.time()
    probe_A(a.model, idx)
    probe_B(a.model, idx)
    probe_C()
    probe_D()
    probe_E()
    RESULTS["meta"] = {"torch": torch.__version__, "device": torch.xpu.get_device_name(0),
                       "n_repeats": N, "seconds": round(time.time() - t0, 1)}
    with open(a.out, "w") as f:
        json.dump(RESULTS, f, indent=1)
    print(f"wrote {a.out} in {RESULTS['meta']['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
