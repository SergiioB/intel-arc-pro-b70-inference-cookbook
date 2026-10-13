#!/usr/bin/env python3
"""Unsealed E8RVQ4 / scalar format quality screen, derived from r4_r0_20261011.py.

Default declarative arms: bf16, exl3, r4_g64, int4_g64_rht, int4_g128_rht.
FMT_ARMS selects optional E8 comparators and registered diagnostics (gptq_plain, r4_g128,
r4_fp, r4_fp32 diagnostics). Residual-fold streams rotate globally; RHT arms
instead transform each linear's weight/Hessian, quantize, and undo both transforms.
No model runtime change is needed for this reconstruction-only quality screen.

Launch only via fmt_screen_launch_20261011.sh, which owns the shared GPU1 lock,
strict oneDNN fp32 math and as-found guards. Damping remains exactly 0.01.
Reverse-order Cholesky computes the correct upper inverse factor without forming
the inverse Gram; failures fall back to the in-place classic fp64 CPU sequence.
FMT_HINV_VERIFY=1 factorizes identical damped Hessians twice and asserts torch.equal;
layers 2/11 additionally compare against classic fp64 and save U/H captures.
R4_DUMP_H must name a container-writable capture directory when proving INT4 codes.

R4_SMOKE=1 (launcher), R4_SMOKE_FULLCALIB=1, R4_SMOKE_LAYERS=0,...,12 selects
the 13-layer prefix with the full 64x512 calibration and 2x832 held-out eval.
R4_HINV_BLK=256, R4_GPTQ_BLOCK=128 and pinned EXL3 ref.py remain unchanged.
FMT_HINV_ONLY=1 skips quantization for a factorization-only diagnostic; these
quality metrics are not format evidence. FMT_SAVE_CODES=1 saves codes on /out.
FMT_FP64_WEIGHTED_COMPARE=1 compares L2/G128-RHT/down weighted GPTQ errors
under device fp32 versus classic fp64 inverse factors. Every quantized tensor
reports original-basis undamped Hessian-weighted error, tr(W H W^T), and numel
in layers.jsonl for reverse water-filling. E8 arms remain flag-gated; lead cancelled
their GPU screening on mathematical grounds. Act-order is not implemented.
SC artifact arms exl3_sc4/exl3_sc5 decode per-tensor K from packed shape, with
one bit-exact XPU-vs-CPU self-check per artifact/K. FMT_CALIB_FREE=1 removes all
calibration streams for full64-layer evaluation; artifact manifest/revision are pinned.
FMT_NATIVE_HEAD=1 uses each EXL3 artifact's quantized head and additionally reports
<arm>_bf16head diagnostics on a common BF16 head. Native head decode is streamed
on128-aligned output columns to fit14gRSS; calibration-free arms also use their
own embedding, norm and unquantized layer tensors. Hidden-state per-layer MSE
is a cumulative discrepancy proxy, not an additive final-logit KL contribution.

Sanity: fp32 fold exactness <=1e-4 at every layer, EXL3 r01 margin >0, and
gptq_plain defect sign only if that arm runs. r4_fp storage-noise metrics are
reported (not gated) only when enabled. NLL gather bounds are retained.
Measurements retain the original schema with additive arm specs, factor proofs,
RSS and 64-layer wall-time projection. Prefix canary margins are diagnostic,
not full-depth task correctness. This is never a sealed run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = Path(SCRIPT_DIR).resolve().parents[1]  # SCRIPT_DIR = <repo>/scripts/tmp

# ------------------------------------------------------------------ pinned identities
BF16 = "/bf16"
EXL3 = "/exl3"
STATES = "/states"
CORPUS = "/corpus/wikitext-2-raw-v1/test-00000-of-00001.parquet"
OUT = "/out"
REFSRC = "/refsrc"                      # b70-k3-build-20260929/src (ro); ref.py pinned below
RUN_DIR = os.environ.get("B70_RUN_DIR", "/run_dir")

REF_SHA256 = "3f614b7c91510d8520202e85dd0a08255370349a4d5ae4e7601edfa6897d1780"
REF_GIT_HEAD = "15ded2f3add148c4db3c900cba7de878238f53bc"
QUESTION_ID = "RQ-27"
SEED = 20260925                          # build_H_5120 seed (design §2.1)
D_MODEL = 5120
BITS, SYM, DESC_ACT, PERCDAMP = 4, True, False, 0.01   # champion contract
GROUPS = {"r4_g128": 128, "r4_g64": 64, "gptq_plain": 128}
N_CALIB, T_CALIB = 64, 512
CALIB_FRAC = 0.60
EVAL_FILE = "tokens_eval.safetensors"
N_EVAL, T_EVAL = 2, 832
CHUNK_LOGITS = 256
MIN_OUT_FREE_FOR_CODES = 10 * (1 << 30)  # keep >= 10 GiB free on /out after saving codes
SMOKE_LAYERS = [1, 3]                    # one GDN + one full-attention block

# --------------------------------------------------------------- diagnostic switches
FP32_DIAG = os.environ.get("R4_FP32", "0") == "1"           # r4_fp32 arm + fold probe
SMOKE_FULLCALIB = os.environ.get("R4_SMOKE_FULLCALIB", "0") == "1"
HINV_BLK = int(os.environ.get("R4_HINV_BLK", "256"))        # blocked Cholesky tile
XPU_RECON = os.environ.get("R4_XPU_RECON", "1") == "1"       # ref.py reconstruct on XPU
BLOCKSIZE = int(os.environ.get("R4_GPTQ_BLOCK", "128"))      # GPTQ column block

R01 = {"id": "r01",
       "prompt": ("What does this Python expression evaluate to? Answer only the "
                  "integer: sum(i * i for i in range(4))"),
       "expected": "14", "foil": "30"}
HELDOUT_MANIFEST_REL = "results/2026/09/b70-margin-requant-pilot1-20260930T0855Z/heldout-manifest.json"

PREFIX = "model.language_model."
QUANT_GDN = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
QUANT_ATT = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
# GPTQ input groups: linears sharing the SAME input tensor share one Hessian and
# one hinv (r4-recon Q1: qkv+z read the input_layernorm output; q/k/v read the
# same stream; gate+up read the post_attention_layernorm output; verified live
# on the first layer by _verify_input_groups).
IGROUPS_GDN = [["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
               ["linear_attn.out_proj"],
               ["mlp.gate_proj", "mlp.up_proj"],
               ["mlp.down_proj"]]
IGROUPS_ATT = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
               ["self_attn.o_proj"],
               ["mlp.gate_proj", "mlp.up_proj"],
               ["mlp.down_proj"]]
READERS_GDN = ["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
               "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"]
READERS_ATT = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]
READERS_MLP = ["mlp.gate_proj.weight", "mlp.up_proj.weight"]
WRITERS_GDN = ["linear_attn.out_proj.weight"]
WRITERS_ATT = ["self_attn.o_proj.weight"]
WRITERS_MLP = ["mlp.down_proj.weight"]
from dataclasses import dataclass

@dataclass(frozen=True)
class ArmSpec:
    name: str
    transform: str
    quantizer: str

ARM_SPECS = {
    s.name: s for s in (
        ArmSpec("bf16", "none", "none"), ArmSpec("exl3", "none", "exl3"),
        ArmSpec("exl3_sc4", "none", "exl3"), ArmSpec("exl3_sc5", "none", "exl3"),
        ArmSpec("r4_g64", "fold", "int4_g64"),
        ArmSpec("int4_g64_rht", "rht128", "int4_g64"),
        ArmSpec("int4_g128_rht", "rht128", "int4_g128"),
        ArmSpec("e8rvq4_rht", "rht128", "e8rvq4"),
        ArmSpec("e8rvq4_fold", "fold", "e8rvq4"),
        ArmSpec("gptq_plain", "none", "int4_g128"),
        ArmSpec("r4_g128", "fold", "int4_g128"),
        ArmSpec("r4_fp", "fold", "none"),
        ArmSpec("r4_fp32", "fold", "none"))}
ARM_ORDER = os.environ.get("FMT_ARMS", "bf16,exl3,r4_g64,int4_g64_rht,int4_g128_rht").split(",")
if FP32_DIAG:
    ARM_ORDER += [n for n in ("r4_fp", "r4_fp32") if n not in ARM_ORDER]
assert len(set(ARM_ORDER)) == len(ARM_ORDER) and all(n in ARM_SPECS for n in ARM_ORDER)
assert "bf16" in ARM_ORDER and "exl3" in ARM_ORDER
QUANT_ARMS = [n for n in ARM_ORDER if ARM_SPECS[n].quantizer.startswith(("int4", "e8"))]
CALIB_ARMS = list(dict.fromkeys(QUANT_ARMS + (["bf16", "exl3"] if os.environ.get("FMT_CALIB_FREE") != "1" else [])))
ROT_ARMS = tuple(n for n in ARM_ORDER if ARM_SPECS[n].transform == "fold")
EXL3_ROOTS = {"exl3": EXL3, "exl3_sc4": "/exl3_sc4", "exl3_sc5": "/exl3_sc5"}
EXL3_ARMS = [n for n in ARM_ORDER if ARM_SPECS[n].quantizer == "exl3"]
GROUPS.update({"int4_g64_rht": 64, "int4_g128_rht": 128})
HINV_PROOFS = []
HINV_VERIFY = os.environ.get("FMT_HINV_VERIFY", "0") == "1"
HINV_ONLY = os.environ.get("FMT_HINV_ONLY", "0") == "1"

def e8_quantize(W, U):
    from e8rvq_20261011 import quantize_e8rvq_ldlq
    return quantize_e8rvq_ldlq(W, U)

def rht128_matrix(n, dev, role=0):
    assert n % 128 == 0, f"RHT dimension {n} not divisible by 128"
    gen = torch.Generator(device="cpu").manual_seed(SEED + n + role * 1000003)
    signs = (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).to(dev, torch.float32)
    h = torch.tensor(sylvester(128), device=dev, dtype=torch.float32) / (128 ** 0.5)
    return signs, h

def rht_left(W, signs, h, inverse=False):
    n, m = W.shape
    v = W.reshape(n // 128, 128, m)
    if inverse:
        return ((h.T @ v).reshape(n, m) * signs[:, None]).contiguous()
    return (h @ (W * signs[:, None]).reshape(n // 128, 128, m)).reshape(n, m)

def rht_weight(W, ti, to, inverse=False):
    # T = block-H @ D; forward T_out W T_in^T, inverse T_out^T W T_in.
    return rht_left(rht_left(W, *to, inverse=inverse).T, *ti, inverse=inverse).T.contiguous()

def rht_hessian(H, ti):
    return rht_left(rht_left(H, *ti).T, *ti).T.contiguous()

log_lines: list[str] = []


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]}Z] {msg}"
    print(line, flush=True)
    log_lines.append(line)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def wjson(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def read_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------- shared helpers (reuse)
sys.path.insert(0, str(REPO / "scripts" / "tmp"))
import hq4_states_20261010 as hst  # noqa: E402  verified helpers, nothing edited in place

import torch  # noqa: E402  module-level: the run_layer decorator needs it
hst.BF16_FARM = BF16  # container mount of the same farm (module-global it reads)


# --------------------------------------------------- Hadamard (copied from rotation_t1t2_common)
def sylvester(n: int):
    import numpy as np
    H = np.array([[1]], dtype=np.int64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H
def paley1(q: int):
    import numpy as np
    if q % 4 != 3:
        raise ValueError("paley1 needs q = 3 (mod 4)")
    H = np.ones((q + 1, q + 1), dtype=np.int64)
    residues = {(j * j) % q for j in range(1, (q - 1) // 2 + 1)}
    for i in range(q):
        for j in range(q):
            H[1 + i, 1 + j] = 1 if ((i - j) % q) in residues else -1
    return H


def assert_hadamard(K):
    import numpy as np
    n = K.shape[0]
    assert K.shape == (n, n)
    assert np.array_equal(K @ K.T, (n * np.eye(n, dtype=np.int64)).astype(np.int64)), "not Hadamard"
    assert set(np.unique(K)) <= {1, -1}


def build_H_5120(seed: int = SEED):
    """H_5120 = D (K_20 kron S_256)/sqrt(5120), D Rademacher row signs; fp32 [5120,5120].

    Verbatim construction from scripts/tmp/rotation_t1t2_common.py (checked against
    r4-recon.md §Q2; nothing edited in place).
    """
    import numpy as np
    import torch
    K = paley1(19)          # 20x20 Paley type-I
    S = sylvester(256)      # 256x256 Sylvester
    assert_hadamard(K)
    assert_hadamard(S)
    B = np.kron(K, S) / np.sqrt(D_MODEL)
    g = torch.Generator().manual_seed(seed)
    D = torch.where(torch.rand(D_MODEL, generator=g) < 0.5, -1.0, 1.0)
    H = torch.from_numpy(B).float() * D.unsqueeze(1)
    return H.contiguous()


# ------------------------------------------------------------------------------- GPTQ
def find_params_sym(w):  # w [out, group] -> per-row scale, zero=8 (classic sym)
    import torch
    maxq = 2 ** BITS - 1
    tmp = torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
    xmin = torch.minimum(w.min(-1)[0], tmp)
    xmax = torch.maximum(w.max(-1)[0], tmp)
    xmax = torch.maximum(xmin.abs(), xmax)
    neg = xmin < 0
    xmin = torch.where(neg, -xmax, xmin)
    deg = (xmin == 0) & (xmax == 0)
    xmin = torch.where(deg, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(deg, torch.full_like(xmax, 1.0), xmax)
    scale = (xmax - xmin) / maxq
    zero = torch.full_like(scale, (maxq + 1) / 2)
    return scale, zero


def gptq_from_hinv_batched(Ws, hinv, groups, blocksize: int = BLOCKSIZE):
    """GPTQ bits=4 sym desc_act=False damp 0.01 for ALL GPTQ arms at once.

    Ws [A, out, in] fp32 or bf16 (one arm per slice); hinv [A, in, in] fp32 upper
    inverse-Hessian factors; groups = per-arm group size. Champion contract
    (classic GPTQ static order, group scales from the running W, lazy batch
    update with the Err1 carry). Batched over arms: one kernel set per column
    for all arms.

    Per-column arithmetic is unchanged; the loop is dispatch-bound, so every step
    writes in place into a preallocated buffer, codes accumulate in fp32 (one cast at
    the end) and the per-arm group scale is a single [A, out] row instead of a
    [A, out, in] broadcast table (which alone was 1.07 GiB at 17408 wide).
    Returns (Wq [A,out,in] fp32, codes [A,out,in] uint8, scales per arm fp32).
    """
    A_dim, out_f, in_f = Ws.shape
    dev = Ws.device
    W = Ws.float().clone()
    codes = torch.zeros(A_dim, out_f, in_f, dtype=torch.float32, device=dev)
    scales = [torch.zeros(out_f, (in_f + g - 1) // g, dtype=torch.float32, device=dev)
              for g in groups]
    S = torch.ones(A_dim, out_f, device=dev)          # current group scale, per arm
    t = torch.empty(A_dim, out_f, device=dev)
    e = torch.empty(A_dim, out_f, device=dev)
    bs = blocksize
    for i1 in range(0, in_f, bs):
        i2 = min(i1 + bs, in_f)
        W1 = W[:, :, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = hinv[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            for a, g in enumerate(groups):
                if col % g == 0:
                    g2 = min(col + g, in_f)
                    s, _ = find_params_sym(W[a, :, col:g2])
                    scales[a][:, col // g] = s
                    S[a] = s
            w = W1[:, :, i]
            torch.div(w, S, out=t)
            torch.round(t, out=t)
            t.add_(8.0)
            t.clamp_(0.0, 15.0)
            codes[:, :, col].copy_(t)
            t.sub_(8.0)
            t.mul_(S)                                  # q = s * (tc - 8)
            torch.sub(w, t, out=e)
            e.div_(Hinv1[:, i, i].unsqueeze(1))         # err = (w - q) / d
            Err1[:, :, i].copy_(e)
            # plain ops: baddbmm with out= aliased to the beta=1 input view was
            # non-deterministic (verified: same input -> different codes per run)
            W1[:, :, i:] -= e.unsqueeze(2) @ Hinv1[:, i, i:].unsqueeze(1)
        # write the block's quantized values back into the running W (group scales are
        # exactly the ones used at quantization time: bs is a multiple of every group)
        for a, g in enumerate(groups):
            idx = torch.arange(i1, i2, device=dev) // g
            W[a, :, i1:i2] = (codes[a, :, i1:i2] - 8.0) * scales[a][:, idx]
        if i2 < in_f:
            W[:, :, i2:] -= Err1 @ hinv[:, i1:i2, i2:]
    return W, codes.to(torch.uint8), scales




def _diag_block_factors(C):
    """b x b SPD device block -> (lower Cholesky factor, its triangular inverse) on the
    device. LAPACK on the host CPU: b^3/3 + b^3 flops per block, ~4 ms at b=256; the
    device path is a D2H fallback anyway, so this is strictly cheaper."""
    Cc = C.detach().to("cpu", torch.float32 if os.environ.get("FMT_REPLAY_OLD") == "1" else torch.float64)
    Lb = torch.linalg.cholesky(Cc)
    Li = torch.linalg.solve_triangular(
        Lb, torch.eye(Lb.shape[0], dtype=Lb.dtype), upper=False)
    return Lb.to(device=C.device, dtype=C.dtype), Li.to(device=C.device, dtype=C.dtype)


def _pad_spd(H, blk: int):
    """Pad an n x n SPD matrix to a multiple of blk with an identity border:
    blkdiag(H, I) is SPD with a block-consistent factor and inverse, so the extra rows
    cost nothing and every tile below is a contiguous view."""
    n = H.shape[0]
    n_pad = -(-n // blk) * blk
    if n_pad == n:
        return H, n
    Hp = torch.zeros(n_pad, n_pad, dtype=H.dtype, device=H.device)
    Hp[:n, :n] = H
    idx = torch.arange(n, n_pad, device=H.device)
    Hp[idx, idx] = 1.0
    return Hp, n


def cholesky_lower_blocked(H_in, blk: int = HINV_BLK):
    """Bordered (left-looking) blocked Cholesky, lower, IN PLACE on a padded device fp32
    SPD matrix; returns the unpadded lower factor. Only gemm-class ops run on the device:

      L_jj = chol(H_jj - L[j,:j] L[j,:j]^T)     (CPU tile)
      L21  = (H[j2:,j:j2] - L[j2:,:j] L[j:j2,:j]^T) L_jj^-T   (addmm + gemm)

    Each trailing block is re-read from the still-pristine H, so no trailing syrk is
    needed and the gemm work is ~n^3/6 rather than the textbook n^3/3. torch's own
    torch.linalg.cholesky has no XPU kernel: it D2H-fell back and cost 22.5 s for the
    17408-wide down_proj Hessian.
    """
    H, n = _pad_spd(H_in, blk)
    n_pad = H.shape[0]
    sc = torch.zeros(n_pad, blk, device=H.device, dtype=H.dtype)
    for j in range(0, n_pad, blk):
        j2 = j + blk
        Cb = sc[:blk, :blk]
        Cb.copy_(H[j:j2, j:j2])
        if j:
            Cb.addmm_(H[j:j2, :j], H[j:j2, :j].transpose(0, 1), alpha=-1.0, beta=1.0)
        Lb, Li = _diag_block_factors(Cb)
        H[j:j2, j:j2] = Lb
        if j2 < n_pad:
            A21 = sc[:n_pad - j2, :blk]
            A21.copy_(H[j2:, j:j2])
            if j:
                A21.addmm_(H[j2:, :j], H[j:j2, :j].transpose(0, 1), alpha=-1.0, beta=1.0)
            H[j2:, j:j2] = A21 @ Li.transpose(0, 1)
        del Lb, Li
    del sc
    # The upper triangle is still the input Hessian; triangular solve needs zeros.
    return H[:n, :n].tril_().contiguous()




def hinv_upper_inplace(H):
    """Reverse-order Cholesky avoids forming the ill-conditioned inverse Gram.

    J H J = L L^T implies U = J L^-1 J and U^T U = H^-1. Unlike
    chol(X^T X), this never rounds a dense inverse before factoring it.
    The damped input is consumed; on any numerical failure batched_hinv uses
    the classic fp64 chain, without changing percdamp.
    """
    L = cholesky_lower_blocked(H.flip((0, 1)))
    eye = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
    X = torch.linalg.solve_triangular(L, eye, upper=False)
    del L, eye
    return X.flip((0, 1)).contiguous()


HINV_FALLBACKS = []      # [{"layer", "arm", "group", "reason", "stats"}, ...]


def hinv_classic_fp64(H_cpu_f32):
    """Single fp64 matrix, in-place classic sequence; no simultaneous L/Hi/U.

    LAPACK may allocate its own workspace. The caller consumes only one Hessian
    at a time and releases this result immediately after its fp32 device copy.
    """
    H64 = H_cpu_f32.double()
    torch.linalg.cholesky(H64, out=H64)
    torch.cholesky_inverse(H64, out=H64)
    torch.linalg.cholesky(H64, upper=True, out=H64)
    return H64


def batched_hinv(Hs, percdamp: float = PERCDAMP, labels=None):
    """List of [in, in] Gram matrices -> (list of upper inverse-Hessian factors U,
    list of per-matrix info dicts).

    U is the classic GPTQ Hinv (H^-1 = U^T U, same object as
    cholesky(cholesky_inverse(cholesky(H)), upper=True) in the reference port).
    Damping is the champion contract (percdamp=0.01 of mean diag; NO adaptive
    damping). Primary path: the device blocked factorization; a CPU fp32 copy of
    the damped H is kept until U is verified, so on ANY failure or non-finite U
    we recompute with the classic sequence in fp64 on CPU and record an
    entry in HINV_FALLBACKS (layer/arm/group + matrix stats). If the fp64 classic
    factorization also fails, that is a real property of the calibration data:
    we raise naming the tensor (no silent fixes).
    """
    outs, infos = [], []
    for i, H in enumerate(Hs):
        lab = labels[i] if labels else (None, None, None)
        dev_h = H.device
        diag = H.diagonal().clone()
        dead = diag == 0
        H[dead, dead] = 1.0
        damp = percdamp * diag.mean()
        H.diagonal().add_(damp)
        del diag
        H_cpu = H.detach().to("cpu", torch.float32, copy=True)   # fallback + dump copy
        tag = f"L{lab[0]}_{lab[2]}_{lab[1]}" if lab[0] is not None else "hessian"
        dd = os.environ.get("R4_DUMP_H")
        if dd and lab[0] in (2, 11):
            from safetensors.torch import save_file
            os.makedirs(dd, exist_ok=True)
            save_file({"H_damped": H_cpu.contiguous()}, f"{dd}/{tag}.safetensors")
        fb = None
        try:
            U = hinv_upper_inplace(H)
            if not torch.isfinite(U).all():
                raise RuntimeError(f"blocked factorization produced non-finite U "
                                   f"({int((~torch.isfinite(U)).sum())} entries)")
            if not bool((U.diagonal() > 0).all()):
                raise RuntimeError("blocked factorization produced non-positive U diagonal")
        except Exception as exc:  # noqa: BLE001  -> classic fp64 fallback
            fb = str(exc)[:200]
            tag = f"L{lab[0]}_{lab[2]}_{lab[1]}" if lab[0] is not None else "hessian"
            dd = os.environ.get("R4_DUMP_H")
            if dd:
                os.makedirs(dd, exist_ok=True)
                from safetensors.torch import save_file
                p = f"{dd}/FAIL_{tag}.safetensors"
                save_file({"H_damped": H_cpu.contiguous()}, p)
                log(f"hinv fallback: dumped failing Hessian to {p}")
            t_fb = time.time()
            try:
                U_cpu = hinv_classic_fp64(H_cpu)
            except Exception as exc64:  # noqa: BLE001  real property: raise loudly
                raise RuntimeError(
                    f"hinv: fp64 classic factorization ALSO failed for "
                    f"layer={lab[0]} arm={lab[2]} group={lab[1]} "
                    f"(blocked: {fb}; fp64: {str(exc64)[:120]}); the damped H is "
                    "genuinely not SPD - real property of this calibration, "
                    "refusing to continue") from exc64
            U = U_cpu.float().to(dev_h)
            del U_cpu
            log(f"hinv fallback ({tag}): blocked failed ({fb[:80]}); classic fp64 "
                f"CPU factorization succeeded in {time.time() - t_fb:.1f}s")
        assert bool(torch.isfinite(U).all()) and bool((U.diagonal() > 0).all()), f"invalid upper factor: {tag}"
        if HINV_VERIFY:
            try:
                U2 = hinv_upper_inplace(H_cpu.to(dev_h))
            except Exception:
                U2_cpu = hinv_classic_fp64(H_cpu)
                U2 = U2_cpu.float().to(dev_h)
                del U2_cpu
            equal = torch.equal(U, U2)
            proof = {"layer": lab[0], "arm": lab[2], "group": lab[1],
                     "n": H_cpu.shape[0], "torch_equal": equal,
                     "finite": bool(torch.isfinite(U).all()),
                     "positive_diagonal": bool((U.diagonal() > 0).all()),
                     "fallback": fb}
            del U2
            if lab[0] in (2, 11):
                U_ref_cpu = hinv_classic_fp64(H_cpu)
                U_ref = U_ref_cpu.float().to(dev_h)
                del U_ref_cpu
                proof["rel_fro_vs_classic_fp64"] = float((U - U_ref).norm() / U_ref.norm())
                if dd:
                    from safetensors.torch import save_file
                    save_file({"U": U.cpu(), "U_classic_fp64": U_ref.cpu()}, f"{dd}/U_{tag}.safetensors")
                del U_ref
            HINV_PROOFS.append(proof)
            wjson(f"{RUN_DIR}/raw/hinv_proofs.json", HINV_PROOFS)
            assert equal, f"hinv nondeterministic: {tag}"
            assert proof["finite"] and proof["positive_diagonal"], proof
        stats = {"fallback": fb,
                 "damp": float(damp),
                 "n_dead": int(dead.sum())}
        if fb is not None:   # log real fallbacks only; per-layer count goes to layers.jsonl
            HINV_FALLBACKS.append({"layer": lab[0], "arm": lab[2], "group": lab[1], **stats})
        infos.append({**stats, "dead_mask": dead.cpu()})
        del H_cpu
        outs.append(U)
    return outs, infos


def hinv_from_HtXtX(H, percdamp: float = PERCDAMP):
    """Single-matrix wrapper (host CPU unit test / smoke self-check); H is consumed."""
    return batched_hinv([H], percdamp)[0][0]


# ------------------------------------------------------------------- auto_gptq packing
def auto_gptq_pack(codes, scales, group: int):
    """Pack sym uint4 codes [out,in] + fp16 scales [out,in/g] into the auto_gptq
    layout consumed by vLLM XPU W4A16 (XPUwNa16LinearKernel):
      qweight int32 [in/8, out]  (pack along the INPUT dim, 8 nibbles/int32)
      scales  fp16  [in/g, out]
      qzeros  int32 [in/g, out/8] encoding zp-1 with zp=8 (sym, static)
      g_idx   int32 [in] sequential (desc_act off)
    """
    import torch
    out_f, in_f = codes.shape
    assert in_f % 8 == 0 and in_f % group == 0 and group % 8 == 0
    c = codes.to(torch.int64)                                   # [out, in]
    qw = torch.zeros(in_f // 8, out_f, dtype=torch.int64, device=codes.device)
    for k in range(8):              # 8 uint4 nibbles per int32, 4-bit lanes
        qw += c[:, k::8].T.contiguous() << (4 * k)
    qw = qw.to(torch.int32)         # uint32 bit pattern; int32 storage per auto_gptq
    sc = scales.to(torch.float16).T.contiguous()                # [in/g, out]
    n_g = in_f // group
    val = 0x77777777          # (zp-1)=7 in every nibble, zp=8 sym
    qz = torch.full((n_g, out_f // 8), val, dtype=torch.int64,
                    device=codes.device).to(torch.int32)
    g_idx = torch.arange(in_f, dtype=torch.int32, device=codes.device)
    return {"qweight": qw.contiguous(), "scales": sc, "qzeros": qz.contiguous(),
            "g_idx": g_idx}


# ------------------------------------------------------------------------ module checks
def verify_modeling_contract(txt_cfg, sd, layer_idx):
    """Live check vs the INSTALLED transformers qwen3_5 modeling code:
    - RMSNorm applies (1.0 + weight)
    - DecoderLayer.forward is the two-residual-add pattern (no other residual writer)
    - every nn.Linear in the built layer is in the known reader/writer lists
    """
    import torch
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    src = Path(M.__file__).read_text(encoding="utf-8")
    assert "1.0 + self.weight" in src, "installed RMSNorm does not use (1+w) gain"
    fwd = src[src.index("class Qwen3_5DecoderLayer"):]
    fwd = fwd[:fwd.index("\nclass ")]
    assert fwd.count("residual + hidden_states") == 2, \
        "DecoderLayer.forward residual pattern changed"
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    layer = Qwen3_5DecoderLayer(txt_cfg, layer_idx)
    lin = {n for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}
    known = {r[:-len(".weight")] for r in
             READERS_GDN + READERS_ATT + READERS_MLP + WRITERS_GDN + WRITERS_ATT + WRITERS_MLP}
    assert lin <= known, f"unknown residual-stream linears: {lin - known}"
    del layer
    return {"transformers": transformers.__version__,
            "modeling_file": str(Path(M.__file__)),
            "residual_adds": 2, "norm_gain": "1.0 + self.weight",
            "linears": sorted(lin)}



# ---- run_layer (copy of hq4_states_20261010.run_layer with a device fix: the fresh
# Qwen3_5TextRotaryEmbedding and the causal mask must live on the arm device; the
# original runs on CPU. Semantics identical: eager attention, manual additive causal
# mask, per-row independent sequences.)
@torch.no_grad()
def run_layer(layer, x, txt_cfg, layer_idx):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    T = x.shape[1]
    is_full = txt_cfg.layer_types[layer_idx] == "full_attention"
    pos = torch.arange(T, dtype=torch.long, device=x.device).view(1, 1, -1)
    text_pid = pos[0]
    mrope_pid = pos.expand(3, 1, -1)
    rotary = Qwen3_5TextRotaryEmbedding(config=txt_cfg).to(x.device)
    cos, sin = rotary(x, mrope_pid)
    if is_full:
        m = torch.zeros(1, 1, T, T, dtype=x.dtype, device=x.device)
        m = m.masked_fill(
            torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1),
            float("-inf"))
        attn_mask = m
    else:
        attn_mask = None
    return layer(x, position_embeddings=(cos, sin), attention_mask=attn_mask,
                 position_ids=text_pid)


def _build_layer_gpu(txt_cfg, layer_idx: int, sd_gpu, dtype=torch.bfloat16):
    """Build a Qwen3_5DecoderLayer directly on the arm device from a GPU state
    dict (skips the CPU round-trip of hst.build_layer; folded tensors stay on
    device). dtype=torch.float32 builds the R4_FP32 diagnostic arm."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    layer = Qwen3_5DecoderLayer(txt_cfg, layer_idx).to(dtype).to(
        next(iter(sd_gpu.values())).device)
    missing, unexpected = layer.load_state_dict(sd_gpu, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not [m for m in missing if "rotary" not in m], f"missing keys: {missing}"
    return layer.eval()


def fold_layer_sd(sd: dict, H, dev, out_dtype=torch.bfloat16):
    """R4 fold: readers W diag(1+w) Q, writers Q^T W, norm gains zeroed.
    fp32 matmuls on GPU, then cast to out_dtype (design §2.1; rotation_t1t2_common
    norm_gain_fold + h_fold combined, gain-then-Q order). out_dtype=torch.float32
    keeps the folded weights in fp32 (the R4_FP32 diagnostic arm)."""
    import torch
    sd = {k: v.to(dev) for k, v in sd.items()}
    Q = H.to(dev).float()
    g_in = 1.0 + sd["input_layernorm.weight"].float()
    g_post = 1.0 + sd["post_attention_layernorm.weight"].float()
    readers_in = READERS_GDN if "linear_attn.in_proj_qkv.weight" in sd else READERS_ATT
    for name in readers_in:
        if name in sd:
            sd[name] = ((sd[name].float() * g_in[None, :]) @ Q).to(out_dtype)
    for name in READERS_MLP:
        if name in sd:
            sd[name] = ((sd[name].float() * g_post[None, :]) @ Q).to(out_dtype)
    for name in WRITERS_GDN + WRITERS_ATT + WRITERS_MLP:
        if name in sd:
            sd[name] = (Q.T @ sd[name].float()).to(out_dtype)
    sd["input_layernorm.weight"] = torch.zeros_like(sd["input_layernorm.weight"])
    sd["post_attention_layernorm.weight"] = torch.zeros_like(
        sd["post_attention_layernorm.weight"])
    return sd


def rmsnorm(x, w, eps: float):
    import torch
    xf = x.float()
    o = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (o * (1.0 + w.float())).to(x.dtype)


# ------------------------------------------------------------------------- EXL3 decode
_DECODE_CHECK = set()              # bit-exact checks once per artifact and K
_DECODE_RECORDS = []


def _weight_orig_from_inner(inner_f16, suh, svh, ref):
    """ref.weight_orig tail VERBATIM (same ops, same order, CPU, fp32): inner fp16
    [k,n] Hadamard-domain reconstruction -> original-basis fp32 weight [k,n]."""
    w = inner_f16.float()
    k, n = w.shape
    h = ref.hadamard128(w.device, torch.float32)
    w = (h @ w.view(k // 128, 128, n)).view(k, n)
    w = w * suh.float().unsqueeze(1)
    w = (w.view(k, n // 128, 128) @ h).view(k, n)
    w = w * svh.float().unsqueeze(0)
    return w


def load_exl3_block_cpu(layer, names, artifact_root):
    """Worker-safe file loading only: every returned tensor stays on CPU."""
    from safetensors.torch import safe_open
    wm = read_json(hst.under(artifact_root, "model.safetensors.index.json"))["weight_map"]
    packed = {}
    for base in names:
        key = f"{PREFIX}layers.{layer}.{base}"
        extra = f"{key}.mul1" if f"{key}.mul1" in wm else f"{key}.mcg"
        want = [f"{key}.trellis", f"{key}.suh", f"{key}.svh", extra]
        shards = {wm[k] for k in want}
        assert len(shards) == 1, f"{key} spans shards: {shards}"
        with safe_open(hst.under(artifact_root, wm[want[0]]), framework="pt", device="cpu") as f:
            tensors = tuple(f.get_tensor(k) for k in want)
        assert all(t.device.type == "cpu" for t in tensors)
        packed[base] = (extra, tensors)
    return packed

def decode_exl3_block(layer: int, names, ref, dev=None, artifact_root=EXL3, packed=None):
    """Bit-exact ref.py reconstruction of one block's quantized linears
    (hq4_f0_20261010.decode_exl3_block semantics; /exl3 mount).

    dev is not None: the integer trellis reconstruction (ref.reconstruct_inner: pure
    int/permute ops, and 93% of the 43 s/layer CPU cost) runs on XPU; the fp32 Hadamard
    sandwich stays on the CPU in ref's exact op order, so the returned fp32 weight is
    bit-identical to the CPU-only path. torch.equal is asserted once per
    (artifact_root,K), including each new mixed-rate code encountered."""
    if dev is not None:
        import threading
        assert threading.current_thread() is threading.main_thread(), "XPU decode is main-thread only"
    if packed is None:
        packed = load_exl3_block_cpu(layer, names, artifact_root)
    weights = {}
    for base in names:
        key = f"{PREFIX}layers.{layer}.{base}"
        extra, tensors = packed.pop(base)
        tre, suh, svh, scal = tensors
        cb_val = int(scal.item()) if scal.numel() == 1 else int(scal)
        cb = (ref.CB_MUL1 if extra.endswith("mul1") and cb_val else
              (ref.CB_MCG if extra.endswith("mcg") and cb_val else ref.CB_3INST))
        k_bits = int(tre.shape[2]) // 16
        assert tre.shape[2] % 16 == 0 and 2 <= k_bits <= 8, f"invalid trellis K on {key}"
        if dev is None:
            w_in_out = ref.weight_orig(tre, suh, svh, k_bits, cb)
        else:
            tre_gpu = tre.to(dev)
            inner_gpu = ref.reconstruct_inner(tre_gpu, k_bits, cb)
            inner = inner_gpu.cpu()
            torch.xpu.synchronize()
            del tre_gpu, inner_gpu
            if (artifact_root, k_bits) not in _DECODE_CHECK:
                inner_ref = ref.reconstruct_inner(tre, k_bits, cb)
                assert torch.equal(inner, inner_ref), \
                    f"{key}: XPU trellis reconstruction is not bit-exact vs CPU"
                w_ref = ref.weight_orig(tre, suh, svh, k_bits, cb)
                assert torch.equal(_weight_orig_from_inner(inner, suh, svh, ref), w_ref), \
                    f"{key}: Hadamard tail changed vs ref.weight_orig"
                del inner_ref, w_ref
                _DECODE_CHECK.add((artifact_root, k_bits))
                _DECODE_RECORDS.append({"root": artifact_root, "K": k_bits, "tensor": key,
                                        "integer_torch_equal": True, "weight_torch_equal": True})
                wjson(f"{RUN_DIR}/raw/exl3_decode_checks.json", _DECODE_RECORDS)
                log(f"EXL3 decode self-check: root={artifact_root} K={k_bits} "
                    f"XPU reconstruction + CPU Hadamard torch.equal on {key}")
            w_in_out = _weight_orig_from_inner(inner, suh, svh, ref)
            del inner
        weights[f"{base}.weight"] = w_in_out.T.contiguous()   # [out, in]
        del tre, suh, svh, w_in_out
    return weights


def artifact_head_logits(root, normalized_rows, ref, dev):
    """Stream output columns on block128 boundaries; exact ref.py head decode."""
    from safetensors.torch import safe_open
    wm = read_json(hst.under(root, "model.safetensors.index.json"))["weight_map"]
    extra = "lm_head.mul1" if "lm_head.mul1" in wm else "lm_head.mcg"
    with safe_open(hst.under(root, wm["lm_head.trellis"]), framework="pt", device="cpu") as f:
        tre = f.get_tensor("lm_head.trellis")
        suh = f.get_tensor("lm_head.suh")
        svh = f.get_tensor("lm_head.svh")
        val = int(f.get_tensor(extra).item())
    cb = ref.CB_MUL1 if extra.endswith("mul1") and val else (ref.CB_MCG if extra.endswith("mcg") and val else ref.CB_3INST)
    K = tre.shape[2] // 16
    assert tre.shape[2] % 16 == 0 and 2 <= K <= 8
    assert svh.numel() % 128 == 0, "head output dimension must be Hadamard128 aligned"
    out = torch.empty(normalized_rows.shape[0], svh.numel(), device=dev, dtype=torch.float32)
    for j in range(0, svh.numel(), 8192):
        j2 = min(j + 8192, svh.numel())
        block = tre[:, j // 16:j2 // 16].contiguous()
        inner = ref.reconstruct_inner(block.to(dev), K, cb).cpu()
        weight = _weight_orig_from_inner(inner, suh, svh[j:j2], ref)
        if (root, K) not in _DECODE_CHECK:
            # A full input axis and one aligned output128 block proves the tail.
            sample = block[:, :8].contiguous()
            expected = ref.reconstruct_inner(sample, K, cb)
            actual = ref.reconstruct_inner(sample.to(dev), K, cb).cpu()
            assert torch.equal(expected, actual), f"head K={K} integer decode mismatch"
            assert torch.equal(_weight_orig_from_inner(actual, suh, svh[j:j+128], ref),
                               ref.weight_orig(sample, suh, svh[j:j+128], K, cb))
            _DECODE_CHECK.add((root, K))
            _DECODE_RECORDS.append({"root": root, "K": K, "tensor": "lm_head",
                                    "integer_torch_equal": True, "weight_torch_equal": True})
            wjson(f"{RUN_DIR}/raw/exl3_decode_checks.json", _DECODE_RECORDS)
            del expected, actual
        weight_gpu = weight.to(torch.bfloat16).to(dev).float()
        out[:, j:j2] = normalized_rows @ weight_gpu
        del block, inner, weight, weight_gpu
    return out

def artifact_tensor(root, key):
    from safetensors.torch import safe_open
    wm = read_json(hst.under(root, "model.safetensors.index.json"))["weight_map"]
    with safe_open(hst.under(root, wm[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key)

def artifact_layer_unquantized(root, layer):
    from safetensors.torch import safe_open
    wm = read_json(hst.under(root, "model.safetensors.index.json"))["weight_map"]
    prefix = f"{PREFIX}layers.{layer}."
    groups = {}
    for key, shard in wm.items():
        if key.startswith(prefix) and key.endswith((".weight", ".bias", ".A_log", ".dt_bias")):
            groups.setdefault(shard, []).append(key)
    out = {}
    for shard, keys in groups.items():
        with safe_open(hst.under(root, shard), framework="pt", device="cpu") as f:
            for key in keys:
                out[key[len(prefix):]] = f.get_tensor(key)
    return out

# ------------------------------------------------------------------------------- data
def build_inputs(smoke: bool):
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import safe_open

    meta = read_json(hst.under(STATES, "corpus_meta.json"))
    parquet = CORPUS  # CORPUS already names the parquet file (see preflight)
    sha = sha256_file(parquet)
    assert sha == meta["sha256"], f"corpus sha mismatch {sha} != {meta['sha256']}"
    rows = pq.read_table(parquet).column("text").to_pylist()
    tok = AutoTokenizer.from_pretrained(BF16, trust_remote_code=True)
    ids_all = tok("\n".join(rows), add_special_tokens=False)["input_ids"]
    assert len(ids_all) == meta["tokens_total"], \
        f"token count {len(ids_all)} != {meta['tokens_total']}"

    n_calib = (N_CALIB if SMOKE_FULLCALIB else 2) if smoke else N_CALIB
    hi = int(len(ids_all) * CALIB_FRAC)
    starts = [int(i * (hi - T_CALIB) / max(n_calib - 1, 1)) for i in range(n_calib)]
    assert starts[-1] + T_CALIB <= hi
    calib = torch.tensor([ids_all[s:s + T_CALIB] for s in starts], dtype=torch.long)

    with safe_open(hst.under(STATES, EVAL_FILE), framework="pt", device="cpu") as f:
        ev = f.get_tensor("input_ids")
    assert tuple(ev.shape) == (N_EVAL, T_EVAL), tuple(ev.shape)
    t_eval = T_EVAL if (not smoke or SMOKE_FULLCALIB) else 64
    response_mask = torch.ones_like(ev, dtype=torch.bool)
    wiki_ev = ev[:, :t_eval].clone()
    trace_info = None
    trace_path = os.environ.get("FMT_HELDOUT_TRACE")
    if trace_path:
        assert "cal_trace" not in Path(trace_path).name, "calibration trace is not held-out eval"
        trace = read_json(trace_path)
        selected, mask_rows, provenance = [], [], []
        for ri, row in enumerate(trace["rows"]):
            prompt, response = row["input_ids"], row["response_ids"]
            if len(prompt) + 2 > t_eval or not response:
                continue
            response = response[:t_eval - len(prompt)]
            sequence = prompt + response
            pad = t_eval - len(sequence)
            selected.append(sequence + [tok.pad_token_id or tok.eos_token_id] * pad)
            mask_rows.append([len(prompt)-1 <= p < len(sequence)-1 for p in range(t_eval)])
            provenance.append({"source_row": ri, "prompt_tokens": len(prompt), "response_tokens": len(response),
                               "conversation": row.get("conversation"), "turn": row.get("turn")})
            if len(selected) == N_EVAL:
                break
        assert len(selected) == N_EVAL, "held-out trace has insufficient fitting rows"
        trace_ev = torch.tensor(selected, dtype=torch.long)
        trace_mask = torch.tensor(mask_rows, dtype=torch.bool)
        ev = torch.cat((wiki_ev, trace_ev))
        response_mask = torch.cat((torch.ones_like(wiki_ev, dtype=torch.bool), torch.zeros_like(trace_mask)))
        build_inputs.trace_mask = torch.cat((torch.zeros_like(wiki_ev, dtype=torch.bool), trace_mask))
        trace_info = {"path": trace_path, "sha256": sha256_file(trace_path),
                      "template_vars": trace.get("template_vars"), "selection": provenance,
                      "scored_response_positions": int(trace_mask.sum()),
                      "rendering": "stored input_ids already include model chat template; no retokenization"}
    build_inputs.response_mask = response_mask[:, :t_eval]
    ev = ev[:, :t_eval]

    held = read_json(str(REPO / HELDOUT_MANIFEST_REL))
    canaries = [dict(id=R01["id"], prompt=R01["prompt"], expected=R01["expected"],
                     foil=R01["foil"])]
    for r in held["rows"]:
        canaries.append(dict(id=r["id"], prompt=r["prompt"], expected=r["expected"],
                             foil=r["foil_answer"]))
    for c in canaries:
        # transformers 5 returns a BatchEncoding for tokenize=True; render text, then tokenize.
        text = tok.apply_chat_template([{"role": "user", "content": c["prompt"]}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        c["ids"] = torch.tensor([ids], dtype=torch.long)
        c["n_prompt"] = len(ids)
        c["exp_first"] = tok(c["expected"], add_special_tokens=False)["input_ids"][0]
        c["foil_first"] = tok(c["foil"], add_special_tokens=False)["input_ids"][0]
        assert c["exp_first"] != c["foil_first"], f"{c['id']}: first tokens equal"
    assert canaries[0]["n_prompt"] == 36, f"r01 rendered {canaries[0]['n_prompt']} tokens"
    info = {"corpus_sha256": sha, "tokens_total": len(ids_all), "calib_starts": starts,
            "eval_shape": list(ev.shape), "r01_prompt_tokens": canaries[0]["n_prompt"],
            "canaries": [{k: c[k] for k in ("id", "n_prompt", "exp_first", "foil_first")}
                         for c in canaries]}
    info["heldout_trace"] = trace_info
    return tok, calib, ev, canaries, info


@torch.no_grad()
def fold_exactness_probe_fp32(txt_cfg, L, sd_cpu, Q, dev, x_bf16):
    """Gated s1 (amended 2026-10-11): fp32 fold exactness on ONE layer.

    Path O: original layer in fp32 (bf16-sourced weight values). Path R: folded
    layer in fp32 (readers W diag(1+w) Q, writers Q^T W, gains zeroed), input
    x @ Q, output @ Q^T. The fold is exact in real arithmetic, so the paths must
    agree to fp32 roundoff; rel L2 <= 1e-4 passes on every quantized layer.
    """
    t0 = time.time()
    x = x_bf16.float()
    xr = (x @ Q).contiguous()
    O = _build_layer_gpu(txt_cfg, L,
                         {k: v.float().to(dev) for k, v in sd_cpu.items()},
                         dtype=torch.float32)
    yO = run_layer(O, x, txt_cfg, L).float()
    del O
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    sd_fold32 = fold_layer_sd(sd_cpu, Q, dev, out_dtype=torch.float32)
    R = _build_layer_gpu(txt_cfg, L, sd_fold32, dtype=torch.float32)
    del sd_fold32
    yR = run_layer(R, xr, txt_cfg, L).float() @ Q.T
    del R
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    d = yO - yR
    out = {"n_tokens": int(x.shape[0] * x.shape[1]),
           "fold_fp32_rel_l2": float(d.norm() / yO.norm().clamp_min(1e-30)),
           "fold_fp32_max_abs": float(d.abs().max()),
           "out_rms": float(yO.pow(2).mean().sqrt()),
           "seconds": round(time.time() - t0, 2)}
    return out

def fold_exactness_probe(txt_cfg, L, sd_cpu, Q, Qb, dev, x_bf16, layers):
    """Layer-level exactness test of the R4 fold with NO bf16 storage in the loop.

    Path O: the ORIGINAL layer in fp32 (bf16-sourced weight values, gain (1+w) applied
            by the installed RMSNorm), input x.
    Path R: the FOLDED layer in fp32 (same weight values, readers W diag(1+w) Q,
            writers Q^T W, gains zeroed), input x @ Q, output @ Q^T.
    The fold is exact in real arithmetic, so the two paths must agree to fp32 roundoff.
    The same comparison in bf16 (the bf16 reference arm's layer vs the r4_fp layer)
    measures the model's own bf16 storage noise: the floor any "R4 arm vs bf16-arm"
    metric is bounded by, and the reason r4_fp/r4_g64 KL cannot reach 1e-7.

    layers: {"fp32_folded", "bf16_plain", "bf16_folded"} already-built layer objects.
    Returns relative-L2 / max-abs differences plus the fp32 storage-noise columns.
    """
    t0 = time.time()
    x = x_bf16.float()
    xr = (x @ Q).contiguous()
    out = {"n_tokens": int(x.shape[0] * x.shape[1]),
           "paths": "O=original fp32, R=folded fp32, both from the same bf16-valued weights"}
    O = _build_layer_gpu(txt_cfg, L,
                         {k: v.float().to(dev) for k, v in sd_cpu.items()},
                         dtype=torch.float32)
    yO32 = run_layer(O, x, txt_cfg, L).float()
    del O
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    yR32 = run_layer(layers["fp32_folded"], xr, txt_cfg, L).float() @ Q.T
    d = yO32 - yR32
    out["fold_fp32_rel_l2"] = float(d.norm() / yO32.norm().clamp_min(1e-30))
    out["fold_fp32_max_abs"] = float(d.abs().max())
    out["out_rms"] = float(yO32.pow(2).mean().sqrt())
    del d, yR32
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    # bf16 columns: the same two paths as the arms actually run them
    xb = x_bf16.contiguous()
    yOb = run_layer(layers["bf16_plain"], xb, txt_cfg, L).float()
    out["bf16_plain_vs_fp32_rel_l2"] = float(
        (yO32 - yOb).norm() / yO32.norm().clamp_min(1e-30))
    yRb = run_layer(layers["bf16_folded"], (xb @ Qb).contiguous(), txt_cfg, L).float() @ Qb.T.float()
    out["bf16_folded_vs_fp32_rel_l2"] = float(
        (yO32 - yRb).norm() / yO32.norm().clamp_min(1e-30))
    out["bf16_folded_vs_bf16_plain_rel_l2"] = float(
        (yOb - yRb).norm() / yOb.norm().clamp_min(1e-30))
    out["fold_fp32_rel_l2_over_bf16_noise"] = (
        None if out["bf16_plain_vs_fp32_rel_l2"] == 0 else
        round(out["fold_fp32_rel_l2"] / out["bf16_plain_vs_fp32_rel_l2"], 8))
    del yOb, yRb, yO32
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    out["seconds"] = round(time.time() - t0, 2)
    return out

class Arm:
    """One arm's hidden-state streams (calib only for GPTQ arms; canaries as
    per-canary tensors because prompt lengths differ)."""
    def __init__(self, name, calib, ev, canaries, rot, emb, dev, rot_mat=None, dtype=None):
        import torch
        self.name = name
        self.rot = rot
        self.dev = dev
        self.Hb = rot_mat
        self.dtype = dtype or torch.bfloat16
        with torch.no_grad():
            # cast BEFORE the sandwich so the fp32 diagnostic arm rotates in fp32
            # (the bf16 embedding values are identical; only storage precision rises)
            e_ev = emb[ev.to(dev)].to(self.dtype)
            e_can = [emb[c["ids"].to(dev)].to(self.dtype) for c in canaries]
            e_cal = (emb[calib.to(dev)].to(self.dtype)
                     if calib is not None else None)
            if rot:
                Hb = self.Hb
                if e_cal is not None:
                    e_cal = e_cal @ Hb
                e_ev = e_ev @ Hb
                e_can = [t @ Hb for t in e_can]
        self.calib = None if e_cal is None else e_cal.contiguous()
        self.eval = e_ev.contiguous()
        self.canary = [t.contiguous() for t in e_can]

        self.can_pos = [c["n_prompt"] - 1 for c in canaries]
        del e_ev, e_can, e_cal
        torch.xpu.synchronize()
        torch.xpu.empty_cache()

    def propagate(self, layer, txt_cfg, L):
        import torch
        with torch.no_grad():
            if self.calib is not None:
                self.calib = run_layer(layer, self.calib, txt_cfg, L)
            self.eval = run_layer(layer, self.eval, txt_cfg, L)
            self.canary = [run_layer(layer, t, txt_cfg, L) for t in self.canary]
            streams = [self.eval] + self.canary + ([self.calib] if self.calib is not None else [])
            for t in streams:
                assert torch.isfinite(t.float().sum()), f"{self.name}: non-finite at L{L}"

    def free_calib(self):
        self.calib = None
        import torch
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


def weighted_tensor_error(name, arm, Wq, H_calib, sd_cpu, Q, dev):
    """Undamped calibration-weighted error in the original linear's basis."""
    Worig = sd_cpu[name + ".weight"].to(dev, torch.float32)
    Horig = H_calib
    reconstructed = Wq.to(torch.bfloat16).float()
    basis = "original linear input"
    if ARM_SPECS[arm].transform == "fold":
        if name + ".weight" in READERS_GDN + READERS_ATT + READERS_MLP:
            norm = "post_attention_layernorm.weight" if name.startswith("mlp.") else "input_layernorm.weight"
            gain = 1.0 + sd_cpu[norm].to(dev, torch.float32)
            # Gains can be exactly zero (observed L7 MLP). Dividing cannot express
            # quantization leakage into that nullspace. Use original-coordinate
            # pre-gain RMSNorm inputs and effective W*gain: no pseudoinverse,
            # no dropped error, and the exact same linear output objective.
            Worig = Worig * gain[None, :]
            reconstructed = reconstructed @ Q.T
            Horig = Q @ H_calib @ Q.T
            basis = "original coordinates, pre-gain RMSNorm input; effective W*gain"
        elif name + ".weight" in WRITERS_GDN + WRITERS_ATT + WRITERS_MLP:
            reconstructed = Q @ reconstructed
    delta = reconstructed - Worig
    numerator = float(((delta @ Horig) * delta).sum())
    denominator = float(((Worig @ Horig) * Worig).sum())
    return {"e_t": numerator / denominator if denominator > 0 else None,
            "tr_W_H_Wt": denominator, "tr_dW_H_dWt": numerator,
            "numel": Worig.numel(), "basis": basis, "hessian": "undamped calibration Gram"}

def replay_hinv(capture_dir, dev):
    """Replay old two-Cholesky path on captured, already-damped real Hessians.

    Diagnostic only: no re-damping and no quality evaluation. Logs which of the
    old factorization stages fails; also checks reduction repeatability directly.
    """
    from safetensors.torch import load_file
    os.environ["FMT_REPLAY_OLD"] = "1"
    reports = []
    for path in sorted(Path(capture_dir).glob("L*.safetensors")):
        if "mlp.down_proj" not in path.name:
            continue
        Hc = load_file(str(path))["H_damped"]
        entry = {"file": str(path), "n": Hc.shape[0], "trials": []}
        last_u = None
        for deterministic in (False, True):
            torch.use_deterministic_algorithms(deterministic)
            for trial in range(2):
                stage = "first_cholesky"
                t = time.time()
                try:
                    H = Hc.to(dev)
                    L = cholesky_lower_blocked(H)
                    stage = "triangular_solve"
                    eye = torch.eye(L.shape[0], device=dev)
                    X = torch.linalg.solve_triangular(L, eye, upper=False)
                    del L, eye
                    stage = "inverse_gram"
                    M = X.T @ X
                    del X
                    asym = float((M - M.T).abs().max())
                    stage = "second_cholesky"
                    U = cholesky_lower_blocked(M).T.contiguous()
                    del M
                    row = {"deterministic": deterministic, "trial": trial,
                           "pass": True, "inverse_gram_max_asymmetry": asym,
                           "torch_equal_previous": torch.equal(U, last_u) if last_u is not None else None}
                    last_u = U
                except Exception as exc:
                    row = {"deterministic": deterministic, "trial": trial,
                           "pass": False, "stage": stage, "error": str(exc)}
                torch.xpu.synchronize()
                row["seconds"] = time.time() - t
                entry["trials"].append(row)
                log(f"replay {path.name}: {row}")
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
        reports.append(entry)
        wjson(f"{RUN_DIR}/raw/hinv_replay.json", reports)
        del Hc, last_u
    return 0

def main() -> int:
    global RUN_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["preflight", "run", "hinv-replay"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-dir", default=RUN_DIR)
    ap.add_argument("--capture-dir")
    args = ap.parse_args()
    RUN_DIR = args.run_dir
    Path(RUN_DIR, "raw").mkdir(parents=True, exist_ok=True)

    if args.mode == "preflight":
        ok = {"bf16_index": Path(hst.under(BF16, "model.safetensors.index.json")).is_file(),
              "exl3_index": Path(hst.under(EXL3, "model.safetensors.index.json")).is_file(),
              "states": Path(hst.under(STATES, EVAL_FILE)).is_file(),
              "corpus": Path(CORPUS).is_file(),
              "ref_py": Path(REFSRC, "exl3xpu", "ref.py").is_file(),
              "heldout": (REPO / HELDOUT_MANIFEST_REL).is_file()}
        wjson(f"{RUN_DIR}/raw/preflight_container.json", ok)
        print(json.dumps(ok))
        return 0 if all(ok.values()) else 2

    # ---------------------------------------------------------------------- run mode
    import torch
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mkldnn.fp32_precision = "ieee"
    torch.use_deterministic_algorithms(True)
    log(f"math: fp32={torch.get_float32_matmul_precision()} mkldnn={torch.backends.mkldnn.fp32_precision} "
        f"ONEDNN_DEFAULT_FPMATH_MODE={os.environ.get('ONEDNN_DEFAULT_FPMATH_MODE')} "
        f"deterministic={torch.are_deterministic_algorithms_enabled()} block={HINV_BLK}")
    dev = "xpu"
    assert torch.xpu.is_available(), "no XPU device visible"
    assert os.environ.get("ONEAPI_DEVICE_SELECTOR") == "level_zero:1", \
        "must run pinned to GPU1 (level_zero:1)"
    torch.xpu.set_device(0)
    torch.manual_seed(SEED)
    if args.mode == "hinv-replay":
        assert args.capture_dir, "hinv-replay requires --capture-dir"
        return replay_hinv(args.capture_dir, dev)

    ident = {"utc": now_iso(), "host": platform.node(), "gpu": torch.xpu.get_device_name(0),
             "torch": torch.__version__, "python": sys.version.split()[0],
             "repo_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
             "selector": os.environ.get("ONEAPI_DEVICE_SELECTOR"), "smoke": args.smoke}
    ident["harness_sha256"] = sha256_file(__file__)
    ident["launcher_sha256"] = sha256_file(str(Path(SCRIPT_DIR, "fmt_screen_launch_20261011.sh")))
    codec_path = Path(SCRIPT_DIR, "e8rvq_20261011.py")
    if codec_path.is_file():
        ident["codec_sha256"] = sha256_file(str(codec_path))
    ident["exl3_artifacts"] = {a: {"root": EXL3_ROOTS[a],
        "manifest_sha256": sha256_file(str(Path(EXL3_ROOTS[a], "ARTIFACTS.sha256"))),
        "revision": (Path(EXL3_ROOTS[a], "REVISION.txt").read_text() if Path(EXL3_ROOTS[a], "REVISION.txt").is_file() else None)}
        for a in EXL3_ARMS}
    wjson(f"{RUN_DIR}/raw/identity_start.json", ident)
    log(f"RQ-27 R0 screen start: {ident}")

    # pinned bit-exact ref.py
    ref_py = os.path.join(REFSRC, "exl3xpu", "ref.py")
    ref_sha = sha256_file(ref_py)
    assert ref_sha == REF_SHA256, f"ref.py sha {ref_sha} != pinned {REF_SHA256}"
    # git is not usable inside the image (ro mount, foreign owner); the launcher
    # verifies REF_GIT_HEAD on the host in preflight. The content pin above is binding.
    head = REF_GIT_HEAD + " (host-verified by launcher preflight)"
    sys.path.insert(0, hst.under(REFSRC, "exl3xpu"))
    import ref  # noqa: F401  pinned bit-exact EXL3 reference
    ident.update({"ref_py_sha256": ref_sha, "ref_git_head": head})
    if any(a.startswith("exl3_sc") for a in EXL3_ARMS):
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        synthetic = []
        for K in (3, 5, 6, 7):
            tre = torch.randint(-32768, 32767, (8, 8, 16*K), dtype=torch.int16, generator=generator)
            suh = torch.randn(128, generator=generator)
            svh = torch.randn(128, generator=generator)
            cpu = ref.reconstruct_inner(tre, K, ref.CB_MUL1)
            xpu = ref.reconstruct_inner(tre.to(dev), K, ref.CB_MUL1).cpu()
            assert torch.equal(cpu, xpu), f"synthetic K={K} decode mismatch"
            assert torch.equal(_weight_orig_from_inner(xpu, suh, svh, ref),
                               ref.weight_orig(tre, suh, svh, K, ref.CB_MUL1))
            synthetic.append({"K": K, "integer_torch_equal": True, "weight_torch_equal": True,
                              "scope": "synthetic128x128 mul1; actual artifacts checked separately"})
        wjson(f"{RUN_DIR}/raw/exl3_synthetic_mixedK_checks.json", synthetic)
        del tre, cpu, xpu

    # farms verified fail-closed
    for root, what in [(BF16, "BF16"), (STATES, "states")] + [(EXL3_ROOTS[a], a) for a in EXL3_ARMS]:
        okn, bad = hst.verify_manifest(root)
        assert not bad, f"{what} manifest verification failed: {bad}"
        log(f"{what} manifest verified ({okn} files)")
    index = read_json(hst.under(BF16, "model.safetensors.index.json"))

    txt_cfg = hst.build_text_config(BF16)
    if args.smoke and os.environ.get("R4_SMOKE_LAYERS"):
        layers = [int(x) for x in os.environ["R4_SMOKE_LAYERS"].split(",")]
        assert layers == sorted(layers) and len(set(layers)) == len(layers), \
            "R4_SMOKE_LAYERS must be an increasing unique list"
        log(f"R4_SMOKE_LAYERS override: {layers}")
    else:
        layers = SMOKE_LAYERS if args.smoke else list(
            range(txt_cfg.num_hidden_layers))
    log(f"layers={layers[:3]}...{layers[-1:]} n={len(layers)}")

    # ---- rotation Q, orthogonality in fp64 to 1e-6 -----------------------------
    Q = None
    Arm.Hb = None
    if ROT_ARMS:
        Q = build_H_5120(SEED).to(dev)
        Q64 = Q.double()
        err = (Q64 @ Q64.T - torch.eye(D_MODEL, device=dev, dtype=torch.float64)).abs().max()
        assert float(err) < 1e-6, f"Q orthogonality {float(err)}"
        log(f"H_5120 built seed {SEED}; fp64 orthogonality err {float(err):.2e}")
        del Q64
        Arm.Hb = Q.to(torch.bfloat16).contiguous()

    # ---- inputs ----------------------------------------------------------------
    tok, calib_ids, eval_ids, canaries, data_info = build_inputs(args.smoke)
    log(f"data: {json.dumps(data_info)[:400]}")

    # ---- modeling-code contract check (design §3) -------------------------------
    contract = verify_modeling_contract(txt_cfg, None, layers[0])
    wjson(f"{RUN_DIR}/raw/modeling_contract.json", contract)
    log(f"modeling contract verified (transformers {contract['transformers']})")

    # ---- streams -----------------------------------------------------------------
    from safetensors.torch import safe_open
    emb_key = f"{PREFIX}embed_tokens.weight"
    with safe_open(hst.under(BF16, index["weight_map"][emb_key]), framework="pt",
                   device="cpu") as f:
        emb_cpu = f.get_tensor(emb_key)
        emb = emb_cpu.to(dev)
        torch.xpu.synchronize()
        del emb_cpu
    arms = {}
    for n in ARM_ORDER:
        if n in EXL3_ARMS and not CALIB_ARMS:
            e_cpu = artifact_tensor(EXL3_ROOTS[n], emb_key)
            e = e_cpu.to(dev)
            torch.xpu.synchronize()
            del e_cpu
        else:
            e = emb
        torch.xpu.synchronize()
        arms[n] = Arm(n, calib_ids if n in CALIB_ARMS else None, eval_ids, canaries,
                      n in ROT_ARMS, e, dev, rot_mat=(Q if n == "r4_fp32" else Arm.Hb),
                      dtype=(torch.float32 if n == "r4_fp32" else torch.bfloat16))
        torch.xpu.synchronize()
        del e
    del emb


    # ---- layer-streamed loop -------------------------------------------------------
    layer_log = []
    t_start = time.time()

    # The worker only loads packed CPU tensors. All copies and XPU integer
    # reconstruction run on the main thread after future.result().
    from concurrent.futures import ThreadPoolExecutor
    ex_pool = ThreadPoolExecutor(max_workers=1)
    ex_time = {}

    def _exl3_prefetch(Lp, arm):
        names = QUANT_GDN if txt_cfg.layer_types[Lp] == "linear_attention" else QUANT_ATT
        t = time.time()
        w = load_exl3_block_cpu(Lp, names, EXL3_ROOTS[arm])
        ex_time[(Lp, arm)] = round(time.time() - t, 2)
        return w

    ex_fut = {(layers[0], a): ex_pool.submit(_exl3_prefetch, layers[0], a) for a in EXL3_ARMS}
    li = 0  # index of current layer

    for L in layers:
        t0 = time.time()
        is_gdn = txt_cfg.layer_types[L] == "linear_attention"
        assert is_gdn or txt_cfg.layer_types[L] == "full_attention", txt_cfg.layer_types[L]
        quant_names = QUANT_GDN if is_gdn else QUANT_ATT
        igroups = IGROUPS_GDN if is_gdn else IGROUPS_ATT
        sd_cpu = hst.load_layer_tensors(index, L)
        ph = {}  # per-phase wall times (lead ask 2026-10-11: find the 170 s)

        # build once, clone on-GPU (deepcopy keeps device params; H2D once)
        import copy as _copy
        t = time.time()
        built = {}
        built["bf16"] = hst.build_layer(txt_cfg, L, sd_cpu).to(dev)
        for a in ARM_ORDER:
            if a != "bf16" and ARM_SPECS[a].transform != "fold" and a not in EXL3_ARMS:
                built[a] = _copy.deepcopy(built["bf16"])
        ph["build_plain"] = round(time.time() - t, 2)
        t = time.time()
        sd_fold = fold_layer_sd(sd_cpu, Q, dev) if ROT_ARMS else None
        for a in ROT_ARMS:
            if a == "r4_fp32":
                sd_fold32 = fold_layer_sd(sd_cpu, Q, dev, out_dtype=torch.float32)
                built[a] = _build_layer_gpu(txt_cfg, L, sd_fold32, dtype=torch.float32)
                del sd_fold32
            else:
                built[a] = _build_layer_gpu(txt_cfg, L, sd_fold)
        ph["fold_build"] = round(time.time() - t, 2)
        for a in EXL3_ARMS:
            t = time.time()
            packed = ex_fut.pop((L, a)).result()
            ex_w = decode_exl3_block(L, quant_names, ref, dev if XPU_RECON else None,
                                    EXL3_ROOTS[a], packed=packed)
            del packed
            ph[f"{a}_decode_wait"] = round(time.time() - t, 2)
            t = time.time()
            sd_ex = dict(sd_cpu)
            if not CALIB_ARMS:
                sd_ex.update(artifact_layer_unquantized(EXL3_ROOTS[a], L))
            for k, v in ex_w.items():
                sd_ex[k] = v.to(torch.bfloat16)
            built[a] = hst.build_layer(txt_cfg, L, sd_ex).to(dev)
            torch.xpu.synchronize()
            ph[f"{a}_build"] = round(time.time() - t, 2)
            del sd_ex, ex_w
            if li + 1 < len(layers):
                ex_fut[(layers[li + 1], a)] = ex_pool.submit(_exl3_prefetch, layers[li + 1], a)
        del sd_fold

        # Calibration Hessians for the GPTQ arms: ONE Hessian per (arm, INPUT GROUP),
        # shared by every member of the group (defect fixed 2026-10-11: the half-written
        # edit registered the probe factory where the accumulating hook belonged, so the
        # code would have raised TypeError on the first layer; and it only probed the
        # first layer). Members of a group read the same tensor (input_layernorm output
        # for qkv+z and q+k+v, post_attention_layernorm output for gate+up), so sharing
        # is exact rather than an approximation; it is verified live on EVERY layer with
        # a (sum, sumsq) fingerprint of the first 64 rows of the first calibration chunk.
        store = {}
        probes = {}
        handles = []
        chunk = {"i": 0}
        xtx_t = {"t": 0.0}

        def _fp(x):
            s = x[:64].float()
            return [round(s.sum().item(), 3), round(s.pow(2).sum().item(), 3)]

        def mk_hook(a, rep):
            def hook(mod, inp, out):
                t_x = time.time()
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                XtX = x.T @ x
                key = (a, rep)
                if key in store:
                    store[key].add_(XtX)
                    del XtX
                else:
                    store[key] = XtX
                if chunk["i"] == 0 and key not in probes:
                    probes[key] = _fp(x)
                torch.xpu.synchronize()
                xtx_t["t"] += time.time() - t_x
            return hook

        def mk_probe(a, member, rep):
            def hook(mod, inp, out):
                if chunk["i"] == 0 and (a, member) not in probes:
                    probes[(a, member)] = _fp(
                        inp[0].detach().reshape(-1, inp[0].shape[-1]))
            return hook

        quant_arms = QUANT_ARMS
        for a in CALIB_ARMS:
            mods_ = dict(built[a].named_modules())
            for grp in igroups:
                handles.append(mods_[grp[0]].register_forward_hook(mk_hook(a, grp[0])))
                for member in grp[1:]:
                    handles.append(mods_[member].register_forward_hook(
                        mk_probe(a, member, grp[0])))
            del mods_
        with torch.no_grad():
            for a in CALIB_ARMS:
                t = time.time()
                cal = arms[a].calib
                assert cal is not None, f"{a}: no calibration stream"
                for i0 in range(0, cal.shape[0], 16):
                    chunk["i"] = i0 // 16
                    run_layer(built[a], cal[i0:i0 + 16], txt_cfg, L)
                torch.xpu.synchronize()
                ph[f"hook_calib_{a}"] = round(time.time() - t, 2)
        for h in handles:
            h.remove()
        ph["xtx"] = round(xtx_t["t"], 2)
        sharing = {"groups": [list(g) for g in igroups], "members_identical": True,
                   "fingerprints_L0": None, "checked_layers": [L]}
        for grp in igroups:
            for a in CALIB_ARMS:
                ref_fp = probes.get((a, grp[0]))
                assert ref_fp is not None, f"hook did not fire on {a}/{grp[0]}"
                for member in grp[1:]:
                    fp = probes.get((a, member))
                    assert fp is not None, f"probe did not fire on {a}/{member}"
                    if fp != ref_fp:
                        sharing["members_identical"] = False
                    assert fp == ref_fp, (
                        f"Hessian sharing broken: {a}/{member} input fingerprint {fp} "
                        f"!= group rep {grp[0]} {ref_fp}")
        if L == layers[0]:
            sharing["fingerprints_L0"] = {f"{a}|{m}": v for (a, m), v in
                                          sorted(probes.items())}
        Hacc = {}
        for a in CALIB_ARMS:
            Hacc[a] = {}
            for grp in igroups:
                key = (a, grp[0])
                assert key in store, f"missing Hessian {key}"
                Hacc[a][grp[0]] = store.pop(key)
        del store, probes



        # GPTQ per INPUT GROUP: one hinv per (arm, group) — members share the input
        # tensor so they share the hinv and are quantized in ONE arm-batched column
        # loop (their out-dims differ: in_proj_qkv 10240 vs in_proj_z 6144, q/k/v
        # 6144/1024/1024, gate/up 17408/17408). Hinv = blocked Cholesky + blocked
        # triangular inverse on the device, all in fp32 (the previous torch
        # cholesky/cholesky_inverse chain D2H-fell back: 22.5 s per 17408 Hessian).
        rel_fros = {}
        weighted_errors = {}
        weighted_seconds = 0.0
        for a in [n for n in ("bf16", "exl3") if n in CALIB_ARMS]:
            reference_mods = dict(built[a].named_modules())
            for grp in igroups:
                H_reference = Hacc[a].pop(grp[0])
                capture = os.environ.get("FMT_CAPTURE_ORIGINAL_H")
                if a == "bf16" and L == 2 and grp[0] in ("mlp.down_proj", "linear_attn.in_proj_qkv") and capture:
                    from safetensors.torch import save_file
                    os.makedirs(capture, exist_ok=True)
                    save_file({"H": H_reference.cpu().contiguous()}, f"{capture}/L2_{grp[0]}.safetensors")
                for n in grp:
                    weighted_errors[f"{a}/{n}"] = weighted_tensor_error(n, a, reference_mods[n].weight.data, H_reference, sd_cpu, Q, dev)
                del H_reference
            del reference_mods
        gptq_t = {}
        mods = {a: dict(built[a].named_modules()) for a in quant_arms}
        for grp in igroups:
            t_group = time.time()
            arm_times = {}
            for a in quant_arms:
                t = time.time()
                spec = ARM_SPECS[a]
                H = Hacc[a].pop(grp[0])
                H_calib = H.clone()
                ti = rht128_matrix(H.shape[0], dev) if spec.transform == "rht128" else None
                if ti is not None:
                    H = rht_hessian(H, ti)
                Us, hi = batched_hinv([H], labels=[(L, grp[0], a)])
                U, info = Us[0], hi[0]
                del H, Us, hi
                torch.xpu.synchronize()
                th = round(time.time() - t, 2)
                tl = time.time()
                for n in grp:
                    w = mods[a][n].weight.data
                    W = w.float().clone()
                    to = rht128_matrix(W.shape[0], dev, role=1) if ti is not None else None
                    if ti is not None:
                        W = rht_weight(W, ti, to)
                    dm = info["dead_mask"]
                    if bool(dm.any()):
                        W[:, dm.to(dev)] = 0.0
                    if HINV_ONLY:
                        del W
                        continue
                    if spec.quantizer == "e8rvq4":
                        Wq, codes, scales = e8_quantize(W, U)
                    else:
                        q, c, ss = gptq_from_hinv_batched(W[None], U[None], [GROUPS[a]])
                        Wq, codes, scales = q[0].float(), c[0], ss[0]
                        del q, c, ss
                    if os.environ.get("FMT_FP64_WEIGHTED_COMPARE", "0") == "1" and L == 2 and a == "int4_g128_rht" and n == "mlp.down_proj":
                        Href = rht_hessian(H_calib, ti)
                        diag_ref = Href.diagonal().clone()
                        Href[diag_ref == 0, diag_ref == 0] = 1.0
                        Href.diagonal().add_(PERCDAMP * diag_ref.mean())
                        Uref_cpu = hinv_classic_fp64(Href.cpu())
                        del Href, diag_ref
                        Uref = Uref_cpu.float().to(dev)
                        del Uref_cpu
                        qr, cr, _ = gptq_from_hinv_batched(W[None], Uref[None], [GROUPS[a]])
                        fp32_weight = rht_weight(Wq, ti, to, inverse=True)
                        fp64_weight = rht_weight(qr[0], ti, to, inverse=True)
                        comparison = {"layer": L, "arm": a, "linear": n,
                                      "fp32": weighted_tensor_error(n, a, fp32_weight, H_calib, sd_cpu, Q, dev),
                                      "classic_fp64": weighted_tensor_error(n, a, fp64_weight, H_calib, sd_cpu, Q, dev),
                                      "code_difference_fraction": float((codes != cr[0]).float().mean())}
                        wjson(f"{RUN_DIR}/raw/fp64_weighted_comparison.json", comparison)
                        del Uref, qr, cr, fp32_weight, fp64_weight
                    if HINV_VERIFY and L in (2, 11) and spec.quantizer.startswith("int4"):
                        from safetensors.torch import load_file
                        tag = f"L{L}_{a}_{grp[0]}"
                        uf = load_file(f"{os.environ['R4_DUMP_H']}/U_{tag}.safetensors")
                        uc = uf["U_classic_fp64"].to(dev)
                        _, cc, _ = gptq_from_hinv_batched(W[None], uc[None], [GROUPS[a]])
                        proof = {"layer": L, "arm": a, "linear": n,
                                 "code_difference_fraction": float((codes != cc[0]).float().mean())}
                        with open(f"{RUN_DIR}/raw/code_proofs.jsonl", "a") as f:
                            f.write(json.dumps(proof) + "\n")
                        del uf, uc, cc
                    if ti is not None:
                        Wq = rht_weight(Wq.float(), ti, to, inverse=True)
                    rel_fros[f"{a}/{n}"] = float((Wq.float() - w.float()).norm() / w.float().norm().clamp_min(1e-30))
                    tw = time.time()
                    weighted_errors[f"{a}/{n}"] = weighted_tensor_error(n, a, Wq, H_calib, sd_cpu, Q, dev)
                    weighted_seconds += time.time() - tw
                    mods[a][n].weight.data = Wq.to(w.dtype).contiguous()
                    if not args.smoke:
                        save_codes(L, n, a, codes, scales)
                    del W, Wq, codes, scales
                del U, H_calib
                torch.xpu.synchronize()
                arm_times[a] = {"hinv": th, "loop": round(time.time() - tl, 2), "total": round(time.time() - t, 2)}
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
            gptq_t["+".join(grp)] = {"total": round(time.time() - t_group, 2), "arms": arm_times,
                                   "hinv": sum(v["hinv"] for v in arm_times.values()),
                                   "loop": sum(v["loop"] for v in arm_times.values())}
        del mods, Hacc
        ph["weighted_error"] = round(weighted_seconds, 2)
        ph["gptq"] = gptq_t

        # s1 (amended 2026-10-11 by lead): the fp32 fold-exactness probe runs on
        # EVERY layer in the normal path (64 tokens, fp32 fold of this layer's
        # real bf16 input distribution); max rel L2 over layers is the gated s1.
        fold_probe = {"fold_fp32_rel_l2": None, "not_applicable": not bool(ROT_ARMS)}
        t = time.time()
        if ROT_ARMS:
            fold_probe = fold_exactness_probe_fp32(
                txt_cfg, L, sd_cpu, Q, dev, arms["bf16"].eval[:1, :64])
            fold_probe["layer"] = L
        ph["fold_probe_fp32"] = round(time.time() - t, 2)
        if FP32_DIAG and L == layers[0]:
            # diagnostic extra (bf16 storage-noise columns) on the first layer
            fp = fold_exactness_probe(
                txt_cfg, L, sd_cpu, Q, Arm.Hb, dev, arms["bf16"].eval[:1, :64],
                {"fp32_folded": built["r4_fp32"], "bf16_plain": built["bf16"],
                 "bf16_folded": built["r4_fp"]})
            fold_probe.update({k: v for k, v in fp.items()
                               if k not in ("fold_fp32_rel_l2",)})
            wjson(f"{RUN_DIR}/raw/fold_exactness.json", fp)
        if ROT_ARMS:
            log(f"L{L:02d} fold probe: rel_l2={fold_probe['fold_fp32_rel_l2']:.2e} "
                f"({ph['fold_probe_fp32']}s)")

        t = time.time()
        for a in ARM_ORDER:
            arms[a].propagate(built[a], txt_cfg, L)
        ph["propagate"] = round(time.time() - t, 2)
        reference_state = arms["bf16"].eval.float()
        state_errors = {a: float((arms[a].eval.float() - reference_state).pow(2).sum() /
                                 reference_state.pow(2).sum().clamp_min(1e-30))
                        for a in EXL3_ARMS}
        del reference_state

        del built
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        dt = time.time() - t0
        layer_log.append({"layer": L, "type": txt_cfg.layer_types[L], "seconds": round(dt, 1),
                          "phases": ph, "rel_fro": rel_fros, "hessian_sharing": sharing,
                          "weighted_tensor_errors": weighted_errors,
                          "eval_hidden_state_relative_mse_vs_bf16": state_errors,
                          "fold_rel_l2": fold_probe["fold_fp32_rel_l2"],
                          "fold_probe": fold_probe, "exl3_decode_s": {a: ex_time.get((L,a)) for a in EXL3_ARMS},
                          "peak_xpu_gib": torch.xpu.max_memory_allocated() / (1 << 30),
                          "peak_rss_gib": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / (1 << 20),
                          "hinv_fallbacks": [e for e in HINV_FALLBACKS if e["layer"] == L]})
        with open(f"{RUN_DIR}/raw/layers.jsonl", "a") as f:
            f.write(json.dumps(layer_log[-1], default=float) + "\n")
        gptq_sum = sum(v["total"] for v in gptq_t.values())
        slowest = max(gptq_t.items(), key=lambda kv: kv[1]["total"], default=("none", {"total": 0}))
        log(f"L{L:02d} {txt_cfg.layer_types[L][:3]} done in {dt:.1f}s "
            f"phases={json.dumps({k: v for k, v in ph.items() if k != 'gptq'})} "
            f"gptq_sum={gptq_sum:.1f}s slowest={slowest[0]}:{slowest[1]['total']}s "
            f"(elapsed {time.time() - t_start:.0f}s)")
        li += 1

    # ---- exit rotation + final norm + lm_head ------------------------------------
    norm_key, head_key = f"{PREFIX}norm.weight", "lm_head.weight"
    with safe_open(hst.under(BF16, index["weight_map"][norm_key]), framework="pt",
                   device="cpu") as f:
        w_norm = f.get_tensor(norm_key).to(dev)
    with safe_open(hst.under(BF16, index["weight_map"][head_key]), framework="pt",
                   device="cpu") as f:
        lm_head = f.get_tensor(head_key).to(dev)          # [vocab, 5120] bf16
    log(f"final norm + lm_head loaded {tuple(lm_head.shape)}")

    def final_logits(arm, native_head=False):
        with torch.no_grad():
            h = arm.eval
            if arm.rot:
                h = h @ arm.Hb.T
            norm = artifact_tensor(EXL3_ROOTS[arm.name], norm_key).to(dev) if arm.name in EXL3_ARMS and not CALIB_ARMS else w_norm
            if native_head:
                hc = torch.stack([t[0, -1, :] for t in arm.canary])
                rows = torch.cat((h.reshape(-1, D_MODEL), hc))
                normalized = rmsnorm(rows, norm, txt_cfg.rms_norm_eps).float()
                all_logits = artifact_head_logits(EXL3_ROOTS[arm.name], normalized, ref, dev)
                return all_logits[:-len(canaries)], all_logits[-len(canaries):]
            outs = []
            for i0 in range(0, h.shape[0] * h.shape[1], CHUNK_LOGITS):
                rows = h.reshape(-1, D_MODEL)[i0:i0 + CHUNK_LOGITS]
                hn = rmsnorm(rows, norm, txt_cfg.rms_norm_eps)
                outs.append(hn.float() @ lm_head.float().T)
            logits = torch.cat(outs)
            # step-0 rows: last prompt position of each canary stream
            hc = torch.stack([t[0, -1, :] for t in arm.canary])
            if arm.rot:
                hc = hc @ arm.Hb.T
            can_rows = rmsnorm(hc, norm, txt_cfg.rms_norm_eps).float() @ \
                lm_head.float().T
        return logits.reshape(-1, logits.shape[-1]), can_rows

    ev_ids = eval_ids.to(dev)
    logits = {}
    can_logits = {}
    metric_order = list(ARM_ORDER)
    for a in EXL3_ARMS:
        if os.environ.get("FMT_NATIVE_HEAD") == "1":
            metric_order.append(a + "_bf16head")
    logits["bf16"], can_logits["bf16"] = final_logits(arms["bf16"])

    # ---- metrics ------------------------------------------------------------------
    import torch.nn.functional as F
    ref_l = logits["bf16"]
    lp_ref = F.log_softmax(ref_l, dim=-1)
    p_ref = lp_ref.exp()
    top_ref = ref_l.argmax(-1)
    nxt = ev_ids[:, 1:].reshape(-1)
    prev_ok = torch.arange(ev_ids.shape[1] - 1, device=dev)
    score_mask = build_inputs.response_mask.to(dev).reshape(-1)
    trace_mask = build_inputs.trace_mask.to(dev).reshape(-1) if os.environ.get("FMT_HELDOUT_TRACE") else None

    arm_json = {}
    for a in metric_order:
        if a != "bf16":
            base = a.removesuffix("_bf16head")
            native = a in EXL3_ARMS and os.environ.get("FMT_NATIVE_HEAD") == "1"
            logits[a], can_logits[a] = final_logits(arms[base], native_head=native)
        with torch.no_grad():
            lp = F.log_softmax(logits[a], dim=-1)
            kl_all = (p_ref * (lp_ref - lp)).sum(-1)
            kl = kl_all[score_mask]
            agreement = logits[a].argmax(-1) == top_ref
            top1 = agreement[score_mask].float().mean()
            trace_metrics = ({"kl_mean": float(kl_all[trace_mask].mean()),
                              "top1_vs_bf16": float(agreement[trace_mask].float().mean()),
                              "positions": int(trace_mask.sum())} if trace_mask is not None else None)
            # NLL gather needs THREE indices (batch, position, vocab id); indexing
            # with (batch, vocab_id) as earlier crashed the device (OOB -> engine
            # reset); order matches nxt = ev_ids[:, 1:].reshape(-1): b0 t1..T-1, b1 ...
            T_ev = ev_ids.shape[1]
            # CPU-side bounds proof BEFORE every device gather (lead ask: no more OOB)
            nxt_max = int(nxt.max())
            assert 0 <= nxt_max < logits[a].shape[-1], f"nxt OOB: {nxt_max}"
            batch_eval = ev_ids.shape[0]
            assert logits[a].shape[0] == batch_eval * T_ev, "logit row count"
            cols = torch.arange(T_ev - 1, device=dev).repeat(batch_eval)
            rowsN = torch.arange(batch_eval, device=dev).repeat_interleave(T_ev - 1)
            assert (rowsN.numel() == cols.numel() == nxt.numel()
                    and int(cols.max()) < T_ev and int(rowsN.max()) < batch_eval), \
                "nll gather shape/bounds"
            nll_values = -lp.reshape(batch_eval, T_ev, -1)[rowsN, cols, nxt]
            nll_mask = build_inputs.response_mask[:, :-1].to(dev).reshape(-1)
            nll = nll_values[nll_mask].mean()
            del logits[a]
            torch.xpu.synchronize()
            torch.xpu.empty_cache()

        V_can = can_logits[a].shape[-1]
        for c in canaries:
            assert 0 <= c["exp_first"] < V_can and 0 <= c["foil_first"] < V_can, \
                f"{c['id']} token id OOB"
            assert c["exp_first"] != c["foil_first"]
        margins = {}
        greedy = {}
        for i, c in enumerate(canaries):
            margins[c["id"]] = float(can_logits[a][i, c["exp_first"]] -
                                     can_logits[a][i, c["foil_first"]])
            greedy[c["id"]] = int(can_logits[a][i].argmax())
        arm_json[a] = {"kl_mean": float(kl.mean()), "kl_p99": float(
            torch.quantile(kl.float(), 0.99)), "top1_vs_bf16": float(top1),
            "eval_next_token_nll": float(nll), "canary_margins_step0": margins,
            "canary_greedy_first_token": greedy,
            "mean_rel_fro": _mean_rel_fro(layer_log, a)}
        arm_json[a]["heldout_response_slice"] = trace_metrics
        arm_json[a]["head_scope"] = ("artifact quantized head, decoded to BF16 weights then FP32 matmul"
            if a in EXL3_ARMS and os.environ.get("FMT_NATIVE_HEAD") == "1" else "common BF16 reference head")

        wjson(f"{RUN_DIR}/raw/arm_{a}.json", arm_json[a])
        log(f"arm {a}: kl_mean={arm_json[a]['kl_mean']:.5f} "
            f"top1={arm_json[a]['top1_vs_bf16']:.4f} r01_margin={margins['r01']:.3f}")
    wjson(f"{RUN_DIR}/raw/arms_all.json", arm_json)

    # ---- sanity BEFORE gate ---------------------------------------------------------
    # s1 amended 2026-10-11 (lead decision; amendment disclosed in plan+ledger):
    # "s1: the fp32 fold-exactness probe on every quantized layer has rel L2
    #  <= 1e-4 (fold exact); r4_fp (bf16 storage, quantization off) KL and
    #  top-1 vs bf16 are reported, not gated, because the same bf16 storage
    #  noise is inside every R4 arm's gated metric."
    r01_bf16 = arm_json["bf16"]["canary_margins_step0"]["r01"]
    fold_l2 = [e["fold_rel_l2"] for e in layer_log
               if e.get("fold_rel_l2") is not None]
    assert fold_l2 or not ROT_ARMS, "fold probe never ran"
    s1_pass = not ROT_ARMS or max(fold_l2) <= 1e-4
    sanity = {
        "s1_fold_exact": {
            "definition": ("s1: the fp32 fold-exactness probe on every quantized "
                           "layer has rel L2 <= 1e-4 (fold exact); r4_fp (bf16 "
                           "storage, quantization off) KL and top-1 vs bf16 are "
                           "reported, not gated, because the same bf16 storage "
                           "noise is inside every R4 arm's gated metric."),
            "pass": bool(s1_pass),
            "max_fold_rel_l2": max(fold_l2) if fold_l2 else None,
            "mean_fold_rel_l2": sum(fold_l2) / len(fold_l2) if fold_l2 else None,
            "n_layers_probed": len(fold_l2)},
        "s1_r4fp_reported_not_gated": ({
            "top1_vs_bf16": arm_json["r4_fp"]["top1_vs_bf16"],
            "kl_mean": arm_json["r4_fp"]["kl_mean"],
            "note": "reported, not gated (bf16 storage noise of the fold)"}
            if "r4_fp" in arm_json else {"not_run": True}),
        "s2_gptq_plain_defect": ({"r01_margin": arm_json["gptq_plain"][
            "canary_margins_step0"]["r01"], "pass": arm_json["gptq_plain"][
            "canary_margins_step0"]["r01"] < 0} if "gptq_plain" in arm_json else
            {"not_run": True, "note": "defect control omitted because gptq_plain is not an active arm"}),
        "s3_exl3_r01_correct": {"r01_margin": arm_json["exl3"][
            "canary_margins_step0"]["r01"], "pass": arm_json["exl3"][
            "canary_margins_step0"]["r01"] > 0},
        "bf16_r01_margin_reference": r01_bf16,
    }
    sanity["all_passed"] = bool(
        sanity["s1_fold_exact"]["pass"] and
        sanity["s2_gptq_plain_defect"].get("pass", True) and
        sanity["s3_exl3_r01_correct"]["pass"])
    wjson(f"{RUN_DIR}/raw/sanity.json", sanity)
    log(f"sanity: {json.dumps(sanity)[:400]}")

    # ---- gate (not evaluated if any sanity fails) ------------------------------------
    gate = {"evaluated": sanity["all_passed"], "passed": False}
    gate["format_quality"] = {
        a: {"kl_ratio_vs_exl3": arm_json[a]["kl_mean"] / max(arm_json["exl3"]["kl_mean"], 1e-30),
            "prefix_kl_le_1.10x_exl3": arm_json[a]["kl_mean"] <= 1.10 * arm_json["exl3"]["kl_mean"]}
        for a in QUANT_ARMS}
    if sanity["all_passed"]:
        ex3 = arm_json["exl3"]
        best = None
        for a in QUANT_ARMS:
            aj = arm_json[a]
            margins_ok = all(v > 0 for v in aj["canary_margins_step0"].values())
            if margins_ok and (best is None or aj["kl_mean"] < arm_json[best]["kl_mean"]):
                best = a
        gate.update({"best_r4_arm": best, "exl3_kl": ex3["kl_mean"],
                     "exl3_top1": ex3["top1_vs_bf16"]})
        if best is not None:
            b = arm_json[best]
            gate.update({
                "r4_kl": b["kl_mean"], "r4_top1": b["top1_vs_bf16"],
                "canary_margins": b["canary_margins_step0"],
                "kl_le_exl3": b["kl_mean"] <= ex3["kl_mean"],
                "top1_ge_exl3_minus_0.1pt": b["top1_vs_bf16"] >= ex3["top1_vs_bf16"] - 0.001,
                "all_canaries_correct": True})
            gate["passed"] = bool(gate["kl_le_exl3"] and gate["top1_ge_exl3_minus_0.1pt"])
        else:
            gate["reason"] = "no R4 arm with all canary margins correct"
    if not QUANT_ARMS:
        gate = {"evaluated": False, "passed": False,
                "reason": "reference artifact evaluation only; no format adoption gate",
                "sc_kl_ratios": {a: arm_json[a]["kl_mean"] / max(arm_json["exl3"]["kl_mean"], 1e-30)
                                 for a in EXL3_ARMS if a != "exl3"}}
        if os.environ.get("FMT_NATIVE_HEAD") == "1":
            gate["head_scope"] = "primary native artifact heads; diagnostic common BF16 head"
            gate["sc_bf16head_kl_ratios"] = {
                a: arm_json[a + "_bf16head"]["kl_mean"] / max(arm_json["exl3_bf16head"]["kl_mean"], 1e-30)
                for a in EXL3_ARMS if a != "exl3"}
    wjson(f"{RUN_DIR}/raw/gate.json", gate)
    log(f"gate: {json.dumps(gate)[:400]}")

    # ---- codes saved? finalize /out pointer + sha256 ----------------------------------
    codes_summary = finalize_codes()
    try:
        peak_gib = round(torch.xpu.max_memory_allocated() / (1 << 30), 1)
        log(f"peak XPU memory allocated: {peak_gib} GiB")
    except Exception:  # noqa: BLE001
        peak_gib = None

    # ---- measurements.json -------------------------------------------------------------
    exl3_kl = max(arm_json["exl3"]["kl_mean"], 1e-30)
    cand = []
    for a in QUANT_ARMS:
        cand.append({"pair": f"eval_kl_ratio_{a}", "value": arm_json[a]["kl_mean"] / exl3_kl})
    plan = {"identities": {"harness_sha256": ident["harness_sha256"],
                           "exl3_ref_sha256": REF_SHA256, "exl3_ref_git_head": REF_GIT_HEAD},
            "controlled": {"bits": 4, "percdamp": PERCDAMP, "seed": SEED,
                           "calibration": data_info, "unsealed": True}}
    codec = Path(SCRIPT_DIR, "e8rvq_20261011.py")
    if codec.is_file() and any(ARM_SPECS[a].quantizer == "e8rvq4" for a in ARM_ORDER):
        plan["identities"]["codec_sha256"] = ident.get("codec_sha256", sha256_file(str(codec)))
    by_type = {kind: [e["seconds"] for e in layer_log if e["type"] == kind]
               for kind in set(e["type"] for e in layer_log)}
    projection = sum(txt_cfg.layer_types.count(kind) * sum(values) / len(values)
                     for kind, values in by_type.items())
    meas = {
        "metric": "eval_final_logit_kl_ratio_r4_best_vs_exl3",
        "unit": "kl_ratio", "direction": "lower",
        "identities": plan.get("identities", {}),
        "controlled": plan.get("controlled", {}),
        "baseline": [{"pair": "eval_kl_ratio_exl3", "value": 1.0}],
        "candidate": cand,
        "checks": {
            "correctness": {"passed": bool(sanity["all_passed"]),
                            "evidence": "raw/sanity.json"},
            "quality": {"passed": bool(gate["passed"]), "evidence": "raw/gate.json"},
            "measurement_validity": {"passed": bool(
                data_info["r01_prompt_tokens"] == 36 and
                data_info["eval_shape"][1] ==
                (T_EVAL if (not args.smoke or SMOKE_FULLCALIB) else 64)),
                "evidence": "raw/measurement_validity.json"},
            "isolation": {"passed": Path(f"{RUN_DIR}/raw/isolation.json").is_file(),
                          "evidence": "raw/isolation.json"},
        },
        "identity": ident, "data": data_info, "arms": arm_json, "gate": gate,
        "sanity": sanity, "codes_saved": codes_summary,
        "layer_seconds_total": round(time.time() - t_start, 1),
        "peak_xpu_gib": peak_gib,
        "peak_xpu_reserved_gib": torch.xpu.max_memory_reserved() / (1 << 30),
        "hinv_fallbacks": HINV_FALLBACKS,
        "hinv_proofs": HINV_PROOFS,
        "arm_specs": {n: vars(ARM_SPECS[n]) for n in ARM_ORDER},
        "peak_rss_gib": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / (1 << 20),
        "projected_64_layer_seconds": projection,
        "projected_timeout_1_25x_seconds": projection * 1.25,
        "layer_type_mean_seconds": {k: sum(v) / len(v) for k, v in by_type.items()},
        "exl3_decode_checks": _DECODE_RECORDS,
        "calibration_free": not bool(CALIB_ARMS),
        "metric_arm_order": metric_order,
    }
    wjson(f"{RUN_DIR}/raw/measurement_validity.json", {
        "data": data_info, "identity": ident, "streams": f"{len(ARM_ORDER)} arms layer-streamed",
        "n_layers": len(layers), "chunk_logits": CHUNK_LOGITS,
        "notes": "r01 rendered with the /bf16 chat template enable_thinking=False; "
                 "corpus sha matched corpus_meta.json; farms fail-closed manifest-verified"})
    wjson(f"{RUN_DIR}/measurements.json", meas)
    if not QUANT_ARMS:
        return 0 if sanity["all_passed"] else 3
    return 0 if (sanity["all_passed"] and gate["passed"]) else (1 if sanity["all_passed"] else 3)


# ---- codes persistence (large files -> /out only; pointer + sha256 in run root) ----
_CODE_STATE = {"arms": {}}


def save_codes(layer, name, arm, codes, scales):
    import shutil
    import torch
    from safetensors.torch import save_file
    if os.environ.get("FMT_SAVE_CODES", "0") != "1":
        return
    st = _CODE_STATE["arms"].setdefault(arm, {"enabled": None, "dir": None,
                                              "files": [], "bytes": 0})
    if st["enabled"] is None:
        st["enabled"] = False
        try:
            st["dir"] = f"{OUT}/fmt-{arm}-codes-20261011"
            os.makedirs(st["dir"], exist_ok=True)
            st["enabled"] = True
            log(f"code saving enabled at {st['dir']} "
                f"(free {shutil.disk_usage(OUT).free >> 30} GiB)")
        except Exception as e:  # noqa: BLE001
            log(f"code saving disabled for {arm}: {e}")
            return
    if not st["enabled"]:
        return
    packed = ({"codes": codes, "scales": scales} if ARM_SPECS[arm].quantizer == "e8rvq4"
              else auto_gptq_pack(codes, scales, GROUPS[arm]))
    n_bytes = sum(t.numel() * t.element_size() for t in packed.values())
    if shutil.disk_usage(OUT).free - n_bytes < MIN_OUT_FREE_FOR_CODES:
        log(f"code saving stopped for {arm}: /out budget exhausted")
        st["enabled"] = False
        return
    key = f"{PREFIX}layers.{layer}.{name}"
    fname = f"{st['dir']}/{key.replace('.', '_')}.safetensors"
    save_file({k: v.cpu().contiguous() for k, v in packed.items()}, fname)
    st["files"].append({"file": fname, "tensor": key, "sha256": sha256_file(fname),
                        "bytes": n_bytes, "transform": ARM_SPECS[arm].transform,
                        "quantizer": ARM_SPECS[arm].quantizer})
    st["bytes"] += n_bytes


def finalize_codes():
    files = [f for st in _CODE_STATE["arms"].values() for f in st["files"]]
    if not files:
        return {"saved": False}
    wjson(f"{RUN_DIR}/raw/codes_manifest.json", files)
    return {"saved": True, "bytes": sum(f["bytes"] for f in files),
            "n_tensors": len(files), "manifest": "raw/codes_manifest.json",
            "note": "Scalar arms use auto_gptq packing; E8 arms use uint16 codes [2,out,in/8] "
                    "and fp16 scales [2,out]. Transform identified per file. RHT/fold codes "
                    "require their declared transform; these are not drop-in runtime artifacts. "
                    "mtp/lm_head untouched"}



def _mean_rel_fro(layer_log, arm):
    vals = [v for e in layer_log for k, v in (e.get("rel_fro") or {}).items()
            if k.startswith(arm + "/")]
    return sum(vals) / len(vals) if vals else None


if __name__ == "__main__":
    sys.exit(main())
