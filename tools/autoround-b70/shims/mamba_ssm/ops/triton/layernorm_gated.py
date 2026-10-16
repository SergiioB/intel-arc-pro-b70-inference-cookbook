"""Pure-torch stand-in for mamba_ssm.ops.triton.layernorm_gated (CPU/XPU export only).

Same math as mamba_ssm's rms_norm_ref: optional SiLU gate, grouped RMS norm,
fp32 compute, output cast back to the input dtype.
"""
import torch
import torch.nn.functional as F


def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    w = weight.float()
    b = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else None
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        out = x * rstd * w
    else:
        xg = x.unflatten(-1, (-1, group_size))
        rstd = torch.rsqrt(xg.square().mean(dim=-1, keepdim=True) + eps)
        out = (xg * rstd).flatten(-2) * w
    if b is not None:
        out = out + b
    if z is not None and norm_before_gate:
        out = out * F.silu(z)
    return out.to(dtype)
