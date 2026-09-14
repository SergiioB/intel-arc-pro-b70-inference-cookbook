#!/usr/bin/env python3
"""XPU GDN mixed spec + non-spec split (v5) — causal_conv1d root cause.

The fused ``torch.ops._xpu_C.gdn_attention`` host binding refuses a batch
that mixes spec-decode tokens with non-spec (prefill + decode) tokens:

    causal_conv1d does not support spec-decode and non-spec (prefill +
    decode) tokens in the same invocation

That is a host ``TORCH_CHECK``, not a missing SYCL kernel. The two launchers
already exist; they cannot share one invocation because Xe2 non-spec
intermediates are chunk-padded and spec intermediates are not, and the fused
wrapper consumes exactly one 5-tensor group.

CUDA already splits this in Python (``QwenGDNLinearAttention._forward_core``).
This patch does the same at the XPU fused-op boundary:

  * homogeneous batch → one fused call (unchanged C1 / no-spec path)
  * mixed batch → compact each group with ``index_select``, one fused call
    per group with ``num_actual_tokens = group_count`` and ``token_indx =
    arange``, then ``index_copy_`` both ``core_attn_out`` **and** ``z``

v1–v4 failed the C++ size-sum / ``narrow`` contract (global indices into a
compact buffer, or unused-side dummy tensors that still counted as tokens).
v5 uses ``None`` for the idle optional side (binding is ``Tensor?``).

Applies to the import path the f01e24f6 image actually uses
(``/workspace/vllm/vllm/_xpu_ops.py``), then site-packages as fallback.
Idempotent (``B70_GDN_MIXED_SPLIT_V5``).
"""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "B70_GDN_MIXED_SPLIT_V5"

CANDIDATES = (
    Path("/workspace/vllm/vllm/_xpu_ops.py"),
    Path("/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"),
    Path("/opt/vllm/vllm/_xpu_ops.py"),
)

OLD = """    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        self.num_k_heads,
        self.num_v_heads,
        self.head_k_dim,
        self.head_v_dim,
        conv_state=self.kv_cache[0],
        ssm_state=self.kv_cache[1],
        conv_weights=conv_weights,
        conv_bias=self.conv1d.bias,
        activation=self.activation,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        num_prefills=num_prefills,  # type: ignore[attr-defined]
        num_decodes=num_decodes,  # type: ignore[attr-defined]
        num_spec_decodes=num_spec_decodes,  # type: ignore[attr-defined]
        has_initial_state=has_initial_state,  # type: ignore[attr-defined]
        non_spec_query_start_loc=non_spec_query_start_loc,  # type: ignore[attr-defined]
        non_spec_token_indx=non_spec_token_indx,  # type: ignore[attr-defined]
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,  # type: ignore[attr-defined]
        spec_query_start_loc=spec_query_start_loc,  # type: ignore[attr-defined]
        spec_token_indx=spec_token_indx,  # type: ignore[attr-defined]
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,  # type: ignore[attr-defined]
        num_actual_tokens=num_actual_tokens,  # type: ignore[attr-defined]
        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )
"""

NEW = '''    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    # B70_GDN_MIXED_SPLIT_V5: fused XPU causal_conv1d is exclusive
    # (spec XOR non-spec). Compact + two fused calls + scatter z/out.
    # Homogeneous batches keep the single call (C1 MTP / no-spec).
    _mixed = (
        num_spec_decodes > 0
        and (num_prefills + num_decodes) > 0
        and non_spec_token_indx is not None
        and spec_token_indx is not None
    )
    _reorder = not self.gqa_interleaved_layout

    def _invoke(
        _out,
        _z,
        _qkvz,
        _ba,
        *,
        n_pref,
        n_dec,
        n_spec,
        n_tok,
        has_init,
        ns_loc,
        ns_idx,
        ns_state,
        sp_loc,
        sp_idx,
        sp_state,
        n_acc,
    ):
        torch.ops._xpu_C.gdn_attention(
            _out,
            _z,
            _qkvz,
            _ba,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            conv_state=self.kv_cache[0],
            ssm_state=self.kv_cache[1],
            conv_weights=conv_weights,
            conv_bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_prefills=n_pref,
            num_decodes=n_dec,
            num_spec_decodes=n_spec,
            has_initial_state=has_init,
            non_spec_query_start_loc=ns_loc,
            non_spec_token_indx=ns_idx,
            non_spec_state_indices_tensor=ns_state,
            spec_query_start_loc=sp_loc,
            spec_token_indx=sp_idx,
            spec_state_indices_tensor=sp_state,
            num_accepted_tokens=n_acc,
            num_actual_tokens=n_tok,
            tp_size=self.tp_size,
            reorder_input=_reorder,
        )

    if not _mixed:
        _invoke(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            n_pref=num_prefills,
            n_dec=num_decodes,
            n_spec=num_spec_decodes,
            n_tok=num_actual_tokens,
            has_init=has_initial_state,
            ns_loc=non_spec_query_start_loc,
            ns_idx=non_spec_token_indx,
            ns_state=non_spec_state_indices_tensor,
            sp_loc=spec_query_start_loc,
            sp_idx=spec_token_indx,
            sp_state=spec_state_indices_tensor,
            n_acc=num_accepted_tokens,
        )
        return

    def _i32(t):
        if t is None:
            return None
        if t.dtype != torch.int32:
            t = t.to(torch.int32)
        return t.contiguous()

    _nsti = _i32(non_spec_token_indx)
    _sti = _i32(spec_token_indx)
    _n_ns = int(_nsti.numel())
    _n_sp = int(_sti.numel())
    _dev = core_attn_out.device
    if not getattr(self, "_b70_gdn_split_logged", False):
        logger.info(
            "[B70] GDN mixed split prefill=%d decode=%d spec=%d n_ns=%d n_sp=%d",
            num_prefills,
            num_decodes,
            num_spec_decodes,
            _n_ns,
            _n_sp,
        )
        self._b70_gdn_split_logged = True

    def _compact(src, idx):
        return src.index_select(0, idx.to(torch.long)).contiguous()

    if _n_ns > 0:
        _z_ns = _compact(z, _nsti)
        _qkvz_ns = _compact(projected_states_qkvz, _nsti)
        _ba_ns = _compact(projected_states_ba, _nsti)
        _out_ns = core_attn_out.new_empty((_n_ns,) + tuple(core_attn_out.shape[1:]))
        _ar_ns = torch.arange(_n_ns, dtype=torch.int32, device=_dev)
        _invoke(
            _out_ns,
            _z_ns,
            _qkvz_ns,
            _ba_ns,
            n_pref=num_prefills,
            n_dec=num_decodes,
            n_spec=0,
            n_tok=_n_ns,
            has_init=has_initial_state,
            ns_loc=non_spec_query_start_loc,
            ns_idx=_ar_ns,
            ns_state=non_spec_state_indices_tensor,
            sp_loc=None,
            sp_idx=None,
            sp_state=None,
            n_acc=None,
        )
        core_attn_out.index_copy_(0, _nsti.to(torch.long), _out_ns)
        z.index_copy_(0, _nsti.to(torch.long), _z_ns)

    if _n_sp > 0:
        _z_sp = _compact(z, _sti)
        _qkvz_sp = _compact(projected_states_qkvz, _sti)
        _ba_sp = _compact(projected_states_ba, _sti)
        _out_sp = core_attn_out.new_empty((_n_sp,) + tuple(core_attn_out.shape[1:]))
        _ar_sp = torch.arange(_n_sp, dtype=torch.int32, device=_dev)
        _invoke(
            _out_sp,
            _z_sp,
            _qkvz_sp,
            _ba_sp,
            n_pref=0,
            n_dec=0,
            n_spec=num_spec_decodes,
            n_tok=_n_sp,
            has_init=None,
            ns_loc=None,
            ns_idx=None,
            ns_state=None,
            sp_loc=spec_query_start_loc,
            sp_idx=_ar_sp,
            sp_state=spec_state_indices_tensor,
            n_acc=num_accepted_tokens,
        )
        core_attn_out.index_copy_(0, _sti.to(torch.long), _out_sp)
        z.index_copy_(0, _sti.to(torch.long), _z_sp)
'''


def resolve_paths() -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for p in CANDIDATES:
        if p.is_file():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(rp)
    try:
        import vllm

        p = (Path(vllm.__file__).resolve().parent / "_xpu_ops.py")
        if p.is_file() and p not in seen:
            found.append(p)
    except Exception:
        pass
    if not found:
        sys.exit("vllm/_xpu_ops.py not found")
    return found


def patch(path: Path | None = None) -> Path:
    targets = [path] if path is not None else resolve_paths()
    last = targets[-1]
    patched = 0
    for p in targets:
        text = p.read_text()
        if MARKER in text:
            print(f"[gdn-split-v5] already patched {p}")
            last = p
            continue
        if OLD not in text:
            sys.exit(f"[gdn-split-v5] anchor not found in {p}")
        p.write_text(text.replace(OLD, NEW, 1))
        print(f"[gdn-split-v5] patched {p}")
        last = p
        patched += 1
    if path is None:
        print(f"[gdn-split-v5] applied to {len(targets)} file(s), new={patched}")
    return last


if __name__ == "__main__":
    patch()
