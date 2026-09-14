#!/usr/bin/env python3
"""Wrap int64 XPU data_ptr stores that overflow signed int64.

On some B70 + older-image paths, ``torch.xpu.data_ptr()`` returns an address
>= 2^63. Assigning that Python int into a **signed int64** tensor raises
``ValueError: Overflow when unpacking long long``. The GDN/mamba kernels
reinterpret the same 64 bits as a pointer, so a two's-complement wrap is
bitwise-correct **only** for int64 slots.

Do **not** wrap the uint64 ``dst_ptrs_np`` / ``src_ptrs_np`` staging arrays.
Those already accept the unsigned ``data_ptr()``; a signed wrap can
OverflowError or store the wrong type.

Fail **closed** if the int64 anchors are missing (do not write a silent
partial patch). Optional on the Qwen3.8 champion image
``f01e24f6`` / kernels 0.1.12.3 — overflow was observed on legacy
``2c427ef`` GGUF drafter logs, not on champion Qwen3.8 campaigns.

Applies every ``vllm/v1/worker/mamba_utils.py`` copy the process can see.
"""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "B70_PTR_WRAP"

CANDIDATES = (
    Path("/workspace/vllm/vllm/v1/worker/mamba_utils.py"),
    Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py"),
    Path("/opt/vllm/vllm/v1/worker/mamba_utils.py"),
)

INT64_SITES = (
    "self.state_base_addrs[idx] = state.data_ptr()",
    "self.block_table_ptrs[i] = bt.data_ptr()",
)

HELPER = '''

def B70_PTR_WRAP_wrap_ptr(value: int) -> int:
    """Wrap unsigned 64-bit pointer into signed int64 (same bits)."""
    value &= 0xFFFFFFFFFFFFFFFF
    if value >= 0x8000000000000000:
        value -= 0x10000000000000000
    return value
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

        p = Path(vllm.__file__).resolve().parent / "v1" / "worker" / "mamba_utils.py"
        if p.is_file() and p not in seen:
            found.append(p)
    except Exception:
        pass
    if not found:
        sys.exit("vllm/v1/worker/mamba_utils.py not found")
    return found


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    missing = [s for s in INT64_SITES if s not in text]
    if missing:
        raise RuntimeError("int64 data_ptr anchors missing: " + "; ".join(missing))
    if "\ndef B70_PTR_WRAP_wrap_ptr" not in text:
        lines = text.splitlines()
        last_import = 0
        for i, ln in enumerate(lines):
            if ln.startswith(("import ", "from ")):
                last_import = i
        lines.insert(last_import + 1, HELPER)
        text = "\n".join(lines)
    text = text.replace(
        "self.state_base_addrs[idx] = state.data_ptr()",
        "self.state_base_addrs[idx] = B70_PTR_WRAP_wrap_ptr(state.data_ptr())",
        1,
    )
    text = text.replace(
        "self.block_table_ptrs[i] = bt.data_ptr()",
        "self.block_table_ptrs[i] = B70_PTR_WRAP_wrap_ptr(bt.data_ptr())",
        1,
    )
    # uint64 dst_ptrs_np / src_ptrs_np stay raw data_ptr() on purpose.
    return text


def patch(path: Path | None = None) -> Path:
    targets = [path] if path is not None else resolve_paths()
    last = targets[-1]
    patched = 0
    for p in targets:
        original = p.read_text()
        if MARKER in original:
            print(f"[ptr-wrap] already patched {p}")
            last = p
            continue
        try:
            updated = patch_text(original)
        except RuntimeError as exc:
            sys.exit(f"[ptr-wrap] fail-closed {p}: {exc}")
        compile(updated, str(p), "exec")
        p.write_text(updated)
        print(f"[ptr-wrap] patched {p} (int64 sites only; uint64 dst_ptrs untouched)")
        last = p
        patched += 1
    if path is None:
        print(f"[ptr-wrap] applied to {len(targets)} file(s), new={patched}")
    return last


if __name__ == "__main__":
    patch()
