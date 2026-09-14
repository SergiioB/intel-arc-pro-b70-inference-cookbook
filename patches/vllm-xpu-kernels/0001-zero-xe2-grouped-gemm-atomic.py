#!/usr/bin/env python3
"""Apply vllm-xpu-kernels#524 part A: zero the Xe2 grouped-GEMM atomic.

Source-only. Does not rebuild the .so. Run against a kernels checkout, then
rebuild per upstream. Idempotent. Fails closed if the public-HEAD anchor is gone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REL = Path("csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp")
OLD = """  at::Tensor atomic_buffer =
      at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));
"""
NEW = """  // Persistent-block scheduler counter. Group 0 also stores 0 in-kernel, but
  // SYCL does not guarantee group 0 runs first; a dirty leftover becomes the
  // starting tile index (especially under XPU graph replay).
  at::Tensor atomic_buffer =
      at::zeros({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));
"""
MARKER = "at::zeros({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt))"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="vllm-xpu-kernels checkout")
    args = parser.parse_args()
    path = args.root / REL
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    text = path.read_text()
    if MARKER in text and "at::empty({static_cast<long>(1)}" not in text:
        print(f"[ok] already applied: {path}")
        return
    if text.count(OLD) != 1:
        raise SystemExit(
            f"anchor mismatch in {path} (count={text.count(OLD)}). "
            "Public HEAD changed or PR #524 already landed — do not force."
        )
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"[ok] at::empty -> at::zeros: {path}")


if __name__ == "__main__":
    main()
