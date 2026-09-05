#!/usr/bin/env python3
"""Apply Muse local paged-decode tuple (historical).

Already in vllm-xpu-kernels main via PR #526 (merged 2026-08-14):
`16,128,64,false,true,false`. Do not run this against current main.

Kept for old checkouts. No-ops if the tuple is present. Not required for
Nemotron DFlash. Source-only; rebuild after applying on a pre-#526 tree.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REL = Path("csrc/xpu/attn/kernel_configs/paged_decode_default.conf")
OLD = """# --- GLM / Seed qgroup=16 decode heads --------------------------------------
# zai-org/GLM-4-9B-chat, zai-org/glm-4-9b-hf
# ByteDance-Seed/Seed-OSS-36B-Instruct
16,128,64,false,false,false
"""
NEW = """# --- GLM / Seed qgroup=16 decode heads --------------------------------------
# zai-org/GLM-4-9B-chat, zai-org/glm-4-9b-hf
# ByteDance-Seed/Seed-OSS-36B-Instruct
# Muse-Glimmer-30B (local attn, qgroup=16, head=128, page=64)
16,128,64,false,false,false
16,128,64,false,true,false
"""
TUPLE = "16,128,64,false,true,false"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="vllm-xpu-kernels checkout")
    args = parser.parse_args()
    path = args.root / REL
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    text = path.read_text()
    if TUPLE in text:
        print(f"[ok] already present: {path}")
        return
    if text.count(OLD) != 1:
        raise SystemExit(
            f"anchor mismatch in {path} (count={text.count(OLD)}). "
            "Public HEAD changed or PR #524 already landed — do not force."
        )
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"[ok] added {TUPLE}: {path}")


if __name__ == "__main__":
    main()
