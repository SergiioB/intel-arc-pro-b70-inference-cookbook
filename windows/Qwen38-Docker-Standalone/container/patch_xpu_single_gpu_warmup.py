#!/usr/bin/env python3
"""Skip vLLM's oneCCL warm-up collective for a single-XPU WSLC server."""

from pathlib import Path
import re


TARGET = Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/xpu_worker.py")
MARKER = "# WSLC_SINGLE_GPU_SKIP_CCL_WARMUP"
CALL_PATTERN = re.compile(
    r"^(?P<indent>[ \t]+)torch\.distributed\.all_reduce\(torch\.zeros\(1\)\.xpu\(\)\)[ \t]*$",
    re.MULTILINE,
)


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"single-GPU WSLC warm-up patch already present in {TARGET}")
        return

    matches = list(CALL_PATTERN.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            f"Refusing to patch {TARGET}: expected one exact XPU oneCCL warm-up "
            f"call, found {len(matches)}. The pinned vLLM source may have changed."
        )

    indent = matches[0].group("indent")
    replacement = "\n".join(
        (
            f"{indent}{MARKER}",
            f"{indent}# A one-rank reduction is a no-op. Avoid invoking oneCCL because WSLC",
            f"{indent}# exposes XPU compute but not the Linux DRM directory its IPC path uses.",
            f"{indent}if self.parallel_config.world_size > 1:",
            f"{indent}    torch.distributed.all_reduce(torch.zeros(1).xpu())",
        )
    )
    patched = CALL_PATTERN.sub(replacement, source, count=1)
    TARGET.write_text(patched, encoding="utf-8")
    compile(TARGET.read_text(encoding="utf-8"), str(TARGET), "exec")
    print(f"patched single-GPU oneCCL warm-up in {TARGET}")


if __name__ == "__main__":
    main()
