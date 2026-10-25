#!/usr/bin/env python3
"""Keep vLLM on XPU when the Windows host also has an NVIDIA GPU (issue #15).

Docker Desktop shares /usr/lib/wsl/lib with the container, and that directory
holds libnvidia-ml.so.1 on any host with an NVIDIA driver installed. vLLM's
CUDA detection then finds the NVIDIA card through NVML, activates next to XPU,
and refuses to start: "Only one platform plugin can be activated, but got:
['cuda', 'xpu']". The mount cannot be dropped because libdxcore lives there too.

With VLLM_TARGET_DEVICE=xpu (set by Start-Qwen38-Docker.ps1), the CUDA check
returns None. Must run before anything imports vllm.
"""

from pathlib import Path


TARGET = Path("/opt/venv/lib/python3.12/site-packages/vllm/platforms/__init__.py")
MARKER = "# B70_XPU_ONLY_PLATFORM"
ANCHOR = "def cuda_platform_plugin() -> str | None:\n    is_cuda = False\n"
REPLACEMENT = (
    "def cuda_platform_plugin() -> str | None:\n"
    f"    {MARKER}: NVML on a WSL host with an NVIDIA card must not\n"
    "    # activate CUDA next to the B70 (cookbook issue #15).\n"
    '    if envs.VLLM_TARGET_DEVICE == "xpu":\n'
    "        return None\n"
    "    is_cuda = False\n"
)


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"XPU-only platform patch already present in {TARGET}")
        return
    count = source.count(ANCHOR)
    if count != 1:
        raise RuntimeError(
            f"Refusing to patch {TARGET}: expected one cuda_platform_plugin anchor, "
            f"found {count}. The pinned vLLM source may have changed."
        )
    if "\nfrom vllm import envs\n" not in source:
        raise RuntimeError(f"Refusing to patch {TARGET}: module does not import vllm.envs.")
    patched = source.replace(ANCHOR, REPLACEMENT, 1)
    compile(patched, str(TARGET), "exec")
    TARGET.write_text(patched, encoding="utf-8")
    print(f"patched XPU-only platform detection in {TARGET}")


if __name__ == "__main__":
    main()
