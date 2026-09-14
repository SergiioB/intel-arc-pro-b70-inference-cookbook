from importlib.metadata import version
import os
import sys

import torch


def main() -> None:
    print(f"vLLM: {version('vllm')}")
    print(f"vLLM XPU kernels: {version('vllm-xpu-kernels')}")
    print(f"PyTorch: {torch.__version__}")
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        sys.exit("Intel XPU is unavailable inside WSLC. Check --gpus all and the Windows Intel driver.")
    name = torch.xpu.get_device_name(0)
    print(f"XPU 0: {name}")
    # Some Windows/WSLC Intel drivers expose only the Battlemage PCI device ID
    # instead of the marketing name. 0xe223 is the Arc Pro B70 (BMG G31).
    normalized_name = name.upper()
    is_b70 = "B70" in normalized_name or "0XE223" in normalized_name
    if not is_b70 and os.environ.get("ALLOW_NON_B70") != "1":
        sys.exit(f"Expected an Arc Pro B70, got {name!r}. Set ALLOW_NON_B70=1 only for deliberate testing.")
    print("Arc Pro B70 identity check: OK")
    probe = torch.ones(16, device="xpu")
    if float(probe.sum().cpu()) != 16.0:
        sys.exit("XPU allocation/compute probe returned an invalid result")
    print("XPU allocation and compute probe: OK")


if __name__ == "__main__":
    main()
