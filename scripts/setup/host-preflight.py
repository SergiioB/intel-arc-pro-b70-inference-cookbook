#!/usr/bin/env python3
"""Host preflight utility for large models, validating RAM, swap, swappiness, and disk storage."""

import shutil
import sys
from pathlib import Path
from typing import NamedTuple


class MemoryInfo(NamedTuple):
    """Memory figures in bytes with psutil.virtual_memory() semantics."""

    total: int
    available: int
    used: int
    percent: float


def parse_meminfo(text: str | None = None) -> dict[str, int]:
    """Parse /proc/meminfo (or injected text) into a field name -> bytes mapping."""
    if text is None:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    fields: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        tokens = rest.split()
        if not tokens:
            continue
        # meminfo reports sizes in kB; store bytes.
        fields[key.strip()] = int(tokens[0]) * 1024
    return fields


def virtual_memory(fields: dict[str, int]) -> MemoryInfo:
    """Derive total/available/used/percent from meminfo fields (psutil semantics)."""
    total = fields.get("MemTotal", 0)
    available = fields.get("MemAvailable", 0)
    used = total - available
    percent = used / total * 100 if total else 0.0
    return MemoryInfo(total=total, available=available, used=used, percent=percent)


def swap_total(fields: dict[str, int]) -> int:
    """Total swap in bytes, matching psutil.swap_memory().total on Linux."""
    return fields.get("SwapTotal", 0)


def main() -> int:
    issues: list[str] = []

    meminfo = parse_meminfo()
    vm = virtual_memory(meminfo)
    total_ram_gb = vm.total / (1024**3)
    free_ram_gb = vm.available / (1024**3)
    total_swap_gb = swap_total(meminfo) / (1024**3)

    print("--- Memory Preflight ---")
    print(f"Total RAM: {total_ram_gb:.1f} GB")
    print(f"Available RAM: {free_ram_gb:.1f} GB")
    print(f"Total Swap: {total_swap_gb:.1f} GB")

    # Large model checks for 16GB host (like 30B+ GGUF load freezes)
    if total_ram_gb < 20:
        if total_swap_gb < 30:
            issues.append(
                f"CRITICAL: Host has {total_ram_gb:.1f} GB RAM and only "
                f"{total_swap_gb:.1f} GB Swap. Loading large models (>15GB) via "
                "CPU mapping or OpenVINO GenAI WILL freeze/kernel-panic this "
                "machine. Add a 32GB+ swapfile."
            )

        # Check swappiness values that cause heavy I/O freezes
        try:
            with open("/proc/sys/vm/swappiness") as f:
                swappiness = int(f.read().strip())
                print(f"vm.swappiness = {swappiness}")
                if swappiness < 60:
                    issues.append(
                        f"WARNING: vm.swappiness is {swappiness}. "
                        "For massive mmap loads on 16GB RAM, raise it "
                        "(e.g., 80) to allow aggressive eviction instead of freezing."
                    )
        except Exception:
            pass

    # Basic root disk check
    root_usage = shutil.disk_usage("/")
    root_percent = root_usage.used / root_usage.total * 100 if root_usage.total else 0.0
    print(f"Root disk free: {root_usage.free / (1024**3):.1f} GB ({root_percent}%)")
    if root_usage.free < (5 * 1024**3):
        issues.append(
            "CRITICAL: Root disk has less than 5GB free. "
            "Swap provisioning or model downloads will fail."
        )

    print()
    if issues:
        print("Preflight WARNINGS/ERRORS:")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print(
        "Preflight OK. Host looks configured to handle large GGUF/model loads "
        "without immediate I/O death."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
