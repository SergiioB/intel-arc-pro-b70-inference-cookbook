#!/usr/bin/env python3
"""Host preflight utility for large models, validating RAM, swap, swappiness, and disk storage."""
import os
import sys
import psutil
from pathlib import Path

def main() -> int:
    issues = []
    
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024**3)
    free_ram_gb = vm.available / (1024**3)
    swap = psutil.swap_memory()
    total_swap_gb = swap.total / (1024**3)
    
    print("--- Memory Preflight ---")
    print(f"Total RAM: {total_ram_gb:.1f} GB")
    print(f"Available RAM: {free_ram_gb:.1f} GB")
    print(f"Total Swap: {total_swap_gb:.1f} GB")
    
    # Large model checks for 16GB host (like 30B+ GGUF load freezes)
    if total_ram_gb < 20:
        if total_swap_gb < 30:
            issues.append(f"CRITICAL: Host has {total_ram_gb:.1f} GB RAM and only {total_swap_gb:.1f} GB Swap. Loading large models (>15GB) via CPU mapping or OpenVINO GenAI WILL freeze/kernel-panic this machine. Add a 32GB+ swapfile.")
            
        # Check swappiness values that cause heavy I/O freezes
        try:
            with open("/proc/sys/vm/swappiness", "r") as f:
                swappiness = int(f.read().strip())
                print(f"vm.swappiness = {swappiness}")
                if swappiness < 60:
                    issues.append(f"WARNING: vm.swappiness is {swappiness}. For massive mmap loads on 16GB RAM, raise it (e.g., 80) to allow aggressive eviction instead of freezing.")
        except Exception:
            pass

    # Basic root disk check
    root_usage = psutil.disk_usage("/")
    print(f"Root disk free: {root_usage.free / (1024**3):.1f} GB ({root_usage.percent}%)")
    if root_usage.free < (5 * 1024**3):
        issues.append(f"CRITICAL: Root disk has less than 5GB free. Swap provisioning or model downloads will fail.")
        
    print()
    if issues:
        print("Preflight WARNINGS/ERRORS:")
        for issue in issues:
            print(f" - {issue}")
        return 1
    
    print("Preflight OK. Host looks configured to handle large GGUF/model loads without immediate I/O death.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
