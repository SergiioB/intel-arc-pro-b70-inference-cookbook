#!/usr/bin/env python3
"""Hardware, driver, multi-GPU topology, and environment doctor for Intel Arc Pro B60/B70."""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys
from typing import Any


def check_os() -> dict[str, Any]:
    system = platform.system()
    release = platform.release()
    return {
        "system": system,
        "release": release,
        "is_linux": system == "Linux",
        "arch": platform.machine(),
    }


def check_render_nodes() -> dict[str, Any]:
    render_nodes = sorted(glob.glob("/dev/dri/renderD*"))
    card_nodes = sorted(glob.glob("/dev/dri/card*"))
    accessible_render: list[str] = []
    for node in render_nodes:
        if os.access(node, os.R_OK | os.W_OK):
            accessible_render.append(node)

    user_groups: list[str] = []
    try:
        import grp
        user_groups = [g.gr_name for g in grp.getgrall() if os.getlogin() in g.gr_mem]
        current_gid = os.getgid()
        user_groups.append(grp.getgrgid(current_gid).gr_name)
        user_groups = sorted(set(user_groups))
    except Exception:
        pass

    return {
        "render_nodes": render_nodes,
        "card_nodes": card_nodes,
        "accessible_render_nodes": accessible_render,
        "can_access_render": len(accessible_render) > 0,
        "user_groups": user_groups,
    }


def check_intel_gpus() -> dict[str, Any]:
    gpus: list[dict[str, Any]] = []
    pci_devices = glob.glob("/sys/bus/pci/devices/*")
    for dev in sorted(pci_devices):
        vendor_file = os.path.join(dev, "vendor")
        device_file = os.path.join(dev, "device")
        if os.path.isfile(vendor_file) and os.path.isfile(device_file):
            try:
                with open(vendor_file, "r") as f:
                    vendor = f.read().strip().lower()
                with open(device_file, "r") as f:
                    device_id = f.read().strip().lower()
                if vendor == "0x8086":
                    # Intel display/compute controller
                    class_file = os.path.join(dev, "class")
                    dev_class = ""
                    if os.path.isfile(class_file):
                        with open(class_file, "r") as f:
                            dev_class = f.read().strip().lower()
                    if dev_class.startswith("0x03"):  # Display / 3D controller
                        driver_link = os.path.join(dev, "driver")
                        driver = os.path.basename(os.readlink(driver_link)) if os.path.islink(driver_link) else "none"
                        max_link_speed = ""
                        max_link_width = ""
                        speed_file = os.path.join(dev, "max_link_speed")
                        width_file = os.path.join(dev, "max_link_width")
                        if os.path.isfile(speed_file):
                            with open(speed_file, "r") as f:
                                max_link_speed = f.read().strip()
                        if os.path.isfile(width_file):
                            with open(width_file, "r") as f:
                                max_link_width = f.read().strip()

                        gpus.append({
                            "pci_slot": os.path.basename(dev),
                            "device_id": device_id,
                            "driver": driver,
                            "max_link_speed": max_link_speed,
                            "max_link_width": max_link_width,
                        })
            except (OSError, UnicodeDecodeError):
                continue
    return {
        "count": len(gpus),
        "devices": gpus,
        "is_dual_or_more": len(gpus) >= 2,
    }


def check_driver_hangs() -> dict[str, Any]:
    wedged_events: list[str] = []
    try:
        completed = subprocess.run(
            ["dmesg", "-T"],
            capture_output=True,
            text=True,
            check=False,
            errors="replace",
        )
        if completed.returncode == 0:
            lines = completed.stdout.splitlines()
            for line in lines[-200:]:
                if re.search(r"\b(xe|i915)\b.*(wedged|ring hang|GPU HANG|GT reset|resetting)", line, re.IGNORECASE):
                    wedged_events.append(line.strip())
    except (FileNotFoundError, PermissionError):
        pass

    return {
        "has_wedged_event": len(wedged_events) > 0,
        "recent_events": wedged_events[-5:],
    }


def check_tools() -> dict[str, Any]:
    def which(cmd: str) -> bool:
        try:
            return subprocess.run(["which", cmd], capture_output=True, check=False).returncode == 0
        except FileNotFoundError:
            return False

    docker_running = False
    if which("docker"):
        try:
            res = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
            docker_running = res.returncode == 0
        except FileNotFoundError:
            docker_running = False

    return {
        "has_docker": which("docker"),
        "docker_running": docker_running,
        "has_sycl_ls": which("sycl-ls"),
        "has_xpu_smi": which("xpu-smi"),
        "has_huggingface_cli": which("huggingface-cli"),
    }


def run_doctor() -> dict[str, Any]:
    return {
        "os": check_os(),
        "render": check_render_nodes(),
        "gpus": check_intel_gpus(),
        "driver_status": check_driver_hangs(),
        "tools": check_tools(),
    }


def print_report(diag: dict[str, Any]) -> int:
    print("=== Intel Arc Pro B60/B70 System Doctor ===\n")
    all_ok = True

    # OS Check
    os_info = diag["os"]
    print(f"OS: {os_info['system']} {os_info['release']} ({os_info['arch']})")
    if not os_info["is_linux"]:
        print("  [WARN] Native Linux kernel recommended for optimal vLLM XPU performance.")

    # GPU Check
    gpus = diag["gpus"]
    print(f"\nIntel GPUs Detected: {gpus['count']}")
    if gpus["count"] == 0:
        print("  [FAIL] No Intel GPUs found via PCI scan (vendor 0x8086, class 0x03*).")
        all_ok = False
    else:
        for idx, g in enumerate(gpus["devices"]):
            print(f"  GPU #{idx}: PCI {g['pci_slot']} (ID: {g['device_id']}) Driver: {g['driver']} PCIe: {g['max_link_speed']} x{g['max_link_width']}")
            if g["driver"] != "xe":
                print(f"    [WARN] Driver is '{g['driver']}' (recommend 'xe' driver for Battlemage B60/B70).")

    # Render Node Access
    render = diag["render"]
    print(f"\nRender Nodes: {len(render['render_nodes'])} found, {len(render['accessible_render_nodes'])} accessible by user")
    if not render["can_access_render"]:
        print("  [FAIL] Current user cannot write to /dev/dri/renderD*. Add user to render group: sudo usermod -aG render $USER")
        all_ok = False
    else:
        print("  [PASS] Render node access verified.")

    # Multi-GPU Topology Warning
    if gpus["is_dual_or_more"]:
        print("\nMulti-GPU Topology (Dual-Card Detected):")
        print("  [INFO] For dual-B70 TP2 on desktop motherboards without P2P, ensure these oneCCL variables are set:")
        print("         -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296")
        print("         -e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296")
        print("         -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296")
        print("         -e CCL_SYCL_ALLTOALL_TMP_BUF=1")
        print("         --cap-add SYS_PTRACE --ipc=host")
        print("         See docs/DUAL-B70-TP2.md for topology details.")

    # Driver Wedged Check
    drv = diag["driver_status"]
    if drv["has_wedged_event"]:
        print("\n  [FAIL] Xe driver wedge or GPU hang detected in dmesg!")
        for ev in drv["recent_events"]:
            print(f"    {ev}")
        print("\n  RECOVERY INSTRUCTIONS:")
        print("  1. Reset card via sysfs or reboot the machine:")
        print("     sudo reboot")
        print("  2. See docs/RELIABILITY-REPORT.md for ring wedge diagnosis.")
        all_ok = False
    else:
        print("\nDriver Health: [PASS] No recent Xe wedge events detected in dmesg buffer.")

    # Tools Check
    tools = diag["tools"]
    print("\nTools & Runtimes:")
    if tools["docker_running"]:
        print("  [PASS] Docker is installed and running.")
    elif tools["has_docker"]:
        print("  [WARN] Docker is installed but daemon is not responding.")
    else:
        print("  [WARN] Docker not found in PATH (needed for vLLM XPU).")

    if tools["has_sycl_ls"]:
        print("  [PASS] sycl-ls found.")
    if tools["has_xpu_smi"]:
        print("  [PASS] xpu-smi found.")
    if tools["has_huggingface_cli"]:
        print("  [PASS] huggingface-cli found.")
    else:
        print("  [WARN] huggingface-cli not found in PATH.")

    print("\n" + ("=" * 40))
    if all_ok:
        print("Status: READY. System is primed for Intel Arc Pro B60/B70 inference.")
        return 0
    else:
        print("Status: ATTENTION REQUIRED. Resolve highlighted issues before running.")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--recover", action="store_true", help="Print recovery instructions for wedged cards")
    args = parser.parse_args()

    if args.recover:
        print("=== Xe Driver Hang Recovery Instructions ===")
        print("1. Terminate all active inference containers and processes:")
        print("   docker rm -f $(docker ps -q --filter name=b70) 2>/dev/null || true")
        print("   killall -9 vllm llama-server 2>/dev/null || true")
        print("2. Check kernel ring buffer:")
        print("   dmesg -T | grep -iE 'xe|wedged|ring hang'")
        print("3. If the card does not recover, reboot the host:")
        print("   sudo reboot")
        return 0

    diag = run_doctor()
    if args.json:
        print(json.dumps(diag, indent=2))
        return 0
    return print_report(diag)


if __name__ == "__main__":
    raise SystemExit(main())
