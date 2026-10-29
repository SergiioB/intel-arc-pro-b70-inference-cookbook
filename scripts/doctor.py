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
from typing import Any

PCI_DEVICES_ROOT = "/sys/bus/pci/devices"
BDF_PATTERN = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
# Live and maximum link state of the B70's internal switch hop: always 2.5 GT/s
# x1, so the GPU function's own link registers never describe the host link.
PCI_INTERNAL_SPEED = "2.5 GT/s PCIe"
PCI_INTERNAL_WIDTH = "1"


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
        import pwd

        username = pwd.getpwuid(os.getuid()).pw_name
        user_groups = [g.gr_name for g in grp.getgrall() if username in g.gr_mem]
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


def read_attr(path: str) -> str:
    """Read one sysfs attribute, or an empty string when it is absent."""
    try:
        with open(path, errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def link_width(value: str) -> int:
    """Numeric lane width from a sysfs width attribute, 0 when unreadable."""
    return int(value) if value.isdigit() else 0


def link_speed_gts(value: str) -> float:
    """Transfer rate in GT/s from a sysfs speed attribute, 0.0 when unreadable."""
    match = re.match(r"([0-9]+(?:\.[0-9]+)?)", value)
    return float(match.group(1)) if match else 0.0


def pcie_hop(device_dir: str, role: str) -> dict[str, Any]:
    """Describe one device on a PCIe path, including its live and maximum link."""
    speed = read_attr(os.path.join(device_dir, "current_link_speed"))
    width = read_attr(os.path.join(device_dir, "current_link_width"))
    return {
        "bdf": os.path.basename(device_dir),
        "role": role,
        "current_speed": speed,
        "current_width": width,
        "max_speed": read_attr(os.path.join(device_dir, "max_link_speed")),
        "max_width": read_attr(os.path.join(device_dir, "max_link_width")),
        # The B70 GPU function and the card's own bridge sit below an internal
        # switch hop that always reports 2.5 GT/s x1, live and maximum. Its link
        # registers describe nothing about the host link, so they are excluded
        # from the effective-link calculation when a real hop exists.
        "link_state_reliable": not (speed == PCI_INTERNAL_SPEED and width == PCI_INTERNAL_WIDTH),
    }


def pcie_chain(device_dir: str) -> list[dict[str, Any]]:
    """Return the root port, every bridge, and the endpoint of one device, top-down."""
    parts = os.path.realpath(device_dir).split(os.sep)
    nodes = [
        os.sep.join(parts[: index + 1])
        for index, part in enumerate(parts)
        if BDF_PATTERN.match(part)
    ]
    hops: list[dict[str, Any]] = []
    for position, node in enumerate(nodes):
        if len(nodes) == 1 or position == len(nodes) - 1:
            role = "endpoint"
        elif position == 0:
            role = "root_port"
        else:
            role = "bridge"
        hops.append(pcie_hop(node, role))
    return hops


def effective_link(hops: list[dict[str, Any]]) -> dict[str, Any]:
    """Narrowest trustworthy hop: the negotiated host link, not the endpoint's register."""
    reliable = [hop for hop in hops if hop["link_state_reliable"]]
    pool = reliable or hops
    if not pool:
        return {"speed": "", "width": "", "source_bdf": "", "reference_bdf": "", "degraded": False}
    narrowest = min(
        pool,
        key=lambda hop: (link_width(hop["current_width"]), link_speed_gts(hop["current_speed"])),
    )
    reference = next((hop for hop in hops if hop["role"] == "root_port"), narrowest)
    degraded = link_width(reference["current_width"]) < link_width(reference["max_width"]) or (
        link_speed_gts(reference["current_speed"]) < link_speed_gts(reference["max_speed"])
    )
    return {
        "speed": narrowest["current_speed"],
        "width": narrowest["current_width"],
        "source_bdf": narrowest["bdf"],
        "reference_bdf": reference["bdf"],
        "degraded": degraded,
    }


def gpu_entry(dev: str, device_id: str) -> dict[str, Any]:
    """Describe one graphics PCI function, including its whole PCIe link chain."""
    driver_link = os.path.join(dev, "driver")
    driver = os.path.basename(os.readlink(driver_link)) if os.path.islink(driver_link) else "none"
    hops = pcie_chain(dev)
    return {
        "pci_slot": os.path.basename(dev),
        "device_id": device_id,
        "driver": driver,
        "pcie": {"chain": hops, "link": effective_link(hops)},
    }


def is_intel_graphics(dev: str) -> tuple[bool, str]:
    """True plus the device ID when `dev` is an Intel display-class PCI function."""
    if read_attr(os.path.join(dev, "vendor")).lower() != "0x8086":
        return False, ""
    if not read_attr(os.path.join(dev, "class")).lower().startswith("0x03"):
        return False, ""
    return True, read_attr(os.path.join(dev, "device")).lower()


def check_intel_gpus() -> dict[str, Any]:
    gpus: list[dict[str, Any]] = []
    for dev in sorted(glob.glob(os.path.join(PCI_DEVICES_ROOT, "*"))):
        matched, device_id = is_intel_graphics(dev)
        if not matched:
            continue
        try:
            gpus.append(gpu_entry(dev, device_id))
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
                pattern = r"\b(xe|i915)\b.*(wedged|ring hang|GPU HANG|GT reset|resetting)"
                if re.search(pattern, line, re.IGNORECASE):
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


def print_pcie_hop(hop: dict[str, Any]) -> None:
    """Print one device on a GPU's PCIe path with live and maximum link state."""
    marker = "" if hop["link_state_reliable"] else "  [internal hop, not the host link]"
    print(
        f"      {hop['bdf']} {hop['role']}: "
        f"{hop['current_speed']} x{hop['current_width']} live / "
        f"{hop['max_speed']} x{hop['max_width']} max{marker}"
    )


def print_gpus(gpus: dict[str, Any]) -> bool:
    """Print the GPU inventory; False when no Intel graphics function was found."""
    print(f"\nIntel GPUs Detected: {gpus['count']}")
    if gpus["count"] == 0:
        print("  [FAIL] No Intel GPUs found via PCI scan (vendor 0x8086, class 0x03*).")
        return False
    for idx, gpu in enumerate(gpus["devices"]):
        link = gpu["pcie"]["link"]
        print(
            f"  GPU #{idx}: PCI {gpu['pci_slot']} (ID: {gpu['device_id']}) Driver: {gpu['driver']}"
        )
        source = f" (from {link['source_bdf']})"
        print(f"    PCIe host link: {link['speed']} x{link['width']}{source}")
        for hop in reversed(gpu["pcie"]["chain"]):
            print_pcie_hop(hop)
        if link["degraded"]:
            print(
                f"    [WARN] Root port {link['reference_bdf']} trained below its maximum width "
                "or rate; check the slot, riser, and BIOS lane bifurcation."
            )
        if gpu["driver"] != "xe":
            print(f"    [WARN] Driver is '{gpu['driver']}' (recommend 'xe' for Battlemage).")
    return True


def print_report(diag: dict[str, Any]) -> int:
    print("=== Intel Arc Pro B60/B70 System Doctor ===\n")
    all_ok = True

    os_info = diag["os"]
    print(f"OS: {os_info['system']} {os_info['release']} ({os_info['arch']})")
    if not os_info["is_linux"]:
        print("  [WARN] Native Linux kernel recommended for optimal vLLM XPU performance.")

    gpus = diag["gpus"]
    if not print_gpus(gpus):
        all_ok = False

    render = diag["render"]
    r_count = len(render["render_nodes"])
    a_count = len(render["accessible_render_nodes"])
    print(f"\nRender Nodes: {r_count} found, {a_count} accessible by user")
    if not render["can_access_render"]:
        print("  [FAIL] Cannot write to /dev/dri/renderD*.")
        print("         Add user to render group: sudo usermod -aG render $USER")
        all_ok = False
    else:
        print("  [PASS] Render node access verified.")

    if gpus["is_dual_or_more"]:
        print("\nMulti-GPU Topology (Dual-Card Detected):")
        print("  [INFO] For dual-B70 TP2 on desktop boards without P2P, set oneCCL thresholds:")
        print("         -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296")
        print("         -e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296")
        print("         -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296")
        print("         -e CCL_SYCL_ALLTOALL_TMP_BUF=1")
        print("         --cap-add SYS_PTRACE --ipc=host")
        print("         See docs/DUAL-B70-TP2.md for topology details.")

    drv = diag["driver_status"]
    if drv["has_wedged_event"]:
        print("\n  [FAIL] Xe driver wedge or GPU hang detected in dmesg!")
        for ev in drv["recent_events"]:
            print(f"    {ev}")
        print("\n  RECOVERY INSTRUCTIONS:")
        print("  1. Reset card via sysfs or reboot the machine: sudo reboot")
        print("  2. See docs/RELIABILITY-REPORT.md for ring wedge diagnosis.")
        all_ok = False
    else:
        print("\nDriver Health: [PASS] No recent Xe wedge events detected in dmesg buffer.")

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
    print("Status: ATTENTION REQUIRED. Resolve highlighted issues before running.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument(
        "--recover", action="store_true", help="Print recovery instructions for wedged cards"
    )
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
