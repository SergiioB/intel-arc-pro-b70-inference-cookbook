#!/usr/bin/env python3
"""Read-only Intel Arc / Battlemage XPU preflight check (stdlib only).

Discovers Intel GPUs by PCI BDF (never remembered hwmon indices), checks
as-found power caps, package temperatures, visible free VRAM (when debugfs is
readable), host MemAvailable, root/data disk headroom, and active inference
processes. Makes zero changes to hardware or processes.

Usage:
  python3 b70_preflight.py
  python3 b70_preflight.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
from typing import Any


def read_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def discover_intel_gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    pci_root = pathlib.Path("/sys/bus/pci/devices")
    if not pci_root.exists():
        return gpus

    for dev in sorted(pci_root.iterdir()):
        vendor = read_text(dev / "vendor")
        cls = read_text(dev / "class")
        if vendor != "0x8086" or not cls or not (cls.startswith("0x0300") or cls.startswith("0x0380")):
            continue
        bdf = dev.name
        device_id = read_text(dev / "device")
        driver = (dev / "driver").resolve().name if (dev / "driver").exists() else None

        drm_nodes: list[str] = []
        drm_dir = dev / "drm"
        if drm_dir.exists():
            drm_nodes = sorted(p.name for p in drm_dir.iterdir() if p.name.startswith(("card", "renderD")))

        hwmon_info: dict[str, Any] = {}
        hwmon_dir = dev / "hwmon"
        if hwmon_dir.exists():
            for hm in sorted(hwmon_dir.iterdir()):
                cap_uw = read_text(hm / "power1_cap")
                if cap_uw and cap_uw.isdigit():
                    hwmon_info["power_cap_w"] = round(int(cap_uw) / 1e6, 1)
                for tf in sorted(hm.glob("temp*_input")):
                    label = read_text(hm / tf.name.replace("_input", "_label")) or tf.stem
                    val = read_text(tf)
                    if val and val.isdigit():
                        hwmon_info[f"temp_{label}_c"] = round(int(val) / 1000.0, 1)

        vram_avail_mib = None
        vram_mm = pathlib.Path(f"/sys/kernel/debug/dri/{bdf}/tile0/vram_mm")
        vtxt = read_text(vram_mm)
        if vtxt:
            m = re.search(r"visible_avail:\s*(\d+)\s*MiB", vtxt)
            if m:
                vram_avail_mib = int(m.group(1))

        gpus.append(
            {
                "pci_bdf": bdf,
                "device_id": device_id,
                "driver": driver,
                "drm_nodes": drm_nodes,
                "vram_visible_avail_mib": vram_avail_mib,
                "hwmon": hwmon_info,
            }
        )
    return gpus


def host_memory_gib() -> dict[str, float]:
    out: dict[str, float] = {}
    meminfo = read_text(pathlib.Path("/proc/meminfo"))
    if not meminfo:
        return out
    for line in meminfo.splitlines():
        for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            if line.startswith(f"{key}:"):
                kb = int(line.split()[1])
                out[f"{key}_gib"] = round(kb / (1024**2), 2)
    return out


def disk_free_gib(paths: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in paths:
        if os.path.exists(p):
            usage = shutil.disk_usage(p)
            out[p] = round(usage.free / (1024**3), 2)
    return out


def active_inference_procs() -> list[str]:
    try:
        res = subprocess.run(
            ["ps", "-eo", "pid,comm,args"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except OSError:
        return []
    hits = []
    pat = re.compile(r"(vllm serve|EngineCore|llama-server|tensorfold|inkling2_serve|sglang)", re.I)
    me = str(os.getpid())
    for line in res.stdout.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if not parts or parts[0] == me:
            continue
        if pat.search(line) and "b70_preflight.py" not in line:
            hits.append(line.strip()[:140])
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    ap.add_argument("--disk", action="append", default=["/"], help="Additional mount path to check")
    args = ap.parse_args()

    disks = list(dict.fromkeys(args.disk))
    report = {
        "intel_gpus": discover_intel_gpus(),
        "host_memory": host_memory_gib(),
        "disk_free_gib": disk_free_gib(disks),
        "active_inference_procs": active_inference_procs(),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=== Intel XPU Read-Only Preflight ===")
    print(f"Host RAM: {report['host_memory']}")
    print(f"Disk free (GiB): {report['disk_free_gib']}")
    print(f"Active inference processes: {len(report['active_inference_procs'])}")
    for proc in report["active_inference_procs"]:
        print(f"  WARN: {proc}")
    print(f"Intel GPUs discovered: {len(report['intel_gpus'])}")
    for g in report["intel_gpus"]:
        vram = f"{g['vram_visible_avail_mib']} MiB" if g["vram_visible_avail_mib"] is not None else "n/a (need debugfs read)"
        print(
            f"  {g['pci_bdf']} dev={g['device_id']} drv={g['driver']} nodes={g['drm_nodes']} "
            f"vram_free={vram} hwmon={g['hwmon']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
