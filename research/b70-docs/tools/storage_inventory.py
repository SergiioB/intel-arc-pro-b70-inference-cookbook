#!/usr/bin/env python3
"""Inventory the B70 data disks against registry/storage-layout.json.

scan     list every top-level entry on each data root with size, last change and
         whether a live config, a running container or a process mentions it.
check    exit 1 when a mount or a top-level entry does not match the registry.
resolve  map a path from an old note (e.g. /mnt/<models-disk>/strata/...) to where it
         lives now, or report that it was deleted and where the record is.

Read-only. Sizes come from `du -sx` (sudo -n when available).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / "registry" / "storage-layout.json"
HOME = Path.home()

# Files whose mentions mean "something live depends on this path".
LIVE_FILES = [
    ROOT / "registry" / "model-status.json",
    HOME / "b70-launcher" / "recipes.json",
    HOME / ".local" / "state" / "b70-launcher" / "settings-override.json",
    HOME / ".bashrc",
    Path("/etc/fstab"),
    Path("/etc/default/llama-hermes"),
]
LIVE_GLOBS = [
    (HOME / ".config" / "systemd" / "user", "*.service"),
    (Path("/etc/systemd/system"), "*.service"),
]
DOCKER_SOCKS = ["/var/run/docker.sock", "/run/<exl3-docker>.sock"]


def load_layout() -> dict:
    return json.loads(LAYOUT.read_text())


def live_text() -> str:
    parts = []
    files = list(LIVE_FILES)
    for base, pat in LIVE_GLOBS:
        if base.is_dir():
            files += sorted(base.glob(pat))
    for f in files:
        try:
            parts.append(f.read_text(errors="replace"))
        except OSError:
            pass
    for sock in DOCKER_SOCKS:
        try:
            ids = subprocess.run(["sudo", "-n", "docker", "-H", f"unix://{sock}", "ps", "-q"],
                                 capture_output=True, text=True, timeout=15).stdout.split()
            if ids:
                parts.append(subprocess.run(
                    ["sudo", "-n", "docker", "-H", f"unix://{sock}", "inspect",
                     "--format", "{{json .Mounts}}"] + ids,
                    capture_output=True, text=True, timeout=15).stdout)
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        parts.append(subprocess.run(["ps", "-eo", "args"], capture_output=True,
                                    text=True, timeout=10).stdout)
    except (OSError, subprocess.TimeoutExpired):
        pass
    # B70-DOCS evidence folders that are symlinks onto a data disk (2026-10-13:
    # two such targets were deleted as scratch because this check was missing).
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            if os.path.islink(p):
                parts.append(os.readlink(p))
    return "\n".join(parts)


def du(path: Path) -> int | None:
    for cmd in (["sudo", "-n", "du", "-sxb", str(path)], ["du", "-sxb", str(path)]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.stdout.strip():
            return int(out.stdout.split()[0])
    return None


def entries(mount: Path):
    for p in sorted(mount.iterdir()):
        if p.name in ("lost+found",):
            continue
        yield p


def scan(layout: dict, with_size: bool) -> list[dict]:
    text = live_text()
    rows = []
    for disk in layout["disks"]:
        mount = Path(disk["mount"])
        if not mount.is_dir():
            rows.append({"disk": disk["name"], "path": str(mount), "error": "not mounted"})
            continue
        for p in entries(mount):
            sub = []
            if p.name in layout["role_dirs"] and p.is_dir() and not p.is_symlink():
                sub = list(entries(p))
            for q in sub or [p]:
                name = q.name
                live = bool(re.search(r"(?<![\w.-])" + re.escape(name) + r"(?![\w.-])", text))
                st = q.lstat()
                rows.append({
                    "disk": disk["name"],
                    "path": str(q),
                    "bytes": du(q) if with_size else None,
                    "modified": time.strftime("%Y-%m-%d", time.localtime(st.st_mtime)),
                    "live_reference": live,
                })
    return rows


def check(layout: dict) -> int:
    errors = []
    mounts = {}
    for line in Path("/proc/mounts").read_text().splitlines():
        dev, target = line.split()[:2]
        mounts[target] = dev
    for disk in layout["disks"]:
        if disk["mount"] not in mounts:
            errors.append(f"{disk['name']}: {disk['mount']} is not mounted")
            continue
        allowed = set(layout["role_dirs"]) | {"lost+found"} | set(disk.get("extra_top_level", []))
        for p in Path(disk["mount"]).iterdir():
            if p.name not in allowed:
                errors.append(f"{disk['name']}: unregistered top-level entry {p}")
    for legacy in layout.get("retired_paths", []):
        if os.path.lexists(legacy["old"]):
            errors.append(f"retired path still exists: {legacy['old']}")
    for e in errors:
        print("error:", e)
    print("OK: storage layout matches registry" if not errors else f"FAIL: {len(errors)} storage problems")
    return 1 if errors else 0


def _swap_prefix(path: str, old: str, new: str) -> str | None:
    if path == old or path.startswith(old.rstrip("/") + "/"):
        return new.rstrip("/") + path[len(old.rstrip("/")):]
    return None


def resolve(layout: dict, path: str) -> int:
    """Old path -> current path, following aliases, the 2026-10-13 remount and moves."""
    p = os.path.normpath(path)
    for alias, target in sorted(layout.get("mount_aliases", {}).items(), key=lambda kv: -len(kv[0])):
        p = _swap_prefix(p, alias, target) or p
    for rec in layout.get("deletions", []):
        for gone in (ROOT / rec["list"]).read_text().split():
            if _swap_prefix(p, gone, gone):
                print(f"deleted {rec['date']}: {gone}\nrecord: {rec['list']}")
                return 1
    for old, new in sorted(layout.get("mount_map", {}).items(), key=lambda kv: -len(kv[0])):
        p = _swap_prefix(p, old, new) or p
    for old, new in sorted(layout.get("path_map", []), key=lambda kv: -len(kv[0])):
        moved = _swap_prefix(p, old, new)
        if moved:
            p = moved
            break
    exists = os.path.lexists(p)
    print(p if exists else f"{p}\nnot found: not in the delete record either; check older cleanup notes")
    return 0 if exists else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--no-size", action="store_true", help="skip du (fast)")
    s.add_argument("--json", action="store_true")
    sub.add_parser("check")
    r = sub.add_parser("resolve")
    r.add_argument("path")
    a = ap.parse_args(argv)
    layout = load_layout()
    if a.cmd == "check":
        return check(layout)
    if a.cmd == "resolve":
        return resolve(layout, a.path)
    rows = scan(layout, not a.no_size)
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    for r in rows:
        if "error" in r:
            print(f"{r['disk']:6} {r['path']}  ERROR {r['error']}")
            continue
        size = f"{r['bytes'] / 2**30:7.1f}G" if r["bytes"] is not None else "      ?"
        print(f"{r['disk']:6} {size} {r['modified']} {'LIVE' if r['live_reference'] else '    '} {r['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
