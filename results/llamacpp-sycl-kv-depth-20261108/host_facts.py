#!/usr/bin/env python3
"""Live host facts for the B70 host. Never hardcode hwmon numbers: they move on reboot.

  host_facts.py hwmon GPU_INDEX     -> /sys/class/hwmon/hwmonN of that B70 (index = PCI order = ZE_AFFINITY_MASK)
  host_facts.py json                -> facts as JSON
  host_facts.py md [OUT]            -> HOST-FACTS.md (default ~/HOST-FACTS.md)

GPU index i is the i-th Intel xe device sorted by PCI address; that is the order
Level Zero enumerates (ZE_AFFINITY_MASK=i), checked 2026-11-04 (mask 1 -> 0000:10:00.0).
"""
import glob, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path


def gpus():
    out = []
    for h in sorted(glob.glob('/sys/class/hwmon/hwmon*')):
        try:
            if Path(h, 'name').read_text().strip() != 'xe':
                continue
        except OSError:
            continue
        pci = os.path.basename(os.path.realpath(Path(h, 'device')))
        def rd(f):
            try:
                return int(Path(h, f).read_text())
            except Exception:
                return None
        out.append({'pci': pci, 'hwmon': h, 'cap_w': (rd('power1_cap') or 0) / 1e6,
                    'temp_c': (rd('temp2_input') or 0) / 1000,
                    'vram_debugfs': f'/sys/kernel/debug/dri/{pci}/tile0/vram_mm'})
    out.sort(key=lambda g: g['pci'])
    for i, g in enumerate(out):
        g['index'] = i
    return out


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ''


def facts():
    mem = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo') if l.split(':')[0] in ('MemTotal', 'MemAvailable')}
    disks = {}
    for m in ['/']:
        if os.path.ismount(m) or m == '/':
            u = shutil.disk_usage(m)
            disks[m] = {'free_gb': round(u.free / 1e9), 'used_pct': round(100 * u.used / u.total)}
    llama = []
    for d in sorted(glob.glob('$ENGINES/llama.cpp-*/build-sycl/bin/llama-server')):
        root = d.split('/build-sycl')[0]
        llama.append({'binary': d, 'git': sh(f'git -C {root} describe --tags --always 2>/dev/null'),
                      'built': time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(d)))})
    tf = sorted(glob.glob('$TF*/src*/zig-out/xpu/bin/tensorfold-native'))
    models = {}
    for root in [
                 '$TF/models']:
        if os.path.isdir(root):
            models[root] = sorted(x for x in os.listdir(root) if not x.endswith('.log'))
    return {'utc': time.strftime('%FT%TZ', time.gmtime()), 'host': platform.node(), 'kernel': platform.release(),
            'cpu': sh("lscpu | sed -n 's/^Model name: *//p'"), 'ram_gib': round(mem['MemTotal'] / 2**20),
            'ram_avail_gib': round(mem['MemAvailable'] / 2**20),
            'board': sh('cat /sys/class/dmi/id/board_name'), 'gpus': gpus(), 'disks': disks,
            'oneapi': sorted(os.listdir('/opt/intel/oneapi/compiler')) if os.path.isdir('/opt/intel/oneapi/compiler') else [],
            'llama_cpp_builds': llama, 'tensorfold_builds': tf,
            'docker_images': sh("docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' 2>/dev/null").splitlines(),
            'lmx': sh('command -v lmx'), 'models': models}


def md(f):
    L = [f"# Host facts — generated {f['utc']} by `host_facts.py md`", '',
         'Do not hand-edit. Regenerate after any hardware, engine or model change.', '',
         f"- Host `{f['host']}`, kernel `{f['kernel']}`, board `{f['board']}`",
         f"- CPU {f['cpu']}; RAM {f['ram_gib']} GiB ({f['ram_avail_gib']} available at capture)",
         f"- oneAPI compilers: {', '.join(f['oneapi'])}; `source /opt/intel/oneapi/setvars.sh --force`",
         f"- lmx: `{f['lmx'] or 'not installed'}`", '', '## GPUs', '',
         '| index (ZE_AFFINITY_MASK) | PCI | hwmon (moves on reboot) | cap W | temp C | VRAM debugfs |', '|---|---|---|---|---|---|']
    for g in f['gpus']:
        L.append(f"| {g['index']} | {g['pci']} | {g['hwmon']} | {g['cap_w']:.0f} | {g['temp_c']:.0f} | `sudo cat {g['vram_debugfs']}` |")
    L += ['', 'Look a card up with `python3 host_facts.py hwmon <index>`; never hardcode hwmonN.', '',
          '## Disks', '', '| mount | free GB | used % |', '|---|---|---|']
    L += [f"| {m} | {d['free_gb']} | {d['used_pct']} |" for m, d in f['disks'].items()]
    L += ['', '## Engine builds', '']
    L += [f"- llama.cpp `{b['git']}` built {b['built']}: `{b['binary']}`" for b in f['llama_cpp_builds']]
    L += [f"- TensorFold: `{t}`" for t in f['tensorfold_builds']]
    L += [f"- docker: `{i}`" for i in f['docker_images']]
    L += ['', '## Models on disk', '']
    for root, xs in f['models'].items():
        L.append(f"- `{root}`: " + ', '.join(f'`{x}`' for x in xs))
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'json'
    if cmd == 'hwmon':
        print(next(g['hwmon'] for g in gpus() if g['index'] == int(sys.argv[2])))
    elif cmd == 'md':
        out = Path(sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/HOST-FACTS.md'))
        out.write_text(md(facts())); print(out)
    else:
        print(json.dumps(facts(), indent=1))
