#!/usr/bin/env python3
"""B70 synchronized hwmon monitor with portable xe/PCI discovery.

Usage: b70-sync-monitor.py OUTPUT.jsonl [interval_seconds]
Writes one metadata record, then fixed-cadence samples using monotonic_ns. Card
power is each interval's energy1_input delta; temperatures remain decimal °C.
SIGTERM writes a final summary record with exact total energy/time average.
"""
import glob, json, os, signal, sys, time

OUT = sys.argv[1]
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
TARGET_PCI = "0000:0b:00.0"
running = True

def stop(*_):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

def read_int(path):
    with open(path) as f:
        return int(f.read().strip())

def read_text(path):
    with open(path) as f:
        return f.read().strip()

def discover():
    candidates = []
    for h in sorted(glob.glob('/sys/class/hwmon/hwmon*')):
        name = read_text(h + '/name') if os.path.exists(h + '/name') else ''
        device = os.path.realpath(h + '/device')
        if name == 'xe' and TARGET_PCI in device and os.path.exists(h + '/energy1_input'):
            candidates.append((h, name, device))
    if len(candidates) != 1:
        raise RuntimeError(f'expected one B70 xe hwmon mapped to {TARGET_PCI}, found {candidates}')
    return candidates[0]

h, name, device = discover()
temps = []
for label_path in sorted(glob.glob(h + '/temp*_label')):
    stem = label_path[:-5]
    input_path = stem + 'input'
    if os.path.exists(input_path):
        temps.append({'key': os.path.basename(stem), 'label': read_text(label_path), 'input': input_path})
energy = h + '/energy1_input'
energy_label = read_text(h + '/energy1_label') if os.path.exists(h + '/energy1_label') else 'unknown'
cap = read_int(h + '/power1_cap') if os.path.exists(h + '/power1_cap') else None

with open(OUT, 'x', buffering=1) as f:
    def emit(obj):
        f.write(json.dumps(obj, separators=(',', ':')) + '\n')
    start_ns = time.monotonic_ns()
    start_energy = read_int(energy)
    prev_ns, prev_energy = start_ns, start_energy
    emit({'type':'metadata','schema':1,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'clock':'CLOCK_MONOTONIC via time.monotonic_ns','interval_s':INTERVAL,'hwmon_path':h,
          'hwmon_name':name,'device_path':device,'pci_bdf':TARGET_PCI,'energy_path':energy,
          'energy_label':energy_label,'energy_unit':'uJ','configured_cap_W':cap/1e6 if cap is not None else None,
          'temperature_sensors':[{'key':x['key'],'label':x['label'],'path':x['input']} for x in temps]})
    deadline = start_ns
    samples = 0
    while running:
        now = time.monotonic_ns()
        e = read_int(energy)
        dt = (now-prev_ns)/1e9
        de = e-prev_energy
        temp_values = {x['label']: read_int(x['input'])/1000.0 for x in temps}
        emit({'type':'sample','monotonic_ns':now,'elapsed_s':round((now-start_ns)/1e9,6),
              'energy1_input_uJ':e,'interval_energy_uJ':de,'interval_s':round(dt,6),
              'card_interval_average_W':round(de/dt/1e6,3) if dt > 0 and de >= 0 else None,
              'temperature_C':temp_values})
        samples += 1
        prev_ns, prev_energy = now, e
        deadline += int(INTERVAL*1e9)
        delay = (deadline-time.monotonic_ns())/1e9
        if delay > 0:
            time.sleep(delay)
    end_ns = time.monotonic_ns()
    end_energy = read_int(energy)
    elapsed = (end_ns-start_ns)/1e9
    delta = end_energy-start_energy
    emit({'type':'summary','monotonic_ns':end_ns,'elapsed_s':round(elapsed,6),'samples':samples,
          'energy_start_uJ':start_energy,'energy_end_uJ':end_energy,'energy_delta_uJ':delta,
          'average_card_draw_W':round(delta/elapsed/1e6,3) if elapsed > 0 and delta >= 0 else None})
