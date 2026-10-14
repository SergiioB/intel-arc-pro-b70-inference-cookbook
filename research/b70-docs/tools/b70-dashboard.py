#!/usr/bin/env python3
"""B70 System Monitor - Draggable Wayland widget with right-click menu.
Position saved to ~/.config/b70-dashboard/position.json"""

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('cairo', '1.0')
from gi.repository import Gtk, GLib, Gdk
import cairo

import os
import re
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from collections import deque

# ─── Config & State ─────────────────────────────────────────────────────────────
REFRESH_MS = 2000
LLAMA_PORTS = [8765, 8080]
VLLM_PORT = 8765  # vLLM MTP profile (drop-in for llama on the same port)
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY", "REDACTED-LAN-KEY")
LLAMA_HEADERS = {"Authorization": f"Bearer {LLAMA_API_KEY}"}
GPU_HWMON = "/sys/class/hwmon/hwmon4"
CPU_HWMON = "/sys/class/hwmon/hwmon1"
NVME_HWMON = "/sys/class/hwmon/hwmon0"
WIDGET_W = 460
WIDGET_H = 1180  # increased for subscriptions section + models partition + dynamic models

# Config dir for position
CONFIG_DIR = Path.home() / ".config" / "b70-dashboard"
CONFIG_FILE = CONFIG_DIR / "position.json"
LOG_FILE = Path("/tmp/b70-dashboard.log")
MOA_STATE_FILE = Path.home() / "b70-dashboard-moa-state.json"

# Layout constants
PAD = 18
BAR_START = 210
BAR_H = 14
ROW_H = 22
FONT_LABEL = "Monospace"

GPU_FREQ_BASE = None
for p in Path("/sys/devices").rglob("tile0/gt0/freq0/cur_freq"):
    GPU_FREQ_BASE = str(p.parent)
    break

# ─── Colors ────────────────────────────────────────────────────────────────────
C_BG          = (0.05, 0.06, 0.10, 0.93)
C_BORDER      = (0.20, 0.25, 0.35, 0.9)
C_TITLE       = (0.40, 0.70, 1.0)
C_SECTION     = (0.55, 0.65, 0.80)
C_LABEL       = (0.52, 0.60, 0.70)
C_VALUE       = (0.90, 0.92, 0.95)
C_VALUE_HOT   = (1.0, 0.65, 0.35)
C_VALUE_CRIT  = (1.0, 0.35, 0.35)
C_VALUE_OK    = (0.45, 0.90, 0.55)
C_BAR_BG      = (0.12, 0.15, 0.20)
C_BAR_FG_OK   = (0.30, 0.85, 0.45)
C_BAR_FG_HOT  = (1.0, 0.60, 0.25)
C_BAR_FG_CRIT = (1.0, 0.30, 0.30)
C_SPARK       = (0.30, 0.65, 1.0)

# ─── Helpers ─────────────────────────────────────────────────────────────────────

def log(msg):
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except: pass

def read_int(path):
    try:
        with open(path) as f: return int(f.read().strip())
    except: return None

def read_str(path):
    try:
        with open(path) as f: return f.read().strip()
    except: return None

def celsius(millideg, thresholds=(60, 80)):
    if millideg is None: return ("N/A", C_LABEL, C_BAR_BG)
    c = millideg / 1000.0
    warn, crit = thresholds
    if c >= crit:   return (f"{c:.0f}°C", C_VALUE_CRIT, C_BAR_FG_CRIT)
    elif c >= warn: return (f"{c:.0f}°C", C_VALUE_HOT, C_BAR_FG_HOT)
    return (f"{c:.0f}°C", C_VALUE, C_BAR_FG_OK)

def bar_color(pct, warn=60, crit=80):
    if pct >= crit:  return C_BAR_FG_CRIT
    elif pct >= warn: return C_BAR_FG_HOT
    return C_BAR_FG_OK

def fmt_bytes(b):
    if b is None: return "N/A"
    for u in ['B','K','M','G','T']:
        if abs(b) < 1024: return f"{b:.0f}{u}" if u == 'B' else f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}P"

def fmt_uptime(s):
    d, s = divmod(int(s), 86400)
    h, s = divmod(s, 3600)
    m, _ = divmod(s, 60)
    return f"{d}d {h}h {m}m" if d else f"{h}h {m}m"

def fmt_speed(bps):
    if not bps: return "0"
    for u in ['B/s','KB/s','MB/s','GB/s']:
        if abs(bps) < 1024: return f"{bps:.0f} {u}" if u == 'B/s' else f"{bps:.1f} {u}"
        bps /= 1024
    return f"{bps:.1f} TB/s"

def load_position():
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                return json.load(f)
    except: pass
    return None

def save_position(x, y):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump({'x': x, 'y': y}, f)

# ─── Data Collectors ───────────────────────────────────────────────────────────

_last_net = None
_last_time = 0
_prev_cpu = {'idle': 0, 'total': 0}

def get_llama_status():
    results = []
    # If vLLM is active on 8765, skip the llama check for that port (avoids
    # false-positive from vLLM's /health endpoint and wasted parse attempts).
    vllm_active = False
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", "vllm-mtp-profile.service"],
                           capture_output=True, text=True, timeout=1)
        vllm_active = r.stdout.strip() == "active"
    except: pass
    ports = [p for p in LLAMA_PORTS if not (vllm_active and p == VLLM_PORT)]
    for port in ports:
        url = f"http://localhost:{port}"
        try:
            req = urllib.request.Request(f"{url}/health", headers=LLAMA_HEADERS, method='GET')
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                health = json.loads(resp.read())
            model_name, model_path, ctx_size, kv_cache = "Unknown", "", "", ""
            active_slots = 0
            kv_k = kv_v = flash = spec = None
            try:
                ps = subprocess.run(["pgrep", "-a", "llama-server"], capture_output=True, text=True, timeout=1)
                for line in ps.stdout.split('\n'):
                    m = re.search(r'--model\s+(\S+)', line)
                    if m:
                        model_path = m.group(1)
                        model_name = os.path.basename(model_path)
                    m = re.search(r'--cache-type-k\s+(\S+)', line)
                    if m: kv_k = m.group(1)
                    m = re.search(r'--cache-type-v\s+(\S+)', line)
                    if m: kv_v = m.group(1)
                    if '--flash-attn' in line: flash = True
                    m_ctx = re.search(r'--ctx-size\s+(\d+)', line)
                    if m_ctx: ctx_size = f"{int(m_ctx.group(1))//1024}K"
                    m_spec = re.search(r'--spec-type\s+(\S+)', line)
                    if m_spec: spec = m_spec.group(1)
            except: pass
            kv_desc = ""
            if kv_k and kv_v: kv_desc = f" | KV {kv_k}/{kv_v}"
            if flash: kv_desc += " | FA"
            if spec: kv_desc += f" | {spec}"
            try:
                req3 = urllib.request.Request(f"{url}/slots", headers=LLAMA_HEADERS, method='GET')
                with urllib.request.urlopen(req3, timeout=1.5) as resp3:
                    slots = json.loads(resp3.read())
                    if isinstance(slots, list) and slots:
                        s0 = slots[0]
                        cache = s0.get('cache', {})
                        if cache:
                            tu = cache.get('tokens_used', 0)
                            tc = cache.get('tokens_capacity', 0)
                            if tc: kv_cache = f"{tu}/{tc} tok"
                        active_slots = sum(1 for s in slots if s.get('is_processing', False))
            except: pass
            results.append({
                'port': port, 'running': True, 'model': model_name,
                'model_path': model_path, 'ctx': ctx_size, 'kv': kv_cache,
                'kv_desc': kv_desc, 'active': active_slots,
                'status': health.get('status', 'ok'),
            })
        except: pass
    return results


def get_vllm_status():
    """vLLM MTP profile (port 8765, Docker, vllm-mtp-profile.service)."""
    url = f"http://localhost:{VLLM_PORT}"
    try:
        req = urllib.request.Request(f"{url}/v1/models", method='GET')
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read())
        first = (data.get("data") or [{}])[0]
        model_id = first.get("id") or "unknown"
        # Spec + ctx from the container command
        ctx = spec = ""
        try:
            ps = subprocess.run(["pgrep", "-af", "vllm"], capture_output=True, text=True, timeout=1)
            for line in ps.stdout.split('\n'):
                m = re.search(r'--max-model-len\s+(\d+)', line)
                if m: ctx = f"{int(m.group(1))//1024}K"
                m = re.search(r'--enable-prefix-caching', line)
                if m: spec = "PC"
                if '--max-num-batched-tokens' in line: spec = (spec + " |" if spec else "") + " 8KBT"
        except: pass
        return [{
            'port': VLLM_PORT, 'running': True, 'model': model_id[:46],
            'model_path': "Docker: intel/vllm:0.21.0-xpu-int4moe",
            'ctx': ctx or "16K", 'kv': "",
            'kv_desc': f"MTP spec" + (f" | {spec}" if spec else ""),
            'active': 0, 'status': 'ok', 'vllm': True,
        }]
    except: pass
    return []


def get_gpu_stats():
    stats = {}
    stats['pkg_temp']   = read_int(f"{GPU_HWMON}/temp2_input")
    stats['vram_temp']  = read_int(f"{GPU_HWMON}/temp3_input")
    stats['mctrl_temp'] = read_int(f"{GPU_HWMON}/temp4_input")
    stats['pcie_temp']  = read_int(f"{GPU_HWMON}/temp5_input")
    cap_uw = read_int(f"{GPU_HWMON}/power1_cap")
    stats['power_cap'] = cap_uw / 1_000_000 if cap_uw else None
    crit_uw = read_int(f"{GPU_HWMON}/power1_crit")
    stats['power_crit'] = crit_uw / 1_000_000 if crit_uw else None
    if GPU_FREQ_BASE:
        stats['freq'] = read_int(f"{GPU_FREQ_BASE}/cur_freq")
        stats['freq_max'] = read_int(f"{GPU_FREQ_BASE}/max_freq") or read_int(f"{GPU_FREQ_BASE}/rp0_freq")
    else:
        stats['freq'] = stats['freq_max'] = None
    stats['vram_used'] = None
    stats['vram_total'] = 32 * 1024**3
    stats['vram_estimated'] = False
    for card in Path("/sys/class/drm").glob("card[0-9]"):
        try:
            used = int((card / "device/mem_info_vram_used").read_text().strip())
            total = int((card / "device/mem_info_vram_total").read_text().strip())
            if total > 0:
                stats['vram_used'] = used
                stats['vram_total'] = total
                break
        except: pass
    if stats['vram_used'] is None:
        try:
            ps = subprocess.run(["pgrep", "-a", "llama-server"], capture_output=True, text=True, timeout=1)
            for line in ps.stdout.split('\n'):
                m = re.search(r'--model\s+(\S+)', line)
                if m and os.path.exists(m.group(1)):
                    stats['vram_used'] = int(os.path.getsize(m.group(1)) * 1.1)
                    stats['vram_estimated'] = True
                    break
        except: pass
    return stats

def get_cpu_stats():
    stats = {}
    stats['tctl'] = read_int(f"{CPU_HWMON}/temp1_input")
    stats['tdie'] = read_int(f"{CPU_HWMON}/temp3_input")
    try:
        with open("/proc/stat") as f:
            fields = list(map(int, f.readline().split()[1:]))
        stats['_idle'] = fields[3] + fields[4]
        stats['_total'] = sum(fields)
    except:
        stats['_idle'] = stats['_total'] = 0
    return stats

def compute_cpu_usage(stats):
    global _prev_cpu
    idle = stats.get('_idle', 0)
    total = stats.get('_total', 0)
    d_idle = idle - _prev_cpu['idle']
    d_total = total - _prev_cpu['total']
    _prev_cpu = {'idle': idle, 'total': total}
    return max(0, (1 - d_idle / d_total) * 100) if d_total > 0 else 0

def get_system_stats():
    stats = {}
    try:
        mi = {}
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                if len(p) >= 2: mi[p[0].rstrip(':')] = int(p[1]) * 1024
        stats['ram_total'] = mi.get('MemTotal', 0)
        stats['ram_used'] = stats['ram_total'] - mi.get('MemAvailable', stats['ram_total'])
        stats['swap_total'] = mi.get('SwapTotal', 0)
        stats['swap_used'] = stats['swap_total'] - mi.get('SwapFree', stats['swap_total'])
    except:
        stats['ram_total'] = stats['ram_used'] = 0
    try:
        st = os.statvfs('/')
        stats['disk_total'] = st.f_blocks * st.f_frsize
        stats['disk_used'] = (st.f_blocks - st.f_bavail) * st.f_frsize
    except:
        stats['disk_total'] = stats['disk_used'] = 0
    try:
        st = os.statvfs('/data/nvme')
        stats['models_total'] = st.f_blocks * st.f_frsize
        stats['models_used'] = (st.f_blocks - st.f_bavail) * st.f_frsize
    except:
        stats['models_total'] = stats['models_used'] = 0
    stats['nvme_temp'] = read_int(f"{NVME_HWMON}/temp1_input")
    try:
        with open("/proc/uptime") as f:
            stats['uptime'] = float(f.read().split()[0])
    except:
        stats['uptime'] = 0
    global _last_net, _last_time
    net_speed = {}
    now = time.time()
    try:
        rx = tx = 0
        for iface in os.listdir("/sys/class/net"):
            if iface == 'lo': continue
            try:
                rx += int(read_str(f"/sys/class/net/{iface}/statistics/rx_bytes") or 0)
                tx += int(read_str(f"/sys/class/net/{iface}/statistics/tx_bytes") or 0)
            except: pass
        if _last_net and now > _last_time:
            dt = now - _last_time
            net_speed['rx'] = max(0, (rx - _last_net['rx']) / dt)
            net_speed['tx'] = max(0, (tx - _last_net['tx']) / dt)
        _last_net = {'rx': rx, 'tx': tx}
        _last_time = now
    except: pass
    stats['net'] = net_speed
    return stats

def fmt_reset(sec):
    """Format seconds until reset."""
    if not sec or sec < 0:
        return ""
    if sec < 3600:
        return f"{sec//60}m"
    if sec < 86400:
        return f"{sec//3600}h{sec%3600//60:02d}m"
    return f"{sec//86400}d{sec%86400//3600}h"


def parse_iso_to_epoch(s):
    """Parse ISO 8601 like 2026-07-02T14:08:27Z to epoch."""
    if not s:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return None


def get_quota_status():
    """Read quota state fetched locally by b70-quota-fetch.py."""
    results = {'providers': [], 'updated_ago': None}
    try:
        raw = MOA_STATE_FILE.read_text()
        state = json.loads(raw)
        updated_ts = state.get('updated_at', 0)
        if updated_ts:
            results['updated_ago'] = int(time.time() - updated_ts)
        now = time.time()
        q = state.get('quota', {})
        # ── Codex (ChatGPT Plus) ──
        cq = q.get('codex', {})
        p_used = cq.get('primary_used_percent')
        w_used = cq.get('weekly_used_percent')
        p_reset = cq.get('primary_reset_after_seconds')
        w_reset = cq.get('weekly_reset_after_seconds')
        
        err_msg = cq.get('error', '')
        needs_login = cq.get('needs_login') or "expired" in err_msg.lower() or "signin" in err_msg.lower() or "signing in" in err_msg.lower() or "401" in err_msg.lower()
        
        has_data = cq.get('ok') or bool(cq.get('plan_type')) or (p_used is not None and p_used > 0) or (w_used is not None and w_used > 0)
        if has_data:
            p_used_val = p_used if p_used is not None else 0
            w_used_val = w_used if w_used is not None else 0
            icon = '🟢' if p_used_val < 85 else '⚠️'
            results['providers'].append({
                'name': 'ChatGPT', 'model': '', 'icon': icon,
                'bars': [
                    ('5h', p_used_val, fmt_reset(p_reset) if p_reset else ''),
                    ('Wk', w_used_val, fmt_reset(w_reset) if w_reset else ''),
                ],
            })
        elif needs_login:
            results['providers'].append({
                'name': 'ChatGPT', 'model': '', 'icon': '🔐',
                'bars': [('5h', -1, 're-auth'), ('Wk', -1, '')],
            })
        else:
            reset_lbl = "offline"
            if err_msg:
                reset_lbl = "error"
            results['providers'].append({
                'name': 'ChatGPT', 'model': '', 'icon': '❌',
                'bars': [('5h', -1, reset_lbl), ('Wk', -1, '')],
            })

        # ── Antigravity (AG) ──
        aq = q.get('antigravity', {})
        groups = aq.get('groups', {}) if aq.get('ok') else {}
        if aq.get('ok') and not groups and 'models' in aq:
            mlist = []
            for key, info in aq['models'].items():
                mid = key.split(':', 1)[1] if ':' in key else key
                rem_pct = info.get('remaining_percent')
                used_pct = 100 - rem_pct if rem_pct is not None else 0.0
                mlist.append({
                    "id": mid,
                    "display_name": info.get("display_name"),
                    "used_percent": used_pct,
                    "reset_time": info.get("reset_time"),
                    "has_quota": True
                })
            groups = {"API_PROVIDER_GOOGLE_GEMINI": mlist}
        
        primary_defaults = {
            "Flash": "gemini-3-flash-agent",
            "Pro": "gemini-pro-agent",
            "Sonnet": "claude-sonnet-4-6",
            "Opus": "claude-opus-4-6-thinking",
            "GPT": "gpt-oss-120b-medium",
            "FLite": "gemini-3.1-flash-lite",
        }
        
        ag_short = {
            "gemini-3-flash-agent": "Flash",
            "gemini-pro-agent": "Pro",
            "claude-sonnet-4-6": "Sonnet",
            "claude-opus-4-6-thinking": "Opus",
            "gpt-oss-120b-medium": "GPT",
            "gemini-3.5-flash-low": "Flash",
            "gemini-3.5-flash-extra-low": "Flash",
            "gemini-3-flash": "Flash",
            "gemini-2.5-flash": "Flash",
            "gemini-2.5-flash-thinking": "Flash",
            "gemini-3.1-flash-image": "Flash",
            "gemini-3.1-pro-high": "Pro",
            "gemini-3.1-pro-low": "Pro",
            "gemini-2.5-pro": "Pro",
            "gemini-2.5-flash-lite": "FLite",
            "gemini-3.1-flash-lite": "FLite",
        }
        
        if groups:
            selected_models = []
            
            by_label = {}
            for mlist in groups.values():
                for m in mlist:
                    mid = m.get("id")
                    if not mid:
                        continue
                    lbl = ag_short.get(mid)
                    if not lbl:
                        continue
                    if lbl not in by_label:
                        by_label[lbl] = []
                    by_label[lbl].append(m)
            
            for lbl, mlist in by_label.items():
                used_models = [m for m in mlist if (m.get("used_percent") or 0) > 0]
                if used_models:
                    default_id = primary_defaults.get(lbl)
                    default_m = next((m for m in used_models if m.get("id") == default_id), None)
                    if default_m:
                        best_m = default_m
                    else:
                        best_m = max(used_models, key=lambda m: m.get("used_percent") or 0)
                    selected_models.append((lbl, best_m))
                else:
                    default_id = primary_defaults.get(lbl)
                    if default_id:
                        default_m = next((m for m in mlist if m.get("id") == default_id), None)
                        if default_m:
                            selected_models.append((lbl, default_m))
                        else:
                            selected_models.append((lbl, mlist[0]))
            
            ordered_labels = ["Flash", "Pro", "Sonnet", "Opus", "GPT", "FLite"]
            bars = []
            seen_lbls = set()
            
            for o_lbl in ordered_labels:
                for lbl, m in selected_models:
                    if lbl == o_lbl:
                        used = m.get("used_percent")
                        reset_epoch = parse_iso_to_epoch(m.get("reset_time"))
                        reset_str = fmt_reset(int(reset_epoch - now)) if reset_epoch else ""
                        if used is not None and m.get("has_quota"):
                            bars.append((lbl, used, reset_str))
                        else:
                            bars.append((lbl, -1, ""))
                        seen_lbls.add(lbl)
            
            for lbl, m in selected_models:
                if lbl not in seen_lbls:
                    used = m.get("used_percent")
                    reset_epoch = parse_iso_to_epoch(m.get("reset_time"))
                    reset_str = fmt_reset(int(reset_epoch - now)) if reset_epoch else ""
                    if used is not None and m.get("has_quota"):
                        bars.append((lbl, used, reset_str))
                    else:
                        bars.append((lbl, -1, ""))
                        
            if bars:
                results['providers'].append({
                    'name': 'Antigravity', 'model': '', 'icon': '',
                    'bars': bars,
                })
        else:
            results['providers'].append({
                'name': 'Antigravity', 'model': '', 'icon': '❌',
                'bars': [],
            })
        # ── xAI / Grok (X Premium) ──
        xq = q.get('xai', {})
        grok_r = xq
        if grok_r.get('ok') and grok_r.get('source') == 'codexbar':
            # Weekly-only window from the X Premium path
            used_w = grok_r.get('weekly_used_percent')
            reset_epoch = parse_iso_to_epoch(grok_r.get('weekly_resets_at'))
            reset_str = fmt_reset(int(reset_epoch - now)) if reset_epoch else ''
            results['providers'].append({
                'name': 'Grok', 'model': '', 'icon': '🟢' if (used_w or 0) < 80 else '⚠️',
                'bars': [('Wk', used_w if used_w is not None else -1, reset_str)],
            })
        elif grok_r.get('ok'):
            rem_r = grok_r.get('remaining_requests', 0) or 0
            lim_r = grok_r.get('limit_requests', 1) or 1
            used_r = max(0, 100 - rem_r / lim_r * 100)
            rem_t = grok_r.get('remaining_tokens', 0) or 0
            lim_t = grok_r.get('limit_tokens', 1) or 1
            used_t = max(0, 100 - rem_t / lim_t * 100)
            reset_r = grok_r.get('reset_requests_sec')
            reset_t = grok_r.get('reset_tokens_sec')
            results['providers'].append({
                'name': 'Grok', 'model': '', 'icon': '🟢' if used_r < 80 else '⚠️',
                'bars': [
                    ('Req', used_r, fmt_reset(int(reset_r)) if reset_r else ''),
                    ('Tok', used_t, fmt_reset(int(reset_t)) if reset_t else ''),
                ],
            })
        else:
            results['providers'].append({
                'name': 'Grok', 'model': '', 'icon': '❌',
                'bars': [('Wk', -1, (grok_r.get('error') or 'down')[:12])],
            })
        # ── Z.AI (GLM) ──
        zq = q.get('zai', {})
        if zq.get('ok'):
            windows = zq.get('token_windows', [])
            bars = []
            for w in windows[:2]:
                unit = w.get('unit', '')
                if unit == 3: label = '5h'
                elif unit == 6: label = 'Wk'
                else: label = f'W{unit}'
                used = w.get('used_percent', 0)
                reset_ms = w.get('next_reset_ms', 0)
                reset_str = ''
                if reset_ms and reset_ms > 0:
                    reset_epoch = reset_ms / 1000
                    reset_str = fmt_reset(int(reset_epoch - now))
                bars.append((label, used, reset_str))
            for w in zq.get('web_windows', [])[:1]:
                used = w.get('used_percent', 0)
                reset_ms = w.get('next_reset_ms', 0)
                reset_str = ''
                if reset_ms and reset_ms > 0:
                    reset_str = fmt_reset(int(reset_ms / 1000 - now))
                usage, limit = w.get('usage'), w.get('limit')
                if usage is not None and limit:
                    reset_str = f"{usage}/{limit} {reset_str}".strip()
                bars.append(('Web', used, reset_str))
            while len(bars) < 2:
                bars.append(('N/A', 0, ''))
            results['providers'].append({
                'name': 'Z.AI', 'model': '', 'icon': '🟢' if all(b[1] < 85 for b in bars) else '⚠️',
                'bars': bars,
            })
        else:
            results['providers'].append({
                'name': 'Z.AI', 'model': '', 'icon': '❌',
                'bars': [('5h', 0, ''), ('Wk', 0, '')],
            })
        # ── OpenCode Go ──
        oq = q.get('opencodego', {})
        if oq.get('ok'):
            bars = []
            for key, label in (('primary', '5h'), ('secondary', 'Wk'), ('tertiary', 'Mo')):
                w = oq.get(key) or {}
                used = w.get('used_percent')
                reset_epoch = parse_iso_to_epoch(w.get('resets_at'))
                reset_str = fmt_reset(int(reset_epoch - now)) if reset_epoch else ''
                bars.append((label, used if used is not None else -1, reset_str))
            ok_bars = [b[1] for b in bars if b[1] >= 0]
            results['providers'].append({
                'name': 'OpenCode Go', 'model': '',
                'icon': '🟢' if all(b < 85 for b in ok_bars) else '⚠️',
                'bars': bars,
            })
        else:
            results['providers'].append({
                'name': 'OpenCode Go', 'model': '', 'icon': '❌',
                'bars': [('5h', -1, ''), ('Wk', -1, ''), ('Mo', -1, '')],
            })
        # ── Xiaomi MiMo (no quota API — liveness only) ──
        mq = q.get('mimo', {})
        if mq.get('ok'):
            results['providers'].append({
                'name': 'MiMo', 'model': '', 'icon': '🟢',
                'bars': [('API', -1, f"{mq.get('models', 0)} models")],
            })
        else:
            results['providers'].append({
                'name': 'MiMo', 'model': '', 'icon': '❌',
                'bars': [('API', -1, (mq.get('error') or 'down')[:12])],
            })
    except Exception as e:
        log(f"quota: {e}")
    return results

# ─── History ───────────────────────────────────────────────────────────────────
HIST_LEN = 60
hist = {
    'gpu_pkg': deque(maxlen=HIST_LEN),
    'cpu_usage': deque(maxlen=HIST_LEN),
    'ram': deque(maxlen=HIST_LEN),
}

# ─── Drawing Widget ────────────────────────────────────────────────────────────

class MonitorArea(Gtk.DrawingArea):
    def __init__(self, on_config_callback=None):
        super().__init__()
        self.set_size_request(WIDGET_W, WIDGET_H)
        self.data = {}
        self.on_config = on_config_callback
        self.connect("draw", self.on_draw)

    def update_data(self, data):
        self.data = data
        self.queue_draw()

    def on_draw(self, widget, cr):
        w = widget.get_allocated_width()
        h = widget.get_allocated_height()

        # Background
        cr.set_source_rgba(*C_BG)
        self._rrect(cr, 0, 0, w, h, 14)
        cr.fill()
        cr.set_source_rgba(*C_BORDER)
        cr.set_line_width(1)
        self._rrect(cr, 0.5, 0.5, w-1, h-1, 14)
        cr.stroke()

        y = PAD

        # Title + live dot
        cr.set_source_rgba(*C_TITLE)
        cr.select_font_face(FONT_LABEL, cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
        cr.set_font_size(16)
        cr.move_to(PAD, y + 4)
        cr.show_text("B70 SYSTEM MONITOR")
        cr.set_source_rgba(0.3, 0.9, 0.4, 0.9)
        cr.arc(w - PAD - 4, y, 5, 0, 6.28318)
        cr.fill()
        y += 28

        # ── INFERENCE ──
        y = self._section(cr, "INFERENCE ENGINE", y, w)
        llama = self.data.get('llama', [])
        if not llama:
            cr.set_source_rgba(*C_VALUE_CRIT)
            cr.set_font_size(12)
            cr.move_to(PAD, y + 4)
            cr.show_text("● No inference server running")
            y += ROW_H
        else:
            for srv in llama:
                if not srv['running']: continue
                cr.set_source_rgba(*C_VALUE_OK)
                cr.set_font_size(12)
                cr.move_to(PAD, y + 4)
                engine = "vLLM MTP" if srv.get('vllm') else "llama.cpp"
                status = f"● {engine} Running"
                if srv.get('active', 0) > 0:
                    status += f" ({srv['active']} active)"
                cr.show_text(status)
                y += ROW_H

                cr.set_source_rgba(*C_VALUE)
                cr.move_to(PAD + 10, y + 4)
                cr.show_text(f"Model:  {srv['model'][:46]}")
                y += 18

                cr.set_source_rgba(*C_LABEL)
                cr.move_to(PAD + 10, y + 4)
                cr.show_text(f"Path:   {srv['model_path'][:50]}")
                y += 18

                cr.set_source_rgba(*C_LABEL)
                cr.move_to(PAD + 10, y + 4)
                cr.show_text(f"Port:   :{srv['port']}  |  Ctx: {srv['ctx']}")
                y += 18

                if srv['kv']:
                    cr.move_to(PAD + 10, y + 4)
                    cr.show_text(f"Slots:  {srv['kv']}")
                    y += 18

                cr.move_to(PAD + 10, y + 4)
                cr.show_text(f"Config:{srv['kv_desc']}")
                y += 22

        # ── GPU ──
        y = self._section(cr, "GPU - ARC PRO B70 32GB", y, w)
        gpu = self.data.get('gpu', {})

        for name, val, thresh in [
            ("Package",  gpu.get('pkg_temp'),   (70, 90)),
            ("VRAM",     gpu.get('vram_temp'),  (85, 100)),
            ("MemCtrl",  gpu.get('mctrl_temp'), (85, 100)),
            ("PCIe",     gpu.get('pcie_temp'),  (75, 90)),
        ]:
            y = self._temp_row(cr, name, val, thresh, y, w)

        # Power
        cap = gpu.get('power_cap')
        if cap is not None:
            cr.set_source_rgba(*C_LABEL)
            cr.set_font_size(12)
            cr.move_to(PAD, y + 4)
            cr.show_text(f"Power:    {cap:.0f}W cap")
            crit = gpu.get('power_crit')
            if crit: cr.show_text(f" / {crit:.0f}W max")
            y += ROW_H

        # Frequency
        freq = gpu.get('freq')
        if freq is not None:
            fmax = gpu.get('freq_max') or freq
            cr.set_source_rgba(*C_LABEL)
            cr.move_to(PAD, y + 4)
            cr.show_text(f"Freq:     {freq}/{fmax} MHz")
            y += ROW_H

        # VRAM
        vram_used = gpu.get('vram_used')
        vram_total = gpu.get('vram_total', 32*1024**3)
        if vram_used is not None:
            pct = vram_used / vram_total * 100 if vram_total else 0
            lbl = f"{fmt_bytes(vram_used)}/{fmt_bytes(vram_total)}"
            if gpu.get('vram_estimated'): lbl += " ~est"
            y = self._bar_row(cr, "VRAM", lbl, pct, y, w, 85, 95)
        else:
            cr.set_source_rgba(*C_LABEL)
            cr.move_to(PAD, y + 4)
            cr.show_text("VRAM:     N/A")
            y += ROW_H

        if len(hist['gpu_pkg']) > 2:
            y = self._spark(cr, list(hist['gpu_pkg']), "GPU temp trend", y, w, 0, 100, "°C")
        y += 4

        # ── CPU ──
        y = self._section(cr, "CPU - RYZEN 7 5700X3D", y, w)
        cpu = self.data.get('cpu', {})
        y = self._temp_row(cr, "Tctl", cpu.get('tctl'), (70, 85), y, w)
        usage = self.data.get('cpu_usage', 0)
        y = self._bar_row(cr, "Usage", f"{usage:.1f}%", usage, y, w, 70, 90)
        if len(hist['cpu_usage']) > 2:
            y = self._spark(cr, list(hist['cpu_usage']), "CPU usage trend", y, w, 0, 100, "%")
        y += 4

        # ── SYSTEM ──
        y = self._section(cr, "SYSTEM", y, w)
        ss = self.data.get('system', {})

        ram_u = ss.get('ram_used', 0)
        ram_t = ss.get('ram_total', 1)
        y = self._bar_row(cr, "RAM", f"{fmt_bytes(ram_u)}/{fmt_bytes(ram_t)}",
                          ram_u/ram_t*100 if ram_t else 0, y, w, 80, 95)
        sw_u = ss.get('swap_used', 0)
        sw_t = ss.get('swap_total', 0)
        if sw_t > 0:
            y = self._bar_row(cr, "Swap", f"{fmt_bytes(sw_u)}/{fmt_bytes(sw_t)}",
                              sw_u/sw_t*100, y, w, 50, 80)
        du = ss.get('disk_used', 0)
        dt = ss.get('disk_total', 1)
        y = self._bar_row(cr, "Disk", f"{du/1024**3:.0f}G/{dt/1024**3:.0f}G",
                          du/dt*100 if dt else 0, y, w, 85, 95)
        mu = ss.get('models_used', 0)
        mt = ss.get('models_total', 0)
        if mt > 0:
            y = self._bar_row(cr, "M-ext4", f"{mu/1024**3:.0f}G/{mt/1024**3:.0f}G",
                              mu/mt*100 if mt else 0, y, w, 85, 95)
        y = self._temp_row(cr, "NVMe", ss.get('nvme_temp'), (60, 75), y, w)

        cr.set_source_rgba(*C_LABEL)
        cr.set_font_size(12)
        cr.move_to(PAD, y + 4)
        cr.show_text(f"Uptime:   {fmt_uptime(ss.get('uptime', 0))}")
        y += ROW_H

        net = ss.get('net', {})
        if net:
            cr.move_to(PAD, y + 4)
            cr.show_text(f"Net:      ↓{fmt_speed(net.get('rx',0))}  ↑{fmt_speed(net.get('tx',0))}")
            y += ROW_H

        # ── SUBSCRIPTIONS ──
        y = self._section(cr, "AI SUBSCRIPTIONS", y, w)
        quota = self.data.get('quota', {})
        providers = quota.get('providers', [])
        updated_ago = quota.get('updated_ago')
        for prov in providers:
            pname = f"{prov['icon']} {prov['name']}"
            cr.set_source_rgba(*C_VALUE)
            cr.set_font_size(12)
            cr.move_to(PAD, y + 4)
            cr.show_text(pname)
            y += 18
            for label, used_pct, reset_str in prov['bars']:
                cr.set_source_rgba(*C_LABEL)
                cr.set_font_size(10)
                cr.move_to(PAD + 8, y + 3)
                cr.show_text(f"{label}")
                cr.set_source_rgba(*C_VALUE)
                cr.move_to(PAD + 54, y + 3)
                if used_pct < 0:
                    cr.show_text("  ~  ")
                else:
                    cr.show_text(f"{used_pct:.0f}%")
                bar_w = w - PAD*2 - 142
                cr.set_source_rgba(*C_BAR_BG)
                self._rrect(cr, PAD + 92, y - 2, bar_w, BAR_H - 4, (BAR_H-4)/2)
                cr.fill()
                if used_pct > 0:
                    fill_w = max(BAR_H-4, bar_w * used_pct / 100)
                    cr.set_source_rgba(*bar_color(used_pct, 70, 85))
                    self._rrect(cr, PAD + 92, y - 2, fill_w, BAR_H - 4, (BAR_H-4)/2)
                    cr.fill()
                if reset_str:
                    cr.set_source_rgba(0.4, 0.45, 0.5, 0.7)
                    cr.move_to(w - PAD - 45, y + 3)
                    cr.show_text(reset_str)
                y += 14
            y += 4
        if updated_ago is not None:
            cr.set_source_rgba(0.3, 0.35, 0.4, 0.6)
            cr.set_font_size(9)
            cr.move_to(PAD, y + 2)
            mins = updated_ago // 60
            if mins < 60:
                cr.show_text(f"Updated {mins}m ago")
            else:
                cr.show_text(f"Updated {mins//60}h {mins%60}m ago")
            y += 14

        # Footer
        cr.set_source_rgba(0.3, 0.35, 0.4, 0.6)
        cr.set_font_size(10)
        cr.move_to(PAD, h - 12)
        cr.show_text("Right-click for options | Drag to move | ESC to close")
        return False

    def _rrect(self, cr, x, y, w, h, r):
        r = min(r, w/2, h/2)
        cr.move_to(x+r, y)
        cr.arc(x+w-r, y+r, r, -1.5708, 0)
        cr.arc(x+w-r, y+h-r, r, 0, 1.5708)
        cr.arc(x+r, y+h-r, r, 1.5708, 3.14159)
        cr.arc(x+r, y+r, r, 3.14159, 4.71239)
        cr.close_path()

    def _section(self, cr, title, y, w):
        cr.set_source_rgba(*C_SECTION)
        cr.select_font_face(FONT_LABEL, cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
        cr.set_font_size(11)
        cr.move_to(PAD, y + 4)
        cr.show_text(f"── {title} ──")
        y += 22
        cr.set_source_rgba(0.15, 0.20, 0.30, 0.5)
        cr.set_line_width(0.5)
        cr.move_to(PAD, y)
        cr.line_to(w - PAD, y)
        cr.stroke()
        return y + 10

    def _temp_row(self, cr, name, val_mdeg, thresh, y, w):
        val_str, text_color, bar_fill = celsius(val_mdeg, thresh)
        cr.select_font_face(FONT_LABEL, cairo.FontSlant.NORMAL, cairo.FontWeight.NORMAL)
        cr.set_font_size(12)
        cr.set_source_rgba(*C_LABEL)
        cr.move_to(PAD, y + 4)
        cr.show_text(f"{name:10s}")
        cr.set_source_rgba(*text_color)
        cr.move_to(PAD + 80, y + 4)
        cr.show_text(val_str)
        if val_mdeg is not None:
            c = val_mdeg / 1000.0
            warn_t, crit_t = thresh
            pct = min(c / (crit_t + 10), 1.0) * 100
            bar_w = w - PAD*2 - BAR_START
            cr.set_source_rgba(*C_BAR_BG)
            self._rrect(cr, PAD + BAR_START, y - 2, bar_w, BAR_H, BAR_H/2)
            cr.fill()
            fill_w = max(BAR_H, bar_w * pct / 100)
            cr.set_source_rgba(*bar_fill)
            self._rrect(cr, PAD + BAR_START, y - 2, fill_w, BAR_H, BAR_H/2)
            cr.fill()
        return y + ROW_H

    def _bar_row(self, cr, name, value_str, pct, y, w, warn, crit):
        cr.select_font_face(FONT_LABEL, cairo.FontSlant.NORMAL, cairo.FontWeight.NORMAL)
        cr.set_font_size(12)
        cr.set_source_rgba(*C_LABEL)
        cr.move_to(PAD, y + 4)
        cr.show_text(f"{name:10s}")
        cr.set_source_rgba(*C_VALUE)
        cr.move_to(PAD + 80, y + 4)
        cr.show_text(value_str)
        bar_w = w - PAD*2 - BAR_START
        cr.set_source_rgba(*C_BAR_BG)
        self._rrect(cr, PAD + BAR_START, y - 2, bar_w, BAR_H, BAR_H/2)
        cr.fill()
        fill_w = max(BAR_H, bar_w * pct / 100)
        cr.set_source_rgba(*bar_color(pct, warn, crit))
        self._rrect(cr, PAD + BAR_START, y - 2, fill_w, BAR_H, BAR_H/2)
        cr.fill()
        return y + ROW_H

    def _spark(self, cr, vals, label, y, w, vmin, vmax, unit):
        cr.select_font_face(FONT_LABEL, cairo.FontSlant.NORMAL, cairo.FontWeight.NORMAL)
        cr.set_font_size(10)
        cr.set_source_rgba(*C_LABEL)
        cr.move_to(PAD, y + 3)
        cr.show_text(label)
        y += 8
        sh = 28
        sw = w - PAD * 2
        rv = max(vmax - vmin, 1)
        cr.set_line_width(1.2)
        cr.set_source_rgba(*C_SPARK, 0.8)
        n = len(vals)
        for i, v in enumerate(vals):
            px = PAD + (i / (HIST_LEN - 1)) * sw
            py = y + sh - ((v - vmin) / rv) * sh
            py = max(y, min(y + sh, py))
            if i == 0: cr.move_to(px, py)
            else: cr.line_to(px, py)
        cr.stroke()
        lx = PAD + ((n - 1) / (HIST_LEN - 1)) * sw
        cr.line_to(lx, y + sh)
        cr.line_to(PAD, y + sh)
        cr.close_path()
        cr.set_source_rgba(*C_SPARK, 0.12)
        cr.fill()
        cr.set_source_rgba(*C_VALUE)
        cr.move_to(w - PAD - 36, y - 5)
        lv = vals[-1]
        cr.show_text(f"{lv:.0f}{unit}" if unit else f"{lv:.1f}")
        return y + sh + 10


class MonitorWindow(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title="B70 Monitor")
        self.set_decorated(False)
        self.set_default_size(WIDGET_W, WIDGET_H)
        self.set_resizable(False)
        self.set_keep_above(False)
        self.set_skip_taskbar_hint(True)
        self.set_skip_pager_hint(True)
        self.set_app_paintable(True)

        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.set_visual(visual)

        css = b'window { background: transparent; }'
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(screen, provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        self.area = MonitorArea()
        self.add(self.area)

        # Event masks on BOTH window and drawing area
        evmask = (Gdk.EventMask.BUTTON_PRESS_MASK |
                  Gdk.EventMask.BUTTON_RELEASE_MASK |
                  Gdk.EventMask.POINTER_MOTION_MASK |
                  Gdk.EventMask.POINTER_MOTION_HINT_MASK)
        self.add_events(evmask)
        self.area.add_events(evmask)
        self._drag_start = None

        # Right-click menu
        menu = Gtk.Menu()

        item_center = Gtk.MenuItem(label="Center on screen")
        item_center.connect("activate", self.center_on_screen)
        menu.append(item_center)

        item_top = Gtk.MenuItem(label="Toggle always-on-top")
        item_top.connect("activate", self.toggle_keep_above)
        menu.append(item_top)

        item_reload = Gtk.MenuItem(label="Reload dashboard")
        item_reload.connect("activate", lambda *_: os.execv(sys.executable, [sys.executable] + sys.argv))
        menu.append(item_reload)

        menu.append(Gtk.SeparatorMenuItem())

        item_close = Gtk.MenuItem(label="Quit")
        item_close.connect("activate", lambda *_: Gtk.main_quit())
        menu.append(item_close)

        menu.show_all()
        self.popup_menu = menu

        # Event handlers on BOTH window and area
        self.connect("button-press-event", self.on_button_press)
        self.area.connect("button-press-event", self.on_button_press)
        self.connect("configure-event", self.on_configure)
        self.connect("key-press-event", self.on_key)

        self._tick()

    def center_on_screen(self, *_):
        display = self.get_display()
        monitor = display.get_primary_monitor()
        if monitor:
            rect = monitor.get_geometry()
        else:
            screen = self.get_screen()
            rect = screen.get_monitor_geometry(0)
        x = (rect.width - WIDGET_W) // 2
        y = (rect.height - WIDGET_H) // 2
        self.move(x, y)
        save_position(x, y)

    def on_configure(self, widget, event):
        save_position(event.x, event.y)

    def toggle_keep_above(self, *_):
        self._keep_above = not getattr(self, '_keep_above', False)
        self.set_keep_above(self._keep_above)

    def on_button_press(self, widget, event):
        if event.button == 1:  # Left click = drag via compositor
            self.begin_move_drag(event.button, int(event.x_root), int(event.y_root),
                                 event.get_time())
        elif event.button == 3:  # Right click = menu
            self.popup_menu.popup_at_pointer()

    def on_key(self, widget, event):
        if event.keyval == Gdk.KEY_Escape:
            Gtk.main_quit()
        return False

    def _tick(self):
        data = {}
        try:
            data['llama'] = get_llama_status()
        except Exception as e:
            log(f"llama: {e}")
            data['llama'] = []
        try:
            data['llama'] += get_vllm_status()
        except Exception as e:
            log(f"vllm: {e}")
        try:
            data['gpu'] = get_gpu_stats()
            if data['gpu'].get('pkg_temp'):
                hist['gpu_pkg'].append(data['gpu']['pkg_temp'] / 1000)
        except Exception as e:
            log(f"gpu: {e}")
            data['gpu'] = {}
        try:
            data['cpu'] = get_cpu_stats()
            data['cpu_usage'] = compute_cpu_usage(data['cpu'])
            hist['cpu_usage'].append(data['cpu_usage'])
        except Exception as e:
            log(f"cpu: {e}")
            data['cpu'] = {}
            data['cpu_usage'] = 0
        try:
            data['system'] = get_system_stats()
            if data['system'].get('ram_total'):
                hist['ram'].append(data['system']['ram_used'])
        except Exception as e:
            log(f"system: {e}")
            data['system'] = {}
        try:
            data['quota'] = get_quota_status()
        except Exception as e:
            log(f"quota: {e}")
            data['quota'] = {}

        self.area.update_data(data)
        GLib.timeout_add(REFRESH_MS, self._tick)


class MonitorApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.b70.monitor')

    def do_activate(self):
        log("Activating window")
        win = MonitorWindow(self)
        win.connect("destroy", Gtk.main_quit)

        # Load saved position
        pos = load_position()
        if pos:
            log(f"Loading saved position: {pos}")
            win.move(pos['x'], pos['y'])
        else:
            log("Centering on screen")
            win.center_on_screen()

        win.show_all()


def main():
    log("Starting B70 Dashboard")
    try:
        app = MonitorApp()
        app.run(sys.argv)
    except Exception as e:
        log(f"Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()