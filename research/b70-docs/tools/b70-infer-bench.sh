#!/bin/bash
API_KEY="${LLAMA_API_KEY:-REDACTED-LAN-KEY}"
URL="http://127.0.0.1:8080/v1/chat/completions"
PROMPT_LEN="${1:-256}"
MAXTOK="${2:-128}"
python3 <<PY
import json, time, urllib.request
key = "$API_KEY"
url = "$URL"
prompt = "Explain PCIe bandwidth and MoE routing. " * (int("$PROMPT_LEN") // 40 + 1)
body = json.dumps({"messages":[{"role":"user","content":prompt[:int("$PROMPT_LEN")]}],"max_tokens":int("$MAXTOK"),"temperature":0.3}).encode()
req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as r:
    data = json.loads(r.read().decode())
t1 = time.time()
tim = data.get("timings") or {}
pt = tim.get("predicted_per_second_ms")
tps = (1000.0/pt) if pt else None
usage = data.get("usage") or {}
comp = usage.get("completion_tokens", 0)
wall = t1 - t0
print(json.dumps({"wall_s": round(wall,2), "completion_tokens": comp, "tok_per_s_est": round(comp/wall,2) if wall else None, "timings": tim, "finish": data.get("choices",[{}])[0].get("finish_reason")}, indent=2))
PY
