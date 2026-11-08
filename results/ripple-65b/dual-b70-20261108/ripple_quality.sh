#!/usr/bin/env bash
# Ripple-65B on 2x B70 (-sm tensor, f16 KV): exact-answer canaries + one coding prompt via llama-server.
set -uo pipefail
OUT=$1; MODEL=${2:-$MODELS/Ripple-65B-Preview-AD-IQ4_XS.gguf}
BIN=$LLAMA/build-sycl/bin; PORT=8050
mkdir -p "$OUT"
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE SYCL_CACHE_PERSISTENT=0 ONEAPI_DEVICE_SELECTOR='level_zero:0,1'
nohup $BIN/llama-server -m "$MODEL" -ngl 99 -fa on -ctk f16 -ctv f16 -sm tensor -c 32768 -t 8 --jinja --host 127.0.0.1 --port $PORT > "$OUT/server.log" 2>&1 < /dev/null &
PID=$!; trap 'kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 5; done
python3 $TESTS/canary.py --base http://127.0.0.1:$PORT --model ripple \
  --prompts $TESTS/prompts_decoded.json > "$OUT/canary.txt" 2>&1
python3 - "$PORT" "$OUT" <<'PY'
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
body = {"model": "ripple", "messages": [{"role": "user", "content": "Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals and returns them sorted. Then give 3 assert tests. Code only."}],
        "max_tokens": 2048, "temperature": 0}
t = time.time()
r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800).read())
m = r["choices"][0]["message"]
json.dump({"wall_s": round(time.time() - t, 1), "usage": r.get("usage"), "timings": r.get("timings"), "content": m.get("content"), "reasoning_len": len(m.get("reasoning_content") or "")}, open(f"{out}/code.json", "w"), indent=1)
print("code prompt done", r.get("timings", {}).get("predicted_per_second"))
PY
