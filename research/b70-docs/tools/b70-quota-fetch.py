#!/usr/bin/env python3
"""Quota fetcher for B70 dashboard. Runs locally on B70, writes JSON state."""
import json, os, subprocess, time, urllib.request, urllib.error, urllib.parse
from pathlib import Path

HOME = Path.home()
STATE_FILE = HOME / "b70-dashboard-moa-state.json"
ENV_FILE = HOME / ".b70-dashboard-env"
CODEX_AUTH = HOME / ".b70-codex-auth.json"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
HERMES_AUTH = HOME / ".b70-hermes-auth.json"
AG_AUTH = HOME / ".b70-ag-auth.json"
OPENCODE_CFG = HOME / ".config" / "opencode" / "opencode.json"
CODEXBAR = HOME / ".local" / "bin" / "codexbar"
MIMO_BASE = "https://token-plan-ams.xiaomimimo.com/v1"

ZAI_QUOTA_URL = "https://api.z.ai/api/monitor/usage/quota/limit"
CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
XAI_URL = "https://api.x.ai/v1/chat/completions"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
CLOUDCODE_BASE = "https://cloudcode-pa.googleapis.com"
AG_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
AG_CLIENT_SECRET = os.environ.get("AG_CLIENT_SECRET", "")  # or AG_CLIENT_SECRET in ~/.b70-dashboard-env


def load_env():
    out = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def req_json(url, data=None, headers=None, timeout=35):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = r.read().decode()
            try:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, json.loads(txt) if txt else {}
            except Exception:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, txt
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace")
        try:
            parsed = json.loads(txt) if txt else {}
        except Exception:
            parsed = txt
        return e.code, {k.lower(): v for k, v in e.headers.items()}, parsed
    except Exception as e:
        return 0, {}, {"error": {"message": str(e)}}


def fetch_codex():
    codex_main = Path.home() / ".codex" / "auth.json"
    if codex_main.exists():
        try:
            if not CODEX_AUTH.exists() or codex_main.stat().st_mtime > CODEX_AUTH.stat().st_mtime:
                CODEX_AUTH.write_text(codex_main.read_text())
        except Exception:
            pass

    if not CODEX_AUTH.exists():
        return {"ok": False, "error": "missing codex auth"}
    try:
        d = json.loads(CODEX_AUTH.read_text())
        t = d.get("tokens", d)
        access = t["access_token"]
        account = t["account_id"]
        refresh_token = t.get("refresh_token")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    client_id = None
    # Decode client_id from id_token if available
    try:
        id_token = t.get("id_token", "")
        if id_token:
            parts = id_token.split(".")
            if len(parts) == 3:
                payload = parts[1] + "=="
                import base64
                decoded = json.loads(base64.urlsafe_b64decode(payload))
                aud = decoded.get("aud")
                if isinstance(aud, list):
                    client_id = aud[0]
                elif aud:
                    client_id = aud
    except Exception:
        pass
    for attempt in range(2):
        status, _, body = req_json(CODEX_USAGE_URL, headers={
            "Authorization": f"Bearer {access}",
            "ChatGPT-Account-Id": account,
            "Accept": "application/json",
            "User-Agent": "B70-Dashboard/1.0",
        })
        if status == 200 and isinstance(body, dict):
            break
        if attempt == 0 and status in (401, 403) and refresh_token and client_id:
            try:
                token_req = urllib.request.Request(CODEX_TOKEN_URL,
                    data=json.dumps({
                        "client_id": client_id,
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                    }).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(token_req, timeout=15) as r:
                    fresh = json.loads(r.read().decode())
                access = fresh.get("access_token", access)
                if fresh.get("refresh_token"):
                    refresh_token = fresh["refresh_token"]
                # Persist new tokens
                t["access_token"] = access
                t["refresh_token"] = refresh_token
                d["tokens"] = t
                d["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                CODEX_AUTH.write_text(json.dumps(d, indent=2))
                continue
            except urllib.error.HTTPError as e:
                err_body = e.read().decode(errors="replace")[:200]
                return {"ok": False, "error": f"token refresh {e.code}", "needs_login": True}
            except Exception as e:
                return {"ok": False, "error": f"token refresh: {e}", "needs_login": True}
        return {"ok": False, "error": str(body)[:200]}
    rl = body.get("rate_limit") or {}
    primary = rl.get("primary_window") or {}
    secondary = rl.get("secondary_window") or {}
    p_used = float(primary.get("used_percent") or 0)
    w_used = float(secondary.get("used_percent") or 0)
    return {
        "ok": True,
        "plan_type": body.get("plan_type"),
        "primary_remaining_percent": max(0.0, 100.0 - p_used),
        "primary_used_percent": p_used,
        "primary_reset_after_seconds": primary.get("reset_after_seconds"),
        "weekly_remaining_percent": max(0.0, 100.0 - w_used),
        "weekly_used_percent": w_used,
        "weekly_reset_after_seconds": secondary.get("reset_after_seconds"),
    }


def fetch_antigravity():
    if not AG_AUTH.exists():
        return {"ok": False, "error": "missing AG auth"}
    try:
        d = json.loads(AG_AUTH.read_text())
        refresh = d["refresh_token"]
        project = d.get("project_id")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    form = urllib.parse.urlencode({
        "client_id": AG_CLIENT_ID,
        "client_secret": AG_CLIENT_SECRET or load_env().get("AG_CLIENT_SECRET", ""),
        "refresh_token": refresh,
        "grant_type": "refresh_token",
    }).encode()
    try:
        req = urllib.request.Request(GOOGLE_TOKEN_URL, data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=30) as r:
            token_data = json.loads(r.read().decode())
        access = token_data["access_token"]
    except Exception as e:
        return {"ok": False, "error": str(e)}
    headers = {"Authorization": f"Bearer {access}", "Content-Type": "application/json",
               "User-Agent": "antigravity"}
    status, _, qm = req_json(f"{CLOUDCODE_BASE}/v1internal:fetchAvailableModels",
        data={"project": project} if project else {}, headers=headers)
    if status != 200 or not isinstance(qm, dict):
        return {"ok": False, "error": str(qm)[:200]}
    models_raw = qm.get("models") or {}
    # Group models by api provider for display
    groups = {}
    for key, info in models_raw.items():
        prov = info.get("apiProvider", "unknown")
        if prov not in groups:
            groups[prov] = []
        qi = info.get("quotaInfo") or {}
        rem = qi.get("remainingFraction")
        has_quota = rem is not None
        try:
            remf = float(rem) if rem is not None else None
        except Exception:
            remf = None
        groups[prov].append({
            "id": key,
            "display_name": info.get("displayName", key),
            "remaining_fraction": remf,
            "used_percent": round((1 - remf) * 100, 1) if remf is not None else None,
            "reset_time": qi.get("resetTime"),
            "has_quota": has_quota,
        })
    return {"ok": True, "groups": groups}


XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"


def _refresh_xai_token(refresh_token):
    """Refresh xAI OAuth access token."""
    form = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": XAI_OAUTH_CLIENT_ID,
        "refresh_token": refresh_token,
    }).encode()
    req = urllib.request.Request(XAI_TOKEN_URL, data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def fetch_xai():
    if not HERMES_AUTH.exists():
        return {"ok": False, "error": "missing hermes auth"}
    try:
        d = json.loads(HERMES_AUTH.read_text())
        cred = d["credential_pool"]["xai-oauth"][0]
        refresh_token = cred["refresh_token"]
        access = cred["access_token"]
    except Exception as e:
        fb = _codexbar_grok()
        if fb:
            return fb
        return {"ok": False, "error": f"no xai credentials: {e}"}
    # Try access token first, refresh if it fails
    for attempt in range(2):
        status, headers, body = req_json(XAI_URL, data={
            "model": "grok-4.20-0309-reasoning",
            "messages": [{"role": "user", "content": "Reply exactly OK."}],
            "max_tokens": 8,
        }, headers={"Authorization": f"Bearer {access}", "Content-Type": "application/json"})
        if status == 200:
            break
        if attempt == 0 and status in (401, 403):
            # Token expired, try refresh
            try:
                refreshed = _refresh_xai_token(refresh_token)
                access = refreshed["access_token"]
                # Update the auth file so future runs use the fresh token
                cred["access_token"] = access
                if refreshed.get("refresh_token"):
                    cred["refresh_token"] = refreshed["refresh_token"]
                    refresh_token = refreshed["refresh_token"]
                d["credential_pool"]["xai-oauth"][0] = cred
                HERMES_AUTH.write_text(json.dumps(d))
                continue
            except Exception as e:
                fb = _codexbar_grok()
                if fb:
                    return fb
                return {"ok": False, "error": f"token refresh failed: {e}"}
    rem_req = headers.get("x-ratelimit-remaining-requests")
    lim_req = headers.get("x-ratelimit-limit-requests")
    rem_tok = headers.get("x-ratelimit-remaining-tokens")
    lim_tok = headers.get("x-ratelimit-limit-tokens")
    reset_req = headers.get("x-ratelimit-reset-requests")
    reset_tok = headers.get("x-ratelimit-reset-tokens")
    ok = status == 200 and isinstance(body, dict) and "choices" in body
    if not ok:
        fb = _codexbar_grok()
        if fb:
            return fb
        err = body.get("error") if isinstance(body, dict) else body
        if isinstance(err, dict):
            err = err.get("message") or err.get("code") or str(err)
        return {"ok": False, "error": f"HTTP {status}: {str(err)[:160]}",
                "remaining_requests": None, "limit_requests": None,
                "remaining_tokens": None, "limit_tokens": None,
                "reset_requests_sec": None, "reset_tokens_sec": None}
    def parse_reset(v):
        if not v:
            return None
        import re as _re
        v = str(v).strip()
        if v.replace(".", "").replace("-", "").isdigit():
            return float(v)
        total = 0
        m = _re.match(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", v)
        if m:
            h, mn, s = m.groups()
            total = (int(h or 0) * 3600) + (int(mn or 0) * 60) + int(s or 0)
        return total if total else None
    return {
        "ok": ok,
        "remaining_requests": int(str(rem_req)) if str(rem_req).isdigit() else None,
        "limit_requests": int(str(lim_req)) if str(lim_req).isdigit() else None,
        "remaining_tokens": int(str(rem_tok)) if str(rem_tok).isdigit() else None,
        "limit_tokens": int(str(lim_tok)) if str(lim_tok).isdigit() else None,
        "reset_requests_sec": parse_reset(reset_req),
        "reset_tokens_sec": parse_reset(reset_tok),
    }


def fetch_zai(env):
    key = env.get("ZAI_API_KEY")
    if not key:
        return {"ok": False, "error": "missing ZAI_API_KEY"}
    status, _, body = req_json(ZAI_QUOTA_URL, headers={
        "Authorization": key, "Accept-Language": "en-US,en",
        "Content-Type": "application/json",
    })
    if status != 200 or not isinstance(body, dict) or not body.get("success"):
        return {"ok": False, "error": str(body)[:200]}
    data = body.get("data") or {}
    windows = []
    web_windows = []
    for item in (data.get("limits") or []):
        if item.get("type") == "TOKENS_LIMIT":
            pct = float(item.get("percentage") or 0)
            windows.append({
                "used_percent": pct,
                "remaining_percent": max(0.0, 100.0 - pct),
                "unit": item.get("unit"),
                "next_reset_ms": item.get("nextResetTime"),
            })
        elif item.get("type") == "TIME_LIMIT":
            # Web/search tool requests (search-prime, web-reader, zread)
            pct = float(item.get("percentage") or 0)
            web_windows.append({
                "used_percent": pct,
                "usage": item.get("currentValue"),
                "limit": item.get("usage"),
                "unit": item.get("unit"),
                "next_reset_ms": item.get("nextResetTime"),
            })
    return {"ok": True, "level": data.get("level"), "token_windows": windows,
            "web_windows": web_windows}


def _codexbar_grok():
    """X Premium / Grok quota via codexbar's grok-cli-proxy source. Fallback for
    when the xai-oauth credentials in hermes auth are expired."""
    if not CODEXBAR.exists():
        return None
    try:
        out = subprocess.run(
            [str(CODEXBAR), "usage", "--provider", "grok", "--format", "json"],
            capture_output=True, text=True, timeout=60)
        arr = json.loads(out.stdout or "[]")
    except Exception:
        return None
    e0 = arr[0] if isinstance(arr, list) and arr else {}
    if e0.get("error"):
        return None
    u = e0.get("usage") or {}
    primary = u.get("primary") or {}
    if primary.get("usedPercent") is None:
        return None
    return {"ok": True, "source": "codexbar",
            "weekly_used_percent": primary.get("usedPercent"),
            "weekly_resets_at": primary.get("resetsAt"),
            "account": u.get("accountEmail")}


def _opencode_provider_key(provider):
    """apiKey for a provider entry inside ~/.config/opencode/opencode.json."""
    try:
        oc = json.loads(OPENCODE_CFG.read_text())
        return (oc.get("provider", {}).get(provider, {}).get("options") or {}).get("apiKey")
    except Exception:
        return None


def fetch_opencodego():
    """OpenCode Go quota via codexbar's api source (5h/weekly/monthly windows)."""
    key = _opencode_provider_key("opencode-go")
    if not key:
        return {"ok": False, "error": "missing opencode-go apiKey"}
    if not CODEXBAR.exists():
        return {"ok": False, "error": "codexbar not installed"}
    try:
        out = subprocess.run(
            [str(CODEXBAR), "usage", "--provider", "opencodego",
             "--source", "api", "--format", "json"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "OPENCODE_API_KEY": key})
        arr = json.loads(out.stdout or "[]")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    e0 = arr[0] if isinstance(arr, list) and arr else {}
    if e0.get("error"):
        return {"ok": False, "error": str(e0["error"])[:200]}
    u = e0.get("usage") or {}
    def win(name):
        w = u.get(name) or {}
        return {"used_percent": w.get("usedPercent"), "resets_at": w.get("resetsAt")}
    return {"ok": True, "primary": win("primary"), "secondary": win("secondary"),
            "tertiary": win("tertiary")}


def fetch_mimo():
    """Xiaomi MiMo token-plan: no quota API exists — report API liveness only."""
    key = _opencode_provider_key("xiaomi-mimo")
    if not key:
        return {"ok": False, "error": "missing mimo apiKey"}
    status, _, body = req_json(f"{MIMO_BASE}/models",
        headers={"Authorization": f"Bearer {key}"})
    if status == 200 and isinstance(body, dict):
        return {"ok": True, "models": len(body.get("data") or [])}
    return {"ok": False, "error": f"HTTP {status}" if status else str(body)[:200]}


def main():
    env = load_env()
    quota = {}
    try:
        quota["codex"] = fetch_codex()
    except Exception as e:
        quota["codex"] = {"ok": False, "error": str(e)}
    try:
        quota["antigravity"] = fetch_antigravity()
    except Exception as e:
        quota["antigravity"] = {"ok": False, "error": str(e)}
    try:
        quota["xai"] = fetch_xai()
    except Exception as e:
        quota["xai"] = {"ok": False, "error": str(e)}
    try:
        quota["zai"] = fetch_zai(env)
    except Exception as e:
        quota["zai"] = {"ok": False, "error": str(e)}
    try:
        quota["opencodego"] = fetch_opencodego()
    except Exception as e:
        quota["opencodego"] = {"ok": False, "error": str(e)}
    try:
        quota["mimo"] = fetch_mimo()
    except Exception as e:
        quota["mimo"] = {"ok": False, "error": str(e)}
    state = {"updated_at": int(time.time()), "quota": quota}
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"Quota state written: {STATE_FILE}")


if __name__ == "__main__":
    main()
