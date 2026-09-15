<img src="../assets/windows-b70-badge.svg" alt="Windows 11 on Intel Arc Pro B70 — Qwen3.8-27B via WSLC / Docker Desktop with MTP4" width="640">

# Windows 11 hosts — Qwen3.8-27B on the Arc Pro B70

Everything in this cookbook runs on Linux. This page is the Windows reference:
the same Qwen3.8-27B GPTQ-INT4 + MTP4 profile, packaged as two standalone
PowerShell kits for **Windows 11**, devised and end-to-end tested by
**Ian Hudson (aitesthive.com)** on 2026-08-18 — in his words, *"I've tested all
the steps, it's 2 days work."* Both kits are vendored unchanged-plus-fixes
under [`windows/`](../../windows/) in this repo.

| <img src="../assets/windows-11-logo.svg" alt="" width="20"> Path | What it is | Status on the B70 |
|---|---|---|
| **Docker Desktop** (Linux/WSL2 engine) | `windows/Qwen38-Docker-Standalone/` — GUI installs Docker, scripts do the rest | **Proven.** ~70 tok/s class decode, stable TTFT (self-reported, below) |
| **Microsoft WSLC** (`wslc.exe` containers) | `windows/Qwen38-WSLC-Standalone/` — no Docker Desktop needed | **Experimental.** Works, but 2.4–2.8× slower; Microsoft bug report prepared (below) |

**TL;DR for a Windows friend with one B70:** install Docker Desktop, extract
the Docker kit, run `.\Setup-Qwen38-Docker.ps1`, wait for the tea-length model
load, point any OpenAI client at `http://127.0.0.1:8000/v1` (model `qwen38`).

**Already on the 2026.08.18 kit?** Jump to
[Upgrade from 2026.08.18](#upgrade-from-20260818--draft-int4--prefix-cache).
You do not re-download the model. You rebuild the image, recreate the
container, and prefix cache stays on for real chat.

---

## The single-B70 memory rule (why the kits refuse to fill the card)

On the Linux bench host the B70 is headless: vLLM may take the whole card. On a
typical Windows machine the **B70 also drives the display**, and that display
memory comes out of the same 32 GB — a 4K desktop (and Docker Desktop's WSL VM,
which can reserve GPU-backed memory of its own) can reserve several GiB before
vLLM starts. Ian hit this directly:

> "I've made the script leave enough memory over for people that own just a
> single B70 as some memory will be used for their display, I noticed on my 4K
> display Docker was potentially reserving more memory than it should."

The measured failure mode (from his WSLC/Docker A/B report): letting vLLM
auto-allocate at `gpu-memory-utilization=0.80` built a **22.07 GiB KV cache**
(536,842-token capacity) and decode collapsed to **25.73 tok/s** while the
desktop starved. Pinning the cache explicitly restored ~70 tok/s.

So both kits pin, instead of negotiating:

| Lever | Value | Effect |
|---|---|---|
| `GPU_MEMORY_UTILIZATION` | **0.75** | Leaves ~8 GiB of the card for Windows, the display server and Docker Desktop's WSL VM |
| `--kv-cache-memory-bytes` | **4617089843** (exactly 4.30 GiB) | No auto-sizing surprise; reported capacity 102,631 tokens at 100K context (1.03× concurrency). Bumped from 4.25 GiB on 2026-08-22: some Windows hosts need 4.26 GiB for 100K (mamba page alignment + padding layers) and failed boot by ~10 MiB (issue #4) |
| `--kv-cache-dtype` | **fp8** | Same requirement as Linux dense 27B — fp16 KV does not fit |
| `MAX_NUM_SEQS` | **1** | Single-user desktop serving; this is **not** the concurrent Cn profile |

This is the Windows analog of the Linux VRAM gates: on a display-attached GPU,
never let the engine size its cache to the card. It is also why the Windows
profile runs 100K context rather than the Linux 131,072.

## Quick start

Both kits need: Windows 11, a current Intel Arc driver, ≥22 GiB free disk for
the ~18.2 GiB model (plus image space), and an ordinary PowerShell window
(admin not required). An `HF_TOKEN` read token is optional — the model is
public; it only improves download rate limits. The complete authored steps are
below; each kit's `README.md` / `README.html` is the canonical copy.

**Docker Desktop (recommended):** install Docker Desktop from
<https://docs.docker.com/desktop/setup/install/windows-install/>, select the
Linux/WSL 2 engine, start it, then:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\Setup-Qwen38-Docker.ps1          # builds image, probes B70, downloads model, starts server
.\Test-CookbookDecode.ps1          # optional: exact-token decode benchmark (cookbook method)
.\Stop-Qwen38-Docker.ps1           # stops server, keeps container + compiled-graph cache
```

**WSLC (experimental):**

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\Setup-Qwen38-WSLC.ps1
.\Stop-Qwen38.ps1 -ReleaseGpuMemory   # terminating the WSLC session is what returns VRAM to Windows
```

Endpoint (both): `http://127.0.0.1:8000/v1`, served model name `qwen38`.
Tool calling is on (`qwen3_coder` parser), vision is off, thinking is on with
a low default reasoning effort (template below).

## Upgrade from 2026.08.18 — draft INT4 + prefix cache

The 18 August kit already works. This is the next image, not a new install.

What changed in **2026.08.19**:

| Piece | 2026.08.18 (what you have) | 2026.08.19 (rebuild this) |
|---|---|---|
| Patches | MTP nightly + MTP boundary | those two, plus mixed-split v5, plus draft-INT4 S+M1 |
| Draft head | BF16 (`B70_MTP_BF16_DRAFT=1`) | still that checkpoint; the **draft** is requantized to INT4 at start (`DRAFT_INT4=1`) |
| Prefix cache | off | **on** (real multi-turn sessions). Turn it off only for a cold decode test |
| Sampling | template only | `--generation-config auto` (Qwen thinking / non-thinking defaults) |
| Image tag | `qwen38-b70-docker:2026.08.18` / `qwen38-b70-friendly:2026.08.18` | `…:2026.08.19` |
| Display-safe VRAM | 0.75 + 4.25 GiB fp8 KV, 100K, MTP4, C1 | same pin; KV default later bumped 4.25 → 4.30 GiB for headroom (see [Troubleshooting](#troubleshooting-windows)) |

Linux, same overlay, C1, n=5, cache **off**: **112.65** vs matched BF16-draft **81.20** tok/s at p512/g128. Prefill is flat. Quality KEEP on the 12/15 coding gate. That is a Linux campaign, not a Windows re-measure. On Windows expect “noticeably faster than your ~70”, not a copied 112.

Prefix cache is for **follow-up turns**, not a first-token speed-up. Leave it on for Pi / Open WebUI / anything with a system prompt. For `Test-CookbookDecode.ps1` (cold unique prompts) set `-PrefixCache 0` so the cell matches the Linux speed card.

**Do not** `docker start` the old container after a rebuild. Docker keeps the 18 August entrypoint until you recreate.

### Docker Desktop (recommended)

In the kit folder (`windows/Qwen38-Docker-Standalone/` if you cloned, or the unzipped kit after `git pull`):

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\Upgrade-Qwen38-Docker.ps1
```

That rebuilds `qwen38-b70-docker:2026.08.19`, **removes** `qwen38-b70-docker-test`, and starts a new container with `DRAFT_INT4=1` and prefix cache **on**. Model files on disk are reused.

Manual equivalent:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\Build-Qwen38Image-Docker.ps1
.\Start-Qwen38-Docker.ps1 -Recreate
# optional: cold decode test (prefix cache off)
.\Start-Qwen38-Docker.ps1 -Recreate -PrefixCache 0
.\Test-CookbookDecode.ps1
```

Confirm the overlay actually ran:

```powershell
docker logs qwen38-b70-docker-test | findstr /C:"draft-INT4"
```

You want: `[start] draft-INT4 S+M1 overlay ENABLED`.

Fall back to the 18 August draft (BF16, still the same model files):

```powershell
.\Start-Qwen38-Docker.ps1 -Recreate -DraftInt4 0
```

### WSLC (experimental)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\Upgrade-Qwen38-WSLC.ps1
```

WSLC already deletes the previous container on start. Confirm with `wslc logs qwen38-b70-friendly`. WSLC was ~26 tok/s on 18 August; this overlay does not fix that runtime.

### Fresh install (never ran the 18 August kit)

Same as before: `.\Setup-Qwen38-Docker.ps1`. Setup now builds **2026.08.19** with the overlay and prefix cache on. You still install Docker Desktop first.

---

## Full step-by-step (as Ian authored and tested it)

Transcribed from the kit READMEs so this page is self-contained. Work in the
kit folder you downloaded (or `windows/Qwen38-*-Standalone/` from this repo).

### Step 0 — shared preparation (both paths)

1. **Open PowerShell in the extracted kit folder** (an ordinary window;
   administrator mode is not normally required).
2. **Optional: a Hugging Face read token** (higher rate limits; the model
   itself is public):
   1. Sign in at <https://huggingface.co/>, open
      <https://huggingface.co/settings/tokens>, **Create new token** → type
      **Read**. It begins with `hf_`.
   2. Set it for this window only:
      `$env:HF_TOKEN = "hf_your_read_token_here"` — or permanently:
      `[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_your_read_token_here", "User")`
      (permanent changes are visible to **new** processes — reopen PowerShell).
   3. Verify without printing it:
      `if ($env:HF_TOKEN) { "HF_TOKEN is set" } else { "HF_TOKEN is not set" }`
   4. Treat it like a password; revoke and replace it if it is ever exposed.
3. **Optional: place the model manually** instead of downloading — copy the
   repository contents into the kit's
   `.\models\Qwen3.8-27B-GPTQ-Int4\` folder. It must contain `config.json`,
   all five `model-0000x-of-00005.safetensors` shards, the shard index and
   tokenizer files.

### Path A — Docker Desktop (recommended)

1. **Install Docker Desktop manually** from
   <https://docs.docker.com/desktop/setup/install/windows-install/> only.
   Choose the **Linux/WSL 2** engine, start Docker Desktop and wait until it
   reports the engine is running.
2. **Check access** in a new ordinary PowerShell window:

   ```powershell
   docker context use desktop-linux
   docker info
   ```

   If `docker info` reports permission denied, an administrator can add your
   Windows user to Docker's access group:

   ```powershell
   net localgroup docker-users "$env:USERNAME" /add
   ```

   Then **sign out and back in (or reboot)**, start Docker Desktop, and run
   `docker info` again.
3. **Run the installer** (start Docker Desktop first):

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\Setup-Qwen38-Docker.ps1
   ```

   It verifies Docker, builds the pinned image, mounts the minimum WSL GPU
   interfaces (`/dev/dxg`, `/usr/lib/wsl/lib`, `/usr/lib/wsl/drivers`,
   read-only; **no `--privileged`**), runs a real XPU calculation, downloads
   the model if necessary, and starts the server. Partial downloads are
   retained for resumption.
4. **Use it.** Endpoint `http://127.0.0.1:8000/v1`, model name `qwen38`,
   automatic tool choice with the `qwen3_coder` parser, vision disabled.
5. **Daily controls:**

   ```powershell
   .\Start-Qwen38-Docker.ps1     # starts or restarts; preserves the compiled-graph cache
   .\Stop-Qwen38-Docker.ps1      # stops; keeps the stopped container for fast restart
   .\Build-Qwen38Image-Docker.ps1   # rebuild the pinned image only
   .\Test-CookbookDecode.ps1        # exact-token decode benchmark
   ```

   The first image build and model download take a long time; model loading is
   several minutes because the 18.2 GiB checkpoint is read from the Windows
   bind mount. Keep the stopped container rather than removing it to preserve
   the compiled graph cache.

### Path B — Microsoft WSLC (experimental)

1. **Prerequisites:** Windows 11 with Microsoft WSLC installed and working,
   Arc Pro B70 with a current Windows driver.
2. **Run the installer:**

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\Setup-Qwen38-WSLC.ps1
   ```

   It builds the pinned image, verifies a real XPU allocation/compute probe
   (including the B70 PCI ID `0xe223` when the driver hides the marketing
   name), downloads the model when absent, and starts the server with the
   friendly profile.
3. **Daily controls:**

   ```powershell
   .\Start-Qwen38.ps1 -MtpTokens 4 -MaxModelLength 100000 -GpuMemoryUtilization 0.75 -KvCacheMemoryGiB 4.3 -KvCacheDtype fp8
   .\Stop-Qwen38.ps1
   .\Stop-Qwen38.ps1 -ReleaseGpuMemory   # terminates the WSLC session to return ALL GPU memory to Windows
   .\Test-B70Gpu.ps1                      # standalone B70 passthrough + XPU probe
   .\Test-CookbookDecode.ps1
   ```

   Startup and model download can take several minutes; the setup script
   prints progress and warns immediately if `HF_TOKEN` is unavailable.

## What the kits pin (provenance)

| Component | Exact value |
|---|---|
| Base image | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` — the **same digest as the Linux Qwen3.8 champion row** in [IMAGE-AND-PATCH-MATRIX.md](../IMAGE-AND-PATCH-MATRIX.md) |
| vLLM / kernels in image | `0.27.2rc1.dev77+gac7509e2b` / `0.1.12.3` (verified in-image, not from the tag) |
| Patches | MTP nightly + boundary (commit `5c6b6b1`), then mixed-split v5 (`db20e00`), then draft-INT4 S+M1 (`aa363ca`). SHA-256-verified at build after LF normalization. WSLC downloads the five files hash-pinned; Docker verifies the bundled copies |
| Model | [`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16) @ revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e` (5 shards) — **same files as 2026.08.18** |
| Draft profile | **Draft-INT4 S+M1 on** (`DRAFT_INT4=1`). The checkpoint still ships a BF16 MTP head; the overlay requantizes only the draft at start. Set `DRAFT_INT4=0` to recover the 18 August BF16-draft path |
| Prefix cache | **On** for serving (`PREFIX_CACHE=1`). Set `PREFIX_CACHE=0` only for a cold unique-prompt decode test |
| Author's test machine | Windows 11 Pro 10.0.26200, WSL 2.9.4.0, Intel driver 32.0.101.8805 (2026-07-07), Docker client 29.7.2 |

Served flags (via `container/start.sh` in both kits): MTP4
(`num_speculative_tokens 4`), `--quantization gptq --dtype float16`,
`--max-model-len 100000`, `--max-num-batched-tokens 8192`,
`--enable-prefix-caching` (default; `--no-enable-prefix-caching` if
`PREFIX_CACHE=0`), `--generation-config auto`, `--language-model-only`,
`--reasoning-parser qwen3`, `--enable-auto-tool-choice --tool-call-parser
qwen3_coder`, XPU graphs on, expandable segments on.

## Line endings (Git for Windows)

The cookbook stores every text file as **LF**. Git for Windows defaults to
`core.autocrlf=true`, so a clone can rewrite working-tree files to CRLF. That
used to fail SHA-256 checks on the five pinned patches and to break
`container/start.sh` inside the Linux image (`/usr/bin/env bash^M`).

The repo now pins LF via `.gitattributes`. The build scripts also rewrite
patches, Dockerfiles, and container scripts to LF before hashing or `COPY`,
and both Dockerfiles strip CR as a last line of defense. If you cloned before
this fix, `git pull` and rebuild (`.\Upgrade-Qwen38-Docker.ps1` or
`.\Upgrade-Qwen38-WSLC.ps1`). Do not re-download the model.

## Windows-specific engineering inside the kits

- **WSLC exposes XPU compute but not Linux `/dev/dri` DRM nodes.** The kits
  force oneCCL away from drmfd IPC (`CCL_ZE_IPC_EXCHANGE=sockets` and friends)
  and ship `patch_xpu_single_gpu_warmup.py`, which skips vLLM's one-rank
  oneCCL warm-up `all_reduce` on `world_size == 1` (a no-op reduction whose
  IPC path would otherwise fail).
- **Docker needs only the minimum GPU surface:** `/dev/dxg` plus read-only
  `/usr/lib/wsl/lib` and `/usr/lib/wsl/drivers`. No `--privileged`.
- **B70 identity check tolerates Windows drivers:** `diagnose.py` accepts the
  marketing name **or** the Battlemage PCI ID `0xe223` (some Windows drivers
  expose only the device ID), then runs a real XPU allocation/compute probe
  before any model load.
- **Low-reasoning default without touching the model:** a build-time copy of
  the checkpoint's `chat_template.jinja` with the single anchor
  `reasoning_effort|default('xhigh')` → `'low'` (the kit refuses to patch if
  the anchor count is not exactly 1 — verified against the pinned revision).
- **Slow first model load is expected:** weights are read from a Windows bind
  mount through WSL's 9P file system; the Docker kit preserves the stopped
  container to keep its compiled-graph cache.

## Troubleshooting (Windows)

**Red `NativeCommandError` text while the container is actually fine.**
vLLM (and `docker`/`wslc` themselves) write progress and INFO lines to stderr,
and PowerShell with `$ErrorActionPreference = "Stop"` can promote that harmless
stderr text into a terminating `NativeCommandError`, killing the readiness loop
of older kit copies. The start and download scripts now capture native stderr
inside a `Continue` scope, so `git pull` fixes it. If you see the red text on an
old copy while `http://127.0.0.1:8000/v1/models` responds, the server is fine —
update the kit and rerun.

**Intel NEO `linear_stream.h` abort under sustained load.** With XPU graphs on
and Level Zero's default immediate command lists, long agent sessions could
abort the engine (`Abort was called at 90 line in file:
.../linear_stream.h`, then `EngineDeadError`). Both start scripts now set
`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` by default — the
reporter-verified workaround from issue #6, which stopped the aborts while
keeping graph-enabled speed (~104–106 tok/s short decode). Existing containers
keep their old env: recreate with `.\Start-Qwen38-Docker.ps1 -Recreate` (or the
WSLC twin) to pick it up.

**Throughput and MTP acceptance degrade over hours, server stays alive (Issue #6).**
In controlled continuous testing without agent harnesses (repeated fixed-shape benchmark runs over hours):
- **INT4 draft (`-DraftInt4 1`):** Experienced a gradual throughput drop followed by a severe speculative token acceptance collapse (falling from ~99% to ~59% by Cycle 5, reducing decode to ~60 tok/s).
- **BF16 draft (`-DraftInt4 0`):** Maintained **100% MTP acceptance** throughout 18 cycles, though underlying runtime state accumulation showed a gradual ~19.9% throughput drift (77.3 -> 61.9 tok/s).
- **Restoration:** A quick container restart (`.\Stop-Qwen38-Docker.ps1` followed by `.\Start-Qwen38-Docker.ps1`) immediately clears accumulated Level Zero / allocator state and completely restores 100% throughput and normal MTP acceptance.

**Recommendation for long-running / agentic workloads:**
1. Use `-DraftInt4 0` (BF16 draft) for multi-hour stability. Keep `-DraftInt4 1` for short-session / benchmark speed.
2. The default **100,000 context** with explicit **4.30 GiB FP8 KV cache** and `GPU_MEMORY_UTILIZATION=0.75` is the proven sweet spot on single-card B70 Windows setups, leaving ~8 GiB for the OS and display.
3. If running continuous 24/7 daemon tasks, schedule a periodic container restart.

## Measured results — self-reported, one Windows machine

> **Evidence class: external self-report.** These numbers are Ian's, measured
> on his Windows 11 machine with `Test-CookbookDecode.ps1` (exact-token
> prompts, streaming client timing, same method as the cookbook). They are not
> a B70-DOCS campaign and have not been independently reproduced. C1, cache
> off, MTP4, 100K, fp8 KV, 4.25 GiB explicit cache.

| Scenario | WSLC | Docker Desktop | Docker advantage |
|---|---:|---:|---:|
| p512/g128 decode, after 60 s idle | 25.76 tok/s | 69.76 tok/s | 2.71× |
| p512/g128 decode, best initial | 48.18 tok/s | 71.84 tok/s (median of 5) | 1.49× |
| p8192/g128 decode | 26.91 tok/s | 64.26 tok/s (median of 3) | 2.39× |
| p8192 time to first token | 348.82 s | 4.42–4.51 s | ~78× |

Docker Desktop consistency on the recommended config: short-prompt median
71.84 (range 71.63–72.94, TTFT 0.356–0.359 s); after idle 69.76 (69.53–71.01);
sustained g1024 68.45 / 67.08; p8192 64.42 / 64.26 / 64.08. MTP acceptance
read 100% throughout (greedy, repetitive filler prompts — the acceptance
maximum; real prompts sample lower on every measured stack).

**Directional only:** the Linux champion in the **BF16-draft** profile class
measured **83.7** tok/s C1 p512/g128 (n=5). Windows Docker at ~70 is ~15%
below that cell — different OS, driver, scheduler and file path, plus a
single-machine self-report versus a controlled campaign. Do not compute a
precise "Windows tax" from these two cells.

The Linux **draft-INT4 S+M1** overlay is a later matched n=5: **112.65** vs
**81.20** at the same p512/g128 cell (cache off). Windows 2026.08.19 ships
that overlay; it has **not** been re-measured on Ian's machine. Do not put
112.65 in a Windows table until `Test-CookbookDecode.ps1` is re-run on
2026.08.19.

The WSLC slowness is documented for Microsoft in
[`windows/Qwen38-WSLC-Standalone/WSLC-Performance-Bug-Report.md`](../../windows/Qwen38-WSLC-Standalone/WSLC-Performance-Bug-Report.md)
(same image, model, driver and benchmark on both runtimes; suspects GPU memory
placement / scheduling inside the WSLC runtime). Until Microsoft responds,
**Docker Desktop is the Windows path**.

## Vendored copies — fixes applied

The originals are Ian's zips (`Qwen38-WSLC-Standalone-2026.08.18.zip`,
`Qwen38-Docker-Standalone-2026.08.18.zip`). The vendored trees started
byte-equal to them except:

1. **`Qwen38-Docker-Standalone/Test-CookbookDecode.ps1`** — added
   `Add-Type -AssemblyName System.Net.Http`. The WSLC twin already had it;
   without it Windows PowerShell 5.1 can fail to resolve `[Net.Http.HttpClient]`
   used by the streaming benchmark.
2. **`Qwen38-WSLC-Standalone/Start-Qwen38.ps1`** — `--publish` changed from
   `${Port}:8000` (all interfaces) to `127.0.0.1:${Port}:8000`, matching the
   Docker kit and the documented endpoint. The API has no authentication and
   must not be LAN-visible by default. *(Standard docker publish syntax, but
   not re-tested on `wslc.exe` — re-run once before redistributing the zip.)*
3. Provenance header added to both kit `README.md` files (this page).

**2026.08.19 overlay** (this page's upgrade): both kits now also ship
`patch_gdn_mixed_split_v5.py`, `patch_draft_lmhead_int4.py`,
`patch_draft_mtp_int4.py`; `container/start.sh` applies them and defaults
`DRAFT_INT4=1` / `PREFIX_CACHE=1`; image tag `2026.08.19`;
`Upgrade-Qwen38-*.ps1` rebuilds and recreates so a leftover 18 August
container cannot silently keep the old entrypoint. The Microsoft bug
report is unchanged evidence.

## Files

```text
windows/
  Qwen38-WSLC-Standalone/     WSLC kit: Setup/Build/Download/Start/Stop, Test-B70Gpu,
                              Smoke-Test, Test-CookbookDecode, bug report
  Qwen38-Docker-Standalone/   Docker Desktop kit: Setup/Build/Download/Start/Stop,
                              Smoke-Test, Test-CookbookDecode
  (both) container/           start.sh, diagnose.py, oneCCL warm-up skip,
                              low-reasoning template patch
  (both) patches/             five hash-pinned files: MTP nightly, MTP boundary,
                              mixed-split v5, draft lmhead INT4, draft MTP INT4
  (both) Upgrade-Qwen38-*.ps1 rebuild image 2026.08.19 and recreate the container
  (both) models/Qwen3.8-27B-GPTQ-Int4/   PLACE_MODEL_FILES_HERE.txt (model not bundled)
```

## Credits

Windows packages devised and tested by **Ian Hudson —
[aitesthive.com](https://aitesthive.com)**, 2026-08-18. Cookbook section,
vendoring and fixes by the B70 cookbook maintainers, 2026-08-19.

Windows and the Windows logo are trademarks of the Microsoft group of
companies. The logo is used here nominatively to identify compatibility with
Microsoft Windows; this independent project is not endorsed by Microsoft.
