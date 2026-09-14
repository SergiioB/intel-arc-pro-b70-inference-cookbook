# Qwen3.8-27B on Docker Desktop

> Vendored from `Qwen38-Docker-Standalone-2026.08.18.zip`, then the
> 2026.08.19 overlay (draft-INT4 S+M1 + mixed-split v5, prefix cache on).
> See [WINDOWS-STANDALONE.md](../../docs/qwen38-27/WINDOWS-STANDALONE.md)
> for the upgrade steps, provenance and measured results.

Windows container devised and tested by Ian Hudson - aitesthive.com

This folder provides the Docker Desktop entry points for the proven Intel Arc
Pro B70 configuration.

## Requirements

- Windows 11 and Docker Desktop using its Linux/WSL2 engine
- Intel Arc Pro B70 with a current Windows graphics driver
- At least 22 GiB free for the model, plus space for the container image
- An ordinary PowerShell window; administrator mode is not normally required
- Optional `HF_TOKEN` for faster, authenticated Hugging Face downloads

## Step 1: Install Docker Desktop manually

Download Docker Desktop for Windows only from Docker's official page:

<https://docs.docker.com/desktop/setup/install/windows-install/>

Install it, choose the Linux/WSL 2 engine, start Docker Desktop and wait until
it reports that the engine is running. In a new ordinary PowerShell window,
check:

```powershell
docker context use desktop-linux
docker info
```

If `docker info` reports permission denied, an administrator can add the Windows
user to Docker's local access group:

```powershell
net localgroup docker-users "$env:USERNAME" /add
```

Sign out of Windows and sign back in—or reboot—after changing group membership.
Then start Docker Desktop and run `docker info` again.

## Getting and setting a Hugging Face token

The model is public, but an authenticated download normally receives higher
rate limits. Hugging Face calls this an **access token** rather than an API key.

1. Sign in at <https://huggingface.co/>.
2. Open <https://huggingface.co/settings/tokens>.
3. Select **Create new token** and choose a **Read** token. No write permission
   is required to download this model.
4. Copy the token when it is displayed. It normally begins with `hf_`.

Set it for only the current PowerShell window:

```powershell
$env:HF_TOKEN = "hf_your_read_token_here"
```

Or save it permanently for your Windows user account:

```powershell
[Environment]::SetEnvironmentVariable(
    "HF_TOKEN",
    "hf_your_read_token_here",
    "User"
)
```

A permanent environment-variable change is visible to **new** processes. Close
and reopen PowerShell before running setup, or also set `$env:HF_TOKEN` in the
current window. Verify presence without printing the secret:

```powershell
if ($env:HF_TOKEN) { "HF_TOKEN is set" } else { "HF_TOKEN is not set" }
```

Treat the token like a password. Do not paste it into screenshots, bug reports,
Git commits or chat messages. Revoke and replace it from the Hugging Face token
page if it is exposed.

## Complete installation

Extract the complete ZIP, start Docker Desktop, then open PowerShell in the
extracted root folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\Setup-Qwen38-Docker.ps1
```

Already running the 2026.08.18 kit? Do **not** re-download the model. Rebuild
and recreate:

```powershell
.\Upgrade-Qwen38-Docker.ps1
```

That turns on draft-INT4 and prefix cache. Restarting the old container is
not an upgrade.

Git for Windows may check out this folder with CRLF. The build script rewrites
the pinned patches, Dockerfile, and container scripts to LF before hashing and
`docker build`, so SHA-256 checks and the Linux `start.sh` shebang still work.
The image Dockerfile also strips CR as a last line of defense.

The installer verifies Docker, builds the pinned image, mounts the minimum WSL
GPU interfaces (`/dev/dxg`, `/usr/lib/wsl/lib` and `/usr/lib/wsl/drivers`), runs
a real XPU calculation, downloads the model if necessary and starts the server.
The container does not require `--privileged`.

The model can instead be copied manually into the shared directory:

```text
.\models\Qwen3.8-27B-GPTQ-Int4
```

Copy the repository contents directly into that folder. It must contain
`config.json`, all five safetensor shards, the shard index and tokenizer files.

## Main controls

```powershell
.\Start-Qwen38-Docker.ps1              # prefix cache on, draft-INT4 on
.\Start-Qwen38-Docker.ps1 -Recreate -DraftInt4 0     # recommended for long multi-hour/agentic sessions (stable 100% acceptance)
.\Start-Qwen38-Docker.ps1 -Recreate    # after an image rebuild
.\Start-Qwen38-Docker.ps1 -Recreate -PrefixCache 0   # cold decode test
.\Stop-Qwen38-Docker.ps1
```

## Other controls

```powershell
.\Build-Qwen38Image-Docker.ps1
.\Upgrade-Qwen38-Docker.ps1
.\Test-CookbookDecode.ps1
```

- Endpoint: `http://127.0.0.1:8000/v1`
- Model name: `qwen38`
- Tools: enabled with the `qwen3_coder` parser
- Vision: disabled
- Profile: MTP4, 100K context, FP8 KV cache, explicit 4.30 GiB cache, prefix cache on

> [!TIP]
> **Long-running stability:** If running continuous 24/7 or agentic sessions over hours, use `-DraftInt4 0` (BF16 draft) to maintain consistent speculative acceptance. If degradation occurs after hours of continuous runtime, restart the container (`.\Stop-Qwen38-Docker.ps1` followed by `.\Start-Qwen38-Docker.ps1`) to clear Level Zero runtime buffers.

The first image build and model download can take a long time. Model loading is
also several minutes because the 18.2 GiB checkpoint is read from the Windows
bind mount. Keep the stopped container for a faster daily restart. After an
**image** upgrade, use `-Recreate` (or `Upgrade-Qwen38-Docker.ps1`) so the new
entrypoint actually runs.
