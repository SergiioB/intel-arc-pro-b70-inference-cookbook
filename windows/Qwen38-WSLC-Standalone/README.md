# Qwen3.8-27B on Microsoft WSLC

> Vendored from `Qwen38-WSLC-Standalone-2026.08.18.zip` (devised and tested by
> Ian Hudson — aitesthive.com), then the 2026.08.19 overlay (draft-INT4 S+M1
> + mixed-split v5, prefix cache on). See
> [WINDOWS-STANDALONE.md](../../docs/qwen38-27/WINDOWS-STANDALONE.md)
> for the upgrade steps, provenance and measured results.

This folder provides the Microsoft WSLC entry points for the reproducible Intel
Arc Pro B70 configuration.

## Requirements

- Windows 11 with Microsoft WSLC installed and working
- Intel Arc Pro B70 with a current Windows graphics driver
- At least 22 GiB free for the model, plus space for the container image
- An ordinary PowerShell window; administrator mode is not normally required
- Optional `HF_TOKEN` for faster, authenticated Hugging Face downloads

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

Extract the complete ZIP, then open PowerShell in the extracted root folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\Setup-Qwen38-WSLC.ps1
```

Already on 2026.08.18? Rebuild without re-downloading the model:

```powershell
.\Upgrade-Qwen38-WSLC.ps1
```

Git for Windows may check out this folder with CRLF. The build script rewrites
downloaded patches, the Dockerfile, and container scripts to LF before hashing
and `wslc build`, so SHA-256 checks and the Linux `start.sh` shebang still work.

The installer builds the pinned image, verifies a real XPU calculation,
downloads the approximately 18.2 GiB model when it is not already present, and
starts the server. Partial downloads are retained for resumption.

The model can instead be copied manually into the shared directory:

```text
.\models\Qwen3.8-27B-GPTQ-Int4
```

Copy the repository contents directly into that folder. It must contain
`config.json`, all five safetensor shards, the shard index and tokenizer files.

## Main controls

```powershell
.\Start-Qwen38.ps1 -MtpTokens 4 -MaxModelLength 100000 -GpuMemoryUtilization 0.75 -KvCacheMemoryGiB 4.3 -KvCacheDtype fp8
.\Start-Qwen38.ps1 -MtpTokens 4 -MaxModelLength 100000 -GpuMemoryUtilization 0.75 -KvCacheMemoryGiB 4.3 -KvCacheDtype fp8 -PrefixCache 0
.\Start-Qwen38.ps1 -MtpTokens 4 -MaxModelLength 100000 -GpuMemoryUtilization 0.75 -KvCacheMemoryGiB 4.3 -KvCacheDtype fp8 -DraftInt4 0
.\Stop-Qwen38.ps1
```

## Other controls

```powershell
.\Upgrade-Qwen38-WSLC.ps1
.\Test-CookbookDecode.ps1
```

- Endpoint: `http://127.0.0.1:8000/v1`
- Model name: `qwen38`
- Tools: automatic tool choice enabled with the `qwen3_coder` parser
- Vision: disabled
- Profile: MTP4, 100K context, FP8 KV cache, explicit 4.30 GiB cache,
  draft-INT4 overlay on, prefix cache on

Startup and model download can take several minutes. The setup script prints
progress and warns immediately if `HF_TOKEN` is unavailable.

For Microsoft performance-report details, see
`.\WSLC-Performance-Bug-Report.md`.
