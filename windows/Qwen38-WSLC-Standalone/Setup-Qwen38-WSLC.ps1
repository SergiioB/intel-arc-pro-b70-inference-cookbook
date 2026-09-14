[CmdletBinding()]
param(
    [string]$ModelDirectory = (Join-Path $PSScriptRoot "models\Qwen3.8-27B-GPTQ-Int4"),
    [string]$HuggingFaceToken = $env:HF_TOKEN,
    [switch]$SkipModelDownload,
    [switch]$SkipServerStart,
    [switch]$RequireHuggingFaceToken,
    [switch]$RemoveStaleLock
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Common.ps1")
$image = "qwen38-b70-friendly:2026.08.19"
Write-QwenBanner "Qwen3.8-27B reproducible WSLC setup"
Write-Host "Windows container devised and tested by Ian Hudson - aitesthive.com" -ForegroundColor Cyan
Write-Host "This builds the exact WSLC comparison profile used for the Microsoft performance report."
Write-HuggingFaceTokenStatus -Token $HuggingFaceToken -RequireToken:$RequireHuggingFaceToken

Write-QwenStep "[1/4] Build pinned WSLC image" "The base image, patches and versions are pinned for repeatability."
& (Join-Path $PSScriptRoot "Build-Qwen38Image.ps1") -Image $image
Write-QwenStep "[2/4] Verify Arc Pro B70" "Runs a real PyTorch XPU allocation and compute probe."
& (Join-Path $PSScriptRoot "Test-B70Gpu.ps1") -Image $image
if (-not $SkipModelDownload) {
    Write-QwenStep "[3/4] Supply the model" "Existing files are reused; otherwise approximately 18.2 GiB is downloaded."
    & (Join-Path $PSScriptRoot "Download-Qwen38Model.ps1") -Image $image -ModelDirectory $ModelDirectory -HuggingFaceToken $HuggingFaceToken -TokenStatusAlreadyShown -RemoveStaleLock:$RemoveStaleLock
} else { Write-Host "[3/4] Model download skipped." -ForegroundColor Yellow }
if (-not $SkipServerStart) {
    Write-QwenStep "[4/4] Start comparable WSLC server" "MTP4, 100K context, automatic tools (qwen3_coder), XPU graphs, FP8 KV cache and explicit 4.25 GiB cache."
    & (Join-Path $PSScriptRoot "Start-Qwen38.ps1") -Image $image -ModelDirectory $ModelDirectory -MtpTokens 4 -MaxModelLength 100000 -GpuMemoryUtilization 0.75 -KvCacheMemoryGiB 4.25 -XpuGraphs On -ExpandableSegments On -KvCacheDtype fp8 -MaxNumSeqs 1
} else { Write-Host "[4/4] Server start skipped." }
Write-QwenBanner "WSLC setup complete"
if (-not $SkipServerStart) { Write-Host "OpenAI-compatible endpoint: http://127.0.0.1:8000/v1" -ForegroundColor Green; Write-Host "Model name: qwen38" }
