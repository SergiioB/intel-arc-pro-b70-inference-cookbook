[CmdletBinding()]
param(
    [string]$Image = "qwen38-b70-friendly:2026.08.19",
    [string]$ModelDirectory = (Join-Path $PSScriptRoot "models\Qwen3.8-27B-GPTQ-Int4"),
    [string]$HuggingFaceToken = $env:HF_TOKEN,
    [switch]$RequireHuggingFaceToken,
    [switch]$RemoveStaleLock,
    [switch]$TokenStatusAlreadyShown
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Common.ps1")

if (-not $TokenStatusAlreadyShown) {
    Write-QwenBanner "Qwen3.8-27B model download"
    Write-HuggingFaceTokenStatus -Token $HuggingFaceToken -RequireToken:$RequireHuggingFaceToken
}

$ModelDirectory = [System.IO.Path]::GetFullPath($ModelDirectory)
$DownloadContainer = "qwen38-friendly-model-download"
$ExpectedRevision = "9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e"
$ExpectedShards = 5

Write-Host "Model:"
Write-Host "  SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
Write-Host ""
Write-Host "Pinned revision:"
Write-Host "  $ExpectedRevision"
Write-Host ""
Write-Host "Download size: approximately 18.2 GiB"
Write-Host "Destination:   $ModelDirectory"
Write-Host ""
Write-Host "This is a large download and may take quite a while." -ForegroundColor Yellow
Write-Host "You might want to go and make a cup of tea." -ForegroundColor Cyan
Write-Host ""
Write-Host "Please leave this PowerShell window open. If interrupted, completed"
Write-Host "and partial data will be retained so Hugging Face can resume later."
Write-Host ""

New-Item -ItemType Directory -Force -Path $ModelDirectory | Out-Null
$root = [IO.Path]::GetPathRoot($ModelDirectory)
$driveName = $root.TrimEnd('\').TrimEnd(':')
$drive = Get-PSDrive -Name $driveName -ErrorAction Stop
$freeGiB = [Math]::Round($drive.Free / 1GB, 1)
Write-Host "[Storage] $freeGiB GiB available on $root"
if ($drive.Free -lt 22GB) { throw "At least 22 GiB of free space is recommended before downloading the model." }
Write-Host "[Storage] Sufficient free space detected." -ForegroundColor Green

$configPath = Join-Path $ModelDirectory "config.json"
$shards = @(Get-ChildItem -LiteralPath $ModelDirectory -Filter "model-*-of-00005.safetensors" -File -ErrorAction SilentlyContinue)
if ((Test-Path -LiteralPath $configPath) -and $shards.Count -eq $ExpectedShards) {
    Write-Host "[Download] All five model shards and config.json are already present." -ForegroundColor Green
    Write-Host "[Download] Nothing needs to be downloaded."
    return
}

# WSLC answers "not found" on stderr; capture in a Continue scope so
# PowerShell cannot promote it into a terminating NativeCommandError (issue #5).
$listText = & { $ErrorActionPreference = "Continue"; (& wslc.exe list --all --no-trunc 2>&1 | Out-String) }
if ($listText -match [regex]::Escape($DownloadContainer)) {
    $downloadInspect = (& { $ErrorActionPreference = "Continue"; (& wslc.exe inspect $DownloadContainer 2>&1 | Out-String) } | ConvertFrom-Json)[0]
    if ($downloadInspect.State.Running) {
        Write-Host "[Download] Another Qwen3.8 download container is already running." -ForegroundColor Yellow
        Write-Host "Container: $DownloadContainer"
        Write-Host "No second downloader will be started because both processes would compete for the same locks."
        throw "An existing model download is already active."
    }
    Write-Host "[Cleanup] Removing an exited download container left by an earlier attempt..."
    & wslc.exe remove --force $DownloadContainer | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not remove the exited $DownloadContainer container." }
}

$partialFiles = @(Get-ChildItem -LiteralPath $ModelDirectory -Recurse -Filter "*.incomplete" -File -ErrorAction SilentlyContinue)
if ($partialFiles.Count -gt 0) {
    Write-Host "[Resume] Found $($partialFiles.Count) partial download file(s)." -ForegroundColor Cyan
    Write-Host "[Resume] They will be retained so Hugging Face can continue the transfer."
}

$lockFiles = @(Get-ChildItem -LiteralPath $ModelDirectory -Recurse -Filter "*.lock" -File -ErrorAction SilentlyContinue)
if ($lockFiles.Count -gt 0) {
    Write-Host "[Lock] Found $($lockFiles.Count) Hugging Face lock file(s)." -ForegroundColor Yellow
    if (-not $RemoveStaleLock) {
        Write-Host "No fixed-name download container is running, so these may be stale."
        $lockFiles.FullName | ForEach-Object { Write-Host "  $_" }
        Write-Host ""
        Write-Host "After confirming no legacy downloader is running, rerun with:"
        Write-Host "  .\Download-Qwen38Model.ps1 -RemoveStaleLock" -ForegroundColor White
        throw "Stale lock removal requires explicit -RemoveStaleLock approval."
    }
    foreach ($lockFile in $lockFiles) {
        Remove-Item -Force -LiteralPath $lockFile.FullName
        Write-Host "[Lock] Removed stale lock: $($lockFile.FullName)"
    }
}

$envArgs = @()
if ($HuggingFaceToken) { $envArgs = @("--env", "HF_TOKEN=$HuggingFaceToken") }
$download = @(
    "run", "--rm", "--name", $DownloadContainer,
    "--volume", "${ModelDirectory}:/download"
) + $envArgs + @(
    "--entrypoint", "/bin/bash", $Image, "-lc",
    "set -e; if command -v hf >/dev/null 2>&1; then hf download SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 --revision $ExpectedRevision --local-dir /download; else huggingface-cli download SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 --revision $ExpectedRevision --local-dir /download; fi"
)

Write-Host "[Download] Transfer started at $(Get-Date -Format T)."
Write-Host "[Download] Hugging Face progress will appear below."
Write-Host ""
try {
    & wslc.exe @download
    if ($LASTEXITCODE -ne 0) { throw "Model download failed with exit code $LASTEXITCODE" }
} finally {
    # --rm normally handles this; explicit cleanup covers interrupted clients.
    $containersAfterDownload = & { $ErrorActionPreference = "Continue"; (& wslc.exe list --all --no-trunc 2>&1 | Out-String) }
    if ($containersAfterDownload -match [regex]::Escape($DownloadContainer)) {
        Write-Host "[Cleanup] Removing the download container..."
        & wslc.exe remove --force $DownloadContainer | Out-Null
    }
}

Write-Host ""
Write-Host "[Download] Model transfer completed." -ForegroundColor Green
Write-Host "[Verify] Checking required model files..."
$shards = @(Get-ChildItem -LiteralPath $ModelDirectory -Filter "model-*-of-00005.safetensors" -File -ErrorAction SilentlyContinue)
if (-not (Test-Path -LiteralPath $configPath)) { throw "Download completed but config.json is missing." }
if ($shards.Count -ne $ExpectedShards) { throw "Expected $ExpectedShards weight shards but found $($shards.Count)." }
Write-Host "[Verify] config.json found." -ForegroundColor Green
Write-Host "[Verify] All five weight shards found." -ForegroundColor Green
Write-Host "[Download] Model is ready at $ModelDirectory" -ForegroundColor Green
