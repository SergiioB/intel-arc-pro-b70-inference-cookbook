[CmdletBinding()]
param(
    [string]$ModelDirectory = (Join-Path $PSScriptRoot "models\Qwen3.8-27B-GPTQ-Int4"),
    [ValidateSet(0, 1)][int]$DraftInt4 = 1,
    [ValidateSet(0, 1)][int]$PrefixCache = 1
)

# Rebuilds the 2026.08.19 WSLC image and starts a new container.
# WSLC Start already removes the previous container.

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Common.ps1")
$image = "qwen38-b70-friendly:2026.08.19"

Write-QwenBanner "Qwen3.8-27B WSLC upgrade (2026.08.19)"
Write-Host "Windows container devised and tested by Ian Hudson - aitesthive.com"
Write-Host ""
Write-Host "This rebuilds the image with mixed-split v5 + draft-INT4 S+M1."
Write-Host "Draft INT4 overlay: $(if ($DraftInt4 -eq 1) { 'ON (default)' } else { 'OFF — BF16 draft' })"
Write-Host "Prefix cache:       $(if ($PrefixCache -eq 1) { 'ON (default, real sessions)' } else { 'OFF — decode-test only' })"
Write-Host ""

Write-QwenStep "[1/2] Build image $image" "Five SHA-256-verified patches downloaded from pinned cookbook commits."
& (Join-Path $PSScriptRoot "Build-Qwen38Image.ps1") -Image $image
if ($LASTEXITCODE -ne 0) { throw "Image build failed." }

Write-QwenStep "[2/2] Start the WSLC server" "MTP4, 100K, FP8 KV, explicit 4.25 GiB cache."
& (Join-Path $PSScriptRoot "Start-Qwen38.ps1") -Image $image -ModelDirectory $ModelDirectory -MtpTokens 4 -MaxModelLength 100000 -GpuMemoryUtilization 0.75 -KvCacheMemoryGiB 4.25 -XpuGraphs On -ExpandableSegments On -KvCacheDtype fp8 -MaxNumSeqs 1 -DraftInt4 $DraftInt4 -PrefixCache $PrefixCache
if ($LASTEXITCODE -ne 0) { throw "Server start failed." }

Write-QwenBanner "Upgrade complete"
Write-Host "Endpoint: http://127.0.0.1:8000/v1"
Write-Host "Model:    qwen38"
Write-Host ""
Write-Host "Confirm the overlay:"
Write-Host "  wslc logs qwen38-b70-friendly"
Write-Host "You want: [start] draft-INT4 S+M1 overlay ENABLED"
Write-Host ""
Write-Host "Then measure:"
Write-Host "  .\Test-CookbookDecode.ps1"
Write-Host "WSLC remains the experimental path (~26 tok/s on 2026.08.18)."
