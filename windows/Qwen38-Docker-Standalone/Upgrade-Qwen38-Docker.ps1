[CmdletBinding()]
param(
    [string]$ModelDirectory = (Join-Path $PSScriptRoot "models\Qwen3.8-27B-GPTQ-Int4"),
    [ValidateSet(0, 1)][int]$DraftInt4 = 1,
    [ValidateSet(0, 1)][int]$PrefixCache = 1
)

# Rebuilds the 2026.08.19 image and recreates the container so the new
# patches and env actually take effect. Restarting the old 2026.08.18
# container is not an upgrade.

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Common.ps1")

Write-QwenBanner "Qwen3.8-27B Docker Desktop upgrade (2026.08.19)"
Write-Host "Windows container devised and tested by Ian Hudson - aitesthive.com"
Write-Host ""
Write-Host "This rebuilds the image with mixed-split v5 + draft-INT4 S+M1,"
Write-Host "then deletes and recreates the container. Your model files stay."
Write-Host "Draft INT4 overlay: $(if ($DraftInt4 -eq 1) { 'ON (default)' } else { 'OFF — BF16 draft' })"
Write-Host "Prefix cache:       $(if ($PrefixCache -eq 1) { 'ON (default, real sessions)' } else { 'OFF — decode-test only' })"
Write-Host ""

Write-QwenStep "[1/2] Build image qwen38-b70-docker:2026.08.19" "Five SHA-256-verified patches; first rebuild downloads only changed layers."
& (Join-Path $PSScriptRoot "Build-Qwen38Image-Docker.ps1")
if ($LASTEXITCODE -ne 0) { throw "Image build failed." }

Write-QwenStep "[2/2] Recreate the server container" "Old 2026.08.18 container cannot pick up new patches by restart."
& (Join-Path $PSScriptRoot "Start-Qwen38-Docker.ps1") -ModelDirectory $ModelDirectory -DraftInt4 $DraftInt4 -PrefixCache $PrefixCache -Recreate
if ($LASTEXITCODE -ne 0) { throw "Server start failed." }

Write-QwenBanner "Upgrade complete"
Write-Host "Endpoint: http://127.0.0.1:8000/v1"
Write-Host "Model:    qwen38"
Write-Host ""
Write-Host "Confirm the overlay in the logs:"
Write-Host '  docker logs qwen38-b70-docker-test | findstr /C:"draft-INT4"'
Write-Host "You want: [start] draft-INT4 S+M1 overlay ENABLED"
Write-Host ""
Write-Host "Then measure:"
Write-Host "  .\Test-CookbookDecode.ps1"
Write-Host "Compare to your previous Docker ~70 tok/s p512/g128 cell."
Write-Host "Linux on the same overlay is 112.65 vs 81.20 (C1, n=5, cache off)."
Write-Host "Windows has not been re-measured with this overlay yet."
