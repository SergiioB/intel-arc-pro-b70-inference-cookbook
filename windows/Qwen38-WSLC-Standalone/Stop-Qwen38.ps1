[CmdletBinding()]
param(
    [string]$ContainerName = "qwen38-b70-friendly",
    [switch]$ReleaseGpuMemory
)

$ErrorActionPreference = "Stop"
if ($ReleaseGpuMemory) {
    Write-Host "[Server] Terminating the WSLC session to release all GPU memory..." -ForegroundColor Cyan
    Write-Host "[Server] This also stops any other containers in the same WSLC session." -ForegroundColor Yellow
    & wslc.exe system session terminate
    if ($LASTEXITCODE -ne 0) { throw "WSLC session termination failed with exit code $LASTEXITCODE" }
    Write-Host "[Server] Qwen3.8 has stopped and the WSLC GPU allocation has been released." -ForegroundColor Green
    return
}

Write-Host "[Server] Stopping $ContainerName..."
& wslc.exe stop $ContainerName
if ($LASTEXITCODE -ne 0) { throw "Container stop failed with exit code $LASTEXITCODE" }
Write-Host "[Server] Qwen3.8 has stopped." -ForegroundColor Green
Write-Host "[Server] To guarantee that WSLC returns VRAM to Windows, use:" -ForegroundColor Cyan
Write-Host "         .\Stop-Qwen38.ps1 -ReleaseGpuMemory"
