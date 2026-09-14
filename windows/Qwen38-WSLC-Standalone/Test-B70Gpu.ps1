[CmdletBinding()]
param([string]$Image = "qwen38-b70-friendly:2026.08.19")

$ErrorActionPreference = "Stop"
Write-Host "[GPU] Passing all available GPUs into a short diagnostic container..."
& wslc.exe run --rm --gpus all --entrypoint python $Image /opt/qwen38/diagnose.py
if ($LASTEXITCODE -ne 0) { throw "B70 GPU diagnostic failed with exit code $LASTEXITCODE" }
Write-Host "[GPU] Arc Pro B70 passthrough and XPU compute are working." -ForegroundColor Green
