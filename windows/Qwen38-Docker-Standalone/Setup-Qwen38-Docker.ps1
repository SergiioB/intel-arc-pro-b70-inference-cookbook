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
$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
Write-QwenBanner "Qwen3.8-27B reproducible Docker Desktop setup"
Write-Host "Windows container devised and tested by Ian Hudson - aitesthive.com" -ForegroundColor Cyan
Write-Host ""
Write-Host "This installer creates the proven B70 profile with text, reasoning and tools enabled."
Write-Host "Vision remains disabled to preserve VRAM and text performance."
Write-Host "[Prerequisite] Docker Desktop must be installed manually first."
Write-Host "Official installer: https://docs.docker.com/desktop/setup/install/windows-install/" -ForegroundColor Cyan
Write-Host "Use Docker Desktop's Linux/WSL 2 engine and leave Docker Desktop running."
Write-Host ""
if (-not (Test-Path -LiteralPath $docker)) {
    $command = Get-Command docker.exe -ErrorAction SilentlyContinue
    if (-not $command) { throw "Docker Desktop was not found. Install it from the official link above, start it, then rerun this script." }
    $docker = $command.Source
}
Initialize-QwenDocker -DockerPath $docker
Write-HuggingFaceTokenStatus -Token $HuggingFaceToken -RequireToken:$RequireHuggingFaceToken

Write-QwenStep "[1/4] Build pinned Docker image" "Cached layers are reused; the first download can take several minutes."
& (Join-Path $PSScriptRoot "Build-Qwen38Image-Docker.ps1")
Write-QwenStep "[2/4] Verify Docker access to the Arc Pro B70" "Uses /dev/dxg and read-only Windows WSL driver-library mounts; privileged mode is not used."
& $docker run --rm --device /dev/dxg -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro -v /usr/lib/wsl/drivers:/usr/lib/wsl/drivers:ro -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/tmp/ucx_install/lib:/opt/venv/lib:/usr/local/lib --entrypoint python qwen38-b70-docker:2026.08.19 -c "import torch; assert torch.xpu.is_available(); x=torch.arange(1048576,device='xpu'); y=(x*2+1).sum(); torch.xpu.synchronize(); print('B70:',torch.xpu.get_device_name(0)); print('XPU allocation and compute probe: OK')"
if ($LASTEXITCODE -ne 0) { throw "Docker could not perform the B70 XPU compute probe." }
if (-not $SkipModelDownload) {
    Write-QwenStep "[3/4] Supply the model" "Existing files are reused; otherwise approximately 18.2 GiB is downloaded."
    & (Join-Path $PSScriptRoot "Download-Qwen38Model-Docker.ps1") -ModelDirectory $ModelDirectory -HuggingFaceToken $HuggingFaceToken -TokenStatusAlreadyShown -RemoveStaleLock:$RemoveStaleLock
} else { Write-Host "[3/4] Model download skipped." -ForegroundColor Yellow }
if (-not $SkipServerStart) {
    Write-QwenStep "[4/4] Start the Docker server" "MTP4, 100K context, tools, XPU graphs, FP8 KV cache and explicit 4.25 GiB cache."
    & (Join-Path $PSScriptRoot "Start-Qwen38-Docker.ps1") -ModelDirectory $ModelDirectory
} else { Write-Host "[4/4] Server start skipped." }
Write-QwenBanner "Docker setup complete"
if (-not $SkipServerStart) { Write-Host "OpenAI-compatible endpoint: http://127.0.0.1:8000/v1" -ForegroundColor Green; Write-Host "Model name: qwen38"; Write-Host "Tools: enabled; vision: disabled" }
