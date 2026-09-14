[CmdletBinding()]
param(
    [string]$Image = "qwen38-b70-docker:2026.08.19",
    [string]$ModelDirectory = (Join-Path $PSScriptRoot "models\Qwen3.8-27B-GPTQ-Int4"),
    [string]$HuggingFaceToken = $env:HF_TOKEN,
    [switch]$RequireHuggingFaceToken,
    [switch]$RemoveStaleLock,
    [switch]$TokenStatusAlreadyShown
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Common.ps1")
$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
if (-not (Test-Path -LiteralPath $docker)) { $docker = (Get-Command docker.exe -ErrorAction Stop).Source }
Initialize-QwenDocker -DockerPath $docker
if (-not $TokenStatusAlreadyShown) { Write-HuggingFaceTokenStatus -Token $HuggingFaceToken -RequireToken:$RequireHuggingFaceToken }

$ModelDirectory = [IO.Path]::GetFullPath($ModelDirectory)
$containerName = "qwen38-docker-model-download"
$revision = "9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e"
New-Item -ItemType Directory -Force -Path $ModelDirectory | Out-Null
$config = Join-Path $ModelDirectory "config.json"
$shards = @(Get-ChildItem -LiteralPath $ModelDirectory -Filter "model-*-of-00005.safetensors" -File -ErrorAction SilentlyContinue)
if ((Test-Path -LiteralPath $config) -and $shards.Count -eq 5) {
    Write-Host "[Download] The complete pinned model is already present; no download is needed." -ForegroundColor Green
    return
}

$root = [IO.Path]::GetPathRoot($ModelDirectory)
$drive = Get-PSDrive -Name $root.TrimEnd('\').TrimEnd(':') -ErrorAction Stop
if ($drive.Free -lt 22GB) { throw "At least 22 GiB of free space is recommended before downloading the model." }
Write-Host "[Download] Model: SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
Write-Host "[Download] Pinned revision: $revision"
Write-Host "[Download] Destination: $ModelDirectory"
Write-Host "[Download] This is approximately 18.2 GiB and may take quite a while." -ForegroundColor Yellow
Write-Host "[Download] You might want to go and make a cup of tea." -ForegroundColor Cyan

$running = (& $docker ps -a --filter "name=^/${containerName}$" --format '{{.Names}}')
if ($running -eq $containerName) {
    $state = (& $docker inspect --format '{{.State.Running}}' $containerName)
    if ($state -eq "true") { throw "The model download container is already running. Follow it with: docker logs -f $containerName" }
    & $docker rm $containerName | Out-Null
}

$locks = @(Get-ChildItem -LiteralPath $ModelDirectory -Recurse -Filter "*.lock" -File -ErrorAction SilentlyContinue)
if ($locks.Count -gt 0 -and -not $RemoveStaleLock) {
    throw "Found $($locks.Count) Hugging Face lock file(s). Confirm no downloader is running, then use -RemoveStaleLock."
}
if ($RemoveStaleLock) { $locks | ForEach-Object { Remove-Item -Force -LiteralPath $_.FullName } }

$args = @("run", "--rm", "--name", $containerName, "--mount", "type=bind,source=$ModelDirectory,target=/download")
if ($HuggingFaceToken) { $args += @("--env", "HF_TOKEN=$HuggingFaceToken") }
$args += @("--entrypoint", "/bin/bash", $Image, "-lc", "set -e; if command -v hf >/dev/null 2>&1; then hf download SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 --revision $revision --local-dir /download; else huggingface-cli download SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 --revision $revision --local-dir /download; fi")
& $docker @args
if ($LASTEXITCODE -ne 0) { throw "Docker model download failed with exit code $LASTEXITCODE." }
$shards = @(Get-ChildItem -LiteralPath $ModelDirectory -Filter "model-*-of-00005.safetensors" -File -ErrorAction SilentlyContinue)
if (-not (Test-Path -LiteralPath $config) -or $shards.Count -ne 5) { throw "Download finished but the required config and five shards were not all found." }
Write-Host "[Download] Model verified and ready." -ForegroundColor Green
