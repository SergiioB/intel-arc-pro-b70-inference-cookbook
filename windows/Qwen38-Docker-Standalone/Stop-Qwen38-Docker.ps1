[CmdletBinding()]
param([string]$ContainerName = "qwen38-b70-docker-test")

$ErrorActionPreference = "Stop"
$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
if (-not (Test-Path -LiteralPath $docker)) {
    $docker = (Get-Command docker.exe -ErrorAction Stop).Source
}
. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-QwenDocker -DockerPath $docker

$containerNames = @(& $docker ps --all --filter "name=^/${ContainerName}$" --format '{{.Names}}')
if ($containerNames -notcontains $ContainerName) {
    Write-Host "[Docker] Container '$ContainerName' does not exist. Nothing needs stopping."
    exit 0
}
$status = (& $docker inspect --format '{{.State.Status}}' $ContainerName)
if ($LASTEXITCODE -ne 0) { throw "Docker could not inspect the existing $ContainerName container." }
if ($status -ne "running") {
    Write-Host "[Docker] Container '$ContainerName' is already stopped."
    exit 0
}

Write-Host "[Docker] Stopping Qwen3.8 and releasing its B70 memory..."
& $docker stop --timeout 15 $ContainerName | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker could not stop $ContainerName." }
Write-Host "[Docker] Server stopped. The container and its compiled cache were preserved for a faster restart."
