[CmdletBinding()]
param(
    [string]$Image = "qwen38-b70-docker:2026.08.19",
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
if (-not (Test-Path -LiteralPath $docker)) { $docker = (Get-Command docker.exe -ErrorAction Stop).Source }
. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-QwenDocker -DockerPath $docker

$expected = @{
    "patch_mtp_nightly.py"  = "4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14"
    "patch_mtp_boundary.py" = "41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50"
    "patch_gdn_mixed_split_v5.py" = "8e4a3cbe5f424f308af74ff215d0fcb8d31f63ac3f07cf359ed2269956c3fc80"
    "patch_draft_lmhead_int4.py" = "db3768becdf1ac52a54dbd27a89ceb42338f8febbd32b1f9c8b6d6ffdd76649b"
    "patch_draft_mtp_int4.py" = "2f6d36c419438862157ad0b625a12264f84ea8c7a6d026d6d0d414aa46a14405"
}
Write-Host "[Docker build] Normalizing copied Linux files to LF, then verifying pinned patches..."
$linuxSources = @(
    (Join-Path $PSScriptRoot "Dockerfile"),
    (Join-Path $PSScriptRoot "container\start.sh"),
    (Join-Path $PSScriptRoot "container\diagnose.py"),
    (Join-Path $PSScriptRoot "container\patch_xpu_single_gpu_warmup.py"),
    (Join-Path $PSScriptRoot "container\patch_xpu_only_platform.py"),
    (Join-Path $PSScriptRoot "container\prepare_low_reasoning_template.py")
)
foreach ($path in $linuxSources) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required image source is missing: $path" }
    ConvertTo-UnixLfFile -Path $path
}

foreach ($name in $expected.Keys) {
    $path = Join-Path $PSScriptRoot "patches\$name"
    if (-not (Test-Path -LiteralPath $path)) { throw "Required pinned patch is missing: $path" }
    ConvertTo-UnixLfFile -Path $path
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    if ($actual -ne $expected[$name]) { throw "Hash mismatch for $name. Expected $($expected[$name]), received $actual" }
    Write-Host "[Docker build] Verified $name" -ForegroundColor Green
}

Write-Host "[Docker build] Building $Image"
Write-Host "[Docker build] The first build downloads several large layers; later builds reuse Docker's cache."
$args = @("build", "--tag", $Image, "--file", (Join-Path $PSScriptRoot "Dockerfile"))
if ($NoCache) { $args += "--no-cache" }
$args += $PSScriptRoot
& $docker @args
if ($LASTEXITCODE -ne 0) { throw "Docker image build failed with exit code $LASTEXITCODE." }
Write-Host "[Docker build] Image ready: $Image" -ForegroundColor Green
