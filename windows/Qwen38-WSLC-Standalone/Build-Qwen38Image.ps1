[CmdletBinding()]
param(
    [string]$Image = "qwen38-b70-friendly:2026.08.19",
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
. (Join-Path $Root "Common.ps1")
$PatchDir = Join-Path $Root "patches"
New-Item -ItemType Directory -Force -Path $PatchDir | Out-Null

Write-Host "[Build] Preparing five pinned compatibility patches..."

$patches = @(
    @{
        Name = "patch_mtp_nightly.py"
        Uri = "https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/5c6b6b1/patches/patch_mtp_nightly.py"
        Sha256 = "4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14"
    },
    @{
        Name = "patch_mtp_boundary.py"
        Uri = "https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/5c6b6b1/patches/patch_mtp_boundary.py"
        Sha256 = "41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50"
    },
    @{
        Name = "patch_gdn_mixed_split_v5.py"
        Uri = "https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/db20e00/patches/patch_gdn_mixed_split_v5.py"
        Sha256 = "8e4a3cbe5f424f308af74ff215d0fcb8d31f63ac3f07cf359ed2269956c3fc80"
    },
    @{
        Name = "patch_draft_lmhead_int4.py"
        Uri = "https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/aa363ca/patches/patch_draft_lmhead_int4.py"
        Sha256 = "ffae41926d5f05f4f38bb985301b5e572092441d06d6063c8820a63a39b8cefc"
    },
    @{
        Name = "patch_draft_mtp_int4.py"
        Uri = "https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/aa363ca/patches/patch_draft_mtp_int4.py"
        Sha256 = "4df179c3e77fd7a248f9b9c0b60217c60caea14ebfd16b7860536fbff3b2a1e9"
    }
)

foreach ($item in $patches) {
    Write-Host "[Build] Downloading $($item.Name)..."
    $destination = Join-Path $PatchDir $item.Name
    $download = "$destination.download"
    Invoke-WebRequest -UseBasicParsing -Uri $item.Uri -OutFile $download

    # Windows PowerShell / Git for Windows may rewrite text as CRLF. The
    # cookbook publishes hashes for the original LF bytes, and the files are
    # copied into a Linux image, so normalize before checking the digest.
    ConvertTo-UnixLfFile -Path $download
    if (Test-Path -LiteralPath $destination) {
        Remove-Item -Force -LiteralPath $destination
    }
    Move-Item -Force -LiteralPath $download -Destination $destination

    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
    if ($actual -ne $item.Sha256) {
        throw "Hash mismatch for $($item.Name). Expected $($item.Sha256), received $actual"
    }
    Write-Host "[Build] Verified $($item.Name)" -ForegroundColor Green
}

Write-Host "[Build] Normalizing Dockerfile and container scripts to LF..."
$linuxSources = @(
    (Join-Path $Root "Dockerfile"),
    (Join-Path $Root "container\start.sh"),
    (Join-Path $Root "container\diagnose.py"),
    (Join-Path $Root "container\patch_xpu_single_gpu_warmup.py"),
    (Join-Path $Root "container\prepare_low_reasoning_template.py")
)
foreach ($path in $linuxSources) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required image source is missing: $path" }
    ConvertTo-UnixLfFile -Path $path
}

Write-Host "[Build] Asking WSLC to build image $Image"
Write-Host "[Build] The first build may download a large base image. Cached layers will be reused."
$args = @("build", "--pull", "--tag", $Image, "--file", (Join-Path $Root "Dockerfile"))
if ($NoCache) { $args += "--no-cache" }
$args += $Root
& wslc.exe @args
if ($LASTEXITCODE -ne 0) { throw "WSLC image build failed with exit code $LASTEXITCODE" }

Write-Host "[Build] Image ready: $Image" -ForegroundColor Green
