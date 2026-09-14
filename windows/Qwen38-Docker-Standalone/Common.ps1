function ConvertTo-UnixLfFile {
    param([Parameter(Mandatory)][string]$Path)

    # Git for Windows defaults to core.autocrlf=true, so a clone can rewrite
    # working-tree bytes to CRLF. The cookbook pins SHA-256 of the original LF
    # files, and Docker COPY of start.sh into a Linux image needs a Unix shebang.
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $offset = 0
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        $offset = 3
    }
    $text = [System.Text.Encoding]::UTF8.GetString($bytes, $offset, $bytes.Length - $offset)
    $text = $text.Replace("`r`n", "`n").Replace("`r", "`n")
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $text, $utf8NoBom)
}

function Write-QwenBanner {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 68) -ForegroundColor Cyan
    Write-Host ("  " + $Title) -ForegroundColor Cyan
    Write-Host ("=" * 68) -ForegroundColor Cyan
    Write-Host ""
}

function Write-QwenStep {
    param([string]$Title, [string]$Detail)
    Write-Host ""
    Write-Host ("-" * 68) -ForegroundColor DarkCyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("-" * 68) -ForegroundColor DarkCyan
    if ($Detail) { Write-Host $Detail }
    Write-Host ""
}

function Write-HuggingFaceTokenStatus {
    param([string]$Token = $env:HF_TOKEN, [switch]$RequireToken)
    if ($Token) {
        Write-Host "[Hugging Face] HF_TOKEN detected." -ForegroundColor Green
        Write-Host "Authenticated downloads and higher rate limits are available."
        return
    }

    Write-Host "[Hugging Face] No HF_TOKEN was detected." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The model is public, so setup can continue, but Hugging Face may"
    Write-Host "apply lower rate limits and the download could take longer."
    Write-Host ""
    Write-Host 'To use a read-only token in this PowerShell session:'
    Write-Host '  $env:HF_TOKEN = "hf_your_token_here"' -ForegroundColor White
    Write-Host ""
    if ($RequireToken) {
        throw "HF_TOKEN is required because -RequireHuggingFaceToken was specified."
    }
    Write-Host "Press Ctrl+C now if you would prefer to set a token first."
    Write-Host "Continuing without authentication in 10 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}

function Initialize-QwenDocker {
    param([Parameter(Mandatory)][string]$DockerPath)

    # A fresh Docker Desktop installation can leave the CLI on the generic
    # Windows docker_engine pipe. Prefer Docker Desktop's Linux engine context
    # for this XPU image without changing the user's global context selection.
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $contexts = @(& $DockerPath context ls --format '{{.Name}}' 2>$null)
        if ($contexts -contains "desktop-linux") {
            $env:DOCKER_CONTEXT = "desktop-linux"
            Write-Host "[Docker] Using Docker Desktop Linux context: desktop-linux" -ForegroundColor Green
        }
        $infoText = (& $DockerPath info 2>&1 | Out-String)
        $infoExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }

    if ($infoExitCode -eq 0) { return }

    Write-Host "[Docker] Docker Desktop's Linux engine could not be accessed." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Docker Desktop manually from:"
    Write-Host "  https://docs.docker.com/desktop/setup/install/windows-install/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Then start Docker Desktop, select its Linux/WSL 2 engine, and wait until it reports that the engine is running."
    Write-Host "Try in a new PowerShell window:"
    Write-Host "  docker context use desktop-linux"
    Write-Host "  docker info"
    Write-Host ""
    Write-Host "If docker info still reports permission denied, an administrator can run:"
    Write-Host '  net localgroup docker-users "$env:USERNAME" /add' -ForegroundColor Yellow
    Write-Host "After group membership changes, sign out of Windows and sign back in (or reboot)."
    Write-Host ""
    throw "Docker Desktop engine access failed. Original diagnostic: $($infoText.Trim())"
}
