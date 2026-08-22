[CmdletBinding()]
param(
    [string]$Image = "qwen38-b70-friendly:2026.08.19",
    [string]$ContainerName = "qwen38-b70-friendly",
    [string]$ModelDirectory = (Join-Path $PSScriptRoot "models\Qwen3.8-27B-GPTQ-Int4"),
    [ValidateSet(0, 1, 2, 4)][int]$MtpTokens = 2,
    [int]$MaxModelLength = 32768,
    [double]$GpuMemoryUtilization = 0.88,
    [ValidateRange(0, 16)][double]$KvCacheMemoryGiB = 0,
    [ValidateSet("On", "Off")][string]$XpuGraphs = "On",
    [ValidateSet("On", "Off")][string]$ExpandableSegments = "On",
    [ValidateSet("fp8", "auto")][string]$KvCacheDtype = "auto",
    [ValidateRange(1, 256)][int]$MaxNumSeqs = 1,
    [ValidateSet(0, 1)][int]$DraftInt4 = 1,
    [ValidateSet(0, 1)][int]$PrefixCache = 1,
    [int]$Port = 8000,
    [int]$ReadyTimeoutMinutes = 10,
    [switch]$NoWait
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Common.ps1")
$ModelDirectory = [System.IO.Path]::GetFullPath($ModelDirectory)
if (-not (Test-Path -LiteralPath (Join-Path $ModelDirectory "config.json"))) {
    throw "Model not found in $ModelDirectory. Run Download-Qwen38Model.ps1 first."
}

# Replacement is intentionally scoped to this project's fixed container name.
# Query first because PowerShell can promote WSLC's harmless "not found"
# stderr response into a terminating NativeCommandError, so every native
# stderr capture below runs in a Continue scope (issue #5).
$containerList = & { $ErrorActionPreference = "Continue"; (& wslc.exe list --all --no-trunc 2>&1 | Out-String) }
if ($containerList -match [regex]::Escape($ContainerName)) {
    Write-Host "[Server] Removing the previous $ContainerName container..."
    & wslc.exe remove --force $ContainerName
    if ($LASTEXITCODE -ne 0) { throw "Could not remove the previous $ContainerName container." }
} else {
    Write-Host "[Server] No previous friendly server container needs cleanup."
}

Write-Host "Configuration:"
Write-Host "  GPU:               Intel Arc Pro B70"
Write-Host "  Quantization:      GPTQ INT4"
Write-Host "  Thinking mode:     Enabled (low reasoning effort)"
Write-Host "  MTP draft tokens:  $(if ($MtpTokens -eq 0) { 'Disabled' } else { $MtpTokens })"
Write-Host "  Maximum context:   $MaxModelLength"
Write-Host "  XPU graphs:        $XpuGraphs"
Write-Host "  Expandable memory: $ExpandableSegments"
Write-Host "  KV cache type:     $KvCacheDtype"
Write-Host "  Scheduler capacity:$MaxNumSeqs sequence(s)"
Write-Host "  GPU memory target: $([Math]::Round($GpuMemoryUtilization * 100))%"
Write-Host "  KV cache budget:   $(if ($KvCacheMemoryGiB -gt 0) { "$KvCacheMemoryGiB GiB (explicit)" } else { 'Automatic from GPU target' })"
Write-Host "  Draft INT4 overlay:$(if ($DraftInt4 -eq 1) { 'ON (default)' } else { 'OFF (BF16 draft)' })"
Write-Host "  Prefix cache:      $(if ($PrefixCache -eq 1) { 'ON (default, real sessions)' } else { 'OFF (decode-test only)' })"
Write-Host "  API port:          $Port"
Write-Host ""
Write-Host "The first startup can take several minutes while vLLM loads the"
Write-Host "weights and prepares its XPU graphs. Another excellent opportunity"
Write-Host "to check on that cup of tea." -ForegroundColor Cyan

$run = @(
    "run", "--detach", "--name", $ContainerName,
    "--gpus", "all",
    # Loopback-only publishing: the endpoint has no authentication, so it must
    # not be reachable from the network by default (matches the Docker package).
    "--publish", "127.0.0.1:${Port}:8000",
    "--shm-size", "16G",
    "--volume", "${ModelDirectory}:/model",
    "--env", "VLLM_TARGET_DEVICE=xpu",
    "--env", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
    "--env", "ZE_AFFINITY_MASK=0",
    "--env", "B70_MTP_BF16_DRAFT=1",
    "--env", "DRAFT_INT4=$DraftInt4",
    "--env", "PREFIX_CACHE=$PrefixCache",
    "--env", "VLLM_XPU_ENABLE_XPU_GRAPH=$(if ($XpuGraphs -eq 'On') { '1' } else { '0' })",
    "--env", "PYTORCH_ALLOC_CONF=expandable_segments:$(if ($ExpandableSegments -eq 'On') { 'True' } else { 'False' })",
    # WSLC exposes the Intel GPU without native Linux /dev/dri DRM nodes.
    # Force oneCCL away from drmfd IPC and apply current B70 stability gates.
    "--env", "CCL_ZE_IPC_EXCHANGE=sockets",
    "--env", "CCL_ATL_TRANSPORT=ofi",
    "--env", "CCL_ENABLE_SYCL_KERNELS=0",
    "--env", "CCL_TOPO_P2P_ACCESS=0",
    "--env", "CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0",
    "--env", "CCL_ZE_CACHE_OPEN_IPC_HANDLES=0",
    "--env", "SYCL_UR_USE_LEVEL_ZERO_V2=0",
    # The default immediate-command-list path can abort in Intel NEO
    # (linear_stream.h) under sustained graph-enabled load; =0 stopped the
    # aborts while keeping graph speed in reporter testing (issue #6).
    "--env", "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0",
    "--env", "TORCH_LLM_ALLREDUCE=1",
    "--env", "MTP_TOKENS=$MtpTokens",
    "--env", "MAX_MODEL_LEN=$MaxModelLength",
    "--env", "KV_CACHE_DTYPE=$KvCacheDtype",
    "--env", "MAX_NUM_SEQS=$MaxNumSeqs",
    "--env", "GPU_MEMORY_UTILIZATION=$GpuMemoryUtilization",
    "--env", "KV_CACHE_MEMORY_BYTES=$(if ($KvCacheMemoryGiB -gt 0) { [int64]($KvCacheMemoryGiB * 1GB) } else { 0 })",
    $Image
)

Write-Host "[Server] Image reference: $Image"
Write-Host "[Server] Model mount:    ${ModelDirectory}:/model"
& wslc.exe @run
if ($LASTEXITCODE -ne 0) { throw "Container start failed with exit code $LASTEXITCODE" }
Write-Host "[Server] Container created successfully." -ForegroundColor Green
Write-Host "[Server] The model is now loading in the background."
Write-Host "[Server] Follow detailed startup with: wslc logs $ContainerName"

if ($NoWait) {
    Write-Host "[Server] Readiness wait skipped because -NoWait was specified."
    return
}

Write-Host "[Server] Waiting up to $ReadyTimeoutMinutes minutes for the API to become ready..."
Write-Host "[Server] vLLM does not expose a reliable overall percentage; loading stages are inferred from its logs."
$deadline = (Get-Date).AddMinutes($ReadyTimeoutMinutes)
$ready = $false
$loadingStage = "Starting the vLLM engine"
$lastDisplayedLoadingStage = $null
$loadingStarted = Get-Date
while ((Get-Date) -lt $deadline) {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5 | Out-Null
        $ready = $true
        break
    } catch {
        # Do not spend the full timeout saying "Still loading" after vLLM has
        # already failed. Inspect the retained container and surface its logs.
        $containerExited = $false
        $containerExitCode = $null
        try {
            $inspectText = & { $ErrorActionPreference = "Continue"; (& wslc.exe inspect $ContainerName 2>&1 | Out-String) }
            $inspectData = $inspectText | ConvertFrom-Json
            $containerState = $inspectData[0].State
            if (-not $containerState.Running) {
                $containerExited = $true
                $containerExitCode = $containerState.ExitCode
            }
        } catch {
            Write-Host "[Server] Could not inspect container state yet: $($_.Exception.Message)" -ForegroundColor Yellow
        }
        if ($containerExited) {
            Write-Host "[Server] The container exited before the API became ready." -ForegroundColor Red
            Write-Host "[Server] Exit code: $containerExitCode"
            Write-Host ""
            Write-Host "[Server] Final log output:" -ForegroundColor Yellow
            & { $ErrorActionPreference = "Continue"; & wslc.exe logs $ContainerName 2>&1 } | Select-Object -Last 100
            throw "Qwen3.8 server startup failed with container exit code $containerExitCode."
        }

        # vLLM does not publish a true loading percentage. Infer a useful
        # human-readable stage from its latest log output instead.
        try {
            $recentLogs = & { $ErrorActionPreference = "Continue"; (& wslc.exe logs $ContainerName 2>&1 | Select-Object -Last 250 | Out-String) }
            $shardMatches = [regex]::Matches(
                $recentLogs,
                '(?im)(?:shard|checkpoint shard)[^\r\n]*?(\d+)\s*(?:of|/)\s*(\d+)'
            )
            if ($recentLogs -match '(?im)(captur(?:e|ing)|prepar(?:e|ing))[^\r\n]*(?:XPU|CUDA)?\s*graphs?') {
                $loadingStage = "Preparing XPU graphs (often the longest final stage)"
            } elseif ($recentLogs -match '(?im)(torch\.compile|compil(?:e|ing|ation)|inductor)[^\r\n]*(?:kernel|graph|model)') {
                $loadingStage = "Compiling and optimising XPU kernels"
            } elseif ($shardMatches.Count -gt 0) {
                $lastShard = $shardMatches[$shardMatches.Count - 1]
                $loadingStage = "Loading model weight shard $($lastShard.Groups[1].Value) of $($lastShard.Groups[2].Value)"
            } elseif ($recentLogs -match '(?im)(loading|reading)[^\r\n]*(weights|safetensors|checkpoint)') {
                $loadingStage = "Loading model weights"
            } elseif ($recentLogs -match '(?im)Initializing a V1 LLM engine|Initializing an LLM engine') {
                $loadingStage = "Initialising the vLLM engine and XPU runtime"
            } elseif ($recentLogs -match '(?im)Resolved architecture|Using max model len|text-only mode') {
                $loadingStage = "Reading and validating the model configuration"
            }
        } catch {
            # Keep the most recently detected stage if this log read fails.
        }
        $elapsed = (Get-Date) - $loadingStarted
        Write-Host ("[Server] Still loading... {0} | Elapsed {1:hh\:mm\:ss}" -f (Get-Date -Format T), $elapsed)
        if ($loadingStage -ne $lastDisplayedLoadingStage) {
            Write-Host "[Server] Estimated stage: $loadingStage"
            $lastDisplayedLoadingStage = $loadingStage
        }
        Start-Sleep -Seconds 60
    }
}
if (-not $ready) {
    throw "The server did not become ready within $ReadyTimeoutMinutes minutes. Inspect it with: wslc logs $ContainerName"
}

Write-QwenBanner "Qwen3.8 is ready!"
Write-Host "OpenAI-compatible endpoint: http://127.0.0.1:$Port/v1"
Write-Host "Served model name:          qwen38"
Write-Host "Run the smoke test with:    .\Smoke-Test.ps1 -Port $Port"
