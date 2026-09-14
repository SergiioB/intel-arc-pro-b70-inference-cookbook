[CmdletBinding()]
param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
Write-Host "[Smoke test] Sending a small deterministic request to Qwen3.8..."
$body = @{
    model = "qwen38"
    messages = @(@{ role = "user"; content = "Reply with exactly: B70 OK" })
    max_tokens = 16
    temperature = 0
} | ConvertTo-Json -Depth 5

$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/v1/chat/completions" -ContentType "application/json" -Body $body
Write-Host "[Smoke test] The API responded successfully." -ForegroundColor Green
$response
