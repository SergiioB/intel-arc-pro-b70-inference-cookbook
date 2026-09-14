[CmdletBinding()]
param(
    [int]$PromptTokens = 512,
    [int]$GenerateTokens = 128,
    [int]$Runs = 5,
    [switch]$SkipWarmup,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$baseUri = "http://127.0.0.1:$Port"

# Windows PowerShell 5.1 does not always load this framework assembly
# automatically before resolving the HttpClient types used below.
Add-Type -AssemblyName System.Net.Http

function Invoke-JsonPost([string]$Uri, [hashtable]$Body) {
    Invoke-RestMethod -Method Post -Uri $Uri -ContentType "application/json" `
        -Body ($Body | ConvertTo-Json -Depth 8) -TimeoutSec 120
}

function Get-SpecCounters {
    try {
        $text = (Invoke-WebRequest -UseBasicParsing -Uri "$baseUri/metrics" -TimeoutSec 10).Content
        $draft = [regex]::Match($text, '(?m)^vllm:spec_decode_num_draft_tokens_total\{[^\r\n]+\}\s+([0-9.eE+-]+)$')
        $accepted = [regex]::Match($text, '(?m)^vllm:spec_decode_num_accepted_tokens_total\{[^\r\n]+\}\s+([0-9.eE+-]+)$')
        return @{ Draft=$(if($draft.Success){[double]$draft.Groups[1].Value}else{0}); Accepted=$(if($accepted.Success){[double]$accepted.Groups[1].Value}else{0}) }
    } catch { return @{ Draft=0; Accepted=0 } }
}

function New-ExactPrompt([int]$Count, [int]$Variant) {
    $words = @(
        "amber","binary","cedar","delta","ember","frost","granite","harbor",
        "indigo","jungle","kernel","lantern","matrix","nebula","orbit","prairie",
        "quartz","river","signal","timber","ultra","vector","willow","xenon","yellow","zenith"
    )
    $parts = [Collections.Generic.List[string]]::new()
    for ($i=0; $i -lt ($Count * 3); $i++) {
        $parts.Add($words[($i * 11 + $Variant * 7) % $words.Count])
    }
    $parts.Add("Write a concise Python function that validates a list of integers, removes duplicates while preserving order, and include a short example.")
    $candidate = $parts -join " "
    $encoded = Invoke-JsonPost "$baseUri/tokenize" @{ model="qwen38"; prompt=$candidate }
    $ids = @($encoded.tokens)
    if ($ids.Count -lt $Count) { throw "Could only construct $($ids.Count) prompt tokens." }
    $selected = @($ids[0..($Count-1)])
    $decoded = Invoke-JsonPost "$baseUri/detokenize" @{ model="qwen38"; tokens=$selected }
    $verified = Invoke-JsonPost "$baseUri/tokenize" @{ model="qwen38"; prompt=$decoded.prompt }
    if (@($verified.tokens).Count -ne $Count) { throw "Exact prompt verification failed." }
    return $decoded.prompt
}

function Invoke-StreamingDecode([string]$Prompt, [int]$MaxTokens) {
    $body = @{
        model="qwen38"; prompt=$Prompt; max_tokens=$MaxTokens
        temperature=0; ignore_eos=$true; stream=$true
        stream_options=@{ include_usage=$true }
    } | ConvertTo-Json -Depth 8

    $client = [Net.Http.HttpClient]::new()
    $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post, "$baseUri/v1/completions")
    $request.Content = [Net.Http.StringContent]::new($body, [Text.Encoding]::UTF8, "application/json")
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $response = $client.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
    $response.EnsureSuccessStatusCode() | Out-Null
    $reader = [IO.StreamReader]::new($response.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
    $firstTokenSeconds = $null
    $completionTokens = 0
    $output = [Text.StringBuilder]::new()
    while (-not $reader.EndOfStream) {
        $line = $reader.ReadLine()
        if (-not $line.StartsWith("data: ")) { continue }
        $data = $line.Substring(6)
        if ($data -eq "[DONE]") { break }
        $chunk = $data | ConvertFrom-Json
        $piece = [string]$chunk.choices[0].text
        if ($piece.Length -gt 0) {
            if ($null -eq $firstTokenSeconds) { $firstTokenSeconds = $clock.Elapsed.TotalSeconds }
            [void]$output.Append($piece)
        }
        if ($chunk.usage -and $chunk.usage.completion_tokens) {
            $completionTokens = [int]$chunk.usage.completion_tokens
        }
    }
    $clock.Stop()
    if ($completionTokens -eq 0) {
        $encodedOutput = Invoke-JsonPost "$baseUri/tokenize" @{ model="qwen38"; prompt=$output.ToString() }
        $completionTokens = @($encodedOutput.tokens).Count
    }
    if ($null -eq $firstTokenSeconds) { throw "The stream returned no generated text." }
    $decodeSeconds = $clock.Elapsed.TotalSeconds - $firstTokenSeconds
    $postFirstTokens = [math]::Max(0, $completionTokens - 1)
    return [pscustomobject]@{
        CompletionTokens=$completionTokens
        TimeToFirstTokenSeconds=[math]::Round($firstTokenSeconds,4)
        PostFirstDecodeSeconds=[math]::Round($decodeSeconds,4)
        PostFirstTokensPerSecond=[math]::Round(($postFirstTokens / $decodeSeconds),2)
    }
}

Write-Host "[Cookbook test] Preparing an exact $PromptTokens-token prompt..."
$warmPrompt = New-ExactPrompt $PromptTokens 99
if (-not $SkipWarmup) {
    Write-Host "[Cookbook test] Same-shape warm-up ($PromptTokens input / $GenerateTokens output)..."
    $warmResult = Invoke-StreamingDecode $warmPrompt $GenerateTokens
    Write-Host ("[Cookbook test] Warm-up completed: {0:N2} post-first tokens/sec (TTFT {1:N3}s)" -f $warmResult.PostFirstTokensPerSecond,$warmResult.TimeToFirstTokenSeconds)
}

$before = Get-SpecCounters
$results = [Collections.Generic.List[object]]::new()
for ($run=1; $run -le $Runs; $run++) {
    Write-Host "[Cookbook test] Measured run $run of $Runs..."
    # Prefix caching is disabled in the server. Reusing the already verified
    # exact-shape prompt avoids spending minutes re-tokenizing long inputs on
    # the host while preserving the GPU workload being measured.
    $result = Invoke-StreamingDecode $warmPrompt $GenerateTokens
    $results.Add([pscustomobject]@{
        Run=$run; PromptTokens=$PromptTokens; CompletionTokens=$result.CompletionTokens
        TimeToFirstTokenSeconds=$result.TimeToFirstTokenSeconds
        PostFirstDecodeSeconds=$result.PostFirstDecodeSeconds
        PostFirstTokensPerSecond=$result.PostFirstTokensPerSecond
    })
    Write-Host ("[Cookbook test] {0:N2} post-first tokens/sec (TTFT {1:N3}s)" -f $result.PostFirstTokensPerSecond,$result.TimeToFirstTokenSeconds) -ForegroundColor Green
}
$after = Get-SpecCounters
$draftDelta = $after.Draft - $before.Draft
$acceptedDelta = $after.Accepted - $before.Accepted
$acceptance = if($draftDelta -gt 0){100*$acceptedDelta/$draftDelta}else{0}
$sorted = @($results.PostFirstTokensPerSecond | Sort-Object)
$median = $sorted[[math]::Floor($sorted.Count/2)]

Write-Host ""
Write-Host ("[Cookbook test] Median: {0:N2} post-first tokens/sec" -f $median) -ForegroundColor Cyan
Write-Host ("[Cookbook test] MTP acceptance: {0:N2}%" -f $acceptance) -ForegroundColor Cyan
$results | Format-Table -AutoSize
