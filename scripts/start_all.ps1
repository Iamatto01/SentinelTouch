$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot

$startServerScript = Join-Path $scriptDir "start_llama_server.ps1"
$startAssistantScript = Join-Path $scriptDir "start_sentinel.ps1"

function Test-LlamaServer {
    $endpoints = @("http://127.0.0.1:8080/health", "http://127.0.0.1:8080/v1/models")
    foreach ($url in $endpoints) {
        try {
            $null = Invoke-WebRequest -Uri $url -Method Get -TimeoutSec 3 -UseBasicParsing
            return $true
        } catch {
            continue
        }
    }
    return $false
}

if (Get-Process -Name "llama-server" -ErrorAction SilentlyContinue) {
    Write-Host "llama.cpp server is already running." -ForegroundColor Green
} else {
    Write-Host "Launching llama.cpp server in a new window..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-File", $startServerScript
    ) | Out-Null

    Write-Host "Waiting for llama.cpp server to become ready..." -ForegroundColor Gray
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        if (Test-LlamaServer) {
            $ready = $true
            break
        }
        Start-Sleep -Seconds 1
    }

    if ($ready) {
        Write-Host "llama.cpp server is ready." -ForegroundColor Green
    } else {
        Write-Host "Server readiness check timed out. Assistant will still try to connect." -ForegroundColor Yellow
    }
}

& $startAssistantScript
exit $LASTEXITCODE
