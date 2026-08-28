$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot

$serverExe = Join-Path $projectRoot "llama.cpp\build\bin\llama-server.exe"
if (-not (Test-Path $serverExe)) {
    Write-Host "Error: llama-server.exe not found at $serverExe" -ForegroundColor Red
    Write-Host "Build llama.cpp first, then run this script again." -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting llama.cpp server..." -ForegroundColor Cyan
Write-Host "Model: prism-ml/Bonsai-8B-gguf:Q1_0" -ForegroundColor Gray

& $serverExe -hf "prism-ml/Bonsai-8B-gguf:Q1_0"
exit $LASTEXITCODE
