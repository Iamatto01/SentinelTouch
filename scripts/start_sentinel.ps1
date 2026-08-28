$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot

$assistantScript = Join-Path $projectRoot "SentinelBonsai.py"
if (-not (Test-Path $assistantScript)) {
    Write-Host "Error: SentinelBonsai.py not found at $assistantScript" -ForegroundColor Red
    exit 1
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        Write-Host "Error: python was not found and .venv\Scripts\python.exe is missing." -ForegroundColor Red
        Write-Host "Create venv and install dependencies first." -ForegroundColor Yellow
        exit 1
    }
    $pythonExe = $pythonCommand.Source
}

Write-Host "Starting SentinelBonsai assistant..." -ForegroundColor Cyan
Write-Host "Python: $pythonExe" -ForegroundColor Gray

& $pythonExe $assistantScript
exit $LASTEXITCODE
