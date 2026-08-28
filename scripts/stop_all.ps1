$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot

$stoppedLlama = 0
$llamaProcesses = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($llamaProcesses) {
    $stoppedLlama = @($llamaProcesses).Count
    $llamaProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
}

$assistantProcesses = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -ieq "python.exe" -or $_.Name -ieq "pythonw.exe") -and
    $_.CommandLine -and
    ($_.CommandLine -match "SentinelBonsai.py" -or $_.CommandLine -match "launcher.py")
}

$stoppedAssistant = 0
foreach ($process in $assistantProcesses) {
    try {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
        $stoppedAssistant++
    } catch {
        continue
    }
}

if ($stoppedLlama -gt 0) {
    Write-Host "Stopped llama-server process count: $stoppedLlama" -ForegroundColor Green
} else {
    Write-Host "No llama-server process found." -ForegroundColor Yellow
}

if ($stoppedAssistant -gt 0) {
    Write-Host "Stopped Sentinel assistant process count: $stoppedAssistant" -ForegroundColor Green
} else {
    Write-Host "No Sentinel assistant python process found." -ForegroundColor Yellow
}
