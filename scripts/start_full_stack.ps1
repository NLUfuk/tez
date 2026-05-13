param(
    [int]$BackendPort = 8765,
    [int]$FrontendPort = 5173,
    [int]$Steps = 50000,
    [int]$Interval = 40,
    [string]$Topology = "small_world",
    [int]$SmallWorldN = 1500,
    [string]$SwcDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Info([string]$Message) {
    Write-Host "[full-stack] $Message" -ForegroundColor Cyan
}

function Stop-ChildProcessTree([int]$ProcessId) {
    try {
        # taskkill /T kills child processes too, /F ensures clean shutdown on Ctrl+C.
        cmd /c "taskkill /PID $ProcessId /T /F >nul 2>nul" | Out-Null
    } catch {
        # Best effort shutdown.
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = "python"
}

$resolvedSwcDir = if ([string]::IsNullOrWhiteSpace($SwcDir)) {
    Resolve-Path (Join-Path $projectRoot "data\morphology")
} else {
    Resolve-Path $SwcDir
}

$logDir = Join-Path $projectRoot "outputs\live_logs"
if (-not (Test-Path $logDir)) {
    New-Item -Path $logDir -ItemType Directory | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backendLog = Join-Path $logDir "backend_$timestamp.log"
$backendErrLog = Join-Path $logDir "backend_$timestamp.err.log"
$frontendLog = Join-Path $logDir "frontend_$timestamp.log"
$frontendErrLog = Join-Path $logDir "frontend_$timestamp.err.log"

Write-Info "Project root: $projectRoot"
Write-Info "Using Python: $pythonExe"
Write-Info "SWC dir: $resolvedSwcDir"

$backendArgs = @(
    "scripts/run_live.py",
    "--mode", "stream",
    "--host", "127.0.0.1",
    "--port", "$BackendPort",
    "--steps", "$Steps",
    "--interval", "$Interval",
    "--topology", "$Topology",
    "--small-world-n", "$SmallWorldN",
    "--swc-dir", "$resolvedSwcDir"
)

Write-Info "Starting backend on http://127.0.0.1:$BackendPort"
$backendProc = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $backendArgs `
    -WorkingDirectory $projectRoot `
    -NoNewWindow `
    -PassThru `
    -RedirectStandardOutput $backendLog `
    -RedirectStandardError $backendErrLog

Write-Info "Starting frontend static server on http://127.0.0.1:$FrontendPort"
$frontendArgs = @("-m", "http.server", "$FrontendPort", "--directory", "web")
$frontendProc = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $frontendArgs `
    -WorkingDirectory $projectRoot `
    -NoNewWindow `
    -PassThru `
    -RedirectStandardOutput $frontendLog `
    -RedirectStandardError $frontendErrLog

$stopRequested = $false
$cleanup = {
    if (-not $stopRequested) {
        $script:stopRequested = $true
        Write-Info "Stopping frontend (PID=$($frontendProc.Id))"
        Stop-ChildProcessTree -ProcessId $frontendProc.Id
        Write-Info "Stopping backend (PID=$($backendProc.Id))"
        Stop-ChildProcessTree -ProcessId $backendProc.Id
        Write-Info "Logs:"
        Write-Info "  Backend:  $backendLog"
        Write-Info "  Backend err: $backendErrLog"
        Write-Info "  Frontend: $frontendLog"
        Write-Info "  Frontend err: $frontendErrLog"
    }
}

try {
    Write-Info "Services are up."
    Write-Info "Visualizer: http://127.0.0.1:$FrontendPort/three_visualizer/index.html"
    Write-Info "Backend API: http://127.0.0.1:$BackendPort/state"
    Write-Info "Press Ctrl+C to stop both services."

    while ($true) {
        if ($backendProc.HasExited) {
            Write-Info "Backend process ended. See logs if this was unexpected."
            break
        }
        if ($frontendProc.HasExited) {
            throw "Frontend server exited unexpectedly (code=$($frontendProc.ExitCode)). See $frontendLog / $frontendErrLog"
        }
        Start-Sleep -Milliseconds 400
    }
}
finally {
    & $cleanup
}
