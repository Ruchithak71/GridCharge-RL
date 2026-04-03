param(
    [switch]$Detach,
    [switch]$NoRestart,
    [int]$RestartDelaySeconds = 2
)

$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment Python not found at $venvPython"
}

$streamlitArgs = @(
    '-m', 'streamlit', 'run', 'dashboard.py',
    '--server.headless', 'true',
    '--server.port', '8501',
    '--server.fileWatcherType', 'none',
    '--browser.gatherUsageStats', 'false'
)

function Stop-StaleDashboard {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -like "*$projectRoot*" -and
            $_.CommandLine -like '*streamlit*dashboard.py*'
        } |
        ForEach-Object {
            try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {}
        }
}

if ($Detach) {
    Stop-StaleDashboard
    $selfPath = Join-Path $projectRoot 'start_dashboard.ps1'
    $detachArgs = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', "`"$selfPath`""
    )

    $proc = Start-Process -FilePath 'powershell.exe' -ArgumentList $detachArgs -WorkingDirectory $projectRoot -WindowStyle Minimized -PassThru
    Write-Host "Detached supervisor started."
    Write-Host "PID: $($proc.Id)"
    Write-Host "URL: http://localhost:8501"
    return
}

Stop-StaleDashboard
Write-Host "Starting dashboard from .venv supervisor..."
Write-Host "URL: http://localhost:8501"
Write-Host "Press Ctrl+C to stop the supervisor."

try {
    do {
        & $venvPython @streamlitArgs
        $exitCode = $LASTEXITCODE

        if ($NoRestart) {
            Write-Host "Dashboard exited with code $exitCode."
            break
        }

        Write-Warning "Dashboard exited with code $exitCode. Restarting in $RestartDelaySeconds second(s)..."
        Start-Sleep -Seconds $RestartDelaySeconds
    } while ($true)
}
catch [System.Management.Automation.PipelineStoppedException] {
    Write-Host "Dashboard supervisor stopped by user."
}
