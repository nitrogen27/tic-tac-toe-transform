[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $repoRoot ".runtime\legacy-ui-api"
$apiPidPath = Join-Path $runtimeDir "api.pid"
$clientPidPath = Join-Path $runtimeDir "client.pid"
$apiLogPath = Join-Path $runtimeDir "api.log"
$clientLogPath = Join-Path $runtimeDir "client.log"
$monitorStartScript = Join-Path $PSScriptRoot "start-training-metrics-monitor.ps1"
$monitorStopScript = Join-Path $PSScriptRoot "stop-training-metrics-monitor.ps1"

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

function Stop-TrackedProcess {
    param([string]$PidPath)

    if (-not (Test-Path $PidPath)) {
        return
    }

    try {
        $pidValue = [int](Get-Content -Path $PidPath -ErrorAction Stop | Select-Object -First 1)
        $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($null -ne $proc) {
            Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 300
        }
    } catch {
    } finally {
        Remove-Item -Path $PidPath -Force -ErrorAction SilentlyContinue
    }
}

function Wait-HttpReady {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 40
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        } catch {
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Ensure-PortFree {
    param([int]$Port)

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($null -ne $connections) {
        $owners = ($connections | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
        throw "Port $Port is already in use by PID(s): $owners"
    }
}

Stop-TrackedProcess -PidPath $apiPidPath
Stop-TrackedProcess -PidPath $clientPidPath
if (Test-Path $monitorStopScript) {
    try {
        & $monitorStopScript -Variant "ttt5" | Out-Null
    } catch {
    }
}

Ensure-PortFree -Port 8080
Ensure-PortFree -Port 5173

$apiCommand = @"
`$env:PYTHONPATH='$repoRoot\apps\api\src;$repoRoot\trainer-lab\src'
Set-Location '$repoRoot'
python -m uvicorn gomoku_api.main:app --host 127.0.0.1 --port 8080 *> '$apiLogPath'
"@

$clientCommand = @"
Set-Location '$repoRoot\client'
npm run dev -- --host 127.0.0.1 --port 5173 *> '$clientLogPath'
"@

$apiProc = Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-Command", $apiCommand
) -PassThru -WindowStyle Hidden

$clientProc = Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-Command", $clientCommand
) -PassThru -WindowStyle Hidden

Set-Content -Path $apiPidPath -Value $apiProc.Id
Set-Content -Path $clientPidPath -Value $clientProc.Id

$apiReady = Wait-HttpReady -Url "http://127.0.0.1:8080/health" -TimeoutSeconds 40
$clientReady = Wait-HttpReady -Url "http://127.0.0.1:5173/" -TimeoutSeconds 40

if (-not $apiReady -or -not $clientReady) {
    throw "Startup failed. API ready: $apiReady, client ready: $clientReady. Check logs in $runtimeDir"
}

if (Test-Path $monitorStartScript) {
    try {
        & $monitorStartScript -Variant "ttt5" | Out-Null
    } catch {
        Write-Warning "Training metrics monitor did not start: $($_.Exception.Message)"
    }
}

Write-Host "API ready:    http://127.0.0.1:8080"
Write-Host "Legacy UI:    http://127.0.0.1:5173"
Write-Host "API PID:      $($apiProc.Id)"
Write-Host "Client PID:   $($clientProc.Id)"
Write-Host "Logs:         $runtimeDir"
