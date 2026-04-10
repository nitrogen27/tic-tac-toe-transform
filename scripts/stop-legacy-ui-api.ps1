[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $repoRoot ".runtime\legacy-ui-api"
$monitorStopScript = Join-Path $PSScriptRoot "stop-training-metrics-monitor.ps1"
$pidPaths = @(
    (Join-Path $runtimeDir "api.pid"),
    (Join-Path $runtimeDir "client.pid")
)

foreach ($pidPath in $pidPaths) {
    if (-not (Test-Path $pidPath)) {
        continue
    }

    try {
        $pidValue = [int](Get-Content -Path $pidPath -ErrorAction Stop | Select-Object -First 1)
        $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($null -ne $proc) {
            Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
        }
    } catch {
    } finally {
        Remove-Item -Path $pidPath -Force -ErrorAction SilentlyContinue
    }
}

if (Test-Path $monitorStopScript) {
    try {
        & $monitorStopScript -Variant "ttt5" | Out-Null
    } catch {
    }
}

Write-Host "Stopped legacy UI + new API processes tracked in $runtimeDir"
