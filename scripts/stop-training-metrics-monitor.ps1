[CmdletBinding()]
param(
    [string]$Variant = "ttt5"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $repoRoot ".runtime\training-metrics\$Variant"
$pidPath = Join-Path $runtimeDir "monitor.pid"

if (-not (Test-Path $pidPath)) {
    Write-Host "No monitor pid file found for $Variant"
    exit 0
}

try {
    $pidValue = [int](Get-Content -Path $pidPath | Select-Object -First 1)
    $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
    }
} finally {
    Remove-Item -Path $pidPath -Force -ErrorAction SilentlyContinue
}

Write-Host "Training metrics monitor stopped for $Variant"
