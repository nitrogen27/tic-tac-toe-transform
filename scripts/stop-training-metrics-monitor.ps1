[CmdletBinding()]
param(
    [string]$Variant = "ttt5"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $repoRoot ".runtime\training-metrics\$Variant"
$pidPath = Join-Path $runtimeDir "monitor.pid"

$stoppedAny = $false

if (Test-Path $pidPath) {
    try {
        $pidValue = [int](Get-Content -Path $pidPath | Select-Object -First 1)
        $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($null -ne $proc) {
            Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
            $stoppedAny = $true
        }
    } finally {
        Remove-Item -Path $pidPath -Force -ErrorAction SilentlyContinue
    }
}

$stale = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and (
        $_.CommandLine -match [regex]::Escape("monitor_training_metrics.py") -and
        ($_.CommandLine -match [regex]::Escape("--variant '$Variant'") -or $_.CommandLine -match [regex]::Escape("--variant $Variant"))
    )
}
foreach ($proc in $stale) {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    $stoppedAny = $true
}

if (-not $stoppedAny) {
    Write-Host "No monitor process found for $Variant"
    exit 0
}

Write-Host "Training metrics monitor stopped for $Variant"
