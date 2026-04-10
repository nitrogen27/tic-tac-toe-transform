[CmdletBinding()]
param(
    [string]$Variant = "ttt5",
    [double]$Interval = 5.0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $repoRoot ".runtime\training-metrics\$Variant"
$pidPath = Join-Path $runtimeDir "monitor.pid"
$outPath = Join-Path $runtimeDir "latest.jsonl"

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

if (Test-Path $pidPath) {
    try {
        $pidValue = [int](Get-Content -Path $pidPath | Select-Object -First 1)
        $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($null -ne $proc) {
            throw "Monitor already running for $Variant (PID $pidValue)"
        }
    } catch {
        Remove-Item -Path $pidPath -Force -ErrorAction SilentlyContinue
    }
}

$cmd = @"
Set-Location '$repoRoot'
python scripts/monitor_training_metrics.py --variant '$Variant' --interval $Interval --out '$outPath'
"@

$proc = Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-Command", $cmd
) -PassThru -WindowStyle Hidden

Set-Content -Path $pidPath -Value $proc.Id

Write-Host "Training metrics monitor started for $Variant"
Write-Host "PID:  $($proc.Id)"
Write-Host "JSONL: $outPath"
