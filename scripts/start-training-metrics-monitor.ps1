[CmdletBinding()]
param(
    [string]$Variant = "ttt5",
    [double]$Interval = 1.0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $repoRoot ".runtime\training-metrics\$Variant"
$pidPath = Join-Path $runtimeDir "monitor.pid"
$outPath = Join-Path $runtimeDir "latest.jsonl"
$monitorOutLogPath = Join-Path $runtimeDir "monitor.out.log"
$monitorErrLogPath = Join-Path $runtimeDir "monitor.err.log"
$pythonExe = (Get-Command python).Source

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

# Clean up any stale monitor processes for this variant before starting a new one.
$stale = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and (
        $_.CommandLine -match [regex]::Escape("monitor_training_metrics.py") -and
        ($_.CommandLine -match [regex]::Escape("--variant '$Variant'") -or $_.CommandLine -match [regex]::Escape("--variant $Variant"))
    )
}
foreach ($proc in $stale) {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 500

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

$startMarker = Get-Date
Remove-Item -Path $outPath -Force -ErrorAction SilentlyContinue
Remove-Item -Path $monitorOutLogPath -Force -ErrorAction SilentlyContinue
Remove-Item -Path $monitorErrLogPath -Force -ErrorAction SilentlyContinue

$proc = Start-Process -FilePath $pythonExe -ArgumentList @(
    (Join-Path $repoRoot "scripts\monitor_training_metrics.py"),
    "--variant", $Variant,
    "--interval", $Interval,
    "--out", $outPath
) -PassThru -WindowStyle Hidden -RedirectStandardOutput $monitorOutLogPath -RedirectStandardError $monitorErrLogPath

Set-Content -Path $pidPath -Value $proc.Id

$deadline = (Get-Date).AddSeconds([Math]::Max(6, [Math]::Ceiling($Interval * 3)))
$startedOk = $false
while ((Get-Date) -lt $deadline) {
    $liveProc = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($null -eq $liveProc) {
        break
    }

    if (Test-Path $outPath) {
        $outFile = Get-Item -Path $outPath -ErrorAction SilentlyContinue
        if ($null -ne $outFile -and $outFile.Length -gt 0 -and $outFile.LastWriteTime -gt $startMarker) {
            $startedOk = $true
            break
        }
    }

    Start-Sleep -Milliseconds 300
}

if (-not $startedOk) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $pidPath -Force -ErrorAction SilentlyContinue
    $logTail = ""
    if (Test-Path $monitorErrLogPath) {
        $logTail = (Get-Content -Path $monitorErrLogPath -Tail 20 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
    } elseif (Test-Path $monitorOutLogPath) {
        $logTail = (Get-Content -Path $monitorOutLogPath -Tail 20 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
    }
    throw "Training metrics monitor failed to start for $Variant. Logs: $monitorOutLogPath / $monitorErrLogPath`n$logTail"
}

Write-Host "Training metrics monitor started for $Variant"
Write-Host "PID:  $($proc.Id)"
Write-Host "JSONL: $outPath"
