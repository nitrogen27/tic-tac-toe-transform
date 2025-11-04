# Скрипт проверки наличия Visual Studio Build Tools

Write-Host "Проверка окружения для сборки TensorFlow.js на Node.js v24..." -ForegroundColor Cyan
Write-Host ""

# Проверка Node.js
$nodeVersion = node -v
Write-Host "Node.js: $nodeVersion" -ForegroundColor Green

# Проверка Visual Studio Build Tools
Write-Host "`nПроверка Visual Studio Build Tools..." -ForegroundColor Yellow
$clPath = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clPath) {
    Write-Host "✓ Visual Studio Build Tools найдены!" -ForegroundColor Green
    Write-Host "  Путь: $($clPath.Source)" -ForegroundColor Gray
} else {
    Write-Host "✗ Visual Studio Build Tools НЕ найдены!" -ForegroundColor Red
    Write-Host "  Необходимо установить Visual Studio Build Tools 2022" -ForegroundColor Yellow
    Write-Host "  Скачать: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Cyan
    Write-Host "  Выберите: 'Desktop development with C++' workload" -ForegroundColor Yellow
}

# Проверка Python
Write-Host "`nПроверка Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python найден: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python НЕ найден!" -ForegroundColor Red
}

# Проверка CUDA (опционально)
Write-Host "`nПроверка CUDA (опционально)..." -ForegroundColor Yellow
$cudaPath = Get-Command nvcc.exe -ErrorAction SilentlyContinue
if ($cudaPath) {
    Write-Host "✓ CUDA Toolkit найден!" -ForegroundColor Green
    $cudaVersion = nvcc --version 2>&1 | Select-Object -First 1
    Write-Host "  $cudaVersion" -ForegroundColor Gray
} else {
    Write-Host "⚠ CUDA Toolkit не найден (GPU поддержка будет недоступна)" -ForegroundColor Yellow
    Write-Host "  Для GPU поддержки: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
}

Write-Host "`n" -ForegroundColor Cyan
Write-Host "После установки Build Tools:" -ForegroundColor Yellow
Write-Host "1. Перезапустите терминал/IDE" -ForegroundColor White
Write-Host "2. Выполните: npm install" -ForegroundColor White

