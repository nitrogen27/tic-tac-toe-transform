# Скрипт для установки и сборки TensorFlow.js с правильным окружением

Write-Host "Настройка окружения для сборки TensorFlow.js..." -ForegroundColor Cyan

$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $vsPath)) {
    Write-Host "✗ Visual Studio Build Tools не найдены!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Build Tools найдены" -ForegroundColor Green

# Очистка
Write-Host "`nОчистка старых установок..." -ForegroundColor Yellow
Remove-Item -Recurse -Force "node_modules\@tensorflow" -ErrorAction SilentlyContinue

# Инициализация окружения и сборка
Write-Host "`nЗапуск сборки TensorFlow.js..." -ForegroundColor Yellow
Write-Host "Это может занять 10-30 минут..." -ForegroundColor Gray

$env:npm_config_python = "C:\Users\nitro\AppData\Local\Programs\Python\Python312\python.exe"

# Запуск через cmd для правильной инициализации окружения
$process = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$vsPath`" && npm install @tensorflow/tfjs-node@4.22.0 @tensorflow/tfjs-node-gpu@4.22.0" -NoNewWindow -Wait -PassThru

if ($process.ExitCode -eq 0) {
    Write-Host "`n✓ Установка завершена!" -ForegroundColor Green
    
    # Проверка файлов
    $cpuFile = "node_modules\@tensorflow\tfjs-node\lib\napi-v8\tfjs_binding.node"
    $gpuFile = "node_modules\@tensorflow\tfjs-node-gpu\lib\napi-v8\tfjs_binding.node"
    
    if (Test-Path $cpuFile) {
        $size = (Get-Item $cpuFile).Length
        Write-Host "✓ CPU модуль найден ($size байт)" -ForegroundColor Green
    } else {
        Write-Host "✗ CPU модуль не найден" -ForegroundColor Red
    }
    
    if (Test-Path $gpuFile) {
        $size = (Get-Item $gpuFile).Length
        Write-Host "✓ GPU модуль найден ($size байт)" -ForegroundColor Green
    } else {
        Write-Host "⚠ GPU модуль не найден (это нормально, если CUDA не установлен)" -ForegroundColor Yellow
    }
    
    Write-Host "`nПопытка загрузки TensorFlow.js..." -ForegroundColor Cyan
    node -e "import('@tensorflow/tfjs-node').then(tf => console.log('✓ Успешно! Backend:', tf.default.getBackend())).catch(e => console.error('✗ Ошибка:', e.message))"
} else {
    Write-Host "`n✗ Ошибка при установке (код выхода: $($process.ExitCode))" -ForegroundColor Red
    exit 1
}



