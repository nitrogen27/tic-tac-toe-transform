# Скрипт для установки Visual Studio Build Tools через winget

Write-Host "Попытка установки Visual Studio Build Tools через winget..." -ForegroundColor Cyan
Write-Host ""

# Проверка наличия winget
$winget = Get-Command winget -ErrorAction SilentlyContinue
if (-not $winget) {
    Write-Host "✗ winget не найден. Используйте ручную установку." -ForegroundColor Red
    Write-Host ""
    Write-Host "Ручная установка:" -ForegroundColor Yellow
    Write-Host "1. Откройте: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Cyan
    Write-Host "2. Скачайте 'Build Tools for Visual Studio 2022'" -ForegroundColor White
    Write-Host "3. При установке выберите 'Desktop development with C++' workload" -ForegroundColor White
    Write-Host "4. После установки перезапустите терминал и выполните: npm install" -ForegroundColor White
    exit 1
}

Write-Host "✓ winget найден, начинаем установку..." -ForegroundColor Green
Write-Host ""

# Запуск установки (требует прав администратора)
Write-Host "Запускаю установку Visual Studio Build Tools..." -ForegroundColor Yellow
Write-Host "Это может занять несколько минут..." -ForegroundColor Yellow
Write-Host ""

try {
    # Установка через winget
    winget install --id Microsoft.VisualStudio.2022.BuildTools `
        --override "--wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --norestart" `
        --accept-package-agreements --accept-source-agreements
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Visual Studio Build Tools успешно установлены!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Следующие шаги:" -ForegroundColor Yellow
        Write-Host "1. Перезапустите терминал/IDE (обязательно!)" -ForegroundColor White
        Write-Host "2. Выполните: npm install" -ForegroundColor White
    } else {
        Write-Host ""
        Write-Host "⚠ Установка через winget не удалась. Используйте ручную установку." -ForegroundColor Yellow
        Write-Host "См. BUILD_SETUP.md для подробных инструкций." -ForegroundColor Yellow
    }
} catch {
    Write-Host ""
    Write-Host "✗ Ошибка при установке: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Используйте ручную установку:" -ForegroundColor Yellow
    Write-Host "https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Cyan
}



