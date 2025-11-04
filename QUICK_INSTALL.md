# Быстрая установка Visual Studio Build Tools

## Вариант 1: Автоматическая установка (через winget)

Откройте PowerShell **от имени администратора** и выполните:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --norestart" --accept-package-agreements --accept-source-agreements
```

Или используйте скрипт:

```powershell
powershell -ExecutionPolicy Bypass -File install-build-tools.ps1
```

## Вариант 2: Ручная установка

1. Откройте в браузере: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
2. Скачайте "Build Tools for Visual Studio 2022"
3. Запустите установщик
4. В выборе компонентов отметьте:
   - ✅ **Desktop development with C++**
   - ✅ **Windows 10/11 SDK** (будет автоматически выбран)
   - ✅ **MSVC v143** (будет автоматически выбран)
5. Нажмите "Install" и дождитесь завершения
6. Перезапустите терминал/IDE

## После установки

1. **Обязательно перезапустите терминал/IDE** (чтобы переменные окружения обновились)
2. Проверьте установку:
   ```powershell
   powershell -ExecutionPolicy Bypass -File check-build-tools.ps1
   ```
3. Установите зависимости проекта:
   ```powershell
   npm install
   ```

## Время установки

- Build Tools: ~5-10 минут (зависит от скорости интернета)
- Сборка TensorFlow.js: ~10-30 минут при первом `npm install`

## Проверка CUDA (опционально, для GPU)

Если у вас есть NVIDIA GPU и вы хотите использовать CUDA:

1. Проверьте наличие CUDA:
   ```powershell
   nvcc --version
   ```

2. Если CUDA не установлен:
   - Скачайте с https://developer.nvidia.com/cuda-downloads
   - Установите совместимую версию (обычно CUDA 11.x или 12.x)



