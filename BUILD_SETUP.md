# Инструкция по установке для Node.js v24 с поддержкой CPU (x86) и CUDA

Для работы проекта на Node.js v24 необходимо собрать нативные модули TensorFlow.js из исходников.

## Требования

1. **Node.js v24.x** (уже установлен)
2. **Visual Studio Build Tools 2022** с компонентом "Desktop development with C++"
3. **Python 3.8-3.11** (уже установлен Python 3.12.0)
4. **CUDA Toolkit** (опционально, только для GPU поддержки)

## Установка Visual Studio Build Tools

1. Скачайте [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)

2. При установке выберите:
   - **Desktop development with C++** workload
   - Включает: MSVC v143, Windows 10/11 SDK, C++ CMake tools

3. После установки перезапустите терминал/IDE

## Установка CUDA Toolkit (для GPU поддержки)

1. Скачайте [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Установите совместимую версию (обычно CUDA 11.x или 12.x)
3. Убедитесь, что драйверы NVIDIA обновлены

## Установка зависимостей

После установки Visual Studio Build Tools выполните:

```bash
# Очистка предыдущих установок
npm cache clean --force
Remove-Item -Recurse -Force node_modules -ErrorAction SilentlyContinue
Remove-Item -Force package-lock.json -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force server/node_modules -ErrorAction SilentlyContinue
Remove-Item -Force server/package-lock.json -ErrorAction SilentlyContinue

# Установка зависимостей
npm install
```

## Проверка установки

После установки при запуске сервера вы должны увидеть:
- `[TFJS] Using tfjs-node-gpu backend (CUDA support)` - если CUDA доступен
- `[TFJS] Using tfjs-node backend (CPU/x86)` - если используется CPU
- `[TFJS] Platform: win32, Architecture: x64`

## Примечания

- Сборка из исходников может занять 10-30 минут в зависимости от скорости интернета и процессора
- Для работы GPU версии необходим NVIDIA GPU с поддержкой CUDA
- Проект автоматически определяет доступный backend (GPU/CPU)

