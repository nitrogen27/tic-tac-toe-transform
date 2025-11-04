# Настройка CUDA для Docker на Windows

## Проблема

Docker Desktop на Windows не передает GPU напрямую в контейнеры через стандартный механизм `deploy.resources`. 

## Решение для Windows

### Вариант 1: Использовать WSL2 с GPU (рекомендуется)

1. **Установите WSL2** (если еще не установлен):
   ```powershell
   wsl --install
   ```

2. **Установите NVIDIA драйверы для WSL**:
   - Скачайте с https://developer.nvidia.com/cuda/wsl
   - Установите драйвер на Windows (не в WSL)

3. **В WSL2 запустите Docker**:
   ```bash
   # В WSL2
   cd /mnt/c/Users/nitro/tic-tac-toe-transform
   docker-compose up --build
   ```

### Вариант 2: Использовать прямой запуск в WSL2

Запустите проект напрямую в WSL2 без Docker:

```bash
# В WSL2
wsl
cd /mnt/c/Users/nitro/tic-tac-toe-transform
nvm use 18
npm install
npm start
```

### Вариант 3: Использовать CPU (текущая работа)

Проект работает на CPU, что достаточно для обучения небольших моделей.

## Текущий статус

- ✅ GPU доступен на хосте (RTX 3060)
- ❌ GPU недоступен в Docker контейнере на Windows
- ✅ Проект работает на CPU backend
- ✅ Производительность достаточна для обучения модели крестиков-ноликов

## Примечание

Для проекта крестики-нолики CPU backend полностью достаточен. GPU ускорение нужно только для больших моделей.



