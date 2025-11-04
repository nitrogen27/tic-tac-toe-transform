# ⚠️ ВАЖНО: Проект настроен только для GPU

Проект теперь по умолчанию запускается с GPU конфигурацией.

## Требования для запуска:

1. **WSL2 должен быть установлен и запущен**
2. **NVIDIA Container Toolkit должен быть установлен в WSL2**
3. **Docker должен работать в WSL2**

## Установка NVIDIA Container Toolkit (если еще не установлен):

```bash
# В WSL2
cd /mnt/c/Users/nitro/tic-tac-toe-transform
chmod +x setup-nvidia-docker.sh
./setup-nvidia-docker.sh
```

## Запуск проекта:

### Вариант 1: Из WSL2 (рекомендуется)

```bash
# В WSL2
cd /mnt/c/Users/nitro/tic-tac-toe-transform
npm run docker:up
```

### Вариант 2: Из Windows (будет использовать WSL2 Docker)

```powershell
# В PowerShell
npm run docker:up
```

**Примечание:** Если Docker Desktop использует WSL2 backend, команды из Windows будут работать с GPU.

## Проверка GPU:

После запуска проверьте логи:

```bash
npm run docker:logs:server
```

Должны увидеть:
- ✅ Успешную загрузку CUDA библиотек
- ✅ `[TFJS] Using tfjs-node-gpu backend (CUDA support)`
- ✅ Нет ошибок о недоступности libcudart.so

## Если GPU недоступен:

Если увидите ошибки о недоступности CUDA, используйте CPU версию временно:

```bash
npm run docker:up:cpu
```

Но сначала **обязательно установите NVIDIA Container Toolkit в WSL2**!



