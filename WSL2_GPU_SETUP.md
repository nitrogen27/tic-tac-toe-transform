# Настройка WSL2 с GPU для Docker

## Шаг 1: Проверка NVIDIA драйверов для WSL

NVIDIA драйверы должны быть установлены на Windows (не в WSL).

### Проверка на Windows:
```powershell
nvidia-smi
```

Если команда работает, драйверы установлены.

### Установка драйверов NVIDIA для WSL (если нужно):

1. Скачайте драйверы с https://developer.nvidia.com/cuda/wsl
2. Установите драйверы на Windows (не в WSL!)
3. Перезапустите компьютер

## Шаг 2: Проверка GPU в WSL2

```bash
# В WSL2
nvidia-smi
```

Должна показаться информация о GPU.

## Шаг 3: Установка NVIDIA Container Toolkit в WSL2

```bash
# В WSL2 (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Шаг 4: Настройка Docker daemon

В WSL2 Docker daemon должен быть настроен для использования NVIDIA runtime.

```bash
# В WSL2
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
EOF

sudo systemctl restart docker
```

## Шаг 5: Проверка

```bash
# В WSL2
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

Если команда выполняется без ошибок и показывает GPU, все настроено правильно.

## Шаг 6: Запуск проекта

```bash
# В WSL2
cd /mnt/c/Users/nitro/tic-tac-toe-transform
docker-compose up --build
```

## Примечания

- Docker Desktop должен использовать WSL2 backend
- Все команды Docker выполняются внутри WSL2
- GPU доступен только в контейнерах, запущенных из WSL2



