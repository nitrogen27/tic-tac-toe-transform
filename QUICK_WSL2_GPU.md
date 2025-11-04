# Быстрая настройка GPU в WSL2

## Шаг 1: Откройте WSL2

```bash
wsl
```

## Шаг 2: Перейдите в проект

```bash
cd /mnt/c/Users/nitro/tic-tac-toe-transform
```

## Шаг 3: Установите NVIDIA Container Toolkit

```bash
chmod +x setup-nvidia-docker.sh
./setup-nvidia-docker.sh
```

Или выполните вручную:

```bash
# Добавление репозитория
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Установка
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Настройка Docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

sudo systemctl restart docker
```

## Шаг 4: Проверка

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

Должна показаться информация о GPU.

## Шаг 5: Запуск проекта с GPU

```bash
# Используйте docker-compose.gpu.yml
docker-compose -f docker-compose.gpu.yml up --build
```

Или переименуйте файл:
```bash
cp docker-compose.gpu.yml docker-compose.yml
docker-compose up --build
```

## Проверка работы GPU

После запуска проверьте логи:

```bash
docker-compose logs server | grep -i "cuda\|gpu\|tensorflow"
```

Должны увидеть сообщения о успешной загрузке CUDA.



