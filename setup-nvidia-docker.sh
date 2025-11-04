#!/bin/bash
# Скрипт установки NVIDIA Container Toolkit в WSL2

set -e

echo "=== Установка NVIDIA Container Toolkit ==="

# Определение дистрибутива
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

echo "Дистрибутив: $distribution"

# Добавление репозитория NVIDIA
echo "Добавление GPG ключа..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

echo "Добавление репозитория..."
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Обновление пакетов
echo "Обновление списка пакетов..."
sudo apt-get update

# Установка NVIDIA Container Toolkit
echo "Установка NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

# Настройка Docker daemon
echo "Настройка Docker daemon..."
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

# Перезапуск Docker
echo "Перезапуск Docker..."
sudo systemctl restart docker

echo ""
echo "=== Проверка установки ==="
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "✓ Установка завершена!"
echo "Теперь можно запустить проект:"
echo "  cd /mnt/c/Users/nitro/tic-tac-toe-transform"
echo "  docker-compose up --build"



