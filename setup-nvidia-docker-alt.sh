#!/bin/bash
# Альтернативный скрипт установки NVIDIA Container Toolkit
# Использует прямой способ установки

set -e

echo "=== Альтернативная установка NVIDIA Container Toolkit ==="

# Обновление пакетов
echo "Обновление списка пакетов..."
sudo apt-get update

# Установка зависимостей
echo "Установка зависимостей..."
sudo apt-get install -y curl gnupg lsb-release

# Попытка установки через прямой метод
echo "Попытка установки через альтернативный метод..."

# Метод 1: Использование прямого GPG ключа (если доступен)
if command -v curl &> /dev/null; then
    echo "Добавление GPG ключа через альтернативный источник..."
    # Используем публичный ключ NVIDIA напрямую
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | sudo apt-key add - || {
        echo "⚠️  Не удалось добавить ключ через curl, пробую альтернативный метод..."
        # Альтернативный метод - скачивание ключа вручную
        echo "Пожалуйста, добавьте репозиторий вручную или проверьте интернет-соединение"
    }
fi

# Метод 2: Прямая установка через apt если доступен репозиторий
echo "Проверка доступности репозитория..."
if sudo apt-cache search nvidia-container-toolkit | grep -q nvidia-container-toolkit; then
    echo "Репозиторий доступен, установка..."
    sudo apt-get install -y nvidia-container-toolkit
else
    echo "⚠️  Репозиторий недоступен. Нужно добавить его вручную."
    echo ""
    echo "Инструкция для ручной установки:"
    echo "1. Убедитесь, что интернет работает в WSL2"
    echo "2. Проверьте DNS: ping google.com"
    echo "3. Выполните команды из WSL2_GPU_SETUP.md"
    exit 1
fi

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
sudo systemctl restart docker || sudo service docker restart

echo ""
echo "=== Проверка установки ==="
docker run --rm --runtime=nvidia nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "✓ Установка завершена!"



