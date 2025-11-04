#!/bin/bash
# Скрипт установки и настройки NVIDIA Container Toolkit для Docker Desktop в WSL2

set -e

# Флаг автоматического режима
AUTO_YES=false
if [[ "$1" == "--yes" || "$1" == "-y" ]]; then
    AUTO_YES=true
fi

echo "=== Установка NVIDIA Container Toolkit ==="
echo ""

# Проверка WSL2
if [ ! -f /proc/version ] || ! grep -qi microsoft /proc/version; then
    echo "⚠ Предупреждение: Похоже, что вы не в WSL2"
fi

# Проверка GPU
echo "1. Проверка GPU в WSL2..."
if ! command -v nvidia-smi > /dev/null 2>&1; then
    echo "❌ nvidia-smi не найден!"
    echo "   Убедитесь, что:"
    echo "   - Драйверы NVIDIA установлены на Windows"
    echo "   - Драйверы поддерживают WSL2"
    echo "   - WSL2 имеет доступ к GPU"
    exit 1
fi

if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ nvidia-smi не может получить доступ к GPU!"
    exit 1
fi

echo "✓ GPU доступен:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/   /'
echo ""

# Определение дистрибутива
echo "2. Определение дистрибутива..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
echo "   Дистрибутив: $distribution"
echo ""

# Добавление репозитория NVIDIA
echo "3. Добавление репозитория NVIDIA..."
if [ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
    echo "   Скачивание GPG ключа..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
fi

echo "   Добавление репозитория..."
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

# Обновление пакетов
echo "4. Обновление списка пакетов..."
sudo apt-get update

# Установка NVIDIA Container Toolkit
echo "5. Установка NVIDIA Container Toolkit..."
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "   ✓ NVIDIA Container Toolkit уже установлен"
    VERSION=$(dpkg -l | grep nvidia-container-toolkit | awk '{print $3}')
    echo "   Версия: $VERSION"
else
    sudo apt-get install -y nvidia-container-toolkit
    echo "   ✓ Установка завершена"
fi
echo ""

# Настройка Docker
echo "6. Настройка Docker..."
DOCKER_DAEMON_JSON="/etc/docker/daemon.json"
BACKUP_FILE="${DOCKER_DAEMON_JSON}.backup.$(date +%Y%m%d_%H%M%S)"

# Создание backup если файл существует
if [ -f "$DOCKER_DAEMON_JSON" ]; then
    echo "   Создание резервной копии существующего daemon.json..."
    sudo cp "$DOCKER_DAEMON_JSON" "$BACKUP_FILE"
    echo "   Backup: $BACKUP_FILE"
fi

# Проверка, есть ли уже настройки для nvidia
if [ -f "$DOCKER_DAEMON_JSON" ] && grep -q "nvidia" "$DOCKER_DAEMON_JSON"; then
    echo "   ⚠ NVIDIA runtime уже настроен в daemon.json"
    echo "   Содержимое daemon.json:"
    sudo cat "$DOCKER_DAEMON_JSON" | sed 's/^/   /'
    echo ""
    if [ "$AUTO_YES" = true ]; then
        REPLY="y"
        echo "   Автоматический режим: перезаписываем"
    else
        read -p "   Перезаписать? (y/N): " -n 1 -r
        echo ""
    fi
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Пропущено"
    else
        sudo tee "$DOCKER_DAEMON_JSON" > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
        echo "   ✓ daemon.json обновлён"
    fi
else
    echo "   Создание/обновление daemon.json..."
    sudo tee "$DOCKER_DAEMON_JSON" > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
    echo "   ✓ daemon.json настроен"
fi
echo ""

# Перезапуск Docker
echo "7. Перезапуск Docker..."
# Для Docker Desktop проверяем, есть ли systemctl
if command -v systemctl > /dev/null 2>&1 && systemctl is-active --quiet docker 2>/dev/null; then
    echo "   Перезапуск через systemctl..."
    sudo systemctl restart docker
    sleep 2
elif command -v service > /dev/null 2>&1; then
    echo "   Перезапуск через service..."
    sudo service docker restart
    sleep 2
else
    echo "   ⚠ Не удалось перезапустить Docker автоматически"
    echo "   Для Docker Desktop:"
    echo "   1. Откройте Docker Desktop"
    echo "   2. Settings > Docker Engine"
    echo "   3. Добавьте в JSON:"
    echo '      "runtimes": {'
    echo '        "nvidia": {'
    echo '          "path": "nvidia-container-runtime",'
    echo '          "runtimeArgs": []'
    echo '        }'
    echo '      }'
    echo "   4. Нажмите 'Apply & Restart'"
    echo ""
    if [ "$AUTO_YES" != true ]; then
        read -p "   Нажмите Enter после перезапуска Docker Desktop..."
    else
        echo "   ⚠ В автоматическом режиме: убедитесь что перезапустили Docker Desktop вручную"
        sleep 2
    fi
fi
echo ""

# Проверка установки
echo "8. Проверка установки..."
echo ""

# Проверка пакетов
echo "   Проверка установленных компонентов:"
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo -n "   ✓ nvidia-container-toolkit: "
    dpkg -l | grep nvidia-container-toolkit | awk '{print $3}'
else
    echo "   ❌ nvidia-container-toolkit не установлен"
fi

if which nvidia-container-runtime > /dev/null 2>&1; then
    echo "   ✓ nvidia-container-runtime: $(which nvidia-container-runtime)"
else
    echo "   ❌ nvidia-container-runtime не найден"
fi

if which nvidia-container-cli > /dev/null 2>&1; then
    echo "   ✓ nvidia-container-cli: $(which nvidia-container-cli)"
else
    echo "   ❌ nvidia-container-cli не найден"
fi
echo ""

# Проверка Docker runtime
echo "   Проверка Docker runtime:"
if docker info 2>/dev/null | grep -qi nvidia; then
    echo "   ✓ NVIDIA runtime доступен в Docker"
    docker info 2>/dev/null | grep -i nvidia | sed 's/^/     /'
else
    echo "   ⚠ NVIDIA runtime не найден в docker info"
    echo "     Это нормально для Docker Desktop, если настройки в GUI"
fi
echo ""

# Тестовая проверка
echo "9. Тестовая проверка GPU в контейнере..."
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "   ✓ Тест с --gpus all: OK"
    
    # Проверка с runtime
    if docker run --rm --runtime=nvidia nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        echo "   ✓ Тест с --runtime=nvidia: OK"
        echo ""
        echo "   Детальная информация:"
        docker run --rm --runtime=nvidia nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/     /'
    else
        echo "   ⚠ Тест с --runtime=nvidia: не работает"
        echo "     Используйте --gpus all вместо --runtime=nvidia"
    fi
else
    echo "   ❌ Тест не прошёл"
    echo "     Проверьте настройки Docker Desktop"
fi
echo ""

echo "=== Установка завершена! ==="
echo ""
echo "Следующие шаги:"
echo "1. Если используется Docker Desktop:"
echo "   - Откройте Docker Desktop > Settings > Docker Engine"
echo "   - Убедитесь, что там настроен NVIDIA runtime (см. выше)"
echo "   - Нажмите 'Apply & Restart'"
echo ""
echo "2. Проверьте установку:"
echo "   ./check-nvidia-docker.sh"
echo ""
echo "3. Запустите проект:"
echo "   cd /mnt/c/Users/nitro/tic-tac-toe-transform"
echo "   npm run docker:up"
echo ""

