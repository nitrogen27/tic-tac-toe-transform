#!/bin/bash
# Скрипт запуска Docker Compose с поддержкой GPU для Docker Desktop

set -e

echo "=== Запуск проекта с GPU поддержкой ==="

# Сначала собираем образы
echo "Сборка образов..."
docker-compose -f docker-compose.gpu.yml build

# Получаем имя проекта (для сети и образов)
PROJECT_NAME=$(docker-compose -f docker-compose.gpu.yml config 2>/dev/null | grep -m1 "^name:" | awk '{print $2}' || basename "$(pwd)" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]-')

# Имя образа сервера (docker-compose создаёт образы без :latest)
IMAGE_NAME="${PROJECT_NAME}-server"

# Запускаем клиент (создаст сеть если нужно)
echo "Запуск клиента..."
docker-compose -f docker-compose.gpu.yml up -d client

# Получаем имя сети
NETWORK_NAME="${PROJECT_NAME}_default"
if ! docker network ls | grep -q "$NETWORK_NAME"; then
    # Если сети нет, создаём
    docker network create "$NETWORK_NAME" 2>/dev/null || true
fi

# Останавливаем старый сервер, если есть
if docker ps -a --format '{{.Names}}' | grep -q "^tic-tac-toe-server$"; then
    echo "Остановка старого сервера..."
    docker stop tic-tac-toe-server 2>/dev/null || true
    docker rm tic-tac-toe-server 2>/dev/null || true
fi

# Получаем полный путь к директории проекта
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)

# Запускаем сервер с GPU
echo "Запуск сервера с GPU..."
docker run -d \
    --name tic-tac-toe-server \
    --gpus all \
    -p 8080:8080 \
    -v "${PROJECT_DIR}/server/saved:/app/server/saved" \
    -e NODE_ENV=production \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e USE_GPU_BIG=1 \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    --restart unless-stopped \
    --network "$NETWORK_NAME" \
    "$IMAGE_NAME"

echo ""
echo "✓ Проект запущен!"
echo ""
echo "Проверка сервера:"
docker ps | grep tic-tac-toe

echo ""
echo "Логи сервера:"
echo "  npm run docker:logs:server"
echo ""
echo "Проверка GPU в контейнере:"
echo "  docker exec tic-tac-toe-server nvidia-smi"

