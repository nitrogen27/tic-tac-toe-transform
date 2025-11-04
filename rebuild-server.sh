#!/bin/bash
# Скрипт пересборки и перезапуска сервера

set -e

echo "=== Остановка сервера ==="
docker-compose -f docker-compose.gpu.yml stop server || true

echo ""
echo "=== Пересборка образа сервера ==="
echo "Это может занять несколько минут..."
docker-compose -f docker-compose.gpu.yml build server

echo ""
echo "=== Запуск сервера ==="
docker-compose -f docker-compose.gpu.yml up -d server

echo ""
echo "=== Проверка статуса ==="
docker ps | grep tic-tac-toe-server

echo ""
echo "✓ Готово! Проверьте логи:"
echo "  docker logs tic-tac-toe-server | tail -30"


