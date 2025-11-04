#!/bin/bash
# Скрипт помощи с настройкой NVIDIA runtime в Docker Desktop

echo "=== Настройка NVIDIA Runtime для Docker Desktop ==="
echo ""

echo "Для Docker Desktop настройки нужно добавить вручную через GUI."
echo ""
echo "Шаги:"
echo ""
echo "1. Откройте Docker Desktop"
echo ""
echo "2. Перейдите: Settings (⚙️) > Docker Engine"
echo ""
echo "3. Найдите или добавьте секцию 'runtimes' в JSON конфигурацию"
echo ""
echo "4. Добавьте следующее в JSON:"
echo ""
cat <<'EOF'
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
echo ""
echo "5. Если уже есть другие настройки, просто добавьте 'runtimes' секцию:"
echo ""
cat <<'EOF'
{
  "builder": {
    ...
  },
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  ...
}
EOF
echo ""
echo "6. Нажмите 'Apply & Restart'"
echo ""
echo "После перезапуска Docker Desktop проверьте:"
echo "  docker info | grep -i nvidia"
echo ""
echo "И тест:"
echo "  docker run --rm --runtime=nvidia nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi"
echo ""

# Проверка текущей конфигурации Docker Desktop
echo "=== Текущая конфигурация Docker Desktop ==="
DOCKER_CONFIG=$(docker info 2>/dev/null | grep -A 20 "Docker Root Dir" || echo "Не удалось получить информацию")

if echo "$DOCKER_CONFIG" | grep -qi nvidia; then
    echo "✓ NVIDIA runtime уже настроен в Docker Desktop!"
else
    echo "⚠ NVIDIA runtime не найден в конфигурации Docker Desktop"
    echo "  Выполните шаги выше для настройки"
fi

echo ""


