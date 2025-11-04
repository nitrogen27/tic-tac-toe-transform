#!/bin/bash
# Скрипт проверки установки NVIDIA Container Toolkit

echo "=== Проверка установки NVIDIA Container Toolkit ==="
echo ""

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Функция для проверки с выводом результата
check() {
    local name=$1
    local command=$2
    
    echo -n "Проверка: $name... "
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((ERRORS++))
        return 1
    fi
}

warn() {
    local name=$1
    local message=$2
    
    echo -e "${YELLOW}⚠ ПРЕДУПРЕЖДЕНИЕ: $name${NC}"
    echo "   $message"
    ((WARNINGS++))
}

# Определение типа Docker (Desktop или daemon)
DOCKER_DESKTOP=false
if docker version 2>/dev/null | grep -qi "desktop\|cloud integration"; then
    DOCKER_DESKTOP=true
    echo -e "${YELLOW}ℹ Обнаружен Docker Desktop${NC}"
fi

# 1. Проверка установки nvidia-container-toolkit
echo "1. Проверка установки пакетов"
if dpkg -l | grep -q nvidia-container-toolkit; then
    check "nvidia-container-toolkit установлен" "dpkg -l | grep -q nvidia-container-toolkit"
    VERSION=$(dpkg -l | grep nvidia-container-toolkit | awk '{print $3}')
    echo "   Версия: $VERSION"
else
    echo -n "Проверка: nvidia-container-toolkit установлен... "
    if [ "$DOCKER_DESKTOP" = true ]; then
        echo -e "${YELLOW}⚠ Пропущено (Docker Desktop может работать без него)${NC}"
        ((WARNINGS++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((ERRORS++))
    fi
fi

if which nvidia-container-runtime > /dev/null 2>&1; then
    check "nvidia-container-runtime установлен" "which nvidia-container-runtime"
else
    echo -n "Проверка: nvidia-container-runtime установлен... "
    if [ "$DOCKER_DESKTOP" = true ]; then
        echo -e "${YELLOW}⚠ Пропущено (Docker Desktop использует свой механизм)${NC}"
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((ERRORS++))
    fi
fi

if which nvidia-container-cli > /dev/null 2>&1; then
    check "nvidia-container-cli установлен" "which nvidia-container-cli"
else
    echo -n "Проверка: nvidia-container-cli установлен... "
    if [ "$DOCKER_DESKTOP" = true ]; then
        echo -e "${YELLOW}⚠ Пропущено (Docker Desktop использует свой механизм)${NC}"
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((ERRORS++))
    fi
fi

# 2. Проверка конфигурации Docker
echo ""
echo "2. Проверка конфигурации Docker"
if [ -f /etc/docker/daemon.json ]; then
    check "daemon.json существует" "test -f /etc/docker/daemon.json"
    
    if grep -q "nvidia" /etc/docker/daemon.json; then
        echo -e "${GREEN}✓ NVIDIA runtime найден в daemon.json${NC}"
        echo "   Содержимое daemon.json:"
        cat /etc/docker/daemon.json | sed 's/^/   /'
    else
        echo -e "${YELLOW}⚠ NVIDIA runtime НЕ найден в daemon.json${NC}"
        if [ "$DOCKER_DESKTOP" = true ]; then
            echo "   (Docker Desktop может управлять GPU через свои настройки)"
        else
            echo "   (Это может быть проблемой для Docker daemon)"
            ((ERRORS++))
        fi
    fi
else
    if [ "$DOCKER_DESKTOP" = true ]; then
        echo -e "${YELLOW}⚠ Файл /etc/docker/daemon.json не существует${NC}"
        echo "   (Docker Desktop управляет настройками через GUI)"
    else
        echo -e "${RED}✗ Файл /etc/docker/daemon.json не существует${NC}"
        ((ERRORS++))
    fi
fi

# 3. Проверка Docker runtime
echo ""
echo "3. Проверка Docker runtime"
if command -v docker > /dev/null 2>&1; then
    if docker info > /dev/null 2>&1; then
        check "Docker работает" "docker info > /dev/null"
        
        # Проверка доступности nvidia runtime
        if docker info 2>/dev/null | grep -q "nvidia"; then
            echo -e "${GREEN}✓ NVIDIA runtime доступен в Docker${NC}"
        else
            echo -e "${YELLOW}⚠ NVIDIA runtime не найден в docker info${NC}"
            echo "   Возможно нужен перезапуск Docker"
            ((WARNINGS++))
        fi
    else
        echo -e "${RED}✗ Docker не работает (нужны права sudo?)${NC}"
        ((ERRORS++))
    fi
else
    echo -e "${RED}✗ Docker не установлен${NC}"
    ((ERRORS++))
fi

# 4. Проверка GPU в WSL2
echo ""
echo "4. Проверка GPU в WSL2"
if command -v nvidia-smi > /dev/null 2>&1; then
    if nvidia-smi > /dev/null 2>&1; then
        echo -e "${GREEN}✓ nvidia-smi работает${NC}"
        echo "   Информация о GPU:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/   /'
    else
        echo -e "${RED}✗ nvidia-smi не может получить доступ к GPU${NC}"
        echo "   Проверьте драйверы NVIDIA на Windows"
        ((ERRORS++))
    fi
else
    echo -e "${RED}✗ nvidia-smi не найден${NC}"
    echo "   Установите драйверы NVIDIA для WSL"
    ((ERRORS++))
fi

# 5. Тестовая проверка Docker с GPU
echo ""
echo "5. Проверка Docker с GPU (тестовый контейнер)"
if [ $ERRORS -eq 0 ] || [ $WARNINGS -lt 3 ]; then
    echo "Запуск тестового контейнера..."
    if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Тестовый контейнер с GPU работает${NC}"
        echo ""
        echo "Детальная информация из контейнера:"
        docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi | sed 's/^/   /'
    else
        echo -e "${RED}✗ Тестовый контейнер с GPU не работает${NC}"
        echo "   Попробуйте с --runtime=nvidia:"
        if docker run --rm --runtime=nvidia nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Работает с --runtime=nvidia${NC}"
        else
            echo -e "${RED}✗ Не работает даже с --runtime=nvidia${NC}"
            ((ERRORS++))
        fi
    fi
else
    warn "Тест Docker пропущен" "Слишком много ошибок"
fi

# Проверка критичности: если тестовый контейнер работает, то всё ОК
GPU_CONTAINER_WORKS=false
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    GPU_CONTAINER_WORKS=true
fi

# Итоговая сводка
echo ""
echo "=== Итоговая сводка ==="
if [ "$GPU_CONTAINER_WORKS" = true ]; then
    if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✓ Все проверки пройдены успешно!${NC}"
    else
        echo -e "${GREEN}✓ GPU в Docker работает!${NC}"
        if [ "$DOCKER_DESKTOP" = true ]; then
            echo "   Docker Desktop успешно использует GPU через WSL2 интеграцию"
        fi
        if [ $ERRORS -gt 0 ]; then
            echo "   Но есть некоторые предупреждения о конфигурации"
        fi
    fi
    echo ""
    echo "Вы можете запустить проект:"
    echo "  cd /mnt/c/Users/nitro/tic-tac-toe-transform"
    echo "  npm run docker:up"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Есть предупреждения, но основная функциональность может работать${NC}"
    echo "Количество предупреждений: $WARNINGS"
    exit 0
else
    echo -e "${RED}✗ Обнаружены проблемы${NC}"
    echo "Количество ошибок: $ERRORS"
    echo "Количество предупреждений: $WARNINGS"
    echo ""
    echo "Рекомендации:"
    if [ "$DOCKER_DESKTOP" = false ]; then
        echo "1. Установите nvidia-container-toolkit в WSL2"
        echo "2. Проверьте, что Docker перезапущен: sudo systemctl restart docker"
        echo "3. Проверьте daemon.json: cat /etc/docker/daemon.json"
    else
        echo "1. Проверьте настройки Docker Desktop: Settings > Resources > WSL Integration"
        echo "2. Убедитесь, что GPU доступен в WSL2: nvidia-smi"
    fi
    echo "3. Проверьте драйверы NVIDIA на Windows: nvidia-smi"
    exit 1
fi

