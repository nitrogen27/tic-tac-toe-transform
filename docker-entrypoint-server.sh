#!/bin/bash
# Docker entrypoint для сервера с оптимизацией GPU

echo "[Entrypoint] Optimizing GPU for maximum performance..."

# Проверяем доступность nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "[Entrypoint] NVIDIA GPU detected"

    # Проверяем текущий power limit
    CURRENT_PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1)
    MAX_PL=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits 2>/dev/null | head -1)
    TARGET_PL=${GPU_POWER_LIMIT:-$MAX_PL}

    echo "[Entrypoint] Current power limit: ${CURRENT_PL}W"
    echo "[Entrypoint] Maximum power limit: ${MAX_PL}W"
    echo "[Entrypoint] Target power limit: ${TARGET_PL}W"

    # Persistence mode — держит GPU инициализированным, убирает задержку на прогрев
    if nvidia-smi -pm 1 &> /dev/null; then
        echo "[Entrypoint] ✓ GPU persistence mode enabled"
    else
        echo "[Entrypoint] ⚠ Could not enable persistence mode (may need host privileges)"
    fi

    # Устанавливаем целевой power limit
    if [ -n "$TARGET_PL" ]; then
        if nvidia-smi -pl "$TARGET_PL" &> /dev/null; then
            echo "[Entrypoint] ✓ Power limit set to ${TARGET_PL}W"
        else
            echo "[Entrypoint] ⚠ Could not set power limit (may need host privileges)"
            echo "[Entrypoint] To set power limit on host, run: sudo nvidia-smi -pl ${TARGET_PL}"
        fi
    fi

    # Устанавливаем максимальные частоты GPU (performance mode)
    # Получаем максимальные поддерживаемые частоты
    MAX_GR_CLK=$(nvidia-smi --query-supported-clocks=gr --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    MAX_MEM_CLK=$(nvidia-smi --query-supported-clocks=mem --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$MAX_GR_CLK" ] && [ -n "$MAX_MEM_CLK" ]; then
        if nvidia-smi -ac "$MAX_MEM_CLK,$MAX_GR_CLK" &> /dev/null; then
            echo "[Entrypoint] ✓ Application clocks locked: GPU=${MAX_GR_CLK}MHz, MEM=${MAX_MEM_CLK}MHz"
        else
            echo "[Entrypoint] ⚠ Could not lock application clocks"
        fi
    fi

    # Выводим информацию о GPU
    echo "[Entrypoint] GPU Info:"
    nvidia-smi --query-gpu=name,power.limit,power.max_limit,clocks.gr,clocks.mem,driver_version --format=csv,noheader 2>/dev/null || true
else
    echo "[Entrypoint] ⚠ nvidia-smi not available"
fi

echo "[Entrypoint] Starting server..."
exec "$@"
