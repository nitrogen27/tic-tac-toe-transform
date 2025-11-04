#!/bin/bash
# Docker entrypoint для сервера с оптимизацией GPU

echo "[Entrypoint] Optimizing GPU for maximum performance..."

# Проверяем доступность nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "[Entrypoint] NVIDIA GPU detected"
    
    # Проверяем текущий power limit
    CURRENT_PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1)
    MAX_PL=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits 2>/dev/null | head -1)
    
    echo "[Entrypoint] Current power limit: ${CURRENT_PL}W"
    echo "[Entrypoint] Maximum power limit: ${MAX_PL}W"
    
    # Пытаемся установить performance режим (persistence mode)
    # Это может не работать из контейнера, но попробуем
    if nvidia-smi -pm 1 &> /dev/null; then
        echo "[Entrypoint] ✓ GPU persistence mode enabled"
    else
        echo "[Entrypoint] ⚠ Could not enable persistence mode (may need host privileges)"
    fi
    
    # Пытаемся установить максимальный power limit
    # Обычно это требует прав на хосте, но попробуем
    if [ -n "$MAX_PL" ] && [ "$CURRENT_PL" != "$MAX_PL" ]; then
        if nvidia-smi -pl "$MAX_PL" &> /dev/null; then
            echo "[Entrypoint] ✓ Power limit set to ${MAX_PL}W"
        else
            echo "[Entrypoint] ⚠ Could not set power limit (may need host privileges)"
            echo "[Entrypoint] To set power limit on host, run: sudo nvidia-smi -pl ${MAX_PL}"
        fi
    else
        echo "[Entrypoint] Power limit already at maximum (${MAX_PL}W)"
    fi
    
    # Выводим информацию о GPU
    echo "[Entrypoint] GPU Info:"
    nvidia-smi --query-gpu=name,power.limit,power.max_limit,driver_version --format=csv,noheader 2>/dev/null || true
else
    echo "[Entrypoint] ⚠ nvidia-smi not available"
fi

echo "[Entrypoint] Starting server..."
exec "$@"

