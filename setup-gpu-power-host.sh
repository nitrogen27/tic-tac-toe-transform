#!/bin/bash
# Скрипт для установки максимального power limit GPU на хосте
# Запускать с правами sudo на хосте (не в контейнере)

echo "=== GPU Power Limit Setup ==="
echo "This script sets the GPU power limit to maximum on the HOST system"
echo ""

# Проверяем доступность nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

# Проверяем права root
if [ "$EUID" -ne 0 ]; then 
    echo "ERROR: This script must be run as root (use sudo)"
    exit 1
fi

# Получаем информацию о GPU
echo "Current GPU status:"
nvidia-smi --query-gpu=name,power.limit,power.max_limit,power.min_limit --format=csv,noheader

echo ""
echo "Setting GPU to maximum performance..."

# Включаем persistence mode
if nvidia-smi -pm 1; then
    echo "✓ GPU persistence mode enabled"
else
    echo "⚠ Failed to enable persistence mode"
fi

# Устанавливаем максимальный power limit
MAX_PL=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | head -1)
if [ -n "$MAX_PL" ]; then
    echo "Setting power limit to ${MAX_PL}W (maximum)..."
    if nvidia-smi -pl "$MAX_PL"; then
        echo "✓ Power limit set to ${MAX_PL}W"
    else
        echo "⚠ Failed to set power limit"
    fi
else
    echo "⚠ Could not determine maximum power limit"
fi

echo ""
echo "Final GPU status:"
nvidia-smi --query-gpu=name,power.limit,power.max_limit,power.draw --format=csv,noheader

echo ""
echo "=== Done ==="
echo "Note: Power limit changes persist until reboot."
echo "To make permanent, add to /etc/rc.local or systemd service"

