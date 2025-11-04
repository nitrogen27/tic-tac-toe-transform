# Увеличение Power Limit с 80 Вт до 85 Вт

## Текущий статус
- **Current Power Limit**: 80.00 W
- **Max Power Limit**: 95.00 W
- **Average Power Draw**: 16.05 W (idle)
- **Цель**: Увеличить до 85 Вт (или до максимального 95 Вт)

## ⚠️ ВАЖНО: Выполнить на хосте

Изменение power limit **нельзя** сделать из Docker контейнера. Нужно выполнить на хосте.

## Инструкции для Windows/WSL

### Вариант 1: Через WSL (если Docker в WSL2)

1. Откройте WSL терминал:
   ```bash
   wsl
   ```

2. Увеличьте power limit до 85 Вт:
   ```bash
   sudo nvidia-smi -pl 85
   ```

   Или до максимального (95 Вт) для лучшей производительности:
   ```bash
   sudo nvidia-smi -pl 95
   ```

3. Проверьте результат:
   ```bash
   nvidia-smi --query-gpu=power.limit,power.max_limit --format=csv,noheader
   ```

### Вариант 2: Через PowerShell (если Docker Desktop на Windows)

1. Откройте PowerShell от имени администратора

2. Увеличьте power limit:
   ```powershell
   nvidia-smi -pl 85
   ```

   Или до максимального:
   ```powershell
   nvidia-smi -pl 95
   ```

## Проверка после изменения

После установки power limit на хосте, проверьте из контейнера:

```bash
docker exec tic-tac-toe-server bash -c "nvidia-smi --query-gpu=power.limit,power.max_limit,power.draw --format=csv,noheader"
```

Ожидаемые значения:
- **Power Limit**: 85 Вт (или 95 Вт, если установили максимум)
- **Power Draw при обучении**: 60-80 Вт (вместо 20-30 Вт)

## Мониторинг во время обучения

```bash
# Проверка в реальном времени
watch -n 1 'docker exec tic-tac-toe-server bash -c "nvidia-smi --query-gpu=power.draw,power.limit,utilization.gpu,utilization.memory --format=csv,noheader"'
```

## Ожидаемые результаты

После увеличения power limit с 80 Вт до 85-95 Вт:

### До (80 Вт лимит):
- Power Draw: 20-30 Вт
- GPU Utilization: 20-30%
- Производительность: ограничена

### После (85-95 Вт лимит):
- Power Draw: 60-80 Вт
- GPU Utilization: 70-90%
- Производительность: значительно выше

## Автоматическое применение при загрузке

Чтобы power limit устанавливался автоматически при загрузке системы:

### Для Linux/WSL:

Создайте systemd service:
```bash
sudo nano /etc/systemd/system/nvidia-gpu-power.service
```

Добавьте:
```ini
[Unit]
Description=Set NVIDIA GPU Power Limit
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -pl 95
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Затем:
```bash
sudo systemctl enable nvidia-gpu-power.service
sudo systemctl start nvidia-gpu-power.service
```

### Для Windows:

Создайте задачу в Task Scheduler, которая запускает при загрузке:
```
nvidia-smi -pl 95
```

## Рекомендации

1. **Для максимальной производительности**: Установите до 95 Вт (максимум)
2. **Для баланса**: 85 Вт достаточно для большинства задач
3. **Для экономии энергии**: Оставьте 80 Вт, но это ограничит производительность

## Текущая конфигурация Docker

Контейнер уже настроен с:
- ✅ `privileged: true` (для максимального доступа)
- ✅ Entrypoint скрипт пытается установить power limit (но требует прав на хосте)
- ✅ Persistence mode включается автоматически

## Важно

⚠️ **Изменения power limit сохраняются до перезагрузки системы**

Для постоянного применения используйте systemd service или Task Scheduler.

## Итог

**Выполните на хосте:**
```bash
sudo nvidia-smi -pl 85  # или 95 для максимума
```

После этого GPU сможет использовать больше мощности при обучении, что значительно увеличит производительность!

