# Исправление ограничения мощности GPU (20-30 Вт → 85 Вт)

## Проблема
GPU использует только 20-30 Вт вместо максимальных 85 Вт, что ограничивает производительность.

## Решение

### Вариант 1: На хосте (рекомендуется)

Запустите скрипт на хосте с правами sudo:
```bash
sudo bash setup-gpu-power-host.sh
```

Или вручную:
```bash
# Включить persistence mode
sudo nvidia-smi -pm 1

# Установить максимальный power limit (обычно 80-85 Вт для RTX 3060 Laptop)
sudo nvidia-smi -pl 85
```

### Вариант 2: Через Docker контейнер (требует privileged)

Контейнер уже настроен с `privileged: true` для изменения power limit из контейнера.

Entrypoint скрипт (`docker-entrypoint-server.sh`) автоматически:
1. Проверяет текущий power limit
2. Пытается установить максимальный power limit
3. Включает persistence mode

**Примечание**: Изменение power limit из контейнера может не работать на всех системах, рекомендуется делать это на хосте.

## Проверка

### Проверить текущий power limit:
```bash
docker exec tic-tac-toe-server bash -c "nvidia-smi --query-gpu=power.limit,power.max_limit,power.draw --format=csv,noheader"
```

### Проверить во время обучения:
```bash
watch -n 1 'docker exec tic-tac-toe-server bash -c "nvidia-smi --query-gpu=power.draw,power.limit,utilization.gpu --format=csv,noheader"'
```

### Ожидаемые значения:
- **Power Limit**: 80-85 Вт (максимум для RTX 3060 Laptop)
- **Power Draw при обучении**: 60-80 Вт
- **GPU Utilization**: 70-90%

## Автоматическое применение при запуске

### Сделать постоянным на хосте:

1. Создать systemd service:
```bash
sudo nano /etc/systemd/system/nvidia-gpu-power.service
```

2. Добавить содержимое:
```ini
[Unit]
Description=Set NVIDIA GPU Power Limit
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -pl 85
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

3. Включить и запустить:
```bash
sudo systemctl enable nvidia-gpu-power.service
sudo systemctl start nvidia-gpu-power.service
```

## Устранение неполадок

### Если power limit не меняется:

1. **Проверьте права**: Требуются права root/sudo
2. **Проверьте модель GPU**: Некоторые модели имеют ограничения
3. **Проверьте драйвер**: Обновите до последней версии
4. **BIOS настройки**: Некоторые ноутбуки ограничивают GPU в BIOS

### Проверка ограничений:
```bash
# Проверить все лимиты
nvidia-smi -q -d POWER | grep -A 10 "Power Limit"

# Проверить, что persistence mode включен
nvidia-smi -q | grep "Persistence Mode"
```

## Текущая конфигурация

- **Модель GPU**: RTX 3060 Laptop
- **Текущий power limit**: Проверьте через `nvidia-smi`
- **Максимальный power limit**: ~80-85 Вт (зависит от модели)
- **Docker**: `privileged: true` для изменения power limit из контейнера

## Важно

⚠️ **Безопасность**: `privileged: true` дает контейнеру полный доступ к хосту. Используйте только если доверяете контейнеру.

Для продакшена рекомендуется:
1. Установить power limit на хосте (скрипт `setup-gpu-power-host.sh`)
2. Убрать `privileged: true` из docker-compose
3. Использовать только необходимые capabilities:
   ```yaml
   cap_add:
     - SYS_ADMIN
   ```

## Результат

После установки максимального power limit:
- ✅ GPU будет использовать 60-80 Вт при обучении
- ✅ Загрузка GPU увеличится до 70-90%
- ✅ Обучение будет проходить быстрее
- ✅ Производительность значительно улучшится

