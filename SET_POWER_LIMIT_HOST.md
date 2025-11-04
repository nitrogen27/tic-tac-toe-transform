# Установка Power Limit GPU на хосте (85 Вт)

## ⚠️ ВАЖНО: Требуется выполнить на хосте (Windows/WSL)

Изменение power limit GPU **нельзя** сделать из Docker контейнера, даже с `privileged: true`. Это нужно сделать на хосте.

## Для Windows (WSL2)

### Вариант 1: Через WSL (если Docker запущен в WSL)

1. Откройте WSL терминал:
```bash
wsl
```

2. Установите power limit:
```bash
sudo nvidia-smi -pl 85
```

3. Включите persistence mode (если еще не включен):
```bash
sudo nvidia-smi -pm 1
```

### Вариант 2: Использовать скрипт на хосте

Если у вас Linux хост, используйте скрипт `setup-gpu-power-host.sh`:

```bash
sudo bash setup-gpu-power-host.sh
```

## Для Windows (нативный)

Если Docker Desktop запущен на Windows без WSL:

1. **Установите NVIDIA drivers** (если еще не установлены)
2. **Откройте PowerShell от имени администратора**
3. **Проверьте доступность nvidia-smi**:
   ```powershell
   nvidia-smi
   ```

4. **Если nvidia-smi доступен, установите power limit**:
   ```powershell
   nvidia-smi -pl 85
   ```

   ⚠️ **Примечание**: На Windows может потребоваться дополнительная настройка или драйверы могут не поддерживать изменение power limit через nvidia-smi.

## Проверка

После установки power limit проверьте:

```bash
docker exec tic-tac-toe-server bash -c "nvidia-smi --query-gpu=power.limit,power.max_limit,power.draw --format=csv,noheader"
```

Ожидаемые значения:
- **Power Limit**: 85 Вт (или максимальный доступный)
- **Power Draw при обучении**: 60-80 Вт

## Автоматическое применение при загрузке

### Для Linux (WSL):

Создайте systemd service или добавьте в `/etc/rc.local`:

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
ExecStart=/usr/bin/nvidia-smi -pl 85
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

Создайте задачу в Task Scheduler, которая запускает:
```
nvidia-smi -pl 85
```

## Текущий статус

- ✅ **Persistence Mode**: Включен (автоматически через entrypoint)
- ⚠️ **Power Limit**: Требуется установка на хосте (85 Вт)
- ✅ **Максимальный доступный**: 95 Вт

## Команды для быстрой проверки

```bash
# Проверить текущий power limit
docker exec tic-tac-toe-server bash -c "nvidia-smi --query-gpu=power.limit,power.max_limit --format=csv,noheader"

# Мониторинг во время обучения
watch -n 1 'docker exec tic-tac-toe-server bash -c "nvidia-smi --query-gpu=power.draw,power.limit,utilization.gpu --format=csv,noheader"'
```

## Если не получается установить power limit

1. **Проверьте права**: Нужны права администратора/root
2. **Проверьте модель GPU**: Некоторые модели имеют ограничения
3. **BIOS настройки**: Некоторые ноутбуки ограничивают GPU в BIOS
4. **Проверьте драйвер**: Обновите до последней версии NVIDIA

## Результат

После установки power limit до 85 Вт:
- ✅ GPU будет использовать 60-80 Вт при обучении
- ✅ Загрузка GPU увеличится до 70-90%
- ✅ Обучение будет проходить значительно быстрее

