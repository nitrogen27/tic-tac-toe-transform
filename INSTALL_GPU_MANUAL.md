# Ручная установка NVIDIA Container Toolkit в WSL2

## Шаг 1: Откройте WSL2 терминал

```bash
wsl
```

## Шаг 2: Исправьте DNS (если нужно)

```bash
sudo bash -c "echo 'nameserver 8.8.8.8' > /etc/resolv.conf"
sudo bash -c "echo 'nameserver 8.8.4.4' >> /etc/resolv.conf"
```

## Шаг 3: Проверьте интернет

```bash
ping -c 2 google.com
ping -c 2 nvidia.github.io
```

Если ping не работает, перезапустите WSL2:
```powershell
# В PowerShell
wsl --shutdown
wsl
```

## Шаг 4: Установите зависимости

```bash
sudo apt-get update
sudo apt-get install -y curl gnupg lsb-release ca-certificates
```

## Шаг 5: Добавьте репозиторий NVIDIA

```bash
# Определите версию дистрибутива
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

# Добавьте GPG ключ
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Добавьте репозиторий
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

## Шаг 6: Установите NVIDIA Container Toolkit

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

## Шаг 7: Настройте Docker

```bash
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
```

## Шаг 8: Перезапустите Docker

```bash
sudo systemctl restart docker
# Или если systemctl не работает:
sudo service docker restart
```

## Шаг 9: Проверка установки

### Автоматическая проверка (рекомендуется)

```bash
chmod +x check-nvidia-docker.sh
./check-nvidia-docker.sh
```

Скрипт проверит:
- ✓ Установлен ли nvidia-container-toolkit
- ✓ Настроен ли Docker daemon.json
- ✓ Доступен ли GPU в WSL2
- ✓ Работает ли Docker с GPU

### Ручная проверка

1. **Проверка установки пакетов:**
```bash
dpkg -l | grep nvidia-container-toolkit
which nvidia-container-runtime
which nvidia-container-cli
```

2. **Проверка конфигурации Docker:**
```bash
cat /etc/docker/daemon.json
```

Должен содержать секцию `"runtimes": { "nvidia": ... }`

3. **Проверка Docker runtime:**
```bash
docker info | grep -i nvidia
```

4. **Проверка GPU в WSL2:**
```bash
nvidia-smi
```

Должна показаться информация о GPU.

5. **Тестовая проверка Docker с GPU:**
```bash
# Вариант 1 (современный синтаксис)
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Вариант 2 (старый синтаксис)
docker run --rm --runtime=nvidia nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

Должна показаться информация о GPU из контейнера.

## Шаг 10: Запустите проект

```bash
cd /mnt/c/Users/nitro/tic-tac-toe-transform
npm run docker:up
```

**Примечание:** Скрипт автоматически использует `--gpus all` для Docker Desktop, так как `runtime: nvidia` не поддерживается без nvidia-container-toolkit.

### Если возникает ошибка "unknown or invalid runtime name: nvidia"

Это означает, что Docker Desktop не поддерживает `runtime: nvidia` напрямую. Решение:
1. Используйте скрипт `docker-gpu-up.sh` (уже включен в `npm run docker:up`)
2. Или установите nvidia-container-toolkit в WSL2 по инструкции выше

## Альтернатива: Если интернет не работает

Если DNS/интернет не работает в WSL2:

1. **Перезапустите WSL2:**
   ```powershell
   wsl --shutdown
   wsl
   ```

2. **Или установите в PowerShell через WSL:**
   ```powershell
   wsl -e bash -c "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
   ```

3. **Проверьте настройки сети Windows**
   - Убедитесь, что Windows может подключиться к интернету
   - Проверьте файрвол
   - Проверьте VPN (может блокировать)


