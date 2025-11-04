# Установка NVIDIA Container Toolkit - Пошаговая инструкция

## Выполните в WSL2 терминале (не через PowerShell):

```bash
cd /mnt/c/Users/nitro/tic-tac-toe-transform
./install-nvidia-toolkit.sh
```

Или выполните вручную:

## Шаг 1: Добавление GPG ключа
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

## Шаг 2: Добавление репозитория
```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

## Шаг 3: Обновление списка пакетов
```bash
sudo apt-get update
```

## Шаг 4: Установка NVIDIA Container Toolkit
```bash
sudo apt-get install -y nvidia-container-toolkit
```

## Шаг 5: Настройка Docker daemon.json
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

## Шаг 6: Настройка Docker Desktop (ВАЖНО!)

Для Docker Desktop нужно также добавить настройки в GUI:

1. Откройте **Docker Desktop**
2. Перейдите в **Settings** (⚙️) > **Docker Engine**
3. Добавьте в JSON конфигурацию:
```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```
4. Нажмите **"Apply & Restart"**

## Шаг 7: Перезапуск Docker (если используется Docker daemon)
```bash
sudo systemctl restart docker
# или
sudo service docker restart
```

## Шаг 8: Проверка установки
```bash
# Проверка пакетов
dpkg -l | grep nvidia-container-toolkit

# Проверка runtime
which nvidia-container-runtime

# Тест GPU в контейнере
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Тест с runtime
docker run --rm --runtime=nvidia nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## Шаг 9: Проверка проекта
```bash
cd /mnt/c/Users/nitro/tic-tac-toe-transform
./check-nvidia-docker.sh
```

## Если что-то не работает:

1. **Проверьте DNS в WSL2:**
```bash
ping -c 2 google.com
ping -c 2 nvidia.github.io
```

2. **Если интернет не работает, перезапустите WSL2:**
```powershell
# В PowerShell
wsl --shutdown
wsl
```

3. **Проверьте драйверы NVIDIA на Windows:**
```powershell
# В PowerShell
nvidia-smi
```


